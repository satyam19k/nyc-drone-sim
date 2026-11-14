"""Stage 4: drone camera paths. CPU only; must not import isaacsim.

One collision-free camera path per altitude, chosen deterministically from a fixed
candidate set and saved as plain data.

Conventions
- World: metres, x east, y north, z up.
- Yaw: heading of the view direction on the ground, degrees counter-clockwise from +x.
- camera_to_world: 4x4, column-vector convention (p_world = M @ p_cam), camera axes as in
  USD: +X right, +Y up, looking along -Z.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from .common import REPO_ROOT, altitude_m, load_config, scene_dir, slugify

log = logging.getLogger("stage4")


# ---------------------------------------------------------------------------
# Pinhole camera
# ---------------------------------------------------------------------------
class Pinhole:
    def __init__(self, cam_cfg: dict):
        self.f_mm = float(cam_cfg["focal_length_mm"])
        self.ha_mm = float(cam_cfg["horizontal_aperture_mm"])
        self.va_mm = float(cam_cfg["vertical_aperture_mm"])
        self.width, self.height = (int(v) for v in cam_cfg["resolution"])
        self.pitch_deg = float(cam_cfg["pitch_deg"])

    @property
    def fov_deg(self) -> tuple[float, float]:
        return (math.degrees(2 * math.atan(self.ha_mm / (2 * self.f_mm))),
                math.degrees(2 * math.atan(self.va_mm / (2 * self.f_mm))))

    @property
    def tan_half(self) -> tuple[float, float]:
        return self.ha_mm / (2 * self.f_mm), self.va_mm / (2 * self.f_mm)

    def intrinsics(self) -> dict:
        fx = self.f_mm / self.ha_mm * self.width
        fy = self.f_mm / self.va_mm * self.height
        return {"focal_length_mm": self.f_mm, "horizontal_aperture_mm": self.ha_mm,
                "vertical_aperture_mm": self.va_mm, "width": self.width, "height": self.height,
                "fx": fx, "fy": fy, "cx": self.width / 2.0, "cy": self.height / 2.0,
                "pitch_deg": self.pitch_deg}


def camera_to_world(pos, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])
    r = np.array([math.sin(y), -math.cos(y), 0.0])
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, u, -f, pos
    return m


def project(m: np.ndarray, cam: Pinhole, pw) -> tuple[float, float, bool]:
    """World point -> normalised image coords in [-1, 1] and an in-front flag."""
    pc = np.linalg.inv(m) @ np.array([pw[0], pw[1], pw[2], 1.0])
    if pc[2] >= 0:
        return float("nan"), float("nan"), False
    th, tv = cam.tan_half
    return float(pc[0] / -pc[2] / th), float(pc[1] / -pc[2] / tv), True


def frustum_ground(m: np.ndarray, cam: Pinhole) -> np.ndarray | None:
    """Ground (z = 0) hits of the four corner rays, or None if any ray misses the ground."""
    th, tv = cam.tan_half
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        d = m[:3, :3] @ np.array([sx * th, sy * tv, -1.0])
        if d[2] >= -1e-9:
            return None
        t = -m[2, 3] / d[2]
        out.append(m[:2, 3] + t * d[:2])
    return np.array(out)


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------
class Obstacles:
    """2 m grid of buildings taller than (altitude - vertical clearance), dilated laterally."""

    def __init__(self, feats: list[dict], alt_m: float, cfg: dict):
        self.R = float(cfg["scene_radius_m"])
        self.res = float(cfg["paths"]["grid_res_m"])
        self.half = self.R + float(cfg["paths"]["grid_margin_m"])
        self.lateral = float(cfg["clearance_m"]["lateral"])
        limit = alt_m - float(cfg["clearance_m"]["vertical"])
        self.tall = [shape(f["geometry"]) for f in feats if f["properties"]["height_m"] > limit]
        n = int(round(2 * self.half / self.res))
        occ = np.zeros((n, n), dtype=bool)
        for g in self.tall:
            minx, miny, maxx, maxy = g.bounds
            x0 = max(int(math.floor((minx + self.half) / self.res)) - 1, 0)
            x1 = min(int(math.ceil((maxx + self.half) / self.res)) + 1, n)
            y0 = max(int(math.floor((miny + self.half) / self.res)) - 1, 0)
            y1 = min(int(math.ceil((maxy + self.half) / self.res)) + 1, n)
            xs = (np.arange(x0, x1) + 0.5) * self.res - self.half
            ys = (np.arange(y0, y1) + 0.5) * self.res - self.half
            X, Y = np.meshgrid(xs, ys)
            # a cell is occupied if the footprint comes within half a cell diagonal of its centre
            near = shapely.dwithin(g, shapely.points(X, Y), self.res * math.sqrt(0.5))
            occ[y0:y1, x0:x1] |= near
        self.occupied = occ
        dist = ndimage.distance_transform_edt(~occ) * self.res if occ.any() else np.full(occ.shape, np.inf)
        self.blocked = dist <= self.lateral
        self.tree = STRtree(self.tall) if self.tall else None

    def cell(self, xy) -> tuple[int, int]:
        return (int(math.floor((xy[1] + self.half) / self.res)),
                int(math.floor((xy[0] + self.half) / self.res)))

    def is_blocked(self, xy) -> bool:
        i, j = self.cell(xy)
        n = self.blocked.shape[0]
        if not (0 <= i < n and 0 <= j < n):
            return True
        return bool(self.blocked[i, j])

    def clearance(self, xy) -> float:
        """Exact horizontal distance to the nearest tall footprint."""
        if self.tree is None:
            return float("inf")
        pt = Point(float(xy[0]), float(xy[1]))
        return float(pt.distance(self.tall[self.tree.nearest(pt)]))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
def frame_times(cfg: dict) -> np.ndarray:
    return np.arange(int(cfg["clip"]["frames"])) / float(cfg["clip"]["fps"])


def flyover(heading_deg: float, alt: float, speed: float, t: np.ndarray, pitch_deg: float):
    """Straight line; the image centre meets the target at the middle frame."""
    h = math.radians(heading_deg)
    d = np.array([math.cos(h), math.sin(h)])
    lead = alt / math.tan(math.radians(-pitch_deg))   # horizontal distance camera -> image centre
    tm = t[len(t) // 2]
    xy = -lead * d + np.outer(speed * (t - tm), d)
    return xy, np.full(len(t), heading_deg)


def orbit(radius: float, mid_angle_deg: float, direction: int, speed: float, t: np.ndarray):
    """Arc around the target, camera yawed toward the target."""
    tm = t[len(t) // 2]
    ang = math.radians(mid_angle_deg) + direction * (speed / radius) * (t - tm)
    xy = radius * np.column_stack([np.cos(ang), np.sin(ang)])
    return xy, np.degrees(ang) + 180.0


def hover_pan(xy0, t: np.ndarray, sweep_deg: float):
    """Fixed position, yaw sweeping across the direction to the target."""
    to_target = math.degrees(math.atan2(-xy0[1], -xy0[0]))
    s = (t - t[0]) / (t[-1] - t[0]) - 0.5
    return np.tile(np.asarray(xy0, float), (len(t), 1)), to_target + sweep_deg * s


def evaluate(xy, yaw, alt, cam: Pinhole, obs: Obstacles, t, R: float, max_yaw_rate: float):
    """Returns None if the candidate breaks a hard constraint, else its score parts."""
    if any(obs.is_blocked(p) for p in xy):
        return None
    # also test the midpoints between frames so a path cannot hop over a thin obstacle
    if any(obs.is_blocked(p) for p in 0.5 * (xy[:-1] + xy[1:])):
        return None
    yaw_rate = np.abs(np.diff(yaw)) / np.diff(t)
    if len(yaw_rate) and yaw_rate.max() > max_yaw_rate:
        return None
    visible = 0
    for p, y in zip(xy, yaw):
        m = camera_to_world([p[0], p[1], alt], y, cam.pitch_deg)
        g = frustum_ground(m, cam)
        if g is None or np.hypot(g[:, 0], g[:, 1]).max() > R:
            return None
        u, v, front = project(m, cam, (0.0, 0.0, 0.0))
        visible += int(front and abs(u) <= 1 and abs(v) <= 1)
    clearance = min(obs.clearance(p) for p in xy)
    return {"min_clearance_m": clearance, "target_visible_frames": visible}


def plan(feats: list[dict], alt_ft: float, cfg: dict) -> dict:
    cam = Pinhole(cfg["camera"])
    alt = altitude_m(alt_ft)
    speed = float(cfg["speed_mps"][int(alt_ft)])
    R = float(cfg["scene_radius_m"])
    t = frame_times(cfg)
    obs = Obstacles(feats, alt, cfg)
    n = len(t)
    pc = cfg["paths"]
    max_yaw = float(pc["max_yaw_rate_deg_s"])
    sweep = float(pc["hover_pan_sweep_deg"])

    cands = []   # fixed generation order is the tie-break order
    for h in range(0, 360, int(pc["flyover_heading_step_deg"])):
        cands.append(("flyover", {"heading_deg": float(h)}, flyover(h, alt, speed, t, cam.pitch_deg)))
    for r in (float(v) for v in pc["orbit_radii_m"]):
        for a in range(0, 360, int(pc["orbit_angle_step_deg"])):
            for d in (1, -1):
                cands.append(("orbit", {"radius_m": r, "mid_angle_deg": float(a), "direction": d},
                              orbit(r, a, d, speed, t)))

    survivors = []
    for order, (kind, params, (xy, yaw)) in enumerate(cands):
        ev = evaluate(xy, yaw, alt, cam, obs, t, R, max_yaw)
        if ev is None or ev["target_visible_frames"] < float(pc["min_target_visible_share"]) * n:
            continue
        survivors.append((-round(ev["min_clearance_m"], 1), -ev["target_visible_frames"], order,
                          kind, params, xy, yaw, ev))
    n_candidates = len(cands)

    if survivors:
        survivors.sort(key=lambda s: s[:3])
        _, _, order, kind, params, xy, yaw, ev = survivors[0]
    else:
        # hover_pan: free cell nearest the target whose whole sweep keeps the frustum in the scene
        free = np.argwhere(~obs.blocked)
        cxy = np.column_stack([(free[:, 1] + 0.5) * obs.res - obs.half,
                               (free[:, 0] + 0.5) * obs.res - obs.half])
        idx = np.lexsort((cxy[:, 1], cxy[:, 0], np.round(np.hypot(cxy[:, 0], cxy[:, 1]), 6)))
        chosen = None
        for i in idx:
            xy, yaw = hover_pan(cxy[i], t, sweep)
            ev = evaluate(xy, yaw, alt, cam, obs, t, R, max_yaw)
            if ev is not None:
                chosen = (xy, yaw, ev, cxy[i])
                break
        if chosen is None:
            raise RuntimeError(f"no feasible camera path at {alt_ft} ft")
        xy, yaw, ev, p0 = chosen
        kind, params, order = "hover_pan", {"position": [float(p0[0]), float(p0[1])],
                                           "sweep_deg": sweep}, -1

    near = [f for f in feats if shape(f["geometry"]).distance(Point(0, 0)) <= float(pc["taller_share_radius_m"])]
    taller = sum(f["properties"]["height_m"] > alt for f in near)
    frames = []
    for i in range(n):
        m = camera_to_world([xy[i, 0], xy[i, 1], alt], float(yaw[i]), cam.pitch_deg)
        frames.append({"index": i, "t": float(t[i]), "position": [float(xy[i, 0]), float(xy[i, 1]), alt],
                       "yaw_deg": float(yaw[i]), "camera_to_world": m.tolist()})
    return {
        "altitude_ft": float(alt_ft), "altitude_m": alt, "speed_mps": speed if kind != "hover_pan" else 0.0,
        "path_type": kind, "path_params": params,
        "clip": {"frames": n, "fps": float(cfg["clip"]["fps"]), "duration_s": n / float(cfg["clip"]["fps"])},
        "intrinsics": cam.intrinsics(),
        "conventions": {"units": "metres", "axes": "x east, y north, z up",
                        "camera_to_world": "column-vector 4x4; camera +X right, +Y up, looks along -Z",
                        "yaw_deg": "view heading on the ground, counter-clockwise from +x"},
        "planner": {"candidates": n_candidates, "survivors": len(survivors), "tie_break_order": order,
                    "min_clearance_m": ev["min_clearance_m"],
                    "target_visible_frames": ev["target_visible_frames"],
                    "obstacle_buildings": len(obs.tall), "grid_res_m": obs.res},
        "taller_share_within_300m": (taller / len(near)) if near else 0.0,
        "taller_count_within_300m": taller, "buildings_within_300m": len(near),
        "seed": cfg["seed"],
        "frames": frames,
    }


def path_file(sdir: Path, alt_ft) -> Path:
    return sdir / f"path_{int(alt_ft)}.json"


def run_stage4(location: str, cfg: dict, out_dir: Path | None = None) -> list[Path]:
    slug = slugify(location)
    sdir = out_dir or scene_dir(cfg, slug)
    feats = json.loads((scene_dir(cfg, slug) / "buildings.geojson").read_text())["features"]
    outs = []
    for alt_ft in cfg["altitudes_ft"]:
        p = plan(feats, alt_ft, cfg)
        p["location"] = slug
        out = path_file(sdir, alt_ft)
        out.write_text(json.dumps(p, indent=1) + "\n")
        log.info("%s ft: %s %s, clearance %.1f m, target visible %d/%d frames, %d/%d candidates survive",
                 alt_ft, p["path_type"], p["path_params"], p["planner"]["min_clearance_m"],
                 p["planner"]["target_visible_frames"], p["clip"]["frames"],
                 p["planner"]["survivors"], p["planner"]["candidates"])
        outs.append(out)
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: drone camera paths")
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_stage4(a.location, load_config(a.config))


if __name__ == "__main__":
    main()
