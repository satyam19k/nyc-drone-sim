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
from scipy import ndimage, sparse
from scipy.sparse import csgraph
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

from .common import (REPO_ROOT, altitude_m, clip_settings, extent_polygon, for_location, load_config, scene_dir,
                     slugify, variant_suffix, view_spec)

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

    def __init__(self, feats: list[dict], alt_m: float, cfg: dict, limit: float | None = None):
        self.R = float(cfg["scene_radius_m"])
        self.res = float(cfg["paths"]["grid_res_m"])
        self.half = self.R + float(cfg["paths"]["grid_margin_m"])
        self.lateral = float(cfg["clearance_m"]["lateral"])
        if limit is None:
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
        self.dist = dist
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
def frame_times(cfg: dict, variant: str | None = None) -> np.ndarray:
    frames, fps = clip_settings(cfg, variant)
    return np.arange(frames) / fps


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

    return path_record(feats, alt_ft, cfg, kind, params, xy, yaw, t, ev, obs, cam,
                       {"candidates": n_candidates, "survivors": len(survivors), "tie_break_order": order})


# ---------------------------------------------------------------------------
# Street routes: long clips that follow the street corridors for several blocks
# ---------------------------------------------------------------------------
class StreetNetwork:
    """Shortest routes along the streets that are flyable at one altitude.

    A street cell lies in the gap between building footprints of any height and is clear of the
    altitude's obstacles. Edge cost grows toward the footprints, so routes keep to the street
    centre line. At a high altitude the camera therefore still follows streets, above the roofs.
    """

    def __init__(self, feats: list[dict], alt_m: float, cfg: dict):
        sc = cfg["street"]
        self.obs = obs = Obstacles(feats, alt_m, cfg)
        ground = Obstacles(feats, alt_m, cfg, limit=float("-inf")).dist    # distance to any footprint
        n = obs.blocked.shape[0]
        c = (np.arange(n) + 0.5) * obs.res - obs.half
        X, Y = np.meshgrid(c, c)
        free = ~obs.blocked & (ground >= float(sc["min_half_width_m"])) & (np.hypot(X, Y) <= obs.R)
        ids = -np.ones((n, n), dtype=np.int64)
        ids[free] = np.arange(int(free.sum()))
        self.cell_xy = np.column_stack([X[free], Y[free]])
        self.cell_clear = ground[free]
        cap = float(sc["centre_cap_m"])
        # room around a cell: to the nearest footprint, and to the edge of the altitude's no-fly zone
        room = np.minimum(ground, obs.dist - obs.lateral)
        w = (cap / np.clip(room, obs.res, cap)) ** 2        # 1 mid-street, larger toward either edge
        rows, cols, vals = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            sa = (slice(0, n - di), slice(max(0, -dj), n - max(0, dj)))
            sb = (slice(di, n), slice(max(0, dj), n + min(0, dj)))
            ok = (ids[sa] >= 0) & (ids[sb] >= 0)
            rows.append(ids[sa][ok]); cols.append(ids[sb][ok])
            vals.append(obs.res * math.hypot(di, dj) * 0.5 * (w[sa][ok] + w[sb][ok]))
        m = len(self.cell_xy)
        self.graph = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))
        # one node per block of node_spacing_m: its free cell farthest from any building
        block = float(sc["node_spacing_m"])
        key = np.floor((self.cell_xy + obs.half) / block).astype(np.int64)
        order = np.lexsort((np.arange(m), -np.round(self.cell_clear, 6), key[:, 1], key[:, 0]))
        first = np.ones(m, dtype=bool)
        first[1:] = np.any(key[order][1:] != key[order][:-1], axis=1)
        self.nodes = np.sort(order[first])

    def routes(self, min_len: float, max_len: float):
        """Yield (i, j, xy) for every node pair whose shortest route has a usable length."""
        for a, src in enumerate(self.nodes):
            cost, pred = csgraph.dijkstra(self.graph, directed=False, indices=int(src), return_predecessors=True)
            for b in range(a + 1, len(self.nodes)):
                dst = int(self.nodes[b])
                if not np.isfinite(cost[dst]) or cost[dst] < min_len:
                    continue
                chain = [dst]
                while chain[-1] != src:
                    chain.append(int(pred[chain[-1]]))
                xy = self.cell_xy[chain[::-1]]
                length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
                if min_len <= length <= max_len:
                    yield a, b, np.asarray(chain[::-1])


def corner_reach(xy: np.ndarray, yaw_deg: np.ndarray, alt: float, cam: Pinhole) -> np.ndarray:
    """Per frame, the largest distance from the origin of the four frustum corner ground hits."""
    th, tv = cam.tan_half
    p = math.radians(cam.pitch_deg)
    y = np.radians(yaw_deg)
    fwd = np.column_stack([np.cos(y), np.sin(y)])
    right = np.column_stack([np.sin(y), -np.cos(y)])
    far = np.zeros(len(xy))
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        # camera-frame ray (sx*th, sy*tv, -1): forward part cos/sin of pitch, up part from sy
        horiz = math.cos(p) - sy * tv * math.sin(p)
        down = -math.sin(p) - sy * tv * math.cos(p)
        if down <= 1e-9:
            return np.full(len(xy), np.inf)
        t = alt / down
        g = xy + t * horiz * fwd + t * sx * th * right
        far = np.maximum(far, np.hypot(g[:, 0], g[:, 1]))
    return far


def smooth_route(xy: np.ndarray, sigma_m: float, step: float, yaw_sigma_m: float):
    """Densify to a fixed step, smooth, and return points, arc length and heading (deg, unwrapped).

    The heading is smoothed further than the track so the view does not swing with every bend.
    """
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    u = np.arange(0.0, s[-1], step)
    d = np.column_stack([np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1])])
    d = ndimage.gaussian_filter1d(d, sigma_m / step, axis=0, mode="nearest")
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(d, axis=0).T))])
    g = np.gradient(d, axis=0)
    head = np.degrees(np.unwrap(np.arctan2(g[:, 1], g[:, 0])))
    return d, arc, ndimage.gaussian_filter1d(head, yaw_sigma_m / step, mode="nearest")


def plan_street(feats: list[dict], alt_ft: float, cfg: dict) -> dict:
    sc, pc = cfg["street"], cfg["paths"]
    cam = Pinhole(cfg["camera"])
    alt = altitude_m(alt_ft)
    speed = float(cfg["speed_mps"][int(alt_ft)])
    R = float(cfg["scene_radius_m"])
    t = frame_times(cfg, "street")
    n = len(t)
    length = speed * (t[-1] - t[0])
    net = StreetNetwork(feats, alt, cfg)
    obs = net.obs
    lead = alt / math.tan(math.radians(-cam.pitch_deg))
    res = net.obs.res
    win = int(math.ceil(length / res)) + 1          # cells that surely span the clip length
    step = max(1, int(round(float(sc["window_step_m"]) / res)))
    span = max(2, int(round(float(sc["tangent_baseline_m"]) / res)))
    tbin, dbin = float(sc["target_bin_m"]), float(sc["detour_bin"])

    # Rank every window of every route: image centre closest to the target, then straightest,
    # then widest street.
    cands = []
    routes = {}
    for order, (a, b, chain) in enumerate(net.routes(length * 1.1, length * float(sc["max_route_factor"]))):
        routes[(a, b)] = chain
        for rev in (0, 1):
            ch = chain[::-1] if rev else chain
            if len(ch) <= win:
                continue
            xy = net.cell_xy[ch]
            tan = xy[np.minimum(np.arange(len(xy)) + span, len(xy) - 1)] - xy[np.maximum(np.arange(len(xy)) - span, 0)]
            tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-9)[:, None]
            look = np.hypot(*(xy + lead * tan).T)
            clear = net.cell_clear[ch]
            run = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
            for i0 in range(0, len(ch) - win, step):
                i1 = i0 + win - 1
                detour = 1.0 - float(np.hypot(*(xy[i1] - xy[i0]))) / (run[i1] - run[i0])
                cands.append((int(look[i0:i0 + win].min() // tbin), int(detour // dbin),
                              -round(float(clear[i0:i0 + win].min()), 1), order, rev, i0, a, b))
    cands.sort()
    chosen = None
    tried = 0
    smoothed = {}
    for _, _, _, order, rev, i0, a, b in cands:
        chain = routes[(a, b)]
        xy0 = net.cell_xy[chain[::-1] if rev else chain]
        start = float(np.hypot(*np.diff(xy0[:i0 + 1], axis=0).T).sum()) if i0 else 0.0
        for sigma in (float(v) for v in sc["smooth_sigma_m"]):
            tried += 1
            if (a, b, rev, sigma) not in smoothed:
                if len(smoothed) > 64:
                    smoothed.clear()
                smoothed[(a, b, rev, sigma)] = smooth_route(xy0, sigma, float(sc["dense_step_m"]),
                                                             float(sc["yaw_sigma_m"]))
            d, arc, head = smoothed[(a, b, rev, sigma)]
            s = start + speed * (t - t[0])
            if s[-1] > arc[-1]:
                break
            xy = np.column_stack([np.interp(s, arc, d[:, 0]), np.interp(s, arc, d[:, 1])])
            yaw = np.interp(s, arc, head)
            if np.abs(np.diff(yaw) / np.diff(t)).max() > float(pc["max_yaw_rate_deg_s"]):
                continue
            if corner_reach(xy, yaw, alt, cam).max() > R:
                break
            ev = evaluate(xy, yaw, alt, cam, obs, t, R, float(pc["max_yaw_rate_deg_s"]))
            if ev is None:
                continue
            chosen = (xy, yaw, ev, {"from": [float(v) for v in xy[0]], "to": [float(v) for v in xy[-1]],
                                    "length_m": float(length), "smooth_sigma_m": sigma})
            break
        if chosen:
            break
    if chosen is None:
        raise RuntimeError(f"no feasible street route at {alt_ft} ft")
    xy, yaw, ev, params = chosen
    out = path_record(feats, alt_ft, cfg, "street", params, xy, yaw, t, ev, obs, cam,
                      {"candidates": len(cands), "survivors": None, "tie_break_order": order,
                       "windows_tried": tried, "network_nodes": int(len(net.nodes))})
    out["variant"] = "street"
    return out


def route_track(route, speed: float, t: np.ndarray, focus: float, yaw_offset_deg: float, cfg: dict):
    """Camera ground track along a drawn route: smoothed, constant speed, one clip long.

    The clip is centred on `focus` (0 to 1 along the route). The view heading is the smoothed
    track heading plus `yaw_offset_deg`, so the camera can look to one side of its travel.
    """
    rc = cfg["route"]
    d, arc, head = smooth_route(np.asarray(route, dtype=np.float64), float(rc["smooth_sigma_m"]),
                                float(rc["dense_step_m"]), float(rc["yaw_sigma_m"]))
    need = speed * (t[-1] - t[0])
    if need > arc[-1]:
        raise RuntimeError(f"route is {arc[-1]:.0f} m long, the clip needs {need:.0f} m")
    s0 = float(np.clip(focus * arc[-1] - need / 2, 0.0, arc[-1] - need))
    s = s0 + speed * (t - t[0])
    xy = np.column_stack([np.interp(s, arc, d[:, 0]), np.interp(s, arc, d[:, 1])])
    return xy, np.interp(s, arc, head) + yaw_offset_deg


def footprint_inside(xy, yaw, alt, cam: Pinhole, extent) -> bool:
    """True when the ground footprint of every frame lies inside the scene extent."""
    import shapely as _sh
    quads = []
    for p, y in zip(xy, yaw):
        g = frustum_ground(camera_to_world([p[0], p[1], alt], float(y), cam.pitch_deg), cam)
        if g is None:
            return False
        quads.append(g)
    return bool(_sh.covers(extent, _sh.polygons(np.asarray(quads))).all())


def plan_route(feats: list[dict], alt_ft: float, cfg: dict, view: str | None = None) -> dict:
    """The place's configured route, or one of its extra views, flown for one long clip."""
    place = view_spec(cfg, view)
    cam = Pinhole(cfg["camera"])
    alt = altitude_m(alt_ft)
    speed = float(cfg["speed_mps"][int(alt_ft)])
    t = frame_times(cfg, "long")
    obs = Obstacles(feats, alt, cfg)
    focus = float(place["focus"][int(alt_ft)])
    xy, yaw = route_track(place["route_m"], speed, t, focus, float(place.get("yaw_offset_deg", 0.0)), cfg)
    if any(obs.is_blocked(p) for p in xy):
        raise RuntimeError(f"{alt_ft} ft: the route passes within the clearance of a tall building")
    if np.abs(np.diff(yaw) / np.diff(t)).max() > float(cfg["paths"]["max_yaw_rate_deg_s"]):
        raise RuntimeError(f"{alt_ft} ft: the route turns faster than the yaw-rate limit")
    if not footprint_inside(xy, yaw, alt, cam, extent_polygon(cfg)):
        raise RuntimeError(f"{alt_ft} ft: a frame's ground footprint leaves the scene extent")
    visible = 0
    for p, y in zip(xy, yaw):
        u, v, front = project(camera_to_world([p[0], p[1], alt], float(y), cam.pitch_deg), cam, (0.0, 0.0, 0.0))
        visible += int(front and abs(u) <= 1 and abs(v) <= 1)
    ev = {"min_clearance_m": min(obs.clearance(p) for p in xy), "target_visible_frames": visible}
    out = path_record(feats, alt_ft, cfg, "route", {"place": place["name"], "view": view or "main", "focus": focus,
                                                    "yaw_offset_deg": float(place.get("yaw_offset_deg", 0.0)),
                                                    "from": [float(v) for v in xy[0]], "to": [float(v) for v in xy[-1]]},
                      xy, yaw, t, ev, obs, cam, {"candidates": 1, "survivors": 1, "tie_break_order": 0})
    out["variant"] = "long"
    return out


def path_record(feats, alt_ft, cfg, kind, params, xy, yaw, t, ev, obs, cam, planner: dict) -> dict:
    alt = altitude_m(alt_ft)
    fps = float(cfg["clip"]["fps"])
    n = len(t)
    near = [f for f in feats if shape(f["geometry"]).distance(Point(0, 0)) <= float(cfg["paths"]["taller_share_radius_m"])]
    taller = sum(f["properties"]["height_m"] > alt for f in near)
    frames = []
    for i in range(n):
        m = camera_to_world([xy[i, 0], xy[i, 1], alt], float(yaw[i]), cam.pitch_deg)
        frames.append({"index": i, "t": float(t[i]), "position": [float(xy[i, 0]), float(xy[i, 1]), alt],
                       "yaw_deg": float(yaw[i]), "camera_to_world": m.tolist()})
    return {
        "altitude_ft": float(alt_ft), "altitude_m": alt,
        "speed_mps": float(cfg["speed_mps"][int(alt_ft)]) if kind != "hover_pan" else 0.0,
        "path_type": kind, "path_params": params,
        "clip": {"frames": n, "fps": fps, "duration_s": n / fps},
        "intrinsics": cam.intrinsics(),
        "conventions": {"units": "metres", "axes": "x east, y north, z up",
                        "camera_to_world": "column-vector 4x4; camera +X right, +Y up, looks along -Z",
                        "yaw_deg": "view heading on the ground, counter-clockwise from +x"},
        "planner": {**planner, "min_clearance_m": ev["min_clearance_m"],
                    "target_visible_frames": ev["target_visible_frames"],
                    "obstacle_buildings": len(obs.tall), "grid_res_m": obs.res},
        "taller_share_within_300m": (taller / len(near)) if near else 0.0,
        "taller_count_within_300m": taller, "buildings_within_300m": len(near),
        "seed": cfg["seed"],
        "frames": frames,
    }


def path_file(sdir: Path, alt_ft, variant: str | None = None, view: str | None = None) -> Path:
    return sdir / f"path_{int(alt_ft)}{variant_suffix(variant, view)}.json"


def run_stage4(location: str, cfg: dict, out_dir: Path | None = None, variant: str | None = None,
               view: str | None = None, altitudes=None) -> list[Path]:
    slug = slugify(location)
    cfg = for_location(cfg, location)
    sdir = out_dir or scene_dir(cfg, slug)
    feats = json.loads((scene_dir(cfg, slug) / "buildings.geojson").read_text())["features"]
    outs = []
    for alt_ft in altitudes or cfg["altitudes_ft"]:
        p = (plan_route(feats, alt_ft, cfg, view) if variant == "long" else
             plan_street(feats, alt_ft, cfg) if variant == "street" else plan(feats, alt_ft, cfg))
        p["location"] = slug
        out = path_file(sdir, alt_ft, variant, view)
        out.write_text(json.dumps(p, indent=1) + "\n")
        log.info("%s ft: %s %s, clearance %.1f m, target visible %d/%d frames, %s/%d candidates survive",
                 alt_ft, p["path_type"], p["path_params"], p["planner"]["min_clearance_m"],
                 p["planner"]["target_visible_frames"], p["clip"]["frames"],
                 p["planner"]["survivors"], p["planner"]["candidates"])
        outs.append(out)
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: drone camera paths")
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, choices=["street", "long"],
                    help="street: clip following the street corridors; long: a place's configured route")
    ap.add_argument("--view", default=None, help="one of the place's extra camera views (with --variant long)")
    ap.add_argument("--altitude", type=int, action="append", help="limit to this altitude in feet; repeatable")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_stage4(a.location, load_config(a.config), variant=a.variant, view=a.view, altitudes=a.altitude)


if __name__ == "__main__":
    main()
