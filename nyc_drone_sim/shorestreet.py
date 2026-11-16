"""Find and score shore-then-street routes: along the shoreline, one smooth turn, then along a
street into the buildings. CPU only.

The route is the ground track of the image-centre point, shared by both altitudes. Each
altitude's camera sits behind that point along the view direction, by altitude / tan(pitch).
"""
from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring, unary_union

from .common import REPO_ROOT, altitude_m, for_location, load_config, scene_dir, slugify
from .paths import Obstacles, Pinhole, smooth_route
from .scout import class_unions, footprint_shares, not_at_grade

log = logging.getLogger("shorestreet")


def camera_xy(ground: np.ndarray, yaw_deg: np.ndarray, alt: float, pitch_deg: float) -> np.ndarray:
    """Camera ground position for an image-centre ground point: behind it along the view."""
    lead = alt / math.tan(math.radians(-pitch_deg))
    y = np.radians(yaw_deg)
    return ground - lead * np.column_stack([np.cos(y), np.sin(y)])


def fillet(a, j, b, radius: float, step: float = 2.0) -> np.ndarray:
    """Points of the arc that joins segment a-j to segment j-b with the given radius."""
    a, j, b = (np.asarray(v, dtype=float) for v in (a, j, b))
    u1, u2 = (a - j) / np.linalg.norm(a - j), (b - j) / np.linalg.norm(b - j)
    turn = math.pi - math.acos(float(np.clip(u1 @ u2, -1, 1)))        # change of heading
    tan_len = radius * math.tan(turn / 2)
    p1, p2 = j + u1 * tan_len, j + u2 * tan_len
    bis = (u1 + u2) / np.linalg.norm(u1 + u2)
    centre = j + bis * radius / math.cos(turn / 2)
    a1 = math.atan2(*(p1 - centre)[::-1])
    a2 = math.atan2(*(p2 - centre)[::-1])
    d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    n = max(3, int(abs(d) * radius / step))
    ang = a1 + d * np.linspace(0, 1, n)
    return np.column_stack([centre[0] + radius * np.cos(ang), centre[1] + radius * np.sin(ang)]), tan_len, math.degrees(turn)


def sync_track(route, speed: float, t: np.ndarray, sc: dict):
    """Image-centre ground points and view heading per frame, at one constant ground speed."""
    d, arc, head = smooth_route(np.asarray(route, dtype=float), float(sc["smooth_sigma_m"]),
                                float(sc["dense_step_m"]), float(sc["yaw_sigma_m"]))
    s = speed * (t - t[0])
    if s[-1] > arc[-1]:
        raise RuntimeError(f"route is {arc[-1]:.0f} m long, the clip needs {s[-1]:.0f} m")
    return np.column_stack([np.interp(s, arc, d[:, 0]), np.interp(s, arc, d[:, 1])]), np.interp(s, arc, head)


def shoreline(U: dict, sc: dict):
    """Curve a fixed distance inland of the water's edge. Piers count as water, and narrow slips
    are closed, so the curve follows the bulkhead rather than every pier."""
    by = {}
    wet = unary_union([U["water"], U.get("pier_only", shapely.Polygon())])
    close = float(sc["slip_close_m"])
    wet = wet.buffer(close).buffer(-close)
    # grow, then shrink back: corners of the curve come out with a radius of at least
    # round_m - inland_m, so following it never needs a sharp turn
    big = float(sc["shore_round_m"])
    return wet.buffer(big).buffer(-(big - float(sc["inland_m"]))).boundary, wet


def street_runs(sdir, sc: dict):
    """Straight at-grade runs of streets with enough travel lanes."""
    feats = json.loads((sdir / "centerlines.geojson").read_text())["features"]
    by = {}
    for f in feats:
        p = f["properties"]
        if (p.get("travel_lanes") or 0) >= int(sc["min_travel_lanes"]) and p.get("rw_type") == 1 \
                and (p.get("from_level") or 13) == 13 and (p.get("to_level") or 13) == 13:
            by.setdefault(p["street"], []).append((LineString(np.asarray(f["geometry"]["coordinates"])[:, :2]), p["travel_lanes"]))
    runs = []
    for name, segs in by.items():
        lanes = float(np.mean([l for _, l in segs]))
        merged = linemerge([s for s, _ in segs])
        for line in getattr(merged, "geoms", [merged]):
            c = np.asarray(line.simplify(4.0).coords)
            start = 0
            for i in range(1, len(c)):
                last = i == len(c) - 1
                bend = 0.0
                if not last:
                    v1, v2 = c[i] - c[i - 1], c[i + 1] - c[i]
                    bend = abs(math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
                if last or bend > float(sc["max_bend_deg"]):
                    run = c[start:i + 1]
                    if np.hypot(*(run[-1] - run[0])) >= float(sc["min_run_m"]):
                        runs.append({"street": name, "lanes": lanes, "a": run[0], "b": run[-1]})
                    start = i
    return runs


def candidates(cfg: dict):
    sc = cfg["shore_street"]
    sdir = scene_dir(cfg, slugify(sc["place"]))
    U = class_unions(sdir)
    surf = json.loads((sdir / "surfaces.geojson").read_text())["features"]
    from shapely.geometry import shape
    U["pier_only"] = unary_union([shape(f["geometry"]) for f in surf if f["properties"]["class"] == "pier"])
    curve, wet = shoreline(U, sc)
    speed, R = float(sc["speed_mps"]), float(sc["turn_radius_m"])
    leg = speed * float(sc["leg_s"])
    out = []
    for run in street_runs(sdir, sc):
        for e, far in ((run["a"], run["b"]), (run["b"], run["a"])):
            axis = (far - e) / np.linalg.norm(far - e)
            probe = LineString([e, e - axis * float(sc["max_gap_m"])])
            hit = probe.intersection(curve)
            if hit.is_empty:
                continue
            pts = [p for p in getattr(hit, "geoms", [hit]) if p.geom_type == "Point"]
            if not pts:
                continue
            j = min(pts, key=lambda p: p.distance(Point(e)))
            piece = next((g for g in getattr(curve, "geoms", [curve]) if g.distance(j) < 0.01), None)
            if piece is None:
                continue
            sj = piece.project(j)
            normal = np.array([-axis[1], axis[0]])
            h2 = math.atan2(axis[1], axis[0])
            for sign in (-1, 1):
                for shift in sc["street_offset_m"]:
                    e0 = np.asarray(e, dtype=float) + normal * float(shift)       # the street leg, moved sideways
                    # Walk the shore curve toward the street. At each point, turn from the curve's
                    # own heading onto the street heading on one arc; take the point whose arc
                    # ends exactly on the street leg's line, so curve, arc and street join smoothly.
                    best = None
                    for s1 in np.arange(max(2.0, sj - 3 * R), min(piece.length - 2.0, sj + 3 * R), 1.0):
                        p0 = np.asarray(piece.interpolate(s1 - sign * 1.0).coords[0])
                        p1 = np.asarray(piece.interpolate(s1).coords[0])
                        # travelling along the curve toward increasing s when sign < 0 ... the
                        # shore leg always ends at s1, arriving from s1 + sign * leg
                        d1 = p1 - np.asarray(piece.interpolate(s1 + sign * 1.0).coords[0])
                        d1 /= np.linalg.norm(d1)
                        h1 = math.atan2(d1[1], d1[0])
                        dh = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
                        if not math.radians(45) <= abs(dh) <= math.radians(135):
                            continue
                        left = np.array([-d1[1], d1[0]]) * (1 if dh > 0 else -1)
                        centre = p1 + left * R
                        a0 = math.atan2(*(p1 - centre)[::-1])
                        ang = a0 + dh * np.linspace(0, 1, max(4, int(abs(dh) * R / 2)))
                        arc = np.column_stack([centre[0] + R * np.cos(ang), centre[1] + R * np.sin(ang)])
                        miss = float((arc[-1] - e0) @ normal)            # sideways distance from the street line
                        along = float((arc[-1] - e0) @ axis)
                        if best is None or abs(miss) < abs(best[0]):
                            best = (miss, s1, arc, along, abs(math.degrees(dh)))
                    if best is None or abs(best[0]) > 1.0:
                        continue
                    _, s1, arc, along, turn = best
                    s0 = s1 + sign * leg
                    if not 0 <= s0 <= piece.length:
                        continue
                    shore = np.asarray(substring(piece, s0, s1).coords)
                    for street_len in sc["street_len_m"]:
                        if along + street_len > np.linalg.norm(far - e):
                            continue                     # would fly past the end of the straight run
                        end = arc[-1] + axis * float(street_len)
                        out.append({"street": run["street"], "lanes": round(run["lanes"], 1), "turn_deg": round(turn, 1),
                                    "gap_m": round(max(0.0, -along), 1), "side": sign, "offset_m": float(shift),
                                    "street_len_m": float(street_len), "route": np.vstack([shore[:-1], arc, end[None]]),
                                    "shore_len_m": leg, "arc_len_m": round(math.radians(turn) * R, 1)})
    return out, U, sdir


def score(cfg: dict, top: int = 12):
    sc = cfg["shore_street"]
    cands, U, sdir = candidates(cfg)
    pcfg = for_location(cfg, sc["place"])
    for g in U.values():
        shapely.prepare(g)
    off = not_at_grade(sdir, float(cfg["scout"]["nongrade_buffer_m"]), U["road"])
    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    cam = Pinhole(cfg["camera"])
    fps, speed = float(cfg["clip"]["fps"]), float(sc["speed_mps"])
    obs = {int(a): Obstacles(feats, altitude_m(a), pcfg) for a in cfg["altitudes_ft"]}
    rows = []
    for c in cands:
        # the smoothed track is a little shorter than the drawn route: size the clip from it
        _, arc, _ = smooth_route(c["route"], float(sc["smooth_sigma_m"]), float(sc["dense_step_m"]), float(sc["yaw_sigma_m"]))
        n = int(arc[-1] / speed * fps)
        t = np.arange(n) / fps
        try:
            G, yaw = sync_track(c["route"], speed, t, sc)
        except RuntimeError:
            continue
        rate = float(np.abs(np.diff(yaw) * fps).max())
        if rate > float(sc["max_yaw_rate_deg_s"]):
            continue
        ok, clear = True, {}
        for a in obs:
            cxy = camera_xy(G, yaw, altitude_m(a), cam.pitch_deg)
            if any(obs[a].is_blocked(p) for p in cxy):
                ok = False
                break
            clear[a] = round(float(min(obs[a].clearance(p) for p in cxy[::8])), 1)
        if not ok:
            continue
        a0 = int(cfg["altitudes_ft"][0])
        sh = footprint_shares(camera_xy(G, yaw, altitude_m(a0), cam.pitch_deg), yaw, altitude_m(a0), cam, U, off, 16)
        fr = np.array(sh["frames"]) / fps
        t1 = c["shore_len_m"] / speed
        t2 = t1 + c["arc_len_m"] / speed
        l1, l2 = fr <= t1, fr >= t2 + 4.0
        w1 = float(np.mean(np.array(sh["water"])[l1]))
        built = float(np.mean((np.array(sh["road"]) + np.array(sh["building"]))[l2]))
        road2 = float(np.mean(np.array(sh["road"])[l2]))
        ng = float(np.max(sh["not_at_grade"]))
        rows.append({**{k: v for k, v in c.items() if k != "route"}, "route": np.round(c["route"], 1).tolist(),
                     "frames": n, "seconds": round(n / fps, 1), "max_yaw_rate": round(rate, 1), "clearance_m": clear,
                     "leg1_water": round(w1, 3), "leg2_road_building": round(built, 3), "leg2_road": round(road2, 3),
                     "not_at_grade_max": round(ng, 3),
                     "score": round(min(w1, 0.5) + built + 0.5 * road2 - 2 * ng, 3)})
    rows.sort(key=lambda r: -r["score"])
    return rows[:top], len(cands)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    rows, n = score(cfg)
    (REPO_ROOT / "reports" / "shore_street_scout.json").write_text(json.dumps({"candidates_built": n, "ranked": rows}, indent=1) + "\n")
    print(f"{n} candidates built; top {len(rows)} that clear both altitudes:")
    for r in rows:
        print(f"{r['street']:<14} lanes {r['lanes']:.1f} turn {r['turn_deg']:>5} gap {r['gap_m']:>5} off {r['offset_m']:+.0f} street {r['street_len_m']:.0f} m "
              f"{r['seconds']:>4} s yaw {r['max_yaw_rate']:>4} water1 {r['leg1_water']:.0%} built2 {r['leg2_road_building']:.0%} "
              f"road2 {r['leg2_road']:.0%} side {r['side']:+d} offgrade {r['not_at_grade_max']:.0%} clear {r['clearance_m']} score {r['score']} start {r['route'][0]} end {r['route'][-1]}")
