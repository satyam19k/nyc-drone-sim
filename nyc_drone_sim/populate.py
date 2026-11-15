"""Place vehicles, pedestrians, trees, street furniture and road markings. CPU only.

Positions come from city records where records exist (trees, hydrants, shelters, kiosks, bike
racks) and from seeded rules where they do not (vehicles, pedestrians, lamps, signals, paint).
The same location, config and seed give the same placement list.

Motion classes used here:
- static: one pose for the whole clip
- recorded animation: a pose per frame, authored here (moving vehicles, walking pedestrians)
Nothing is independently simulated.

Outputs in the scene directory:
  placements.json   every placed object: class, model, pose, scale, rule, motion class
  motion.npz        recorded poses, [object, frame, (x, y, yaw_deg)], with matching ids
  markings.geojson  road paint polygons
"""
from __future__ import annotations

import argparse
import json
import logging
import math

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .assets import load_manifest
from .common import for_location, load_config, scene_dir, slugify

log = logging.getLogger("populate")
MPH = 0.44704


def rects(x, y, yaw_deg, length, width) -> np.ndarray:
    """Oriented rectangles as an array of shapely polygons (all inputs broadcast)."""
    x, y, yaw, length, width = np.broadcast_arrays(*[np.asarray(v, dtype=np.float64) for v in (x, y, yaw_deg, length, width)])
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    hl, hw = length / 2, width / 2
    corners = np.stack([np.stack([x + sx * hl * c - sy * hw * s, y + sx * hl * s + sy * hw * c], axis=-1)
                        for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))], axis=-2)
    return shapely.polygons(corners)


def rect1(x, y, yaw_deg, length, width) -> Polygon:
    return np.atleast_1d(rects(x, y, yaw_deg, length, width))[0]


class Grid:
    """Spatial hash of footprints for incremental overlap tests."""

    def __init__(self, cell: float = 12.0):
        self.cell, self.cells, self.geoms = cell, {}, []

    def _keys(self, g):
        x0, y0, x1, y1 = g.bounds
        return [(i, j) for i in range(int(x0 // self.cell) - 1, int(x1 // self.cell) + 2)
                for j in range(int(y0 // self.cell) - 1, int(y1 // self.cell) + 2)]

    def near(self, g, dist: float) -> bool:
        seen = set()
        for k in self._keys(g):
            for i in self.cells.get(k, ()):
                if i not in seen:
                    seen.add(i)
                    if self.geoms[i].distance(g) < dist:
                        return True
        return False

    def add(self, g):
        self.geoms.append(g)
        x0, y0, x1, y1 = g.bounds
        for i in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
            for j in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                self.cells.setdefault((i, j), []).append(len(self.geoms) - 1)


# ---------------------------------------------------------------------------
# Road network
# ---------------------------------------------------------------------------
class Network:
    def __init__(self, lines: list[dict], road, cfg: dict):
        pc = cfg["populate"]
        self.pc, self.road = pc, road
        shapely.prepare(self.road)
        self.segs = []
        for f in lines:
            p = f["properties"]
            xy = np.asarray(f["geometry"]["coordinates"], dtype=np.float64)[:, :2]
            line = LineString(xy)
            # at ground level only: the scene is flat, so bridges, viaducts, ramps and tunnels carry nothing
            grade = int(pc["ground_level_code"])
            drivable = (p["rw_type"] in pc["drivable_rw_types"] and p["trafdir"] in ("FT", "TF", "TW")
                        and (p["travel_lanes"] or 0) >= 1
                        and (p.get("from_level") or grade) == grade and (p.get("to_level") or grade) == grade)
            self.segs.append({"id": str(p["physicalid"]), "xy": xy, "line": line, "len": line.length, "p": p,
                              "drivable": drivable})
        self.nodes: dict[tuple, list] = {}
        for i, s in enumerate(self.segs):
            for end in (0, 1):
                self.nodes.setdefault(self.key(s["xy"][0 if end == 0 else -1]), []).append((i, end))
        self.tree = STRtree([s["line"] for s in self.segs])
        self.drivable = [s for s in self.segs if s["drivable"]]
        self.drivable_tree = STRtree([s["line"] for s in self.drivable])
        self._node_tree = None
        for s in self.segs:
            self._cross_section(s)

    @staticmethod
    def key(pt) -> tuple:
        return (round(float(pt[0]), 1), round(float(pt[1]), 1))

    def degree(self, pt) -> int:
        return len(self.nodes.get(self.key(pt), ()))

    def frame_at(self, s: dict, d: float):
        """Point, unit tangent and unit left normal at arc length d along the digitised direction."""
        a = np.asarray(s["line"].interpolate(max(d - 0.5, 0.0)).coords[0])
        b = np.asarray(s["line"].interpolate(min(d + 0.5, s["len"])).coords[0])
        t = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        return np.asarray(s["line"].interpolate(d).coords[0]), t, np.array([-t[1], t[0]])

    def half_width(self, c, n) -> float:
        """Distance from c along n to the first point that is not roadbed."""
        if not shapely.contains_xy(self.road, c[0], c[1]):
            return 0.0
        reach = float(self.pc["max_half_width_m"])
        out = LineString([c, c + n * reach]).difference(self.road)
        return reach if out.is_empty else float(Point(c).distance(out))

    def _cross_section(self, s: dict):
        pc = self.pc
        m = float(pc["intersection_margin_m"])
        s["d"], s["left"], s["right"] = np.array([]), np.array([]), np.array([])
        if not s["drivable"] or s["len"] < 4.0:
            return
        if s["len"] < 2 * m + 2:
            # a short link between two intersections: one cross-section at its middle gives it
            # lanes, so routes can pass through; it gets no parking, paint or lamps of its own
            d = np.array([s["len"] / 2])
        else:
            d = np.arange(m, s["len"] - m + 1e-6, float(pc["cross_section_step_m"]))
        left, right = [], []
        for v in d:
            c, _, n = self.frame_at(s, v)
            left.append(self.half_width(c, n))
            right.append(self.half_width(c, -n))
        s["d"], s["left"], s["right"] = d, np.array(left), np.array(right)
        ok = (s["left"] > 0.5) & (s["right"] > 0.5)
        if not ok.any():
            return
        hl, hr = float(np.median(s["left"][ok])), float(np.median(s["right"][ok]))
        p = s["p"]
        pk = p["park_lanes"] or 0
        s["park_sides"] = {2: (1, -1), 1: (-1,)}.get(pk, ())       # +1 = left of the digitised direction
        pw = float(pc["vehicles"]["park_lane_width_m"])
        lim_l = hl - (pw if 1 in s["park_sides"] else 0.0)
        lim_r = hr - (pw if -1 in s["park_sides"] else 0.0)
        width = lim_l + lim_r
        nl = min(int(p["travel_lanes"]), int(width // float(pc["vehicles"]["min_lane_width_m"])))
        lanes = []
        for k in range(max(nl, 0)):
            off = lim_l - (k + 0.5) * width / nl                   # signed offset, + = left
            if p["trafdir"] == "FT":
                direction = 1
            elif p["trafdir"] == "TF":
                direction = -1
            else:
                # two-way: keep right. A lane on the centre line is only used on a one-lane street.
                if abs(off) < 0.5 and nl > 1:
                    continue
                direction = -1 if off > 0 else 1
            lanes.append({"offset": off, "dir": direction, "width": width / nl})
        s["lanes"] = lanes
        s["speed"] = float(pc["vehicles"]["speed_factor"]) * MPH * float(
            p["posted_speed_mph"] or pc["vehicles"]["default_speed_mph"])

    def heading(self, s: dict, direction: int, at_end: bool) -> np.ndarray:
        xy = s["xy"] if direction == 1 else s["xy"][::-1]
        v = xy[-1] - xy[-2] if at_end else xy[1] - xy[0]
        return v / max(np.linalg.norm(v), 1e-9)

    def successor(self, i: int, direction: int, r: float):
        """Next (segment, direction) straight ahead that has a lane at the same left offset r."""
        s = self.segs[i]
        end_pt = s["xy"][-1] if direction == 1 else s["xy"][0]
        h = self.heading(s, direction, True)
        best = None
        for j, end in self.nodes.get(self.key(end_pt), ()):
            if j == i:
                continue
            t = self.segs[j]
            d2 = 1 if end == 0 else -1
            # the next segment needs a lane within one lane change of this one
            if not any(ln["dir"] == d2 and abs(ln["offset"] * d2 - r) <= float(self.pc["vehicles"]["lane_shift_max_m"])
                       for ln in t.get("lanes", ())):
                continue
            bend = math.degrees(math.acos(float(np.clip(h @ self.heading(t, d2, False), -1, 1))))
            if bend <= float(self.pc["vehicles"]["chain_max_bend_deg"]) and (best is None or bend < best[0]):
                best = (bend, j, d2)
        return None if best is None else (best[1], best[2])

    def follows_its_street(self, xy: np.ndarray, yaw_deg: np.ndarray, limit_deg: float, margin: float) -> bool:
        """False if, away from intersections, a pose lies nearest a street it is not aligned with.

        Where two streets meet at a shallow angle a lane can run closer to the other street's
        centre line than to its own; a vehicle there reads as driving askew across that street.
        """
        if self._node_tree is None:
            self._node_pts = shapely.points(np.array([k for k in sorted(self.nodes)]))
            self._node_tree = STRtree(self._node_pts)
        pts = shapely.points(xy[:, 0], xy[:, 1])
        near_node = shapely.distance(pts, self._node_pts[self._node_tree.nearest(pts)]) < margin
        seg_i = self.drivable_tree.nearest(pts)
        for k in np.nonzero(~near_node)[0]:
            s = self.drivable[int(seg_i[k])]
            _, t, _ = self.frame_at(s, s["line"].project(pts[k]))
            h = np.array([math.cos(math.radians(yaw_deg[k])), math.sin(math.radians(yaw_deg[k]))])
            if abs(float(h @ t)) < math.cos(math.radians(limit_deg)):
                return False
        return True

    def nearest(self, x: float, y: float) -> dict:
        return self.segs[int(self.tree.nearest(Point(x, y)))]


# ---------------------------------------------------------------------------
# Vehicles
# ---------------------------------------------------------------------------
class Fleet:
    def __init__(self, manifest: dict, cfg: dict, rng: np.random.Generator):
        self.rng = rng
        vc = cfg["populate"]["vehicles"]
        self.by_group = {}
        for name in sorted(manifest["assets"]):
            a = manifest["assets"][name]
            if a["class"] == "vehicle":
                self.by_group.setdefault(a["group"], []).append((name, a["size_m"][0], a["size_m"][1], a["size_m"][2]))
        self.groups = sorted(g for g in vc["group_weights"] if g in self.by_group)
        w = np.array([float(vc["group_weights"][g]) for g in self.groups])
        self.weights = w / w.sum()
        self.parked = [g for g in self.groups if g in vc["parked_groups"]]
        wp = np.array([float(vc["group_weights"][g]) for g in self.parked])
        self.parked_w = wp / wp.sum()

    def pick(self, parked: bool):
        groups, w = (self.parked, self.parked_w) if parked else (self.groups, self.weights)
        g = groups[int(self.rng.choice(len(groups), p=w))]
        models = self.by_group[g]
        return models[int(self.rng.integers(len(models)))]


def place_parked(net: Network, fleet: Fleet, lots: list, cfg: dict, rng, grid: Grid, out: list, buildings):
    vc = cfg["populate"]["vehicles"]
    gap0, gap1 = (float(v) for v in vc["parked_gap_m"])
    for s in net.segs:
        for side in s.get("park_sides", ()):
            hw = s["left"] if side == 1 else s["right"]
            d = float(s["d"][0])
            while d < s["d"][-1]:
                name, length, width, _ = fleet.pick(True)
                slot = length + rng.uniform(gap0, gap1)
                take = rng.random() < float(vc["parked_occupancy"])
                mid = d + slot / 2
                d += slot
                if not take or mid + length / 2 > s["d"][-1]:
                    continue
                near = (s["d"] >= mid - length / 2 - 3) & (s["d"] <= mid + length / 2 + 3)
                if not near.any() or hw[near].min() < width + 2 * float(vc["kerb_gap_m"]):
                    continue
                c, t, n = net.frame_at(s, mid)
                off = hw[near].min() - float(vc["kerb_gap_m"]) - width / 2
                pos = c + side * n * off
                direction = {"FT": 1, "TF": -1}.get(s["p"]["trafdir"], -side)   # two-way: keep right
                yaw = math.degrees(math.atan2(direction * t[1], direction * t[0]))
                g = rect1(pos[0], pos[1], yaw, length, width)
                if not shapely.covers(net.road, g) or grid.near(g, float(vc["clearance_m"])) or g.intersects(buildings):
                    continue
                # where two streets merge, the far kerb can belong to the other street: a parked
                # vehicle must be nearer its own centre line than any other
                if net.drivable[int(net.drivable_tree.nearest(Point(pos[0], pos[1])))] is not s:
                    continue
                grid.add(g)
                out.append({"class": "vehicle", "asset": name, "x": float(pos[0]), "y": float(pos[1]), "yaw_deg": yaw,
                            "scale": 1.0, "motion": "static", "rule": "parked_kerb", "segment": s["id"]})
    sw, sl = (float(v) for v in vc["lot_slot_m"])
    for lot in lots:
        mrr = lot.minimum_rotated_rectangle
        if mrr.geom_type != "Polygon":
            continue
        e = np.asarray(mrr.exterior.coords)[:4]
        u, v = e[1] - e[0], e[3] - e[0]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < sw or lv < sl:
            continue
        u, v = u / lu, v / lv
        yaw = math.degrees(math.atan2(v[1], v[0]))
        for j in range(int(lv // (sl + 6.0)) + 1):
            for i in range(int(lu // sw)):
                name, length, width, _ = fleet.pick(True)
                if rng.random() >= float(vc["lot_occupancy"]):
                    continue
                pos = e[0] + u * (i + 0.5) * sw + v * (j * (sl + 6.0) + sl / 2)
                g = rect1(pos[0], pos[1], yaw, length, width)
                if not lot.covers(g) or grid.near(g, float(vc["clearance_m"])) or g.distance(buildings) < 0.5:
                    continue
                grid.add(g)
                out.append({"class": "vehicle", "asset": name, "x": float(pos[0]), "y": float(pos[1]), "yaw_deg": yaw,
                            "scale": 1.0, "motion": "static", "rule": "parked_lot", "segment": None})


def lane_routes(net: Network) -> list[dict]:
    """Maximal straight-through routes, one per lane. Each lane of each segment belongs to one route."""
    states = [(i, k) for i, s in enumerate(net.segs) for k in range(len(s.get("lanes", ())))]
    succ = {}
    for i, k in states:
        ln = net.segs[i]["lanes"][k]
        r = ln["offset"] * ln["dir"]
        nxt = net.successor(i, ln["dir"], r)
        if nxt is not None:
            j, d2 = nxt
            cand = [(abs(l2["offset"] * d2 - r), k2) for k2, l2 in enumerate(net.segs[j]["lanes"]) if l2["dir"] == d2]
            succ[(i, k)] = (j, min(cand)[1])
    has_pred = set(succ.values())
    order = [s for s in states if s not in has_pred] + [s for s in states if s in has_pred]
    seen, routes = set(), []
    for start in order:
        if start in seen:
            continue
        # the route's own lanes: those no earlier route has claimed. Vehicles start only there.
        chain, cur = [], start
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            cur = succ.get(cur)
        own = len(chain)
        # the road ahead: where the route joins lanes another route already claimed it carries on
        # along them, so its vehicles can keep driving instead of ending at the join
        ahead = set(chain)
        while cur is not None and cur not in ahead:
            ahead.add(cur)
            chain.append(cur)
            cur = succ.get(cur)
        i0, k0 = chain[0]
        pts, centre_pts, own_len = [], [], 0.0
        for n_state, (i, k) in enumerate(chain):
            if n_state < own:
                own_len += net.segs[i]["len"]
            ln = net.segs[i]["lanes"][k]
            xy = net.segs[i]["xy"] if ln["dir"] == 1 else net.segs[i]["xy"][::-1]
            centre_pts.extend(xy.tolist() if not centre_pts else xy[1:].tolist())
            # each segment contributes its own lane line; lanes can sit at different offsets from
            # one segment to the next, and the join is smoothed below into a short lane change
            lane = LineString(xy).offset_curve(ln["offset"] * ln["dir"])
            if lane.is_empty or lane.geom_type != "LineString":
                pts = []
                break
            pts.extend(list(lane.coords))
        if len(pts) < 2:
            continue
        centre = LineString(centre_pts)
        raw = LineString(pts)
        if raw.length < 20:
            continue
        u = np.arange(0.0, raw.length, 1.0)
        dense = np.array([c for c in shapely.get_coordinates(shapely.line_interpolate_point(raw, u))])
        sm = ndimage.gaussian_filter1d(dense, float(net.pc["vehicles"]["lane_change_sigma_m"]), axis=0, mode="nearest")
        line = LineString(sm)
        routes.append({"line": line, "centre": centre, "speed": net.segs[i0]["speed"], "own_length": own_len,
                       "travel_lanes": max(int(net.segs[i]["p"]["travel_lanes"] or 1) for i, _ in chain),
                       "segments": [net.segs[i]["id"] for i, _ in chain]})
    return routes


def place_moving(net: Network, fleet: Fleet, cfg: dict, rng, grid: Grid, out: list, tracks: list, t: np.ndarray, buildings):
    vc = cfg["populate"]["vehicles"]
    clear = float(vc["clearance_m"])
    static_tree = STRtree(grid.geoms) if grid.geoms else None
    placed_xy = np.zeros((0, len(t), 2))
    placed_r = np.zeros(0)
    placed_poly = []
    span = float(t[-1] - t[0])
    why = {"routes": 0, "routes_long_enough": 0, "tried": 0, "left_roadbed": 0, "askew_or_at_building": 0,
           "hit_parked": 0, "hit_moving": 0}
    for route in lane_routes(net):
        line, total, v = route["line"], route["line"].length, route["speed"]
        # Every vehicle drives at the route's speed for the whole recording and must still be on
        # the route at the end, so none stops or runs off a route end. A vehicle that would reach
        # the end is simply not placed.
        last = min(total - float(vc["stop_margin_m"]) - v * span, route["own_length"])
        why["routes"] += 1
        why["routes_long_enough"] += last > 5.0
        spacing = float(vc["arterial_spacing_m"] if route["travel_lanes"] >= int(vc["arterial_lanes"]) else vc["moving_spacing_m"])
        s0 = last - rng.exponential(spacing)
        while s0 > 5.0:
            name, length, width, _ = fleet.pick(False)
            s = s0 + v * (t - t[0])
            pts = shapely.line_interpolate_point(line, s)
            ahead = shapely.line_interpolate_point(line, np.minimum(s + 1.0, total))
            behind = shapely.line_interpolate_point(line, np.maximum(s - 1.0, 0.0))
            xy = np.column_stack([shapely.get_x(pts), shapely.get_y(pts)])
            yaw = np.degrees(np.arctan2(shapely.get_y(ahead) - shapely.get_y(behind), shapely.get_x(ahead) - shapely.get_x(behind)))
            polys = rects(xy[:, 0], xy[:, 1], yaw, length, width)
            why["tried"] += 1
            ok = bool(shapely.covers(net.road, polys).all())
            why["left_roadbed"] += not ok
            if ok:
                step = slice(None, None, int(vc["alignment_check_every"]))
                ok = (not shapely.intersects(buildings, polys).any()
                      and net.follows_its_street(xy[step], yaw[step], float(vc["max_street_angle_deg"]),
                                                 float(cfg["populate"]["intersection_margin_m"])))
                why["askew_or_at_building"] += not ok
            if ok and static_tree is not None:
                ok = static_tree.query(polys, predicate="dwithin", distance=clear).shape[1] == 0
                why["hit_parked"] += not ok
            before = ok
            if ok and len(placed_r):
                dist = np.linalg.norm(placed_xy - xy[None], axis=2)
                reach = placed_r + math.hypot(length, width) / 2 + clear
                for m, f in zip(*np.nonzero(dist < reach[:, None])):
                    if placed_poly[m][f].distance(polys[f]) < clear:
                        ok = False
                        break
            why["hit_moving"] += before and not ok
            if ok:
                placed_xy = np.concatenate([placed_xy, xy[None]])
                placed_r = np.append(placed_r, math.hypot(length, width) / 2)
                placed_poly.append(polys)
                out.append({"class": "vehicle", "asset": name, "x": float(xy[0, 0]), "y": float(xy[0, 1]),
                            "yaw_deg": float(yaw[0]), "scale": 1.0, "motion": "recorded", "rule": "moving_lane",
                            "segment": route["segments"][0], "speed_mps": v})
                tracks.append(np.column_stack([xy, yaw]))
            s0 = s0 - length - float(vc["queue_gap_m"]) - rng.exponential(spacing)
    return placed_xy, placed_r, why


# ---------------------------------------------------------------------------
# Trees, furniture, pedestrians, markings
# ---------------------------------------------------------------------------
def place_trees(trees: list[dict], manifest: dict, cfg: dict, road, buildings, out: list, lawns=(), rng=None) -> dict:
    tc = cfg["populate"]["trees"]
    models = {n: a for n, a in manifest["assets"].items() if a["class"] == "vegetation"}
    forms = sorted({a["form"] for a in models.values()})
    dropped = moved = 0
    edge = road.boundary
    for tr in trees:
        if shapely.contains_xy(road, tr["x"], tr["y"]):
            # kerbside pits are often recorded just inside the roadbed line: step such a tree
            # back over the kerb; one further into the road is not placed
            pt = Point(tr["x"], tr["y"])
            near = shapely.ops.nearest_points(pt, edge)[1]
            d = pt.distance(near)
            if d > float(tc["kerb_snap_m"]) or d < 1e-6:
                dropped += 1
                continue
            step = float(tc["kerb_setback_m"])
            tr = {**tr, "x": round(near.x + (near.x - pt.x) / d * step, 3), "y": round(near.y + (near.y - pt.y) / d * step, 3)}
            if shapely.contains_xy(road, tr["x"], tr["y"]):
                dropped += 1
                continue
            moved += 1
        if shapely.contains_xy(buildings, tr["x"], tr["y"]):
            dropped += 1
            continue
        genus = tr["species"].split(" ")[0] if tr["species"] else ""
        h = int(tr["id"]) if tr["id"].isdigit() else sum(map(ord, tr["id"]))
        form = tc["genus_form"].get(genus) or forms[h % len(forms)]
        dbh = tr["dbh_in"] if tr["dbh_in"] and tr["dbh_in"] > 0 else float(tc["default_dbh_in"])
        age = "y" if dbh < float(tc["young_below_dbh_in"]) else "m"
        cands = sorted(n for n, a in models.items() if a["form"] == form and a["age"] == age) or \
            sorted(n for n, a in models.items() if a["form"] == form) or sorted(models)
        name = cands[h % len(cands)]
        a = models[name]
        crown = float(np.clip(float(tc["crown_base_m"]) + float(tc["crown_per_dbh_cm"]) * dbh * 2.54, *tc["crown_range_m"]))
        scale = crown / a["crown_diameter_m"]
        scale = float(np.clip(scale, tc["height_range_m"][0] / a["height_m"], tc["height_range_m"][1] / a["height_m"]))
        out.append({"class": "vegetation", "asset": name, "x": tr["x"], "y": tr["y"], "yaw_deg": float((h * 47) % 360),
                    "scale": round(scale, 4), "motion": "static", "rule": "tree_record", "record": tr["id"],
                    "height_m": round(a["height_m"] * scale, 2), "crown_m": round(a["crown_diameter_m"] * scale, 2)})
    stats = {"records": len(trees), "dropped_in_road_or_building": dropped, "moved_over_kerb": moved, "scattered_in_parks": 0}
    # Parks: where the tree records leave a park thin, scatter trees over its lawn by rule.
    if lawns:
        rec_xy = np.array([[o["x"], o["y"]] for o in out if o["class"] == "vegetation"]).reshape(-1, 2)
        names = sorted(n for n, a in models.items() if a["age"] == "m")
        gap = float(tc["park_tree_spacing_m"])
        for k, lawn in enumerate(lawns):
            if lawn.area < float(tc["park_min_area_m2"]):
                continue
            have = int(shapely.contains_xy(lawn, rec_xy[:, 0], rec_xy[:, 1]).sum()) if len(rec_xy) else 0
            want = int(lawn.area / 10000.0 * float(tc["park_trees_per_ha"]))
            # stay off the lawn edge and clear of any building standing in the park
            inner = lawn.buffer(-float(tc["park_edge_m"])).difference(buildings.buffer(float(tc["park_edge_m"])))
            if want <= have or inner.is_empty:
                continue
            x0, y0, x1, y1 = inner.bounds
            n_try = int((want - have) * 6 * (x1 - x0) * (y1 - y0) / max(inner.area, 1.0)) + 20
            xs, ys = rng.uniform(x0, x1, n_try), rng.uniform(y0, y1, n_try)
            ok = shapely.contains_xy(inner, xs, ys)
            placed_xy = [tuple(p) for p in rec_xy]
            added = 0
            for x, y in zip(xs[ok], ys[ok]):
                if added >= want - have:
                    break
                if any((x - px) ** 2 + (y - py) ** 2 < gap * gap for px, py in placed_xy[-400:]):
                    continue
                name = names[int(rng.integers(len(names)))]
                a = models[name]
                scale = float(rng.uniform(*tc["park_tree_scale"]))
                out.append({"class": "vegetation", "asset": name, "x": round(float(x), 3), "y": round(float(y), 3),
                            "yaw_deg": float(rng.uniform(0, 360)), "scale": round(scale, 4), "motion": "static",
                            "rule": "park_scatter", "record": None, "height_m": round(a["height_m"] * scale, 2),
                            "crown_m": round(a["crown_diameter_m"] * scale, 2)})
                placed_xy.append((x, y))
                added += 1
            stats["scattered_in_parks"] += added
    return stats


def place_furniture(records: list[dict], net: Network, cfg: dict, walk, buildings, out: list) -> dict:
    fc = cfg["populate"]["furniture"]
    stats = {}

    def add(kind, x, y, yaw, rule, ident=None):
        if not shapely.contains_xy(walk, x, y):
            p = shapely.ops.nearest_points(Point(x, y), walk)[1]
            if p.distance(Point(x, y)) > float(fc["snap_m"]):
                stats[f"{kind}_dropped"] = stats.get(f"{kind}_dropped", 0) + 1
                return
            away = np.array([p.x - x, p.y - y])
            away = away / max(np.linalg.norm(away), 1e-9)
            x, y = p.x + 0.3 * away[0], p.y + 0.3 * away[1]
            if not shapely.contains_xy(walk, x, y):
                stats[f"{kind}_dropped"] = stats.get(f"{kind}_dropped", 0) + 1
                return
        if shapely.contains_xy(buildings, x, y):
            stats[f"{kind}_dropped"] = stats.get(f"{kind}_dropped", 0) + 1
            return
        stats[kind] = stats.get(kind, 0) + 1
        out.append({"class": "furniture", "asset": f"furn_{kind}", "x": round(float(x), 3), "y": round(float(y), 3),
                    "yaw_deg": float(yaw), "scale": 1.0, "motion": "static", "rule": rule, "record": ident})

    for r in records:
        s = net.nearest(r["x"], r["y"])
        _, t, _ = net.frame_at(s, s["line"].project(Point(r["x"], r["y"])))
        add(r["kind"], r["x"], r["y"], math.degrees(math.atan2(t[1], t[0])), "city_record", r["id"])
    for s in net.segs:
        if not len(s["d"]):
            continue
        for d in np.arange(s["d"][0], s["d"][-1], float(fc["lamp_spacing_m"])):
            k = int(np.argmin(np.abs(s["d"] - d)))
            c, t, n = net.frame_at(s, float(d))
            for side, hw in ((1, s["left"][k]), (-1, s["right"][k])):
                if hw <= 0.5:
                    continue
                p = c + side * n * (hw + float(fc["lamp_setback_m"]))
                add("street_lamp", p[0], p[1], math.degrees(math.atan2(-side * n[1], -side * n[0])), "lamp_rule")
    for key, ends in sorted(net.nodes.items()):
        legs = [(i, e) for i, e in ends if net.segs[i]["drivable"] and len(net.segs[i]["d"]) >= 2]
        if len(legs) < 3:
            continue
        for i, e in legs[:2]:
            s = net.segs[i]
            k = 0 if e == 0 else -1
            c, t, n = net.frame_at(s, float(s["d"][k]))
            side = -1 if e == 1 else 1                                 # right-hand side of the approach
            hw = s["right"][k] if side == -1 else s["left"][k]
            p = c + side * n * (hw + float(fc["signal_setback_m"]))
            add("traffic_signal", p[0], p[1], math.degrees(math.atan2(-side * n[1], -side * n[0])), "signal_rule")
    return stats


def build_markings(net: Network, cfg: dict) -> list[dict]:
    mc = cfg["populate"]["markings"]
    lw, cw = float(mc["lane_line_width_m"]), float(mc["centre_line_width_m"])
    dash, gap = (float(v) for v in mc["dash_m"])
    bar_w, bar_l = (float(v) for v in mc["crosswalk_bar_m"])
    feats = []

    def add(g, kind, seg):
        if not g.is_empty and shapely.covers(net.road, g):
            feats.append({"type": "Feature", "properties": {"kind": kind, "segment": seg},
                          "geometry": shapely.geometry.mapping(shapely.set_precision(g, 1e-4))})

    for s in net.segs:
        lanes = s.get("lanes") or []
        if not lanes or len(s["d"]) < 2:
            continue
        d0, d1 = float(s["d"][0]), float(s["d"][-1])
        for a, b in zip(lanes[:-1], lanes[1:]):
            off = (a["offset"] + b["offset"]) / 2
            if a["dir"] != b["dir"]:                                   # centre line between opposing lanes
                if d1 - d0 < 1.0:
                    continue
                part = shapely.ops.substring(s["line"], d0, d1).offset_curve(off)
                add(part.buffer(cw / 2, cap_style="flat"), "centre_line", s["id"])
            else:
                d = d0
                while d + dash <= d1:
                    c, t, n = net.frame_at(s, d + dash / 2)
                    p = c + n * off
                    add(rect1(p[0], p[1], math.degrees(math.atan2(t[1], t[0])), dash, lw), "lane_line", s["id"])
                    d += dash + gap
        # crosswalks at both ends that meet an intersection
        for end, d in ((0, d0), (1, d1)):
            if net.degree(s["xy"][0 if end == 0 else -1]) < 3:
                continue
            k = 0 if end == 0 else -1
            dd = d - (1 if end == 0 else -1) * (float(cfg["populate"]["intersection_margin_m"]) - float(mc["crosswalk_setback_m"]) - bar_l / 2)
            dd = float(np.clip(dd, bar_l / 2, s["len"] - bar_l / 2))
            c, t, n = net.frame_at(s, dd)
            yaw = math.degrees(math.atan2(t[1], t[0]))
            o = -s["right"][k] + bar_w
            while o <= s["left"][k] - bar_w:
                p = c + n * o
                add(rect1(p[0], p[1], yaw, bar_l, bar_w), "crosswalk", s["id"])
                o += float(mc["crosswalk_pitch_m"])
    return feats


def place_pedestrians(net: Network, manifest: dict, cfg: dict, rng, walk, buildings, static_veh: Grid,
                      mover_xy, mover_r, t, out: list, tracks: list) -> dict:
    pc = cfg["populate"]["pedestrians"]
    R = float(cfg["scene_radius_m"])
    models = sorted(n for n, a in manifest["assets"].items() if a["class"] == "person")
    area = walk.buffer(-float(pc["edge_margin_m"])).difference(buildings.buffer(float(pc["edge_margin_m"])))
    shapely.prepare(area)
    boost = float(pc["near_intersection_boost"])
    n_try = int(area.area * float(pc["density_per_m2"]) * (1 + boost) / (area.area / (4 * R * R)))
    xs, ys = rng.uniform(-R, R, n_try), rng.uniform(-R, R, n_try)
    u = rng.random(n_try)
    inside = shapely.contains_xy(area, xs, ys)
    corners = np.array([k for k, ends in sorted(net.nodes.items()) if len(ends) >= 3])
    ctree = STRtree(shapely.points(corners))
    veh_tree = STRtree(static_veh.geoms) if static_veh.geoms else None
    cell, taken = float(pc["spacing_m"]), set()
    stats = {"static": 0, "walking": 0}
    for x, y, uu in zip(xs[inside], ys[inside], u[inside]):
        pt = Point(x, y)
        dc = pt.distance(ctree.geometries[ctree.nearest(pt)])
        if uu > (1 + boost * math.exp(-dc / float(pc["boost_range_m"]))) / (1 + boost):
            continue
        key = (int(x // cell), int(y // cell))
        if any((key[0] + i, key[1] + j) in taken for i in (-1, 0, 1) for j in (-1, 0, 1)):
            continue
        clear = float(pc["vehicle_clearance_m"])
        if veh_tree is not None and len(veh_tree.query(pt, predicate="dwithin", distance=clear)):
            continue
        name = models[int(rng.integers(len(models)))]
        yaw = float(rng.uniform(0, 360))
        walking = rng.random() < float(pc["walking_share"])
        speed = float(rng.uniform(*pc["walk_speed"]))
        flip = rng.random() < 0.5
        track = None
        if walking:
            s = net.nearest(x, y)
            _, tan, _ = net.frame_at(s, s["line"].project(pt))
            for sign in ((-1, 1) if flip else (1, -1)):
                dxy = sign * tan * speed
                px, py = x + dxy[0] * t, y + dxy[1] * t
                chk = np.arange(0, len(t), int(pc["track_check_every"]))
                chk = np.unique(np.append(chk, len(t) - 1))
                if shapely.contains_xy(area, px[chk], py[chk]).all() and (
                        veh_tree is None or veh_tree.query(shapely.points(px[chk], py[chk]), predicate="dwithin",
                                                           distance=clear).shape[1] == 0):
                    yaw = math.degrees(math.atan2(dxy[1], dxy[0]))
                    track = np.column_stack([px, py, np.full(len(t), yaw)])
                    break
        if len(mover_r):
            p0 = track[:, :2] if track is not None else np.array([[x, y]])
            if (np.linalg.norm(mover_xy - p0[None], axis=2) < (mover_r + clear)[:, None]).any():
                continue
        taken.add(key)
        rec = {"class": "person", "asset": name, "x": float(x), "y": float(y), "yaw_deg": yaw, "scale": 1.0,
               "motion": "recorded" if track is not None else "static",
               "rule": "walking" if track is not None else "standing"}
        if track is not None:
            rec["speed_mps"] = speed
            tracks.append(track)
            stats["walking"] += 1
        else:
            stats["static"] += 1
        out.append(rec)
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_populate(location: str, cfg: dict, radius: float | None = None, out_dir=None) -> dict:
    slug = slugify(location)
    cfg = for_location(cfg, location)
    sdir = scene_dir(cfg, slug)
    odir = out_dir or sdir
    pc = cfg["populate"]
    manifest = load_manifest(cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    surf = json.loads((sdir / "surfaces.geojson").read_text())["features"]
    by = {}
    for f in surf:
        by.setdefault(f["properties"]["class"], []).append(shape(f["geometry"]))
    clip = Point(0, 0).buffer(radius, quad_segs=64) if radius else None

    def union(classes):
        u = unary_union([g for c in classes for g in by.get(c, [])])
        return u.intersection(clip) if clip is not None else u

    road = union(["road"])
    buildings = unary_union([shape(f["geometry"]) for f in json.loads((sdir / "buildings.geojson").read_text())["features"]])
    shapely.prepare(buildings)
    lines = json.loads((sdir / "centerlines.geojson").read_text())["features"]
    net = Network(lines, road, cfg)
    t = np.arange(int(pc["frames"])) / float(cfg["clip"]["fps"])

    objects, veh_tracks, ped_tracks = [], [], []
    grid = Grid()
    fleet = Fleet(manifest, cfg, rng)
    lots = [g.intersection(clip) if clip is not None else g for g in by.get("parking", [])]
    place_parked(net, fleet, [g for g in lots if not g.is_empty and g.geom_type == "Polygon"], cfg, rng, grid, objects, buildings)
    n_parked = len(objects)
    mover_xy, mover_r, mover_why = place_moving(net, fleet, cfg, rng, grid, objects, veh_tracks, t, buildings)
    n_moving = len(objects) - n_parked

    trees = json.loads((sdir / "trees.json").read_text())
    furn = json.loads((sdir / "furniture.json").read_text())
    if radius:
        trees = [r for r in trees if math.hypot(r["x"], r["y"]) <= radius]
        furn = [r for r in furn if math.hypot(r["x"], r["y"]) <= radius]
    lawns = [g for g in by.get("lawn", []) if g.geom_type == "Polygon"]
    tree_stats = place_trees(trees, manifest, cfg, road, buildings, objects, lawns, rng)
    furn_walk = union(pc["furniture"]["surfaces"])
    shapely.prepare(furn_walk)
    furn_stats = place_furniture(furn, net, cfg, furn_walk, buildings, objects)
    walk = union(pc["pedestrians"]["surfaces"])
    ped_stats = place_pedestrians(net, manifest, cfg, rng, walk, buildings, grid, mover_xy, mover_r, t, objects, ped_tracks)
    markings = build_markings(net, cfg)

    ids, k = [], {}
    for o in objects:
        k[o["class"]] = k.get(o["class"], 0) + 1
        o["id"] = f"{o['class']}_{k[o['class']]:05d}"
        for key in ("x", "y", "yaw_deg"):
            o[key] = round(float(o[key]), 4)
    moving = [o for o in objects if o["motion"] == "recorded"]
    tracks = veh_tracks + ped_tracks
    assert len(moving) == len(tracks)
    arr = np.round(np.asarray(tracks, dtype=np.float64), 4).astype(np.float32) if tracks else np.zeros((0, len(t), 3), np.float32)
    ids = np.array([o["id"] for o in moving])
    np.savez_compressed(odir / "motion.npz", ids=ids, poses=arr)
    stats = {"vehicles_parked": n_parked, "vehicles_moving": n_moving, "moving_candidates": mover_why, "trees": tree_stats, "furniture": furn_stats,
             "pedestrians": ped_stats, "markings": {kk: sum(1 for m in markings if m["properties"]["kind"] == kk)
                                                    for kk in ("lane_line", "centre_line", "crosswalk")},
             "by_class": k}
    out = {"location": slug, "seed": int(cfg["seed"]), "radius_m": radius or float(cfg["scene_radius_m"]),
           "radius_limit_m": radius,
           "frames": int(pc["frames"]), "fps": float(cfg["clip"]["fps"]),
           "asset_revision": manifest["source_revision"], "stats": stats, "objects": objects}
    (odir / "placements.json").write_text(json.dumps(out, separators=(",", ":"), sort_keys=True) + "\n")
    (odir / "markings.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": markings},
                                                      separators=(",", ":"), sort_keys=True) + "\n")
    log.info("placed %s", json.dumps(stats))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Place objects in the scene")
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--radius", type=float, default=None, help="limit placement to this radius (prototype)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_populate(a.location, load_config(a.config), a.radius)


if __name__ == "__main__":
    main()
