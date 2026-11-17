"""Checks for the populated scene: placement rules, grounding, labels, instances, depth, budget."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import shapely
from pxr import Usd, UsdGeom, UsdSemantics
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from _checks import REPO_ROOT, CheckList

from nyc_drone_sim.assets import load_manifest
from nyc_drone_sim.common import for_location, altitude_m, load_config, scene_dir, slugify
from nyc_drone_sim.populate import rects, run_populate
from nyc_drone_sim.postrender import bundle_dir, id_to_class

NEW_OBJECT_CLASSES = ("vehicle", "person", "vegetation", "furniture")


def project(c2w: np.ndarray, intr: dict, pts: np.ndarray):
    """World points -> pixel coordinates and distance along the optical axis."""
    pc = (np.linalg.inv(c2w) @ np.column_stack([pts, np.ones(len(pts))]).T).T
    z = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        px = intr["cx"] + intr["fx"] * pc[:, 0] / z
        py = intr["cy"] - intr["fy"] * pc[:, 1] / z
    return px, py, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--run", required=True, help="render run of the populated scene")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, choices=["street", "long"], help="clip variant of the render run")
    ap.add_argument("--only", default=None, help="comma-separated check ids for a partial run; writes no report")
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    sdir = scene_dir(cfg, slug)
    g = CheckList("populate", slug, a.variant, run=a.run)
    man = load_manifest(cfg)["assets"]
    placed = json.loads((sdir / "placements.json").read_text())
    objs = placed["objects"]
    motion = np.load(sdir / "motion.npz")
    poses = {i: p for i, p in zip(motion["ids"].tolist(), motion["poses"])}
    surf = json.loads((sdir / "surfaces.geojson").read_text())["features"]
    by = {}
    for f in surf:
        by.setdefault(f["properties"]["class"], []).append(shape(f["geometry"]))
    U = {c: unary_union(v) for c, v in by.items()}
    bfeats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    bgeoms = [shape(f["geometry"]) for f in bfeats]
    buildings = unary_union(bgeoms)
    drive = unary_union([U["road"]] + ([U["parking"]] if "parking" in U else []))
    for geom in (buildings, drive, U["road"]):
        shapely.prepare(geom)
    veh = [o for o in objs if o["class"] == "vehicle"]
    size = {o["id"]: man[o["asset"]]["size_m"] for o in objs}
    nframes = int(placed["frames"])
    alts = [int(x) for x in cfg["altitudes_ft"]]

    def p1():
        barea = buildings.area
        r = U["road"].intersection(buildings).area / barea
        s = U["sidewalk"].intersection(buildings).area / barea
        both = unary_union([U["road"], U["sidewalk"]]).intersection(buildings).area / barea
        ok = r < 0.01 and s < 0.01
        return ok, (f"overlap with building footprints as a share of building area, judged per layer: roadbed {r:.2%}, "
                    f"sidewalk {s:.2%} (limit 1% each); the two together {both:.2%}")

    def p13():
        # A projection or datum offset between layers shows as overlap piled on one side of the
        # buildings, and as a small shift that removes much of it. Two datasets that merely
        # disagree spread their overlap around, and no shift helps.
        det, ok = [], True
        for cls in ("sidewalk", "road"):
            layer = U[cls]
            base = layer.intersection(buildings).area
            vec, tot = np.zeros(2), 0.0
            for b in bgeoms:
                if not layer.intersects(b):
                    continue
                o = layer.intersection(b)
                d = np.array([o.centroid.x - b.centroid.x, o.centroid.y - b.centroid.y]) if o.area > 0.05 else np.zeros(2)
                n = np.linalg.norm(d)
                if n > 1e-6:
                    vec += o.area * d / n
                    tot += o.area
            one_sided = float(np.linalg.norm(vec) / tot) if tot else 0.0
            shifts = {(dx, dy): shapely.affinity.translate(layer, dx, dy).intersection(buildings).area
                      for dx in (-1.0, -0.5, 0.0, 0.5, 1.0) for dy in (-1.0, -0.5, 0.0, 0.5, 1.0)}
            best = min(shifts, key=shifts.get)
            gain = 1 - shifts[best] / base if base else 0.0
            # with almost no overlap to begin with, a ratio means nothing: a few square metres
            # moving in or out is not an offset. Judge only overlap above 0.1% of building area.
            material = base / buildings.area >= 0.001
            good = gain < 0.15 or not material
            ok &= good
            det.append(f"{cls}: overlap {base / buildings.area:.2%} of building area, one-sidedness {one_sided:.2f} "
                       f"(0 even, 1 all on one side); best shift within 1 m is east {best[0]:+.1f}, north {best[1]:+.1f} m, "
                       f"which removes {gain:.0%} of the overlap (a real offset would remove far more than 15%)"
                       + ("" if material else "; overlap is under 0.1% of building area, too small to judge by ratio"))
        return ok, "; ".join(det)

    vids = [o["id"] for o in veh]
    vlen = np.array([size[i][0] for i in vids])
    vwid = np.array([size[i][1] for i in vids])
    is_mover = np.array([i in poses for i in vids])
    static_polys = rects([o["x"] for o in veh], [o["y"] for o in veh], [o["yaw_deg"] for o in veh], vlen, vwid)[~is_mover]
    mover_track = np.stack([poses[i] for i in vids if i in poses]) if is_mover.any() else np.zeros((0, nframes, 3))
    static_tree = STRtree(static_polys)

    def mover_polys(f):
        return rects(mover_track[:, f, 0], mover_track[:, f, 1], mover_track[:, f, 2], vlen[is_mover], vwid[is_mover])

    def overlapping(tree, geoms, same: bool) -> int:
        """Pairs of footprints sharing more than a hair of area."""
        ia, ib = tree.query(geoms, predicate="intersects")
        if same:
            keep = ia < ib
            ia, ib = ia[keep], ib[keep]
        if not len(ia):
            return 0
        return int((shapely.area(shapely.intersection(geoms[ia], tree.geometries[ib])) > 1e-6).sum())

    def p2():
        outside = int((~shapely.covers(drive, static_polys)).sum())
        in_building = int(shapely.intersects(buildings, static_polys).sum())
        overlaps = overlapping(static_tree, static_polys, True)
        worst_frame = None
        for f in range(nframes if len(mover_track) else 0):
            pol = mover_polys(f)
            in_building += int(shapely.intersects(buildings, pol).sum())
            bad = int((~shapely.covers(drive, pol)).sum()) + overlapping(static_tree, pol, False) + overlapping(STRtree(pol), pol, True)
            if bad:
                outside += int((~shapely.covers(drive, pol)).sum())
                overlaps += bad - int((~shapely.covers(drive, pol)).sum())
                worst_frame = f if worst_frame is None else worst_frame
        ok = outside == 0 and overlaps == 0 and in_building == 0
        return ok, (f"{len(veh)} vehicles ({int((~is_mover).sum())} static, {int(is_mover.sum())} moving, every one of "
                    f"{nframes} frames tested): footprints outside roadbed and parking lots {outside}, overlapping pairs {overlaps}, "
                    f"touching a building {in_building}"
                    + (f"; first bad frame {worst_frame}" if worst_frame is not None else ""))

    lines = json.loads((sdir / "centerlines.geojson").read_text())["features"]
    segs = [(LineString(np.asarray(f["geometry"]["coordinates"])[:, :2]), f["properties"]) for f in lines]
    # streets at ground level: a tunnel or viaduct centre line that runs under or over a street is
    # not the street a vehicle on the surface is following
    grade = int(cfg["populate"]["ground_level_code"])
    drivable = [(ln, p) for ln, p in segs if p["rw_type"] in cfg["populate"]["drivable_rw_types"]
                and p["trafdir"] in ("FT", "TF", "TW")
                and (p.get("from_level") or grade) == grade and (p.get("to_level") or grade) == grade]
    dtree = STRtree([ln for ln, _ in drivable])
    ends = shapely.points(np.array([c for ln, _ in segs for c in (ln.coords[0], ln.coords[-1])]))
    etree = STRtree(ends)

    def street_dir(x, y):
        pt = Point(x, y)
        j = int(dtree.nearest(pt))
        ln, p = drivable[j]
        d = ln.project(pt)
        a_, b_ = ln.interpolate(max(d - 0.5, 0)), ln.interpolate(min(d + 0.5, ln.length))
        t = np.array([b_.x - a_.x, b_.y - a_.y])
        t /= max(np.linalg.norm(t), 1e-9)
        side = t[0] * (y - ln.interpolate(d).y) - t[1] * (x - ln.interpolate(d).x)   # > 0: left of the digitised direction
        return t, p["trafdir"], side, float(pt.distance(ends[int(etree.nearest(pt))]))

    def p3():
        tol = math.cos(math.radians(10))
        margin = float(cfg["populate"]["intersection_margin_m"])
        bad_align = bad_dir = tested = moving_tested = 0
        worst = 0.0
        for o in veh:
            if o["rule"] not in ("parked_kerb", "moving_lane"):
                continue
            track = poses.get(o["id"])
            samples = [(o["x"], o["y"], o["yaw_deg"])] if track is None else track[::16].tolist() + [track[-1].tolist()]
            for x, y, yaw in samples:
                t, trafdir, side, dnode = street_dir(x, y)
                if dnode < margin:
                    continue                      # inside an intersection the nearest centre line may be the cross street
                h = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
                c = float(h @ t)
                tested += 1
                worst = max(worst, math.degrees(math.acos(min(1.0, abs(c)))))
                bad_align += abs(c) < tol
                if track is not None:
                    moving_tested += 1
                    want = {"FT": 1, "TF": -1}.get(trafdir)
                    if want is None:                 # two-way: keep right; on the centre line either way is fine
                        if abs(side) < 0.5:
                            continue
                        want = 1 if side < 0 else -1
                    bad_dir += (c > 0) != (want > 0)
        ok = bad_align == 0 and bad_dir == 0 and tested > 0
        return ok, (f"{tested} on-street poses tested away from intersections: worst angle to the street {worst:.2f} deg "
                    f"(limit 10), misaligned {bad_align}; {moving_tested} moving poses, against the direction of travel {bad_dir}")

    def p4():
        peds = [o for o in objs if o["class"] == "person"]
        walk = unary_union([U[c] for c in cfg["populate"]["pedestrians"]["surfaces"] if c in U])
        shapely.prepare(walk)
        marks = [shape(m["geometry"]) for m in json.loads((sdir / "markings.geojson").read_text())["features"]
                 if m["properties"]["kind"] == "crosswalk"]
        cross = unary_union(marks) if marks else None
        body = 0.3
        frames = sorted(set(range(0, nframes, 4)) | {nframes - 1})
        # positions of every pedestrian at every tested frame: [pedestrian, frame, 2]
        xy = np.stack([poses[o["id"]][frames, :2] if o["id"] in poses else np.tile([o["x"], o["y"]], (len(frames), 1))
                       for o in peds]).astype(np.float64)
        on = shapely.contains_xy(walk, xy[..., 0], xy[..., 1])
        if cross is not None:
            on |= shapely.contains_xy(cross, xy[..., 0], xy[..., 1])
        off = int((~on).any(1).sum())
        in_building = int(shapely.contains_xy(buildings, xy[..., 0], xy[..., 1]).any(1).sum())
        touched = np.zeros(len(peds), dtype=bool)
        for k, f in enumerate(frames):
            pts = shapely.points(xy[:, k, 0], xy[:, k, 1])
            ia, _ = static_tree.query(pts, predicate="dwithin", distance=body)
            touched[ia] = True
            if len(mover_track):
                ia, _ = STRtree(mover_polys(f)).query(pts, predicate="dwithin", distance=body)
                touched[ia] = True
        hit = int(touched.sum())
        ok = off == 0 and in_building == 0 and hit == 0 and len(peds) > 0
        walking = sum(o["id"] in poses for o in peds)
        return ok, (f"{len(peds)} pedestrians ({walking} walking, tested at {len(frames)} frames each): off sidewalk, plaza "
                    f"and crosswalk {off}, inside a building {in_building}, touching a vehicle {hit}")

    def p5():
        trees = [o for o in objs if o["class"] == "vegetation"]
        bad = sum(bool(shapely.contains_xy(U["road"], o["x"], o["y"]) or shapely.contains_xy(buildings, o["x"], o["y"])) for o in trees)
        scattered = sum(o["rule"] == "park_scatter" for o in trees)
        h = np.array([o["height_m"] for o in trees])
        share = 1 - bad / max(len(trees), 1)
        ok = share >= 0.95 and len(trees) > 0 and h.min() >= 3.0 and h.max() <= 30.0
        st = placed["stats"]["trees"]
        return ok, (f"{len(trees)} trees placed from {st['records']} records ({st['moved_over_kerb']} recorded just inside "
                    f"the roadbed line were moved onto the kerb, {st['dropped_in_road_or_building']} deeper in the roadbed or "
                    f"inside a building were not placed; {scattered} more were scattered over park lawns by rule); "
                    f"placed trees outside roadbed and buildings {share:.1%}; "
                    f"heights {h.min():.1f}-{h.max():.1f} m, median {np.median(h):.1f} m")

    stage = Usd.Stage.Open(str(sdir / "scene.usdc"))

    def p6():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        worst, n, bad = 0.0, 0, []
        for group in stage.GetPrimAtPath("/World/Objects").GetChildren():
            for prim in group.GetChildren():
                z = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2]
                n += 1
                if abs(z) > worst:
                    worst = abs(z)
                if abs(z) > 0.05:
                    bad.append(prim.GetName())
        ok = n == len(objs) and not bad
        return ok, f"{n} placed models read back from the USD (placement list has {len(objs)}): largest |lowest z| {worst * 100:.2f} cm, over 5 cm: {len(bad)} {bad[:3]}"

    def bundles():
        return {k: bundle_dir(cfg, a.run, slug, k) for k in alts}

    def p7():
        counts, bad = {}, []
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            found, q = [], prim
            while q and q.GetPath().pathElementCount > 0:
                api = UsdSemantics.LabelsAPI(q, "class")
                attr = api.GetLabelsAttr()
                if attr and attr.HasAuthoredValue():
                    found.append(list(attr.Get()))
                    break
                q = q.GetParent()
            if len(found) != 1 or len(found[0]) != 1:
                bad.append(str(prim.GetPath()))
            else:
                counts[found[0][0]] = counts.get(found[0][0], 0) + 1
        det = [f"USD: {sum(counts.values())} meshes (model copies included) each resolve to one class {dict(sorted(counts.items()))}; without exactly one: {len(bad)}"]
        ok = not bad
        # rendered: every new class with a visible object in a sampled frame must appear in that frame
        for k, B in bundles().items():
            b = json.loads((B / "bundle.json").read_text())
            path = json.loads((B / "path.json").read_text())
            intr = path["intrinsics"]
            ids = {c: i for i, c in id_to_class(B).items()}
            n = len(path["frames"])
            expect, seen, hits, vis = {}, {}, {}, {}
            for i in sorted(set(np.linspace(0, n - 1, 8).round().astype(int).tolist())):
                c2w = np.array(path["frames"][i]["camera_to_world"])
                sem = np.load(B / "semantic" / f"semantic_{i:04d}.npz")["data"]
                depth = np.load(B / "metric_depth" / f"depth_{i:04d}.npy")
                for cls in NEW_OBJECT_CLASSES:
                    os_ = [o for o in objs if o["class"] == cls]
                    if not os_:
                        continue
                    hgt = np.array([man[o["asset"]]["size_m"][2] * o["scale"] for o in os_])
                    xy = np.array([poses[o["id"]][i, :2] if o["id"] in poses else (o["x"], o["y"]) for o in os_])
                    px, py, z = project(c2w, intr, np.column_stack([xy, hgt * 0.75]))
                    inside = (z > 0) & (px >= 8) & (px < intr["width"] - 8) & (py >= 8) & (py < intr["height"] - 8)
                    for j in np.nonzero(inside)[0]:
                        u, v = int(px[j]), int(py[j])
                        if depth[v, u] < z[j] - 2.5:
                            continue                                   # something nearer hides it
                        vis[cls] = vis.get(cls, 0) + 1
                        win = sem[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
                        hits[cls] = hits.get(cls, 0) + int(ids.get(cls, -1) in win)
                        expect.setdefault(cls, set()).add(i)
                    if ids.get(cls, -1) in sem:
                        seen.setdefault(cls, set()).add(i)
            missing = {c: sorted(expect[c] - seen.get(c, set())) for c in expect if expect[c] - seen.get(c, set())}
            ok &= not missing and all(c in ids for c in expect)
            surf_seen = sorted(c for c in ids if c in cfg["scene"]["flat_classes"])
            det.append(f"{k} ft render: classes in the segmentation {sorted(ids)}; per class, visible objects whose pixel is "
                       f"labelled with the class: " + ", ".join(f"{c} {hits.get(c, 0)}/{vis[c]}" for c in sorted(vis))
                       + f"; frames where a visible class is absent: {missing or 'none'}; ground-level classes {surf_seen}")
        return ok, "; ".join(det)

    pdir = sdir / "populate_probe" / "scene"
    probe_result = sdir / "probe" / "probe_result.json"
    MISSING = "not evaluated: probe renders missing"

    def frame_view(B, i):
        """Camera matrix, semantic ids, class-of-id map and vehicle positions for frame i of a clip."""
        b = json.loads((B / "bundle.json").read_text())
        c2w = np.array(json.loads((B / "path.json").read_text())["frames"][i]["camera_to_world"])
        ids = id_to_class(B)
        sem = np.load(B / "semantic" / f"semantic_{i:04d}.npz")["data"]
        xy = np.array([poses[o["id"]][i][:2] if o["id"] in poses else (o["x"], o["y"]) for o in veh])
        return c2w, sem, ids, xy

    def p8_clip():
        # Two neighbouring vehicles in a frame of the clip itself: the writer's instance
        # segmentation must give them different colours.
        from PIL import Image
        B = bundles()[alts[0]]
        n = json.loads((B / "bundle.json").read_text())["clip"]["frames"]
        tops = np.array([size[o["id"]][2] * 0.8 for o in veh])
        for i in range(0, n, max(1, n // 12)):
            c2w, sem, ids, xy = frame_view(B, i)
            H, W = sem.shape
            px, py, z = project(c2w, intrinsics(), np.column_stack([xy, tops]))
            inside = np.nonzero((z > 0) & (px > 8) & (px < W - 8) & (py > 8) & (py < H - 8))[0]
            seen = [j for j in inside if ids.get(int(sem[int(py[j]), int(px[j])])) == "vehicle"]
            seg = None
            for a_ in seen:
                near = [b_ for b_ in seen if b_ != a_ and 2.5 < np.hypot(*(xy[a_] - xy[b_])) < 9.0]
                if not near:
                    continue
                b_ = near[0]
                if seg is None:
                    seg = np.asarray(Image.open(B / "clip_0000" / "segmentation" / f"segmentation_{i:04d}.png").convert("RGB"))
                cols = []
                for j in (a_, b_):
                    win = seg[int(py[j]) - 2:int(py[j]) + 3, int(px[j]) - 2:int(px[j]) + 3].reshape(-1, 3)
                    c, cnt = np.unique(win, axis=0, return_counts=True)
                    cols.append(tuple(int(v) for v in c[np.argmax(cnt)]))
                return cols[0] != cols[1], (
                    f"from the {alts[0]} ft clip, frame {i}: {veh[a_]['id']} ({veh[a_]['asset']}) and {veh[b_]['id']} "
                    f"({veh[b_]['asset']}), {np.hypot(*(xy[a_] - xy[b_])):.1f} m apart, projected with the saved camera pose; "
                    f"instance colours {cols[0]} vs {cols[1]}, distinct {cols[0] != cols[1]}")
        return None, "not evaluated: no frame of the clip shows two neighbouring vehicles"

    def p9_clip():
        # Flat-roofed vehicles in a frame of the clip: metric depth at the projected roof centre
        # against the depth expected from the model's roof height and the saved camera pose.
        from nyc_drone_sim.shotlist import roof_height
        lib = REPO_ROOT / cfg["assets"]["library_dir"]
        B = bundles()[alts[0]]
        n = json.loads((B / "bundle.json").read_text())["clip"]["frames"]
        flat_roof = [j for j, o in enumerate(veh) if man[o["asset"]]["group"] in ("van", "bus", "truck")]
        roofs, tested = {}, []
        for i in range(0, n, max(1, n // 12)):
            c2w, sem, ids, xy = frame_view(B, i)
            H, W = sem.shape
            for j in flat_roof:
                o = veh[j]
                if o["asset"] not in roofs:
                    roofs[o["asset"]] = roof_height(lib / man[o["asset"]]["file"])
                h = roofs[o["asset"]] * float(o.get("scale", 1.0))
                px, py, z = project(c2w, intrinsics(), np.array([[xy[j][0], xy[j][1], h]]))
                u, v = int(px[0]), int(py[0])
                if not (z[0] > 0 and 8 < u < W - 8 and 8 < v < H - 8):
                    continue
                if any(ids.get(int(k)) != "vehicle" for k in np.unique(sem[v - 2:v + 3, u - 2:u + 3])):
                    continue                      # hidden by a tree or at the roof's edge
                d = np.load(B / "metric_depth" / f"depth_{i:04d}.npy")
                tested.append((abs(float(d[v, u]) - float(z[0])), o["id"], o["asset"], i, float(z[0]), float(d[v, u]), h))
                if len(tested) >= 5:
                    break
            if len(tested) >= 5:
                break
        if not tested:
            return None, "not evaluated: no frame of the clip shows an unobstructed flat-roofed vehicle"
        worst = max(tested)
        return worst[0] <= 0.3, (
            f"from the {alts[0]} ft clip: {len(tested)} flat-roofed vehicles, roof centre projected with the saved camera pose; "
            f"largest depth error {worst[0] * 100:.1f} cm (limit 30) on {worst[1]} ({worst[2]}) in frame {worst[3]}: roof "
            f"{worst[6]:.3f} m above ground, expected depth {worst[4]:.3f} m, rendered {worst[5]:.3f} m")

    def shot(name):
        plan = json.loads((pdir / "shots.json").read_text())
        s = next(x for x in plan["groups"][0]["shots"] if x["name"] == name)
        if (pdir / f"{name}.npz").stat().st_mtime < (sdir / "scene.usdc").stat().st_mtime:
            raise RuntimeError(f"{name} was rendered before the current scene.usdc was built")
        return s, np.load(pdir / f"{name}.npz"), json.loads((pdir / f"{name}.info.json").read_text())

    cam_intr = None

    def intrinsics():
        nonlocal cam_intr
        if cam_intr is None:
            from nyc_drone_sim.paths import Pinhole
            cam_intr = Pinhole(cfg["camera"]).intrinsics()
        return cam_intr

    def p8():
        if not (pdir / "scene_pair.npz").exists():
            return None, MISSING
        s, d, info = shot("scene_pair")
        c2w = np.array(s["camera_to_world"])
        out, ok = [], True
        for key in ("instance_id", "instance"):
            sets = []
            for o in s["objects"]:
                top = man[o["asset"]]["size_m"][2]
                px, py, _ = project(c2w, intrinsics(), np.array([[o["x"], o["y"], top * 0.8]]))
                u, v = int(px[0]), int(py[0])
                win = d[key][v - 4:v + 5, u - 4:u + 5]
                ids, cnt = np.unique(win, return_counts=True)
                sets.append(set(ids[cnt >= 3].tolist()))
            distinct = bool(sets[0]) and bool(sets[1]) and not (sets[0] & sets[1])
            if key == "instance_id":
                ok &= distinct                     # the writer's segmentation uses this annotator
            out.append(f"{key}: ids {sorted(sets[0])} vs {sorted(sets[1])}, distinct {distinct}")
        o0, o1 = s["objects"]
        return ok, f"two copies of {o0['asset']} ({o0['id']}, {o1['id']}) in one frame; " + "; ".join(out)

    def p9():
        if not (pdir / "scene_roof.npz").exists():
            return None, MISSING
        s, d, info = shot("scene_roof")
        H, W = d["depth"].shape[:2]
        got = float(np.mean(d["depth"][H // 2 - 1:H // 2 + 1, W // 2 - 1:W // 2 + 1]))
        cam_h = np.array(s["camera_to_world"])[2, 3]
        want = cam_h - s["roof_height_m"]
        ids = {int(i): v.get("class") for i, v in info["semantic"]["idToLabels"].items()}
        lab = ids.get(int(d["semantic"][H // 2, W // 2]))
        ok = abs(got - want) <= 0.3 and lab == "vehicle"
        return ok, (f"{s['objects'][0]['asset']} ({s['objects'][0]['id']}): roof {s['roof_height_m']:.3f} m above ground in its "
                    f"model file, camera at {cam_h:.2f} m, expected depth {want:.3f} m, rendered {got:.3f} m "
                    f"(error {abs(got - want) * 100:.1f} cm, limit 30); centre pixel class '{lab}'")

    def p10():
        with tempfile.TemporaryDirectory() as td:
            run_populate(a.location, cfg, placed.get("radius_limit_m"), out_dir=Path(td))
            again = json.loads((Path(td) / "placements.json").read_text())
            m2 = np.load(Path(td) / "motion.npz")
            same_list = again["objects"] == objs
            same_bytes = (Path(td) / "placements.json").read_bytes() == (sdir / "placements.json").read_bytes()
            same_motion = np.array_equal(m2["ids"], motion["ids"]) and np.array_equal(m2["poses"], motion["poses"])
            same_marks = (Path(td) / "markings.geojson").read_bytes() == (sdir / "markings.geojson").read_bytes()
        ok = same_list and same_bytes and same_motion and same_marks
        return ok, (f"rebuilt with seed {placed['seed']}: placement list identical {same_list} ({len(objs)} objects), file "
                    f"byte-identical {same_bytes}, recorded motion identical {same_motion}, road paint identical {same_marks}")

    def p11():
        if not probe_result.exists():
            return None, MISSING
        det, ok = [], True
        pr = json.loads((sdir / "probe" / "probe_result.json").read_text())
        if (sdir / "probe" / "probe_result.json").stat().st_mtime < (sdir / "scene.usdc").stat().st_mtime:
            raise RuntimeError("probe outputs are older than the current scene.usdc")
        share = pr["vram_peak_mib"] / pr["vram_total_mib"]
        ok &= share < 0.8
        det.append(f"probe job: peak VRAM {pr['vram_peak_mib']:.0f} of {pr['vram_total_mib']:.0f} MiB ({share:.1%}), sampled every "
                   f"0.5 s while the full scene loaded and rendered on {pr['gpu_name']}; scene load {pr['scene_load_s']:.1f} s")
        for k, B in bundles().items():
            b = json.loads((B / "bundle.json").read_text())
            name, used, total = [v.strip() for v in b["gpu"].split(",")][:3]
            ok &= float(used) / float(total) < 0.8
            det.append(f"{k} ft clip: {float(used):.0f} MiB in use when the render ended ({float(used) / float(total):.1%}), "
                       f"{b['timing_s']['render'] / b['clip']['frames']:.2f} s per frame over {b['clip']['frames']} frames, "
                       f"load and warm-up {b['timing_s']['load_and_warmup']:.0f} s, "
                       f"{b['mean_objects_moved_per_frame']:.0f} of {b['moving_objects']} moving objects updated per frame")
        return ok, "; ".join(det) + "; limit 80% of the card"

    def p12():
        if not probe_result.exists():
            return None, MISSING
        det, ok = [], True
        for st in ("data", "scene", "probe", "path", "render"):
            if st == "probe":
                res = sdir / "probe" / "probe_result.json"
                if res.stat().st_mtime < (sdir / "scene.usdc").stat().st_mtime:
                    ok = False
                    det.append("probe: probe outputs are older than the current scene.usdc")
                    continue
            cmd = [sys.executable, str(REPO_ROOT / "checks" / f"{st}.py"), "--location", a.location]
            if st == "render":
                cmd += ["--run", a.run]
            if st in ("path", "render") and a.variant:
                cmd += ["--variant", a.variant]
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines_ = [ln for ln in r.stdout.splitlines() if ln.startswith("[")]
            fails = [ln.split(":")[0] for ln in lines_ if ln.startswith("[FAIL]")]
            warns = [ln.split(":")[0] for ln in lines_ if ln.startswith("[WARN]")]
            ok &= r.returncode == 0
            det.append(f"{st}: exit {r.returncode}, {len(lines_)} checks, failed {fails or 'none'}, warnings {warns or 'none'}")
        return ok, "; ".join(det)

    for gid, fn in [("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5), ("P6", p6), ("P7", p7),
                    ("P8", p8), ("P8c", p8_clip), ("P9", p9), ("P9c", p9_clip), ("P10", p10), ("P11", p11),
                    ("P12", p12), ("P13", p13)]:
        if a.only is None or gid in a.only.split(","):
            g.run(gid, fn)
    if a.only is not None:
        sys.exit(0 if all(r["status"] != "FAIL" for r in g.results) else 1)
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
