"""Probe point selection for the depth probe. CPU only, from the building data, no rendering."""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.strtree import STRtree

from .common import altitude_m, for_location, load_config, scene_dir, slugify

log = logging.getLogger("probes")


def pick_probes(feats: list[dict], cfg: dict, objects: list[dict] | None = None, water=None) -> dict:
    """`objects` are the placed models of a populated scene; the ground probe avoids them.

    `water` is the scene's water surface, if any. The ground probe stays on land: a frame of
    open water has nothing in it to judge exposure or labels by.
    """
    pc = cfg["probe"]
    geoms = [shape(f["geometry"]) for f in feats]
    tree = STRtree(geoms)
    obj_xy = np.array([[o["x"], o["y"]] for o in objects or []], dtype=np.float64).reshape(-1, 2)
    # an object keeps the probe away by its own half-size (a tree by its crown) plus the margin
    obj_r = np.array([o.get("crown_m", 6.0) / 2 for o in objects or []], dtype=np.float64) + float(pc["ground_object_clear_m"])

    # P_ground: nearest grid point to the target with no building within the free radius
    step, free, R = float(pc["ground_search_step_m"]), float(pc["ground_free_radius_m"]), float(cfg["scene_radius_m"])
    ax = np.arange(-R / 2, R / 2 + step, step)
    X, Y = np.meshgrid(ax, ax)
    pts = shapely.points(X.ravel(), Y.ravel())
    order = np.lexsort((Y.ravel(), X.ravel(), np.round(np.hypot(X.ravel(), Y.ravel()), 6)))
    p_ground = None
    for i in order:
        near = tree.query(pts[i], predicate="dwithin", distance=free)
        if len(near) == 0 and water is not None and pts[i].distance(water) < free:
            continue
        if len(near) == 0 and len(obj_xy):
            if (np.hypot(obj_xy[:, 0] - X.ravel()[i], obj_xy[:, 1] - Y.ravel()[i]) < obj_r).any():
                continue
        if len(near) == 0:
            p_ground = (float(X.ravel()[i]), float(Y.ravel()[i]))
            nearest = float(pts[i].distance(geoms[tree.nearest(pts[i])]))
            break
    if p_ground is None:
        raise RuntimeError("no ground probe point found")

    # P_roof: well inside the footprint of a building with a measured height in range
    lo, hi = (float(v) for v in pc["roof_height_range_m"])
    inset = float(pc["roof_inset_m"])
    cands = []
    for f, g in zip(feats, geoms):
        p = f["properties"]
        if p["height_imputed"] or not (lo <= p["height_m"] <= hi):
            continue
        inner = g.buffer(-inset)
        if inner.is_empty:
            continue
        rp = inner.representative_point()
        # the point must be covered by this building only
        hits = tree.query(rp, predicate="intersects")
        if len(hits) != 1:
            continue
        cands.append((round(float(np.hypot(rp.x, rp.y)), 6), p["doitt_id"], rp, p))
    if not cands:
        raise RuntimeError("no roof probe building found")
    cands.sort(key=lambda c: c[:2])
    _, _, rp, p = cands[0]

    above = float(pc["roof_camera_above_m"])
    alts = [altitude_m(a) for a in cfg["altitudes_ft"]]
    probes = [
        {"name": f"ground_{int(a_ft)}ft", "position": [p_ground[0], p_ground[1], a_m],
         "expect_depth_m": a_m, "expect_label": "ground"}
        for a_ft, a_m in zip(cfg["altitudes_ft"], alts)
    ]
    probes.append({"name": "roof", "position": [float(rp.x), float(rp.y), p["height_m"] + above],
                   "expect_depth_m": above, "expect_label": "building",
                   "doitt_id": p["doitt_id"], "bin": p["bin"], "height_m": p["height_m"]})
    return {"p_ground": {"xy": list(p_ground), "nearest_building_m": nearest},
            "p_roof": {"xy": [float(rp.x), float(rp.y)], "doitt_id": p["doitt_id"],
                       "height_m": p["height_m"], "inset_m": inset},
            "probes": probes}


def run(location: str, cfg: dict):
    cfg = for_location(cfg, location)
    sdir = scene_dir(cfg, slugify(location))
    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    placed = sdir / "placements.json"
    objects = json.loads(placed.read_text())["objects"] if cfg.get("populate", {}).get("enabled") and placed.exists() else None
    water = None
    if objects is not None and (sdir / "surfaces.geojson").exists():
        from shapely.ops import unary_union
        w = [shape(f["geometry"]) for f in json.loads((sdir / "surfaces.geojson").read_text())["features"]
             if f["properties"]["class"] == "water"]
        water = unary_union(w) if w else None
    out = pick_probes(feats, cfg, objects, water)
    (sdir / "probes.json").write_text(json.dumps(out, indent=1) + "\n")
    for pr in out["probes"]:
        log.info("%s at %s, expect %.2f m, %s", pr["name"], [round(v, 2) for v in pr["position"]],
                 pr["expect_depth_m"], pr["expect_label"])
    return sdir / "probes.json"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(a.location, load_config(a.config))
