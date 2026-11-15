"""Prepare the camera shots that verify the populated scene in the renderer. CPU only.

Two shot lists are written under <scene>/populate_probe/:
  unit/   a small test stage with one model placed three ways (plain reference, instanceable
          reference, point instancer), a tree, a pedestrian, a lamp and a car that moves
          between shots
  scene/  nadir shots in the real scene: over two copies of the same parked vehicle model, and
          over a flat-roofed vehicle whose roof height is known from its model file
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSemantics, UsdShade, Vt

from .assets import library_dir, load_manifest
from .common import altitude_m, for_location, load_config, scene_dir, slugify
from .dress import pose_matrix
from .paths import camera_to_world

UNIT_CAR = "veh_taxi_0"
UNIT_TREE = "tree_P_m"


def label(prim, name: str):
    UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr([name])


def roof_height(asset_file: Path) -> float:
    """Height of a model's top surface straight above its centre, from the model file."""
    stage = Usd.Stage.Open(str(asset_file))
    top = 0.0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            p = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
            t = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
            # barycentric coordinates of the origin in each triangle's xy projection
            den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
            ok = np.abs(den) > 1e-12
            den = np.where(ok, den, 1.0)
            u = ((b[:, 1] - c[:, 1]) * (-c[:, 0]) + (c[:, 0] - b[:, 0]) * (-c[:, 1])) / den
            v = ((c[:, 1] - a[:, 1]) * (-c[:, 0]) + (a[:, 0] - c[:, 0]) * (-c[:, 1])) / den
            inside = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
            if inside.any():
                z = u * a[:, 2] + v * b[:, 2] + (1 - u - v) * c[:, 2]
                top = max(top, float(z[inside].max()))
    return top


def build_unit_stage(cfg: dict, out: Path) -> dict:
    lib = library_dir(cfg)
    man = load_manifest(cfg)["assets"]
    sc = cfg["scene"]
    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    g = UsdGeom.Mesh.Define(stage, "/World/Ground")
    g.CreatePointsAttr([Gf.Vec3f(-200, -200, 0), Gf.Vec3f(200, -200, 0), Gf.Vec3f(200, 200, 0), Gf.Vec3f(-200, 200, 0)])
    g.CreateFaceVertexCountsAttr([3, 3])
    g.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    g.CreateDisplayColorAttr([Gf.Vec3f(*sc["surface_colors"]["sidewalk"])])
    label(g.GetPrim(), "ground")
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(sc["sun_intensity"]))
    sun.CreateAngleAttr(float(sc["sun_angle_deg"]))
    sx = UsdGeom.Xformable(sun)
    sx.AddRotateZOp().Set(float(sc["sun_azimuth_deg"]))
    sx.AddRotateXOp().Set(90.0 - float(sc["sun_elevation_deg"]))
    UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(float(sc["dome_intensity"]))

    objects = []

    def place(path, asset, x, y, yaw, cls, instanceable):
        xf = UsdGeom.Xform.Define(stage, path)
        xf.GetPrim().GetReferences().AddReference(str(lib / man[asset]["file"]))
        xf.GetPrim().SetInstanceable(instanceable)
        xf.AddTransformOp().Set(pose_matrix(x, y, yaw, 1.0))
        label(xf.GetPrim(), cls)
        objects.append({"path": path, "asset": asset, "class": cls, "x": x, "y": y, "yaw_deg": yaw,
                        "method": "instanceable" if instanceable else "reference"})

    for k, x in enumerate((-9.0, 0.0, 9.0)):
        place(f"/World/Ref/car_{k}", UNIT_CAR, x, 0.0, 0.0, "vehicle", False)
        place(f"/World/Inst/car_{k}", UNIT_CAR, x, 7.0, 0.0, "vehicle", True)
    pi = UsdGeom.PointInstancer.Define(stage, "/World/Points")
    proto = UsdGeom.Xform.Define(stage, "/World/Points/Prototypes/car")
    proto.GetPrim().GetReferences().AddReference(str(lib / man[UNIT_CAR]["file"]))
    pi.CreatePrototypesRel().AddTarget(proto.GetPath())
    pi.CreateProtoIndicesAttr([0, 0, 0])
    pi.CreatePositionsAttr([Gf.Vec3f(x, 14.0, 0.0) for x in (-9.0, 0.0, 9.0)])
    label(pi.GetPrim(), "vehicle")
    for k, x in enumerate((-9.0, 0.0, 9.0)):
        objects.append({"path": f"/World/Points[{k}]", "asset": UNIT_CAR, "class": "vehicle", "x": x, "y": 14.0,
                        "yaw_deg": 0.0, "method": "point_instancer"})
    place("/World/Mover/car", UNIT_CAR, -9.0, -8.0, 0.0, "vehicle", True)
    place("/World/Tree/tree", UNIT_TREE, -24.0, 6.0, 0.0, "vegetation", True)
    place("/World/People/ped_a", "ped_00", 18.0, 0.0, 0.0, "person", True)
    place("/World/People/ped_b", "ped_00", 18.0, 3.0, 90.0, "person", True)
    place("/World/Furniture/lamp", "furn_street_lamp", 18.0, 9.0, 180.0, "furniture", True)
    place("/World/Furniture/hydrant", "furn_hydrant", 22.0, 0.0, 0.0, "furniture", True)
    stage.GetRootLayer().Save()
    return {"objects": objects, "car": man[UNIT_CAR], "tree": man[UNIT_TREE], "ped": man["ped_00"],
            "lamp": man["furn_street_lamp"]}


def close_view(sdir: Path, cfg: dict, placed: list[dict], alt: float) -> dict | None:
    import shapely
    from shapely.geometry import LineString, shape
    from shapely.strtree import STRtree
    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    geoms = [shape(f["geometry"]) for f in feats]
    heights = [f["properties"]["height_m"] for f in feats]
    tree = STRtree(geoms)
    pitch = float(cfg["camera"]["pitch_deg"])
    back = alt / math.tan(math.radians(-pitch))
    trees = np.array([[o["x"], o["y"]] for o in placed if o["class"] == "vegetation"])
    peds = np.array([[o["x"], o["y"]] for o in placed if o["class"] == "person"])
    for o in placed:
        if o["class"] != "vehicle" or o["motion"] != "static" or math.hypot(o["x"], o["y"]) > 400:
            continue
        dt = np.hypot(trees[:, 0] - o["x"], trees[:, 1] - o["y"])
        dp = np.hypot(peds[:, 0] - o["x"], peds[:, 1] - o["y"])
        if not (((dt > 4) & (dt < 10)).any() and (dp < 9).sum() >= 3):
            continue
        for yaw in range(0, 360, 30):
            cx, cy = o["x"] - back * math.cos(math.radians(yaw)), o["y"] - back * math.sin(math.radians(yaw))
            ray = LineString([(cx, cy), (o["x"], o["y"])])
            clear = True
            for j in tree.query(ray, predicate="intersects"):
                # the sight line descends from the camera to the target; it must pass above the
                # building at the point of the crossing nearest the target, where the line is lowest
                pts = shapely.get_coordinates(ray.intersection(geoms[j]))
                if not len(pts):
                    continue
                far = float(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy).max())
                if alt * (1 - far / back) < heights[j] + 3.0:
                    clear = False
                    break
            if clear:
                return {"name": "scene_close", "camera_to_world": camera_to_world([cx, cy, alt], float(yaw), pitch).tolist(),
                        "moves": [], "objects": [o], "target": [o["x"], o["y"]]}
    return None


def run(location: str, cfg: dict) -> None:
    slug = slugify(location)
    cfg = for_location(cfg, location)
    sdir = scene_dir(cfg, slug)
    alt = altitude_m(min(cfg["altitudes_ft"]))
    lib = library_dir(cfg)
    man = load_manifest(cfg)["assets"]

    udir = sdir / "populate_probe" / "unit"
    udir.mkdir(parents=True, exist_ok=True)
    info = build_unit_stage(cfg, udir / "unit.usda")
    nadir = camera_to_world([0.0, 4.0, alt], 90.0, -90.0)
    shots = []
    for k, x in enumerate((-9.0, -4.0, 1.0)):
        shots.append({"name": f"unit_nadir_{k}", "camera_to_world": nadir.tolist(), "mover_x": x,
                      "moves": [{"path": "/World/Mover/car", "matrix": [float(v) for row in pose_matrix(x, -8.0, 0.0, 1.0) for v in row]}]})
    shots.append({"name": "unit_oblique", "camera_to_world": camera_to_world([0.0, -25.0, 30.0], 90.0, -45.0).tolist(),
                  "moves": []})
    (udir / "shots.json").write_text(json.dumps({"groups": [{"name": "unit", "stage": str(udir / "unit.usda"), "shots": shots}],
                                                 "unit": info, "camera_height_m": alt}, indent=1) + "\n")

    # the same test stage under the earlier, higher sun, for a before-and-after of the shadows
    before = cfg["scene"].get("sun_elevation_before_deg")
    if before is not None:
        bdir = sdir / "populate_probe" / "unit_sun_before"
        bdir.mkdir(parents=True, exist_ok=True)
        build_unit_stage({**cfg, "scene": {**cfg["scene"], "sun_elevation_deg": before}}, bdir / "unit.usda")
        (bdir / "shots.json").write_text(json.dumps({"groups": [{"name": "unit_sun_before", "stage": str(bdir / "unit.usda"),
                                                                 "shots": [shots[0], shots[-1]]}],
                                                     "sun_elevation_deg": before}, indent=1) + "\n")

    placed = json.loads((sdir / "placements.json").read_text())["objects"]
    trees = [(o["x"], o["y"], o["crown_m"]) for o in placed if o["class"] == "vegetation"]

    def open_sky(o, margin=1.5):
        return all(math.hypot(o["x"] - tx, o["y"] - ty) > c / 2 + margin for tx, ty, c in trees)

    cars = [o for o in placed if o["class"] == "vehicle" and o["motion"] == "static" and open_sky(o)]
    pair = None
    for i, a in enumerate(cars):
        for b in cars[i + 1:]:
            d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if a["asset"] == b["asset"] and 6.0 <= d <= 25.0:
                pair = (a, b)
                break
        if pair:
            break
    flat = next((o for o in cars if man[o["asset"]]["group"] in ("van", "bus", "truck")), None)
    sshots = []
    if pair:
        mx, my = (pair[0]["x"] + pair[1]["x"]) / 2, (pair[0]["y"] + pair[1]["y"]) / 2
        sshots.append({"name": "scene_pair", "camera_to_world": camera_to_world([mx, my, alt], 90.0, -90.0).tolist(),
                       "moves": [], "objects": [pair[0], pair[1]]})
    if flat:
        sshots.append({"name": "scene_roof", "camera_to_world": camera_to_world([flat["x"], flat["y"], alt], 90.0, -90.0).tolist(),
                       "moves": [], "objects": [flat], "roof_height_m": roof_height(lib / man[flat["asset"]]["file"])})
    # an oblique view at the working altitude and pitch, aimed at a spot that has a parked vehicle,
    # a tree and pedestrians close together, with no building in the line of sight
    close = close_view(sdir, cfg, placed, alt)
    if close:
        sshots.append(close)
    qdir = sdir / "populate_probe" / "scene"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "shots.json").write_text(json.dumps({"groups": [{"name": "scene", "stage": str(sdir / "scene.usdc"), "shots": sshots}],
                                                 "camera_height_m": alt}, indent=1) + "\n")
    print(f"unit shots {len(shots)}, scene shots {[s['name'] for s in sshots]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run(a.location, load_config(a.config))
