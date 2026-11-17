"""Add street-level content to the USD scene: ground surfaces, road paint and placed models.

CPU only. Called by the scene builder when `populate.enabled` is set and the layer and
placement files exist. Surfaces are cut so that no two of them overlap, and all lie exactly on
z = 0; road paint sits `populate.markings.height_m` above the road.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union

from .assets import library_dir, load_manifest

CLASS_ROOT = {"vehicle": "Vehicles", "person": "People", "vegetation": "Vegetation", "furniture": "Furniture"}


def triangulate(polys: list[Polygon], z: float) -> tuple[np.ndarray, np.ndarray]:
    P, T, base = [], [], 0
    for p in polys:
        rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in p.interiors]
        verts = np.vstack(rings).astype(np.float64)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        tri = earcut.triangulate_float64(verts, ends).reshape(-1, 3).astype(np.int64)
        # every triangle faces up
        a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
        flip = ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]
        P.append(np.column_stack([verts, np.full(len(verts), z)]))
        T.append(tri + base)
        base += len(verts)
    return np.vstack(P), np.vstack(T)


def polygons(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    return [p for q in getattr(g, "geoms", []) for p in polygons(q)]


def pose_matrix(x: float, y: float, yaw_deg: float, scale: float) -> Gf.Matrix4d:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    # row-vector convention: p_world = p_local * M
    return Gf.Matrix4d(c * scale, s * scale, 0, 0, -s * scale, c * scale, 0, 0, 0, 0, scale, 0, x, y, 0, 1)


def ground_remainder(sdir: Path, half: float):
    """The ground square minus every street surface, so that nothing is coplanar twice."""
    surf = json.loads((sdir / "surfaces.geojson").read_text())["features"]
    covered = unary_union([shape(f["geometry"]) for f in surf])
    return polygons(box(-half, -half, half, half).difference(covered.buffer(0)))


def add_street_layers(stage, sdir: Path, cfg: dict, make_mesh, make_material, label) -> dict:
    """make_mesh, make_material and label are the scene builder's own helpers."""
    pc, sc = cfg["populate"], cfg["scene"]
    stats = {"triangles": 0}
    looks = "/World/Looks"

    surf = json.loads((sdir / "surfaces.geojson").read_text())["features"]
    UsdGeom.Xform.Define(stage, "/World/Surfaces")
    by = {}
    for f in surf:
        by.setdefault(f["properties"]["class"], []).append(shape(f["geometry"]))
    for cls in sorted(by):
        pts, tris = triangulate([p for g in by[cls] for p in polygons(g)], 0.0)
        color = sc["surface_colors"][cls]
        mesh = make_mesh(stage, f"/World/Surfaces/{cls}", pts, tris, np.tile([0.0, 0.0, 1.0], (len(tris), 1)), color)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            make_material(stage, f"{looks}/Surface_{cls}", color=color, roughness=float(sc["surface_roughness"].get(cls, 0.9))))
        label(mesh.GetPrim(), cls)
        stats["triangles"] += len(tris)
    stats["surface_classes"] = sorted(by)

    marks = json.loads((sdir / "markings.geojson").read_text())["features"]
    if marks:
        UsdGeom.Xform.Define(stage, "/World/Markings")
        z = float(pc["markings"]["height_m"])
        for kind, color in sorted(sc["marking_colors"].items()):
            polys = [shape(m["geometry"]) for m in marks if m["properties"]["kind"] == kind]
            if not polys:
                continue
            pts, tris = triangulate(polys, z)
            mesh = make_mesh(stage, f"/World/Markings/{kind}", pts, tris, np.tile([0.0, 0.0, 1.0], (len(tris), 1)), color)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
                make_material(stage, f"{looks}/Marking_{kind}", color=color, roughness=0.8))
            stats["triangles"] += len(tris)
        label(stage.GetPrimAtPath("/World/Markings"), "road_marking")
    stats["markings"] = len(marks)

    placed = json.loads((sdir / "placements.json").read_text())
    manifest = load_manifest(cfg)["assets"]
    rel = os.path.relpath(library_dir(cfg), sdir)
    mode = pc["instancing"]
    UsdGeom.Xform.Define(stage, "/World/Objects")
    for root in CLASS_ROOT.values():
        UsdGeom.Xform.Define(stage, f"/World/Objects/{root}")
    counts, inst_tris = {}, 0
    for o in placed["objects"]:
        a = manifest[o["asset"]]
        path = f"/World/Objects/{CLASS_ROOT[o['class']]}/{o['id']}"
        x = UsdGeom.Xform.Define(stage, path)
        prim = x.GetPrim()
        prim.GetReferences().AddReference(f"{rel}/{a['file']}")
        if mode == "instanceable":
            prim.SetInstanceable(True)
        x.AddTransformOp().Set(pose_matrix(o["x"], o["y"], o["yaw_deg"], o["scale"]))
        prim.CreateAttribute("asset", Sdf.ValueTypeNames.String).Set(o["asset"])
        prim.CreateAttribute("motion", Sdf.ValueTypeNames.String).Set(o["motion"])
        label(prim, o["class"])
        counts[o["class"]] = counts.get(o["class"], 0) + 1
        inst_tris += a["triangles"]
    stats.update({"objects": counts, "object_triangles_if_not_instanced": inst_tris, "instancing": mode,
                  "asset_library": rel})
    return stats
