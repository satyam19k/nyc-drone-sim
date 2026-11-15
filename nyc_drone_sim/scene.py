"""Stage 2: open data to USD. CPU only (usd-core); must not import isaacsim.

One closed, outward-facing triangle mesh per building, extruded from z = 0 to height_m,
plus ground, preview materials, semantic labels and lights. Metres, Z up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import mapbox_earcut as earcut
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

from . import dress
from .common import for_location, load_config, scene_dir, slugify

log = logging.getLogger("stage2")

try:
    from pxr import UsdSemantics
    HAVE_USD_SEMANTICS = True
except ImportError:  # labels are then applied in Stage 3
    HAVE_USD_SEMANTICS = False


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _ring(coords) -> np.ndarray:
    a = np.asarray(coords, dtype=np.float64)[:, :2]
    if len(a) > 1 and np.allclose(a[0], a[-1]):
        a = a[:-1]
    return a


def _signed_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def triangulate_polygon(rings: list[np.ndarray]) -> np.ndarray:
    """Triangulate a polygon with holes. Returns (T, 3) indices into the stacked ring
    vertices, every triangle counter-clockwise in the xy plane."""
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = np.asarray(earcut.triangulate_float64(verts, ends), dtype=np.int64).reshape(-1, 3)
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri = tri[cross != 0]
    cross = cross[cross != 0]
    flip = cross < 0
    tri[flip] = tri[flip][:, ::-1]
    return tri


def extrude(multipolygon_coords, height: float):
    """Closed solid from z = 0 to height for a GeoJSON MultiPolygon coordinate array.

    Vertices are not shared between roof, base and walls, so shading stays faceted.
    Returns points (N, 3), triangle indices (T, 3), per-triangle normals (T, 3).
    """
    pts, tris, nrm = [], [], []
    off = 0
    for poly in multipolygon_coords:
        rings = [_ring(r) for r in poly]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue
        # exterior counter-clockwise, holes clockwise
        if _signed_area(rings[0]) < 0:
            rings[0] = rings[0][::-1]
        for i in range(1, len(rings)):
            if _signed_area(rings[i]) > 0:
                rings[i] = rings[i][::-1]
        flat = np.vstack(rings)
        n = len(flat)
        cap = triangulate_polygon(rings)
        # roof: +z
        pts.append(np.column_stack([flat, np.full(n, height)]))
        tris.append(cap + off)
        nrm.append(np.tile([0.0, 0.0, 1.0], (len(cap), 1)))
        off += n
        # base: -z (reverse winding)
        pts.append(np.column_stack([flat, np.zeros(n)]))
        tris.append(cap[:, ::-1] + off)
        nrm.append(np.tile([0.0, 0.0, -1.0], (len(cap), 1)))
        off += n
        # walls: for a ring wound with the solid on its left, outward is to the right
        for r in rings:
            a = r
            b = np.roll(r, -1, axis=0)
            d = b - a
            ln = np.hypot(d[:, 0], d[:, 1])
            keep = ln > 0
            a, b, d, ln = a[keep], b[keep], d[keep], ln[keep]
            m = len(a)
            out = np.column_stack([d[:, 1] / ln, -d[:, 0] / ln, np.zeros(m)])
            quad = np.empty((m, 4, 3))
            quad[:, 0] = np.column_stack([a, np.zeros(m)])
            quad[:, 1] = np.column_stack([b, np.zeros(m)])
            quad[:, 2] = np.column_stack([b, np.full(m, height)])
            quad[:, 3] = np.column_stack([a, np.full(m, height)])
            pts.append(quad.reshape(-1, 3))
            base = off + 4 * np.arange(m)
            tris.append(np.column_stack([base, base + 1, base + 2]))
            tris.append(np.column_stack([base, base + 2, base + 3]))
            nrm.append(out)
            nrm.append(out)
            off += 4 * m
    return np.vstack(pts), np.vstack(tris), np.vstack(nrm)


def building_color(doitt_id: int, base, jitter: float) -> tuple[float, float, float]:
    h = hashlib.sha256(str(doitt_id).encode()).digest()
    v = (h[0] / 255.0 - 0.5) * 2 * jitter           # brightness
    t = (h[1] / 255.0 - 0.5) * jitter               # warm/cool tint
    c = np.clip(np.array(base) + v + np.array([t, 0.0, -t]), 0.05, 0.95)
    return tuple(float(x) for x in c)


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------
def _material(stage, path: str, color=None, use_display_color: bool = False, roughness=0.8):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    diffuse = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    if use_display_color:
        rd = UsdShade.Shader.Define(stage, path + "/DisplayColor")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        rd.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*color))
        diffuse.ConnectToSource(rd.CreateOutput("result", Sdf.ValueTypeNames.Float3))
    else:
        diffuse.Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def _mesh(stage, path: str, pts: np.ndarray, tris: np.ndarray, nrm: np.ndarray, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    p32 = pts.astype(np.float32)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p32))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(nrm.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([p32.min(0), p32.max(0)])))
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                 UsdGeom.Tokens.constant)
    pv.Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return mesh


def facade_textures(tex_dir: Path, fc: dict) -> dict:
    """One-bay, one-floor tiles: a wall with a window, and a glass wall with spandrel and mullions."""
    from PIL import Image, ImageDraw
    tex_dir.mkdir(parents=True, exist_ok=True)
    n = int(fc["tile_px"])
    out = {}
    for name, (wall, glass, wx, wy) in {"masonry": ((200, 198, 194), (62, 78, 94), 0.46, 0.5),
                                        "glass": ((158, 163, 168), (100, 124, 146), 0.88, 0.72)}.items():
        im = Image.new("RGB", (n, n), wall)
        d = ImageDraw.Draw(im)
        x0, y0 = n * (1 - wx) / 2, n * (1 - wy) / 2
        d.rectangle([x0, y0, n - x0 - 1, n - y0 - 1], fill=glass)
        d.line([n / 2, y0, n / 2, n - y0 - 1], fill=wall, width=max(1, n // 32))
        f = tex_dir / f"facade_{name}.png"
        im.save(f)
        out[name] = f
    return out


def _texture_material(stage, path: str, file: str, tint, roughness: float):
    """UsdPreviewSurface whose colour is a tiled texture multiplied by a tint."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    rd = UsdShade.Shader.Define(stage, path + "/St")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    tx = UsdShade.Shader.Define(stage, path + "/Tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(float(tint[0]), float(tint[1]), float(tint[2]), 1.0))
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def facade_coords(pts: np.ndarray, tris: np.ndarray, nrm: np.ndarray, fc: dict):
    """Texture coordinates per vertex, and the wall and roof triangle indices.

    Walls: one tile per bay along the wall and per floor up it. Roof and base: metres.
    Vertices are not shared between faces of different orientation, so this is exact.
    """
    st = np.zeros((len(pts), 2))
    wall = np.abs(nrm[:, 2]) < 0.5
    t = np.column_stack([-nrm[:, 1], nrm[:, 0]])                      # along the wall
    for k in range(3):
        v = tris[wall, k]
        st[v, 0] = (pts[v, 0] * t[wall, 0] + pts[v, 1] * t[wall, 1]) / float(fc["bay_m"])
        st[v, 1] = pts[v, 2] / float(fc["floor_m"])
        v = tris[~wall, k]
        st[v] = pts[v, :2] / float(fc["roof_tile_m"])
    return st, np.nonzero(wall)[0], np.nonzero(nrm[:, 2] > 0.5)[0]


def _label(prim, label: str) -> None:
    labels = UsdSemantics.LabelsAPI.Apply(prim, "class")
    labels.CreateLabelsAttr([label])


def dressed(cfg: dict, sdir: Path | None) -> bool:
    """True when the street layers and placements exist and the config asks for them."""
    return bool(sdir is not None and cfg.get("populate", {}).get("enabled")
                and all((sdir / n).exists() for n in ("surfaces.geojson", "markings.geojson", "placements.json")))


def build_scene(features: list[dict], cfg: dict, out_usd: Path, sdir: Path | None = None) -> dict:
    sc = cfg["scene"]
    R = float(cfg["scene_radius_m"])
    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    looks = "/World/Looks"
    UsdGeom.Scope.Define(stage, looks)
    mat_b = _material(stage, looks + "/Building", color=sc["building_color_base"], use_display_color=True)
    mat_g = _material(stage, looks + "/Ground", color=sc["ground_color"], roughness=0.9)

    fc = sc.get("facade") or {}
    walls, roofs = [], []
    if fc.get("enabled"):
        tiles = facade_textures(out_usd.parent / "tex", fc)
        for k, tint in enumerate(fc["wall_tints"]):
            for style in ("masonry", "glass"):
                walls.append((style, _texture_material(stage, f"{looks}/Wall_{style}_{k}", f"tex/{tiles[style].name}", tint, 0.6 if style == "masonry" else 0.25)))
        roofs = [_material(stage, f"{looks}/Roof_{k}", color=c, roughness=0.9) for k, c in enumerate(fc["roof_colors"])]
    broot = UsdGeom.Xform.Define(stage, "/World/Buildings")
    n_tri = 0
    max_h = 0.0
    names = set()
    for f in features:
        p = f["properties"]
        h = float(p["height_m"])
        pts, tris, nrm = extrude(f["geometry"]["coordinates"], h)
        name = f"B_{p['doitt_id']}"
        if name in names:
            name = f"{name}_{p['objectid']}"
        names.add(name)
        color = building_color(p["doitt_id"], sc["building_color_base"], sc["building_color_jitter"])
        mesh = _mesh(stage, f"/World/Buildings/{name}", pts, tris, nrm, color)
        prim = mesh.GetPrim()
        prim.CreateAttribute("bin", Sdf.ValueTypeNames.String).Set(str(p["bin"]))
        prim.CreateAttribute("height_m", Sdf.ValueTypeNames.Double).Set(h)
        prim.CreateAttribute("height_imputed", Sdf.ValueTypeNames.Bool).Set(bool(p["height_imputed"]))
        if walls:
            # appearance only: texture coordinates and face groups; points and faces are untouched
            st, wall_faces, roof_faces = facade_coords(pts, tris, nrm, fc)
            pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(st.astype(np.float32)))
            hsh = hashlib.sha256(str(p["doitt_id"]).encode()).digest()
            # tall buildings lean to glass walls, low ones to masonry
            glassy = h >= float(fc["glass_above_m"]) and hsh[2] < 200 or hsh[2] < 25
            pool = [m for style, m in walls if (style == "glass") == glassy]
            for nm_, faces, mat in (("walls", wall_faces, pool[hsh[3] % len(pool)]), ("roof", roof_faces, roofs[hsh[4] % len(roofs)])):
                sub_ = UsdGeom.Subset.CreateGeomSubset(mesh, nm_, UsdGeom.Tokens.face, Vt.IntArray.FromNumpy(faces.astype(np.int32)),
                                                       UsdShade.Tokens.materialBind)
                UsdShade.MaterialBindingAPI.Apply(sub_.GetPrim()).Bind(mat)
            UsdGeom.Subset.SetFamilyType(mesh, UsdShade.Tokens.materialBind, UsdGeom.Tokens.nonOverlapping)
        n_tri += len(tris)
        max_h = max(max_h, h)
    UsdShade.MaterialBindingAPI.Apply(broot.GetPrim()).Bind(mat_b)

    half = sc["ground_side_factor"] * R / 2.0
    street = dressed(cfg, sdir)
    if street:
        # the ground is what is left of the square once the street surfaces are cut out of it
        gpts, gtris = dress.triangulate(dress.ground_remainder(sdir, half), 0.0)
    else:
        gpts = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], float)
        gtris = np.array([[0, 1, 2], [0, 2, 3]])
    ground = _mesh(stage, "/World/Ground", gpts, gtris, np.tile([0.0, 0.0, 1.0], (len(gtris), 1)),
                   sc["ground_color"])
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(mat_g)
    n_tri += len(gtris)

    if HAVE_USD_SEMANTICS:
        _label(broot.GetPrim(), "building")   # inherited by every child mesh
        _label(ground.GetPrim(), "ground")
    street_stats = None
    if street:
        if not HAVE_USD_SEMANTICS:
            raise RuntimeError("the populated scene needs UsdSemantics in usd-core")
        street_stats = dress.add_street_layers(stage, sdir, cfg, _mesh, _material, _label)
        n_tri += street_stats["triangles"]

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(sc["sun_intensity"]))
    sun.CreateAngleAttr(float(sc["sun_angle_deg"]))
    sx = UsdGeom.Xformable(sun)
    # a distant light shines along its local -Z; tilt from straight down, then spin to azimuth
    sx.AddRotateZOp().Set(float(sc["sun_azimuth_deg"]))
    sx.AddRotateXOp().Set(90.0 - float(sc["sun_elevation_deg"]))
    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(float(sc["dome_intensity"]))

    stage.GetRootLayer().Save()
    return {
        "building_count": len(features),
        "triangle_count": n_tri,
        "max_height_m": max_h,
        "labels_applied_in_stage2": HAVE_USD_SEMANTICS,
        "file_bytes": out_usd.stat().st_size,
        "street": street_stats,
    }


def run_stage2(location: str, cfg: dict, out_dir: Path | None = None) -> Path:
    slug = slugify(location)
    cfg = for_location(cfg, location)
    sdir = out_dir or scene_dir(cfg, slug)
    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    loc = json.loads((sdir / "location.json").read_text())
    out_usd = sdir / "scene.usdc"
    if out_usd.exists():
        out_usd.unlink()
    meta = build_scene(feats, cfg, out_usd, sdir)
    meta.update({
        "location": slug,
        "origin": {"lat": loc["lat"], "lon": loc["lon"], "projection": loc["projection"]},
        "scene_radius_m": float(cfg["scene_radius_m"]),
        "ground_count": 1,
        "imputed_count": sum(bool(f["properties"]["height_imputed"]) for f in feats),
    })
    (sdir / "scene_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    log.info("stage 2 done: %d buildings, %d triangles, %.1f MB -> %s", meta["building_count"],
             meta["triangle_count"], meta["file_bytes"] / 1e6, out_usd)
    return out_usd


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 2: open data to USD")
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_stage2(a.location, load_config(a.config))


if __name__ == "__main__":
    main()
