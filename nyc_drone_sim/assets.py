"""Build the USD model library used to populate the scene. CPU only.

Sources (downloaded by scripts/fetch_assets.sh into data/assets/valdrada):
- vehicles: Valdrada `fleet24` glTF models (CARLA-derived and original NYC vehicles, CC BY 4.0)
- trees: Valdrada `bxtrees` street trees (CC BY 4.0; leaf and bark textures CC0)
- pedestrians: Valdrada `peds24/rb27`, converted from the Microsoft Rocketbox library (MIT)
- street furniture: generated here from simple shapes

Every model is written as one .usdc with default prim /Asset, in metres, Z up, standing on
z = 0, centred in x and y, facing +X. Vehicle and pedestrian textures are not converted; their
parts get flat colours by material role. Trees keep their bark and leaf textures.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from .common import REPO_ROOT, load_config
from .gltf import Glb

log = logging.getLogger("assets")


def to_zup(p: np.ndarray) -> np.ndarray:
    """glTF frame (+Y up, +Z forward) -> scene frame (+Z up, +X forward). Keeps handedness."""
    return np.column_stack([p[:, 2], p[:, 0], p[:, 1]])


# ---------------------------------------------------------------------------
# USD writing
# ---------------------------------------------------------------------------
class AssetWriter:
    def __init__(self, path: Path):
        self.path = path
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        self.root = UsdGeom.Xform.Define(self.stage, "/Asset")
        self.stage.SetDefaultPrim(self.root.GetPrim())
        UsdGeom.Scope.Define(self.stage, "/Asset/Looks")
        self.n = 0
        self.tris = 0
        self.lo = np.full(3, np.inf)
        self.hi = np.full(3, -np.inf)

    def material(self, name: str, color, roughness=0.6, metallic=0.0, texture: str | None = None,
                 cutout: float | None = None):
        mat = UsdShade.Material.Define(self.stage, f"/Asset/Looks/{name}")
        sh = UsdShade.Shader.Define(self.stage, f"/Asset/Looks/{name}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        if texture is None:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in color]))
        else:
            rd = UsdShade.Shader.Define(self.stage, f"/Asset/Looks/{name}/St")
            rd.CreateIdAttr("UsdPrimvarReader_float2")
            rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            tx = UsdShade.Shader.Define(self.stage, f"/Asset/Looks/{name}/Tex")
            tx.CreateIdAttr("UsdUVTexture")
            tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
            tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                rd.ConnectableAPI(), "result")
            rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
            if cutout is not None:
                tx.CreateOutput("a", Sdf.ValueTypeNames.Float)
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(tx.ConnectableAPI(), "a")
                sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(float(cutout))
        sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def mesh(self, pts: np.ndarray, tris: np.ndarray, mat, color, st: np.ndarray | None = None):
        tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
        if len(tris) == 0:
            return
        m = UsdGeom.Mesh.Define(self.stage, f"/Asset/M{self.n:02d}")
        self.n += 1
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, dtype=np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(tris, dtype=np.int32).ravel()))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDoubleSidedAttr(True)
        m.CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in color])])
        m.CreateExtentAttr([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())])
        if st is not None:
            pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(st, dtype=np.float32)))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        self.tris += len(tris)
        self.lo = np.minimum(self.lo, pts.min(0))
        self.hi = np.maximum(self.hi, pts.max(0))

    def save(self) -> dict:
        self.stage.GetRootLayer().Save()
        return {"file": self.path.name, "triangles": int(self.tris),
                "size_m": [round(float(v), 4) for v in (self.hi - self.lo)],
                "min_m": [round(float(v), 4) for v in self.lo], "max_m": [round(float(v), 4) for v in self.hi]}


def ground_and_centre(parts: list[np.ndarray]) -> list[np.ndarray]:
    """Shift so the bounding box is centred in x, y and the model stands on z = 0."""
    allp = np.vstack(parts)
    # ignore a handful of stray vertices below the body when finding the ground contact
    z0 = float(np.percentile(allp[:, 2], 0.05))
    off = np.array([(allp[:, 0].min() + allp[:, 0].max()) / 2, (allp[:, 1].min() + allp[:, 1].max()) / 2, z0])
    out = []
    for p in parts:
        q = p - off
        q[:, 2] = np.maximum(q[:, 2], 0.0)
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Vehicles
# ---------------------------------------------------------------------------
ROLE_LOOK = {   # material role -> (colour, roughness, metallic)
    "glass": ((0.04, 0.05, 0.06), 0.1, 0.0), "lens": ((0.75, 0.75, 0.75), 0.2, 0.0),
    "lamp": ((0.8, 0.8, 0.75), 0.3, 0.0), "lampinner": ((0.05, 0.05, 0.05), 0.5, 0.0),
    "detail": ((0.07, 0.07, 0.075), 0.7, 0.0), "interior": ((0.05, 0.05, 0.05), 0.8, 0.0),
    "chrome": ((0.6, 0.6, 0.62), 0.25, 1.0), "metal": ((0.45, 0.45, 0.47), 0.4, 1.0),
    "siren": ((0.6, 0.05, 0.05), 0.3, 0.0), "sign": ((0.85, 0.5, 0.1), 0.5, 0.0),
    "licenseplate": ((0.85, 0.85, 0.8), 0.5, 0.0), "plate": ((0.85, 0.85, 0.8), 0.5, 0.0),
}


def material_role(name: str) -> str:
    n = name.lower()
    for key in ("lampinner", "licenseplate", "glass", "lens", "lamp", "siren", "chrome", "metal",
                "interior", "sign", "plate", "paint", "bodywork", "detail"):
        if key in n:
            return "paint" if key == "bodywork" else key
    # the first material of the original NYC models is the painted body, named after the model
    return "paint"


def build_vehicle(src: Path, out: Path, paint, lod: int) -> dict:
    prims = Glb(src).primitives(lod)
    pts = ground_and_centre([to_zup(p["points"]) for p in prims])
    w = AssetWriter(out)
    mats = {}
    for p, q in zip(prims, pts):
        role = material_role(p["material"].get("name", ""))
        color, rough, metal = (tuple(paint), 0.35, 0.0) if role == "paint" else ROLE_LOOK.get(role, ROLE_LOOK["detail"])
        if role not in mats:
            mats[role] = w.material(role, color, rough, metal)
        w.mesh(q, p["triangles"], mats[role], color)
    return w.save()


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------
def _quantised(data: bytes, spec, nv: int, box) -> np.ndarray:
    # signed 16-bit, -32767..32767 spanning the box
    q = np.frombuffer(data, np.int16, nv * 3, spec[0]).reshape(nv, 3).astype(np.float64) / 32767.0
    lo, hi = np.array(box[0]), np.array(box[1])
    return (lo + hi) / 2 + q * (hi - lo) / 2


def build_tree(src: Path, out: Path, tex_dir: Path, level: str, cutout: float) -> dict:
    raw = gzip.decompress(src.read_bytes())
    n = struct.unpack_from("<I", raw, 0)[0]
    head = json.loads(raw[4:4 + n])
    data = raw[4 + n:]
    bark, leaf = head[level]["bark"], head[level]["leaf"]
    w = AssetWriter(out)

    bp = to_zup(_quantised(data, bark["pos"], bark["nv"], bark["box"]))
    bst = np.frombuffer(data, np.float32, bark["nv"] * 2, bark["uv"][0]).reshape(-1, 2)
    bidx = np.frombuffer(data, np.uint32, bark["ni"], bark["idx"][0]).reshape(-1, 3).astype(np.int64)
    lp = to_zup(_quantised(data, leaf["pos"], leaf["nv"], leaf["box"]))
    lst = np.frombuffer(data, np.uint16, leaf["nv"] * 2, leaf["uv"][0]).reshape(-1, 2) / 65535.0
    # each leaf card is an outline of `per` vertices in ring order; triangulate it as a fan
    per = leaf["nv"] // leaf["cards"]
    base = np.arange(leaf["cards"])[:, None] * per
    lidx = np.vstack([base + np.array([0, k, k + 1]) for k in range(1, per - 1)])

    # centre on the trunk base and stand on z = 0
    low = bp[bp[:, 2] < bp[:, 2].min() + 0.5]
    off = np.array([low[:, 0].mean(), low[:, 1].mean(), max(float(bp[:, 2].min()), 0.0)])
    bp, lp = bp - off, lp - off
    bp[:, 2] = np.maximum(bp[:, 2], 0.0)

    bark_tex = tex_dir / f"bark_{head['bark']}_col.jpg"
    leaf_tex = tex_dir / f"{head['F']}_leaf_col.png"
    leaf_rgb = np.asarray(Image.open(leaf_tex).convert("RGBA"), dtype=np.float64)
    opaque = leaf_rgb[..., 3] > 255 * cutout
    leaf_color = ((leaf_rgb[..., :3][opaque].mean(0) / 255.0) ** 2.2).tolist()      # sRGB -> linear, roughly
    w.mesh(bp, bidx, w.material("bark", (0.2, 0.15, 0.1), 0.9, texture=f"tex/{bark_tex.name}"), (0.2, 0.15, 0.1), bst)
    w.mesh(lp, lidx, w.material("leaf", leaf_color, 0.7, texture=f"tex/{leaf_tex.name}", cutout=cutout),
           leaf_color, lst)
    info = w.save()
    span = np.percentile(lp[:, :2], 99, axis=0) - np.percentile(lp[:, :2], 1, axis=0)
    info.update({"height_m": round(float(max(bp[:, 2].max(), lp[:, 2].max())), 3),
                 "crown_diameter_m": round(float(span.mean()), 3),
                 "form": head["F"], "age": head["size"]})
    return info


# ---------------------------------------------------------------------------
# Pedestrians
# ---------------------------------------------------------------------------
ARM_CHAINS = {"L": (5, range(5, 28)), "R": (33, range(33, 56))}     # upper-arm joint, joints of the arm
CLOTH_TOP = [(0.08, 0.09, 0.14), (0.55, 0.55, 0.57), (0.35, 0.07, 0.07), (0.07, 0.2, 0.33), (0.02, 0.02, 0.02),
             (0.6, 0.5, 0.35), (0.12, 0.25, 0.14), (0.65, 0.63, 0.6)]
CLOTH_LOW = [(0.03, 0.04, 0.08), (0.02, 0.02, 0.02), (0.12, 0.12, 0.13), (0.2, 0.17, 0.12)]
SKIN = [(0.55, 0.36, 0.26), (0.36, 0.22, 0.15), (0.2, 0.12, 0.08), (0.62, 0.45, 0.36)]
HAIR = [(0.02, 0.015, 0.01), (0.1, 0.06, 0.03), (0.3, 0.22, 0.12), (0.25, 0.25, 0.25)]


def lower_arms(p: np.ndarray, joints: np.ndarray, weights: np.ndarray, target_deg: float) -> np.ndarray:
    """Swing each arm about its shoulder so it hangs `target_deg` from vertical (file frame, +Y up).

    The models come in a spread-arm rest pose. Vertices follow the rotation in proportion to
    their skin weight on the arm joints, which keeps the shoulder smooth.
    """
    out = p.copy()
    for side, (upper, chain) in ARM_CHAINS.items():
        chain = np.array(list(chain))
        wsum = (weights * np.isin(joints, chain)).sum(1)
        arm = wsum > 0.5
        if arm.sum() < 20:
            continue
        sign = 1.0 if p[arm, 0].mean() > 0 else -1.0
        inner = p[arm][np.argsort(sign * p[arm, 0])[:max(10, arm.sum() // 20)]]
        pivot = inner.mean(0)
        far = p[arm][np.argsort(-sign * p[arm, 0])[:max(10, arm.sum() // 20)]].mean(0)
        v = far - pivot
        current = math.atan2(sign * v[0], -v[1])                # angle from straight down, outward positive
        ang = sign * (current - math.radians(target_deg))        # rotation about the forward axis
        c, s = math.cos(ang), math.sin(ang)
        rel = p - pivot
        rot = np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]]) + pivot
        out = out + wsum[:, None] * (rot - p)
    return out


def build_pedestrian(src: Path, out: Path, lod: int, look: int, arm_deg: float) -> dict:
    prims = Glb(src).primitives(lod, extra=("_CLS", "_PART", "JOINTS_0", "WEIGHTS_0"))
    P, T, C = [], [], []
    base = 0
    for pr in prims:
        body = pr["_PART"].ravel() == 0                       # drop the carried bags and phones
        p = lower_arms(pr["points"], pr["JOINTS_0"].astype(int), np.asarray(pr["WEIGHTS_0"], dtype=np.float64), arm_deg)
        keep = body[pr["triangles"]].all(1)
        P.append(p)
        T.append(pr["triangles"][keep] + base)
        C.append(pr["_CLS"].ravel())
        base += len(p)
    p = ground_and_centre([to_zup(np.vstack(P))])[0]
    tris, cls = np.vstack(T), np.concatenate(C)
    height = float(p[np.unique(tris), 2].max())
    tri_cls = cls[tris[:, 0]]
    tri_z = p[tris].mean(1)[:, 2]
    w = AssetWriter(out)
    groups = {
        "top": ((tri_cls == 1) & (tri_z >= 0.52 * height), CLOTH_TOP[look % len(CLOTH_TOP)]),
        "low": ((tri_cls == 1) & (tri_z < 0.52 * height), CLOTH_LOW[(look // 2) % len(CLOTH_LOW)]),
        "skin": ((tri_cls == 0) | (tri_cls == 2), SKIN[(look // 3) % len(SKIN)]),
        "hair": (tri_cls == 3, HAIR[(look // 5) % len(HAIR)]),
    }
    for name, (sel, color) in groups.items():
        if sel.any():
            used, inv = np.unique(tris[sel], return_inverse=True)
            w.mesh(p[used], inv.reshape(-1, 3), w.material(name, color, 0.8), color)
    return w.save()


# ---------------------------------------------------------------------------
# Street furniture from simple shapes
# ---------------------------------------------------------------------------
def box(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    p = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float64)
    t = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return p, t


def cylinder(x, y, z0, z1, r0, r1=None, n=12):
    r1 = r0 if r1 is None else r1
    a = np.arange(n) * 2 * math.pi / n
    p = np.vstack([np.column_stack([x + r0 * np.cos(a), y + r0 * np.sin(a), np.full(n, z0)]),
                   np.column_stack([x + r1 * np.cos(a), y + r1 * np.sin(a), np.full(n, z1)]),
                   [[x, y, z0], [x, y, z1]]])
    i = np.arange(n)
    j = (i + 1) % n
    t = np.vstack([np.column_stack([i, j, n + j]), np.column_stack([i, n + j, n + i]),
                   np.column_stack([np.full(n, 2 * n), j, i]), np.column_stack([np.full(n, 2 * n + 1), n + i, n + j])])
    return p, t


def merge(parts):
    P, T, base = [], [], 0
    for p, t in parts:
        P.append(p)
        T.append(t + base)
        base += len(p)
    return np.vstack(P), np.vstack(T)


def furniture_shapes() -> dict:
    """name -> list of (colour, roughness, metallic, (points, triangles)). Sizes in metres."""
    red, dark, steel, green = (0.5, 0.05, 0.04), (0.06, 0.06, 0.07), (0.45, 0.46, 0.48), (0.05, 0.16, 0.09)
    return {
        "hydrant": [(red, 0.5, 0.0, merge([cylinder(0, 0, 0, 0.6, 0.11), cylinder(0, 0, 0.6, 0.75, 0.11, 0.04),
                                           box((-0.2, -0.05, 0.35), (0.2, 0.05, 0.45))]))],
        "bike_rack": [(steel, 0.3, 1.0, merge([box((-0.4, -0.03, 0), (-0.34, 0.03, 0.85)),
                                               box((0.34, -0.03, 0), (0.4, 0.03, 0.85)),
                                               box((-0.4, -0.03, 0.79), (0.4, 0.03, 0.85))]))],
        "kiosk": [(dark, 0.3, 0.0, box((-0.14, -0.45, 0), (0.14, 0.45, 2.9))),
                  ((0.25, 0.45, 0.75), 0.2, 0.0, box((-0.16, -0.38, 1.2), (0.16, 0.38, 2.6)))],
        "bus_shelter": [(dark, 0.4, 0.0, merge([box((-2.1, -0.75, 2.35), (2.1, 0.75, 2.5)),
                                                box((-2.1, 0.68, 0), (-2.0, 0.75, 2.35)),
                                                box((2.0, 0.68, 0), (2.1, 0.75, 2.35)),
                                                box((-2.1, -0.75, 0), (-2.0, -0.68, 2.35)),
                                                box((2.0, -0.75, 0), (2.1, -0.68, 2.35))])),
                        ((0.35, 0.42, 0.45), 0.1, 0.0, box((-2.0, 0.69, 0.3), (2.0, 0.73, 2.2)))],
        "street_lamp": [(steel, 0.4, 1.0, merge([cylinder(0, 0, 0, 8.5, 0.11, 0.07),
                                                 box((0, -0.05, 8.35), (2.2, 0.05, 8.5))])),
                        ((0.85, 0.85, 0.8), 0.3, 0.0, box((1.7, -0.15, 8.2), (2.4, 0.15, 8.36)))],
        "traffic_signal": [(green, 0.5, 0.0, merge([cylinder(0, 0, 0, 6.2, 0.1, 0.08),
                                                    box((0, -0.05, 5.9), (4.5, 0.05, 6.05))])),
                           ((0.75, 0.6, 0.05), 0.5, 0.0, merge([box((4.3, -0.18, 5.0), (4.6, 0.18, 6.0)),
                                                                box((-0.15, -0.18, 2.6), (0.15, 0.18, 3.6))]))],
        "trash_can": [(green, 0.6, 0.0, cylinder(0, 0, 0, 0.9, 0.28, 0.32))],
    }


def build_furniture(name: str, shapes, out: Path) -> dict:
    w = AssetWriter(out)
    for k, (color, rough, metal, (p, t)) in enumerate(shapes):
        w.mesh(p, t, w.material(f"m{k}", color, rough, metal), color)
    return w.save()


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
def library_dir(cfg: dict) -> Path:
    return REPO_ROOT / cfg["assets"]["library_dir"]


def load_manifest(cfg: dict) -> dict:
    return json.loads((library_dir(cfg) / "manifest.json").read_text())


def build_library(cfg: dict) -> dict:
    ac = cfg["assets"]
    src = REPO_ROOT / ac["source_dir"] / "Content" / "models"
    out = library_dir(cfg)
    if out.exists():
        shutil.rmtree(out)
    (out / "tex").mkdir(parents=True)
    assets = {}

    for kind, spec in ac["vehicles"].items():
        for ci, paint in enumerate(spec["paints"]):
            name = f"veh_{kind}_{ci}"
            info = build_vehicle(src / "fleet24" / f"{kind}.glb", out / f"{name}.usdc", paint, int(ac["vehicle_lod"]))
            assets[name] = {**info, "class": "vehicle", "group": spec["group"], "model": kind,
                            "source": f"Valdrada fleet24/{kind}.glb", "licence": "CC BY 4.0"}

    tdir = src / "bxtrees"
    for f in sorted(tdir.glob("bark_*_col.jpg")):
        shutil.copy(f, out / "tex" / f.name)
    for f in sorted(tdir.glob("*_leaf_col.webp")):
        Image.open(f).convert("RGBA").save(out / "tex" / f.with_suffix(".png").name)
    for f in sorted(tdir.glob("*.bin")):
        name = f"tree_{f.stem}"
        info = build_tree(f, out / f"{name}.usdc", out / "tex", ac["tree_level"], float(ac["leaf_cutout"]))
        assets[name] = {**info, "class": "vegetation", "source": f"Valdrada bxtrees/{f.name}",
                        "licence": "CC BY 4.0; textures CC0"}

    bodies = sorted((src / "peds24" / "rb27" / "bodies").glob("*.glb"))
    adults = [b for b in bodies if "Child" not in b.stem]
    pick = [adults[int(round(i))] for i in np.linspace(0, len(adults) - 1, int(ac["pedestrian_models"]))]
    for k, f in enumerate(pick):
        name = f"ped_{k:02d}"
        info = build_pedestrian(f, out / f"{name}.usdc", int(ac["pedestrian_lod"]), k, float(ac["arm_angle_deg"]))
        assets[name] = {**info, "class": "person", "model": f.stem,
                        "source": f"Valdrada peds24/rb27/bodies/{f.name} (Microsoft Rocketbox)", "licence": "MIT"}

    for fname, shapes in furniture_shapes().items():
        name = f"furn_{fname}"
        info = build_furniture(fname, shapes, out / f"{name}.usdc")
        assets[name] = {**info, "class": "furniture", "model": fname, "source": "generated", "licence": "n/a"}

    for name, a in assets.items():
        a["sha256"] = hashlib.sha256((out / a["file"]).read_bytes()).hexdigest()
    manifest = {"source_revision": ac["source_revision"], "assets": assets}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert the downloaded models into the USD library")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    m = build_library(load_config(a.config))
    by = {}
    for v in m["assets"].values():
        by.setdefault(v["class"], []).append(v["triangles"])
    for k, t in sorted(by.items()):
        log.info("%s: %d models, %d-%d triangles", k, len(t), min(t), max(t))


if __name__ == "__main__":
    main()
