"""Optional weather and look for a render (sim/render.py --sim-weather and --look), applied after
the app has started.

Everything here changes only how the open stage is shaded and lit, in memory: the scene file is
not written, no geometry moves, and no semantic label changes. With `--sim-weather clear` and
`--look standard` (the defaults) nothing in this module runs.

Weather (from the renderer, not from the video model):
  fog_light, fog_heavy   the renderer's distance fog under an overcast sky
  wet                    darker, smoother road and paving under an overcast sky
Look:
  enhanced               sky-coloured dome light with a soft sun, exposure, rippled reflective
                         water, ground textures sized in metres, and wall, glass and roof variation

All numbers come from the `appearance` block of the config.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

WEATHERS = ("clear", "fog_light", "fog_heavy", "wet")
LOOKS = ("standard", "enhanced")


# ---------------------------------------------------------------- lights and renderer settings

def _lights(stage):
    sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Sun"))
    sky = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Sky"))
    if not sun or not sky:
        raise RuntimeError("the scene has no /World/Sun or /World/Sky light")
    return sun, sky


def _scale(attr, factor: float) -> None:
    attr.Set(float(attr.Get()) * float(factor))


def overcast(stage, spec: dict) -> None:
    """Weak, wide sun and a brighter grey sky, so shadows nearly vanish."""
    sun, sky = _lights(stage)
    _scale(sun.GetIntensityAttr(), spec["sun_intensity_factor"])
    sun.GetAngleAttr().Set(float(spec["sun_angle_deg"]))
    _scale(sky.GetIntensityAttr(), spec["dome_intensity_factor"])
    sky.CreateColorAttr().Set(Gf.Vec3f(*[float(v) for v in spec["dome_color"]]))


def fog(settings, spec: dict) -> None:
    """The renderer's distance fog. Setting names are those of omni.rtx.settings.core 0.7.2."""
    settings.set("/rtx/fog/enabled", True)
    settings.set("/rtx/fog/fogColor", [float(v) for v in spec["color"]])
    settings.set("/rtx/fog/fogColorIntensity", float(spec["color_intensity"]))
    settings.set("/rtx/fog/fogStartDist", float(spec["start_m"]))
    settings.set("/rtx/fog/fogEndDist", float(spec["end_m"]))
    settings.set("/rtx/fog/fogDistanceDensity", float(spec["density"]))
    settings.set("/rtx/fog/fogHeightDensity", 0.0)


# ---------------------------------------------------------------- materials

def _surface_shader(stage, material: str):
    sh = UsdShade.Shader(stage.GetPrimAtPath(f"/World/Looks/{material}/Surface"))
    return sh if sh and sh.GetPrim().IsValid() else None


def _set_input(sh, name: str, type_name, value) -> None:
    inp = sh.GetInput(name) or sh.CreateInput(name, type_name)
    if inp.HasConnectedSource():
        inp.DisconnectSource()
    inp.Set(value)


def _mesh_colour(stage, surface: str):
    """The single display colour of one ground-surface mesh, or None when there is no such mesh."""
    prim = stage.GetPrimAtPath(f"/World/Surfaces/{surface}")
    if not prim.IsValid():
        return None, None
    pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    vals = pv.Get() if pv else None
    return (pv, Gf.Vec3f(vals[0])) if vals else (None, None)


def wet(stage, spec: dict) -> list[str]:
    """Darker and smoother ground surfaces. Returns the surfaces that were changed."""
    changed = []
    for name in spec["surfaces"]:
        sh = _surface_shader(stage, f"Surface_{name}")
        pv, col = _mesh_colour(stage, name)
        if sh is None or pv is None:
            continue
        pv.Set(Vt.Vec3fArray([Gf.Vec3f(*[float(c) * float(spec["darken"]) for c in col])]))
        _set_input(sh, "roughness", Sdf.ValueTypeNames.Float, float(spec["roughness"]))
        changed.append(name)
    return changed


def apply_weather(stage, settings, cfg: dict, name: str) -> dict:
    ap = cfg["appearance"]["weather"]
    done = {"weather": name}
    if name == "clear":
        return done
    overcast(stage, ap["overcast"])
    if name in ("fog_light", "fog_heavy"):
        fog(settings, ap[name])
        done["fog"] = ap[name]
    elif name == "wet":
        done["wet_surfaces"] = wet(stage, ap["wet"])
    else:
        raise ValueError(f"unknown weather {name!r}")
    return done


# ---------------------------------------------------------------- enhanced look

def _noise(rng, n: int, octaves) -> np.ndarray:
    """Tiling value noise in 0..1: random grids of a few sizes, upsampled with wrap-around."""
    out = np.zeros((n, n))
    total = 0.0
    for cells, weight in octaves:
        g = rng.random((cells, cells))
        idx = np.arange(n) * cells / n
        i0 = np.floor(idx).astype(int) % cells
        i1 = (i0 + 1) % cells
        t = idx - np.floor(idx)
        t = t * t * (3 - 2 * t)
        rows = g[i0][:, i0] * (1 - t)[None, :] + g[i0][:, i1] * t[None, :]
        rows1 = g[i1][:, i0] * (1 - t)[None, :] + g[i1][:, i1] * t[None, :]
        out += weight * (rows * (1 - t)[:, None] + rows1 * t[:, None])
        total += weight
    return out / total


def _save(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


def make_textures(tex_dir: Path, lk: dict, seed: int) -> dict:
    """Generated tiling textures: a grey variation map per ground kind and a ripple normal map."""
    tex_dir.mkdir(parents=True, exist_ok=True)
    n = int(lk["texture_px"])
    files = {}
    for k, (kind, spec) in enumerate(sorted(lk["ground"].items())):
        rng = np.random.default_rng(seed + 101 * (k + 1))
        v = _noise(rng, n, [(int(c), float(w)) for c, w in spec["octaves"]])
        v = (v - v.mean()) / max(v.std(), 1e-9)
        tone = np.clip(1.0 + float(spec["contrast"]) * v, 0.0, 2.0) / 2.0      # 0.5 = unchanged
        _save(np.repeat(tone[:, :, None], 3, axis=2), tex_dir / f"{kind}_tone.png")
        files[kind] = tex_dir / f"{kind}_tone.png"
    rng = np.random.default_rng(seed + 7)
    h = _noise(rng, n, [(int(c), float(w)) for c, w in lk["water"]["octaves"]])
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * float(lk["water"]["ripple_strength"]) * n / 2
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * float(lk["water"]["ripple_strength"]) * n / 2
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    _save(np.dstack([-gx * nz, -gy * nz, nz]) * 0.5 + 0.5, tex_dir / "water_normal.png")
    files["water_normal"] = tex_dir / "water_normal.png"
    return files


def _world_st(stage, surface: str, tile_m: float) -> bool:
    """Give a ground-surface mesh texture coordinates equal to its position in metres / tile size."""
    prim = stage.GetPrimAtPath(f"/World/Surfaces/{surface}")
    if not prim.IsValid():
        return False
    pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
    st = Vt.Vec2fArray.FromNumpy((pts[:, :2] / float(tile_m)).astype(np.float32))
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    pv.Set(st)
    return True


def _texture(stage, material: str, name: str, file: Path, colour_space: str, scale, bias=None):
    base = f"/World/Looks/{material}"
    rd_path = f"{base}/WorldSt"
    rd = UsdShade.Shader(stage.GetPrimAtPath(rd_path))
    if not rd or not rd.GetPrim().IsValid():
        rd = UsdShade.Shader.Define(stage, rd_path)
        rd.CreateIdAttr("UsdPrimvarReader_float2")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tx = UsdShade.Shader.Define(stage, f"{base}/{name}")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(file)))
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colour_space)
    tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*scale))
    if bias is not None:
        tx.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*bias))
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.GetOutput("result"))
    return tx


def enhance_ground(stage, lk: dict, files: dict) -> list[str]:
    """Ground surfaces: the class colour times a tiling variation map, with its own roughness."""
    changed = []
    for surface, kind in lk["surface_kind"].items():
        sh = _surface_shader(stage, f"Surface_{surface}")
        _, col = _mesh_colour(stage, surface)
        if sh is None or col is None or not _world_st(stage, surface, lk["ground"][kind]["tile_m"]):
            continue
        tint = lk.get("surface_tint", {}).get(surface, [1.0, 1.0, 1.0])
        # the map stores 0.5 for "unchanged", so the class colour is doubled
        scale = [2.0 * float(col[i]) * float(tint[i]) for i in range(3)] + [1.0]
        tx = _texture(stage, f"Surface_{surface}", "Tone", files[kind], "raw", scale)
        sh.GetInput("diffuseColor").ConnectToSource(tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        _set_input(sh, "roughness", Sdf.ValueTypeNames.Float, float(lk["ground"][kind]["roughness"]))
        changed.append(surface)
    return changed


def enhance_water(stage, lk: dict, files: dict) -> bool:
    w = lk["water"]
    sh = _surface_shader(stage, "Surface_water")
    if sh is None or not _world_st(stage, "water", w["tile_m"]):
        return False
    _set_input(sh, "diffuseColor", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*[float(v) for v in w["color"]]))
    _set_input(sh, "roughness", Sdf.ValueTypeNames.Float, float(w["roughness"]))
    _set_input(sh, "metallic", Sdf.ValueTypeNames.Float, 0.0)
    tx = _texture(stage, "Surface_water", "Ripple", files["water_normal"], "raw", (2.0, 2.0, 2.0, 1.0), (-1.0, -1.0, -1.0, 0.0))
    (sh.GetInput("normal") or sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f)).ConnectToSource(
        tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    return True


def enhance_buildings(stage, lk: dict) -> dict:
    """Wall tones, reflective glass and darker roofs, on the materials the scene already has."""
    b = lk["buildings"]
    count = {"walls": 0, "glass": 0, "roofs": 0}
    looks = stage.GetPrimAtPath("/World/Looks")
    for prim in looks.GetChildren():
        name = prim.GetName()
        sh = _surface_shader(stage, name)
        if sh is None:
            continue
        tex = UsdShade.Shader(stage.GetPrimAtPath(f"/World/Looks/{name}/Tex"))
        idx = int(name.rsplit("_", 1)[1]) if name.rsplit("_", 1)[-1].isdigit() else 0
        if name.startswith("Wall_masonry"):
            tone = b["wall_tones"][idx % len(b["wall_tones"])]
            if tex and tex.GetPrim().IsValid():
                tex.GetInput("scale").Set(Gf.Vec4f(float(tone[0]), float(tone[1]), float(tone[2]), 1.0))
            _set_input(sh, "roughness", Sdf.ValueTypeNames.Float, float(b["wall_roughness"]))
            count["walls"] += 1
        elif name.startswith("Wall_glass"):
            _set_input(sh, "roughness", Sdf.ValueTypeNames.Float, float(b["glass_roughness"]))
            _set_input(sh, "metallic", Sdf.ValueTypeNames.Float, float(b["glass_metallic"]))
            count["glass"] += 1
        elif name.startswith("Roof"):
            f = float(b["roof_darken"])
            diffuse = sh.GetInput("diffuseColor")
            if tex and tex.GetPrim().IsValid():
                s = tex.GetInput("scale").Get()
                tex.GetInput("scale").Set(Gf.Vec4f(s[0] * f, s[1] * f, s[2] * f, 1.0))
            elif diffuse and not diffuse.HasConnectedSource() and diffuse.Get() is not None:
                c = diffuse.Get()
                diffuse.Set(Gf.Vec3f(c[0] * f, c[1] * f, c[2] * f))
            _set_input(sh, "roughness", Sdf.ValueTypeNames.Float, float(b["roof_roughness"]))
            count["roofs"] += 1
    return count


def enhance_lights(stage, settings, lk: dict) -> None:
    sun, sky = _lights(stage)
    s = lk["sky"]
    sun.GetAngleAttr().Set(float(s["sun_angle_deg"]))                       # a wider sun softens shadow edges
    _scale(sun.GetIntensityAttr(), s["sun_intensity_factor"])
    sun.CreateColorAttr().Set(Gf.Vec3f(*[float(v) for v in s["sun_color"]]))
    _scale(sky.GetIntensityAttr(), s["dome_intensity_factor"])
    sky.CreateColorAttr().Set(Gf.Vec3f(*[float(v) for v in s["dome_color"]]))


def apply_look(stage, settings, cfg: dict, name: str, tex_dir: Path) -> dict:
    done = {"look": name}
    if name == "standard":
        return done
    if name != "enhanced":
        raise ValueError(f"unknown look {name!r}")
    lk = cfg["appearance"]["enhanced"]
    files = make_textures(tex_dir, lk, int(cfg["seed"]))
    enhance_lights(stage, settings, lk)
    done["ground"] = enhance_ground(stage, lk, files)
    done["water"] = enhance_water(stage, lk, files)
    done["buildings"] = enhance_buildings(stage, lk)
    done["textures"] = {k: str(v) for k, v in files.items()}
    return done
