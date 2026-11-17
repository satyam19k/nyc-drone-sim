"""CPU only: every module imports, and a tiny made-up input builds into a USD scene."""
import importlib
import importlib.util
import json
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import nyc_drone_sim
from nyc_drone_sim.common import load_config

# modules that need the simulator or the video model's environment, not the CPU one
NEEDS_GPU_STACK = ("isaacsim", "omni", "torch", "cosmos_transfer2", "imageio_ffmpeg", "cv2")


def _modules():
    for m in pkgutil.walk_packages(nyc_drone_sim.__path__, "nyc_drone_sim."):
        yield m.name


@pytest.mark.parametrize("name", sorted(_modules()))
def test_module_imports(name):
    spec = importlib.util.find_spec(name)
    src = Path(spec.origin).read_text()
    if "from isaacsim import" in src:
        # scripts run by the simulator's own Python; they start Isaac Sim when imported
        compile(src, spec.origin, "exec")
        return
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in NEEDS_GPU_STACK:
            pytest.skip(f"{name} needs {e.name}")
        raise


def _square(x, y, s):
    return [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]]


def _building(i, x, y, h):
    return {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [_square(x, y, 12.0)]},
            "properties": {"doitt_id": 1000 + i, "objectid": i, "bin": str(2000 + i),
                           "height_m": h, "height_imputed": False}}


def test_tiny_scene_holds_buildings_and_road(tmp_path):
    from pxr import Usd

    from nyc_drone_sim.scene import run_scene

    cfg = load_config()
    cfg["scene_radius_m"] = 60.0
    cfg["populate"]["enabled"] = True
    cfg["scene"].setdefault("facade", {})["enabled"] = False
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "manifest.json").write_text(json.dumps({"assets": {}}))
    cfg["assets"]["library_dir"] = str(lib)

    sdir = tmp_path / "scene"
    sdir.mkdir()
    heights = (10.0, 25.0, 40.0)
    feats = [_building(i, x, 15.0, h) for i, (x, h) in enumerate(zip((-40.0, -10.0, 20.0), heights))]
    (sdir / "buildings.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    (sdir / "location.json").write_text(json.dumps({"lat": 40.7, "lon": -74.0, "projection": "+proj=aeqd"}))
    road = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-50, -10], [50, -10], [50, 0], [-50, 0], [-50, -10]]]},
            "properties": {"class": "road"}}
    (sdir / "surfaces.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [road]}))
    (sdir / "markings.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    (sdir / "placements.json").write_text(json.dumps({"objects": []}))

    out = run_scene("tiny", cfg, out_dir=sdir)

    stage = Usd.Stage.Open(str(out))
    names = sorted(p.GetName() for p in stage.GetPrimAtPath("/World/Buildings").GetChildren())
    assert names == ["B_1000", "B_1001", "B_1002"]
    assert stage.GetPrimAtPath("/World/Buildings/B_1002").GetAttribute("height_m").Get() == 40.0
    assert stage.GetPrimAtPath("/World/Surfaces/road").IsValid()
    meta = json.loads((sdir / "scene_meta.json").read_text())
    assert meta["building_count"] == 3 and meta["max_height_m"] == 40.0
    assert meta["street"]["surface_classes"] == ["road"]
