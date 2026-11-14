"""Stage 3: load scene.usdc in Isaac Sim and render nadir probes (RGB, metric depth,
semantic segmentation). GPU, headless. Evaluation happens in checks/stage_3.py."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--location", default="120 Broadway")
ap.add_argument("--config", default=None)
ap.add_argument("--run", default=None)
args, _ = ap.parse_known_args()

# SimulationApp must exist before any other Omniverse import.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from nyc_drone_sim.common import load_config, scene_dir, slugify  # noqa: E402

DEPTH_ANNOTATOR = "distance_to_image_plane"   # metres, float32 (omni.replicator.core docs)


def gpu_query(fields: str) -> list[str]:
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return [v.strip() for v in out[0].split(",")]


class VramPeak(threading.Thread):
    def __init__(self, period_s: float = 0.5):
        super().__init__(daemon=True)
        self.period, self.peak, self.stop_flag = period_s, 0.0, False

    def run(self):
        while not self.stop_flag:
            try:
                self.peak = max(self.peak, float(gpu_query("memory.used")[0]))
            except Exception:
                pass
            time.sleep(self.period)


def main():
    cfg = load_config(args.config)
    slug = slugify(args.location)
    sdir = scene_dir(cfg, slug)
    out_dir = sdir / "probe"
    out_dir.mkdir(exist_ok=True)
    probes = json.loads((sdir / "probes.json").read_text())["probes"]
    pc, cc = cfg["probe"], cfg["camera"]

    gpu_name, vram_total = gpu_query("name,memory.total")
    mon = VramPeak()
    mon.start()

    t0 = time.time()
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(sdir / "scene.usdc")):
        raise RuntimeError("open_stage failed")
    for _ in range(10):
        simulation_app.update()
    stage = ctx.get_stage()
    load_s = time.time() - t0
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)

    # A USD camera looks along local -Z with +Y up; unrotated in a Z-up stage it looks at the ground.
    cam = UsdGeom.Camera.Define(stage, "/World/ProbeCamera")
    cam.CreateFocalLengthAttr(float(cc["focal_length_mm"]))
    cam.CreateHorizontalApertureAttr(float(cc["horizontal_aperture_mm"]))
    cam.CreateVerticalApertureAttr(float(cc["vertical_aperture_mm"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(*[float(v) for v in pc["clipping_range_m"]]))
    translate = UsdGeom.Xformable(cam).AddTranslateOp()

    rp = rep.create.render_product(str(cam.GetPath()), tuple(int(v) for v in cc["resolution"]))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator(DEPTH_ANNOTATOR)
    seg = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    for a in (rgb, depth, seg):
        a.attach(rp)

    results = []
    # The first captured frame can come back empty while the annotators initialise, so render
    # one warm-up frame at the first probe pose and discard it.
    translate.Set(Gf.Vec3d(*probes[0]["position"]))
    rep.orchestrator.step(rt_subframes=int(pc["rt_subframes"]))
    for pr in probes:
        translate.Set(Gf.Vec3d(*pr["position"]))
        rep.orchestrator.step(rt_subframes=int(pc["rt_subframes"]))
        s = seg.get_data()
        for name, arr in (("rgb", rgb.get_data()), ("depth", depth.get_data()), ("seg", s["data"])):
            if np.asarray(arr).size == 0:
                raise RuntimeError(f"probe {pr['name']}: annotator {name} returned no data")
        np.savez_compressed(
            out_dir / f"{pr['name']}.npz",
            rgb=np.asarray(rgb.get_data()),
            depth=np.asarray(depth.get_data(), dtype=np.float32),
            seg=np.asarray(s["data"]),
        )
        results.append({**pr, "file": f"{pr['name']}.npz", "seg_info": s["info"]})
        print(f"probe {pr['name']} rendered", flush=True)

    rep.orchestrator.wait_until_complete()
    mon.stop_flag = True
    import importlib.metadata as md
    (out_dir / "probe_result.json").write_text(json.dumps({
        "location": slug, "run": args.run,
        "isaacsim_version": md.version("isaacsim"),
        "replicator_version": getattr(rep, "__version__", None),
        "depth_annotator": DEPTH_ANNOTATOR, "depth_unit": "metres",
        "stage_meters_per_unit": mpu, "stage_up_axis": str(up),
        "gpu_name": gpu_name, "vram_total_mib": float(vram_total), "vram_peak_mib": mon.peak,
        "scene_load_s": load_s, "resolution": list(cc["resolution"]),
        "rt_subframes": int(pc["rt_subframes"]), "probes": results,
    }, indent=1, default=str) + "\n")
    print(f"probe done: load {load_s:.1f} s, peak VRAM {mon.peak:.0f} / {vram_total} MiB on {gpu_name}", flush=True)


try:
    main()
finally:
    simulation_app.close()
