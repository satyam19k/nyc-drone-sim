"""Render a list of prepared camera shots and save every annotator output. GPU, headless.

The shot list is written on the CPU by nyc_drone_sim.shotlist; this script only opens each
stage, moves the listed prims, places the camera and captures RGB, metric depth, semantic
segmentation and both instance segmentations. All evaluation happens in checks/populate.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--shots", required=True, help="shot list JSON")
ap.add_argument("--config", default=None)
args, _ = ap.parse_known_args()

# SimulationApp must exist before any other Omniverse import.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from nyc_drone_sim.common import load_config  # noqa: E402

DEPTH_ANNOTATOR = "distance_to_image_plane"   # metres, float32


def gpu_used() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    return [v.strip() for v in out.split(",")]


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def main():
    cfg = load_config(args.config)
    cc, pc = cfg["camera"], cfg["probe"]
    plan = json.loads(Path(args.shots).read_text())
    out_dir = Path(args.shots).parent
    report = {"groups": []}
    if len(plan["groups"]) != 1:
        raise RuntimeError("one stage per run: split the shot list")
    for group in plan["groups"]:
        t0 = time.time()
        ctx = omni.usd.get_context()
        if not ctx.open_stage(group["stage"]):
            raise RuntimeError(f"open_stage failed: {group['stage']}")
        for _ in range(10):
            simulation_app.update()
        stage = ctx.get_stage()
        cam = UsdGeom.Camera.Define(stage, "/World/ShotCamera")
        cam.CreateFocalLengthAttr(float(cc["focal_length_mm"]))
        cam.CreateHorizontalApertureAttr(float(cc["horizontal_aperture_mm"]))
        cam.CreateVerticalApertureAttr(float(cc["vertical_aperture_mm"]))
        cam.CreateClippingRangeAttr(Gf.Vec2f(*[float(v) for v in pc["clipping_range_m"]]))
        cam_op = UsdGeom.Xformable(cam).AddTransformOp()
        rp = rep.create.render_product(str(cam.GetPath()), tuple(int(v) for v in cc["resolution"]))
        ann = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb"),
            "depth": rep.AnnotatorRegistry.get_annotator(DEPTH_ANNOTATOR),
            "semantic": rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False}),
            "instance_id": rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False}),
            "instance": rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False}),
        }
        for a in ann.values():
            a.attach(rp)

        def pose(shot):
            for mv in shot.get("moves", []):
                op = UsdGeom.Xformable(stage.GetPrimAtPath(mv["path"])).GetOrderedXformOps()[0]
                op.Set(Gf.Matrix4d(*[float(v) for v in mv["matrix"]]))
            m = np.asarray(shot["camera_to_world"], dtype=np.float64)
            cam_op.Set(Gf.Matrix4d(*m.T.flatten().tolist()))    # column-vector -> USD row-vector

        # one discarded frame: the first capture can come back empty while annotators initialise
        pose(group["shots"][0])
        rep.orchestrator.step(rt_subframes=int(pc["rt_subframes"]))
        load_s = time.time() - t0
        shots = []
        for shot in group["shots"]:
            t1 = time.time()
            pose(shot)
            rep.orchestrator.step(rt_subframes=int(shot.get("rt_subframes", pc["rt_subframes"])))
            data = {k: a.get_data() for k, a in ann.items()}
            arrays, infos = {}, {}
            for k, d in data.items():
                arr = np.asarray(d["data"] if isinstance(d, dict) else d)
                if arr.size == 0:
                    raise RuntimeError(f"shot {shot['name']}: annotator {k} returned no data")
                arrays[k] = arr.astype(np.float32) if k == "depth" else arr
                if isinstance(d, dict):
                    infos[k] = jsonable(d.get("info", {}))
            np.savez_compressed(out_dir / f"{shot['name']}.npz", **arrays)
            (out_dir / f"{shot['name']}.info.json").write_text(json.dumps(infos, default=str) + "\n")
            shots.append({"name": shot["name"], "seconds": time.time() - t1})
            print(f"shot {shot['name']} rendered", flush=True)
        rep.orchestrator.wait_until_complete()
        used, total, name = gpu_used()
        report["groups"].append({"name": group["name"], "stage": group["stage"], "load_s": load_s,
                                 "vram_used_mib": float(used), "vram_total_mib": float(total), "gpu": name,
                                 "shots": shots})
    import importlib.metadata as md
    report["isaacsim_version"] = md.version("isaacsim")
    (out_dir / "shots_result.json").write_text(json.dumps(report, indent=1) + "\n")
    print("shots done", flush=True)


# Closing the app ends the process, which would swallow an exception raised in main(): print it
# first and leave with a failing exit code so the batch job stops.
try:
    main()
except BaseException:
    import os
    import sys
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
simulation_app.close()
