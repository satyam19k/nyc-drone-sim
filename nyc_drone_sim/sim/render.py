"""Stage 5: render one clip bundle for one location and altitude. GPU, headless.

The camera is driven per frame from path_<altitude>.json; the timeline is not used to move it.
Layout written under runs/<run_id>/<location>/<altitude_ft>/:
  clip_0000/{rgb,depth,segmentation,shaded_seg,edges}/ + one MP4 each   (CosmosWriter)
  metric_depth/  camera/  semantic/  rerender_frame0.npz  path.json  bundle.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--location", default="120 Broadway")
ap.add_argument("--altitude", type=int, required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--config", default=None)
args, _ = ap.parse_known_args()

# SimulationApp must exist before any other Omniverse import.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from nyc_drone_sim.common import REPO_ROOT, load_config, scene_dir, slugify  # noqa: E402

DEPTH_ANNOTATOR = "distance_to_image_plane"   # metres, float32


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def main():
    cfg = load_config(args.config)
    slug = slugify(args.location)
    sdir = scene_dir(cfg, slug)
    rc, cc = cfg["render"], cfg["camera"]
    path_src = sdir / f"path_{args.altitude}.json"
    path = json.loads(path_src.read_text())
    frames = path["frames"]
    fps = float(cfg["clip"]["fps"])

    out = REPO_ROOT / cfg["output"]["runs_dir"] / args.run / slug / str(args.altitude)
    if out.exists():
        shutil.rmtree(out)
    for d in ("metric_depth", "camera", "semantic"):
        (out / d).mkdir(parents=True)
    shutil.copy(path_src, out / "path.json")

    t0 = time.time()
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(sdir / "scene.usdc")):
        raise RuntimeError("open_stage failed")
    for _ in range(10):
        simulation_app.update()
    stage = ctx.get_stage()

    # CosmosWriter takes its video frame rate from the timeline.
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(len(frames) - 1)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_time_codes_per_second(fps)

    cam = UsdGeom.Camera.Define(stage, "/World/DroneCamera")
    cam.CreateFocalLengthAttr(float(cc["focal_length_mm"]))
    cam.CreateHorizontalApertureAttr(float(cc["horizontal_aperture_mm"]))
    cam.CreateVerticalApertureAttr(float(cc["vertical_aperture_mm"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(*[float(v) for v in rc["clipping_range_m"]]))
    xform_op = UsdGeom.Xformable(cam).AddTransformOp()

    def set_pose(frame):
        # path.json is column-vector (p_world = M @ p_cam); USD matrices are row-vector, so transpose.
        m = np.asarray(frame["camera_to_world"], dtype=np.float64)
        xform_op.Set(Gf.Matrix4d(*m.T.flatten().tolist()))

    rp = rep.create.render_product(str(cam.GetPath()), tuple(int(v) for v in cc["resolution"]))
    depth = rep.AnnotatorRegistry.get_annotator(DEPTH_ANNOTATOR)
    cam_params = rep.AnnotatorRegistry.get_annotator("camera_params")
    sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    for a in (depth, cam_params, sem, rgb):
        a.attach(rp)

    subframes = int(rc["rt_subframes"])

    def step():
        rep.orchestrator.step(pause_timeline=False, rt_subframes=subframes)

    # Warm-up at the frame-0 pose before the writer exists: the first captured frames can be empty.
    set_pose(frames[0])
    for _ in range(int(rc["warmup_frames"])):
        step()

    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(out))
    writer = rep.WriterRegistry.get("CosmosWriter")
    writer.initialize(backend=backend, use_instance_id=bool(rc["use_instance_id"]), segmentation_mapping=None,
                      canny_threshold_low=int(rc["canny_threshold_low"]),
                      canny_threshold_high=int(rc["canny_threshold_high"]))
    writer.attach(rp)

    # A throwaway clip at the frame-0 pose lets the writer's own annotators initialise. Its last
    # frame doubles as the independent second render of frame 0.
    for _ in range(int(rc["warmup_frames"])):
        step()
    rerender = np.asarray(rgb.get_data())
    if rerender.size == 0:
        raise RuntimeError("rgb annotator returned no data after warm-up")
    np.savez_compressed(out / "rerender_frame0.npz", rgb=rerender)
    rep.orchestrator.wait_until_complete()
    writer.next_clip()                      # warm-up frames end up in clip_0000

    load_s = time.time() - t0
    t1 = time.time()
    sem_info = None
    for i, fr in enumerate(frames):
        set_pose(fr)
        step()
        d = np.asarray(depth.get_data(), dtype=np.float32)
        s = sem.get_data()
        if d.size == 0 or np.asarray(s["data"]).size == 0:
            raise RuntimeError(f"frame {i}: annotator returned no data")
        np.save(out / "metric_depth" / f"depth_{i:04d}.npy", d)
        np.savez_compressed(out / "semantic" / f"semantic_{i:04d}.npz", data=np.asarray(s["data"]))
        sem_info = s["info"]
        (out / "camera" / f"camera_{i:04d}.json").write_text(json.dumps(
            {"index": i, "t": fr["t"], "requested_camera_to_world": fr["camera_to_world"],
             "camera_params": jsonable(cam_params.get_data())}, indent=1, default=str) + "\n")
        if i % 10 == 0:
            print(f"frame {i}/{len(frames)}", flush=True)
    rep.orchestrator.wait_until_complete()
    writer.next_clip()                      # encodes the real clip (clip_0001)
    writer.detach()
    render_s = time.time() - t1

    # Drop the warm-up clip and move the real one to clip_0000.
    shutil.rmtree(out / "clip_0000")
    (out / "clip_0001").rename(out / "clip_0000")

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,driver_version",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    import importlib.metadata as md
    (out / "bundle.json").write_text(json.dumps({
        "location": slug, "run_id": args.run,
        "origin": json.loads((sdir / "scene_meta.json").read_text())["origin"],
        "altitude_ft": args.altitude, "altitude_m": path["altitude_m"],
        "path_type": path["path_type"], "path_params": path["path_params"],
        "clip": {"frames": len(frames), "fps": fps, "duration_s": len(frames) / fps,
                 "resolution": list(cc["resolution"])},
        "sha256": {"scene.usdc": sha256(sdir / "scene.usdc"), "path.json": sha256(out / "path.json")},
        "isaacsim_version": md.version("isaacsim"),
        "render": {"renderer": "RTX default (as launched by SimulationApp headless)",
                   "rt_subframes": subframes, "warmup_frames": int(rc["warmup_frames"]),
                   "use_instance_id": bool(rc["use_instance_id"]),
                   "canny_threshold_low": int(rc["canny_threshold_low"]),
                   "canny_threshold_high": int(rc["canny_threshold_high"]),
                   "metric_depth_annotator": DEPTH_ANNOTATOR, "metric_depth_unit": "metres",
                   "metric_depth_format": "metric_depth/depth_NNNN.npy float32"},
        "semantic_info": jsonable(sem_info),
        "camera": {k: cc[k] for k in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm", "pitch_deg")},
        "seed": cfg["seed"], "gpu": gpu,
        "timing_s": {"load_and_warmup": load_s, "render": render_s},
        "mp4_from_writer": sorted(p.name for p in (out / "clip_0000").glob("*.mp4")),
    }, indent=1, default=str) + "\n")
    print(f"render done: {len(frames)} frames in {render_s:.1f} s -> {out}", flush=True)


try:
    main()
finally:
    simulation_app.close()
