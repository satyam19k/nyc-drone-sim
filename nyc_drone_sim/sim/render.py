"""Render one clip bundle for one location and altitude. GPU, headless.

The camera is driven per frame from path_<altitude>.json (path_<altitude>_street.json with
--variant street); the timeline is not used to move it.
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
ap.add_argument("--variant", default=None, choices=["street", "long"])
ap.add_argument("--view", default=None, help="one of the place's extra camera views; use its own --run")
args, _ = ap.parse_known_args()

# SimulationApp must exist before any other Omniverse import.
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

from nyc_drone_sim.common import for_location, REPO_ROOT, load_config, scene_dir, slugify, variant_suffix  # noqa: E402

from omni.replicator.core.scripts.writers_default.cosmos import CosmosWriter  # noqa: E402

DEPTH_ANNOTATOR = "distance_to_image_plane"   # metres, float32


class CityCosmosWriter(CosmosWriter):
    """CosmosWriter with a configurable range for its colourised depth picture.

    The stock writer colourises depth over a fixed 0.1-100 m range, which leaves a city seen
    from 45-150 m almost entirely black. Everything else is unchanged.
    """

    def __init__(self, *a, depth_near: float, depth_far: float, **k):
        super().__init__(*a, **k)
        idx = [i for i, an in enumerate(self.annotators) if getattr(an, "name", None) == "depth"]
        if len(idx) != 1:
            idx = [3]   # position of the depth annotator in CosmosWriter 0.0.2
        self.annotators[idx[0]] = rep.AnnotatorRegistry.get_annotator("distance_to_camera").augment(
            "ColorizeDepth", near=float(depth_near), far=float(depth_far), name="depth")


rep.WriterRegistry.register(CityCosmosWriter)


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
    cfg = for_location(load_config(args.config), args.location)
    slug = slugify(args.location)
    sdir = scene_dir(cfg, slug)
    rc, cc = cfg["render"], cfg["camera"]
    path_src = sdir / f"path_{args.altitude}{variant_suffix(args.variant, args.view)}.json"
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

    # Recorded motion: poses per frame for moving vehicles and walking pedestrians. Only objects
    # near the camera are updated; the rest keep the pose they were built with.
    movers = None
    motion_src = sdir / "motion.npz"
    if cfg.get("populate", {}).get("enabled") and motion_src.exists():
        mo = np.load(motion_src)
        placed = {o["id"]: o for o in json.loads((sdir / "placements.json").read_text())["objects"]}
        roots = {"vehicle": "Vehicles", "person": "People"}
        ops, scales = [], []
        for oid in mo["ids"].tolist():
            prim = stage.GetPrimAtPath(f"/World/Objects/{roots[placed[oid]['class']]}/{oid}")
            if not prim.IsValid():
                raise RuntimeError(f"moving object {oid} is not in the stage")
            ops.append(UsdGeom.Xformable(prim).GetOrderedXformOps()[0])
            scales.append(float(placed[oid]["scale"]))
        movers = {"ops": ops, "poses": mo["poses"], "scale": scales,
                  "reach": float(rc["motion_radius_per_altitude"]) * path["altitude_m"] + float(rc["motion_radius_base_m"])}
        if movers["poses"].shape[1] < len(frames):
            raise RuntimeError("recorded motion is shorter than the clip")
    moved_total = 0

    def set_pose(frame):
        nonlocal moved_total
        # path.json is column-vector (p_world = M @ p_cam); USD matrices are row-vector, so transpose.
        m = np.asarray(frame["camera_to_world"], dtype=np.float64)
        xform_op.Set(Gf.Matrix4d(*m.T.flatten().tolist()))
        if movers is not None:
            pose = movers["poses"][:, frame["index"], :]
            near = np.nonzero(np.hypot(pose[:, 0] - m[0, 3], pose[:, 1] - m[1, 3]) <= movers["reach"])[0]
            for k in near.tolist():
                x, y, yaw = (float(v) for v in pose[k])
                c, s_, sc_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), movers["scale"][k]
                movers["ops"][k].Set(Gf.Matrix4d(c * sc_, s_ * sc_, 0, 0, -s_ * sc_, c * sc_, 0, 0, 0, 0, sc_, 0, x, y, 0, 1))
            moved_total += len(near)

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
    writer = rep.WriterRegistry.get("CityCosmosWriter")
    writer.initialize(backend=backend, use_instance_id=bool(rc["use_instance_id"]), segmentation_mapping=None,
                      depth_near=rc["depth_colorize_near_m"], depth_far=rc["depth_colorize_far_m"],
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
    sem_info, sem_labels = None, {}
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
        # each frame's info lists only the classes visible in that frame: keep the union, or a
        # class that has left the view by the last frame would be missing from the bundle
        sem_labels.update(sem_info.get("idToLabels", {}))
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
        "location": slug, "run_id": args.run, "variant": args.variant or "default", "view": args.view or "main",
        "origin": json.loads((sdir / "scene_meta.json").read_text())["origin"],
        "altitude_ft": args.altitude, "altitude_m": path["altitude_m"],
        "path_type": path["path_type"], "path_params": path["path_params"],
        "clip": {"frames": len(frames), "fps": fps, "duration_s": len(frames) / fps,
                 "resolution": list(cc["resolution"])},
        "sha256": {"scene.usdc": sha256(sdir / "scene.usdc"), "path.json": sha256(out / "path.json"),
                   **({"placements.json": sha256(sdir / "placements.json"), "motion.npz": sha256(motion_src),
                       "asset_manifest.json": sha256(REPO_ROOT / cfg["assets"]["library_dir"] / "manifest.json")}
                      if movers is not None else {})},
        "populated": movers is not None,
        "moving_objects": 0 if movers is None else len(movers["ops"]),
        "mean_objects_moved_per_frame": 0 if movers is None else moved_total / max(len(frames), 1),
        "isaacsim_version": md.version("isaacsim"),
        "render": {"renderer": "RTX default (as launched by SimulationApp headless)",
                   "rt_subframes": subframes, "warmup_frames": int(rc["warmup_frames"]),
                   "use_instance_id": bool(rc["use_instance_id"]),
                   "canny_threshold_low": int(rc["canny_threshold_low"]),
                   "canny_threshold_high": int(rc["canny_threshold_high"]),
                   "depth_colorize_range_m": [float(rc["depth_colorize_near_m"]), float(rc["depth_colorize_far_m"])],
                   "metric_depth_annotator": DEPTH_ANNOTATOR, "metric_depth_unit": "metres",
                   "metric_depth_format": "metric_depth/depth_NNNN.npy float32"},
        "semantic_info": jsonable({**sem_info, "idToLabels": sem_labels}),
        "camera": {k: cc[k] for k in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm", "pitch_deg")},
        "seed": cfg["seed"], "gpu": gpu,
        "timing_s": {"load_and_warmup": load_s, "render": render_s},
        "mp4_from_writer": sorted(p.name for p in (out / "clip_0000").glob("*.mp4")),
    }, indent=1, default=str) + "\n")
    print(f"render done: {len(frames)} frames in {render_s:.1f} s -> {out}", flush=True)


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
