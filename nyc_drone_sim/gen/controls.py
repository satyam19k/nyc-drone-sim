"""Build the control videos the video model needs from data already in a clip bundle. CPU only.

Writes into <bundle>/gen/inputs/ and never touches clip_0000/:
  depth_control.mp4    inverse metric depth, near = bright, one fixed range for the whole clip
  edges_geometry.mp4   Canny on the segmentation frames combined with Canny on the depth control:
                       object, building and surface outlines, with no cast shadows in it
  inputs.json          the depth range, frame counts and the bundle hash
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import bundle_dir, bundle_hash, inputs_dir, load_gen_config, video_info, write_video


def depth_range(bdir: Path, cfg: dict) -> tuple[float, float]:
    """Near and far of the clip: percentiles of metric depth over frames sampled along the clip."""
    cc = cfg["controls"]
    files = sorted((bdir / "metric_depth").glob("depth_*.npy"))
    sample = np.concatenate([np.load(f)[::4, ::4].ravel() for f in files[::int(cc["depth_sample_step"])]])
    sample = sample[np.isfinite(sample) & (sample > 0)]
    near, far = np.percentile(sample, cc["depth_percentiles"])
    return float(near), float(far)


def depth_grey(d: np.ndarray, near: float, far: float) -> np.ndarray:
    """Inverse depth scaled so near is 255 and far is 0, as monocular depth models draw it."""
    inv = 1.0 / np.clip(d, near, far)
    return np.round(255.0 * (inv - 1.0 / far) / (1.0 / near - 1.0 / far)).astype(np.uint8)


def build_inputs(bdir: Path, cfg: dict) -> dict:
    import cv2
    cc = cfg["controls"]
    out = inputs_dir(bdir)
    out.mkdir(parents=True, exist_ok=True)
    depth_files = sorted((bdir / "metric_depth").glob("depth_*.npy"))
    seg_files = sorted((bdir / "clip_0000" / "segmentation").glob("segmentation_*.png"))
    if len(depth_files) != len(seg_files) or not depth_files:
        raise RuntimeError(f"{bdir}: {len(depth_files)} depth frames, {len(seg_files)} segmentation frames")
    near, far = depth_range(bdir, cfg)
    lo, hi = int(cc["canny_low"]), int(cc["canny_high"])

    def depth_frames():
        for f in depth_files:
            g = depth_grey(np.load(f), near, far)
            yield np.repeat(g[:, :, None], 3, axis=2)

    def edge_frames():
        for fd, fs in zip(depth_files, seg_files):
            g = depth_grey(np.load(fd), near, far)
            seg = cv2.imread(str(fs), cv2.IMREAD_COLOR)
            e = np.zeros(g.shape, np.uint8)
            for ch in range(3):                       # an outline in any colour channel counts
                e |= cv2.Canny(seg[:, :, ch], lo, hi)
            e |= cv2.Canny(g, lo, hi)
            yield np.repeat(e[:, :, None], 3, axis=2)

    n_d = write_video(depth_frames(), out / "depth_control.mp4", cc["fps"], cc["crf"])
    n_e = write_video(edge_frames(), out / "edges_geometry.mp4", cc["fps"], cc["crf"])
    meta = {"bundle": str(bdir), "bundle_sha256": bundle_hash(bdir), "frames": len(depth_files),
            "depth_control": {"file": "depth_control.mp4", "frames": n_d, "near_m": near, "far_m": far,
                              "encoding": "inverse metric depth along the optical axis, near = 255, far = 0, "
                                          "fixed range for the whole clip",
                              "percentiles": cc["depth_percentiles"]},
            "edges_geometry": {"file": "edges_geometry.mp4", "frames": n_e, "canny": [lo, hi],
                               "source": "segmentation frames and the depth control"},
            "decoded": {k: video_info(out / k) for k in ("depth_control.mp4", "edges_geometry.mp4")}}
    (out / "inputs.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True, help="e.g. runs/run_006/hudson-park-run")
    ap.add_argument("--altitude", type=int, action="append", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    for alt in a.altitude:
        m = build_inputs(bundle_dir(a.run_folder, alt), cfg)
        print(f"{alt} ft: {m['frames']} frames, depth range {m['depth_control']['near_m']:.1f} to "
              f"{m['depth_control']['far_m']:.1f} m -> {inputs_dir(bundle_dir(a.run_folder, alt))}")
