"""Cut a window of frames out of a clip bundle, with every control video for that window. CPU only.

  python -m nyc_drone_sim.gen.windows --run-folder runs/run_009/canal-shore-street --altitude 150 \
      --name shoreline --start 114 --frames 93

Writes <bundle>/gen/windows/<name>/ with rgb.mp4, segmentation.mp4, edges_rendered.mp4,
edges_geometry.mp4, depth_control.mp4, depth_control_smooth.mp4 and window.json. Frames come from
the bundle's PNG sequences and metric depth, never from a re-compressed video, and the depth range
is the whole clip's, so a window's depth picture matches the same frames of the full clip.

The smooth depth control is the same picture with a fixed dither pattern added before rounding to
8 bits. A plain 8-bit ramp over flat ground has contour lines, one grey level high and up to the
full image width long; the dither breaks them up, and because the pattern is the same in every
frame it does not flicker.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import bundle_dir, bundle_hash, load_gen_config, video_info, write_video
from .controls import depth_grey, depth_range


def dither_pattern(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Fixed noise in [-0.5, 0.5), the same for every frame of every clip."""
    return np.random.default_rng(seed).random(shape, dtype=np.float32) - 0.5


def depth_grey_smooth(d: np.ndarray, near: float, far: float, pattern: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.clip(d, near, far)
    g = 255.0 * (inv - 1.0 / far) / (1.0 / near - 1.0 / far)
    return np.clip(np.floor(g + 0.5 + pattern), 0, 255).astype(np.uint8)


def contour_lines(grey: np.ndarray, mask: np.ndarray) -> dict:
    """How banded a depth picture is over the masked pixels: lengths of the horizontal runs along
    which the grey level steps from one row to the next."""
    step = (np.abs(np.diff(grey.astype(np.int16), axis=0)) >= 1) & mask[1:] & mask[:-1]
    runs = []
    for row in step:
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        runs.extend((edges[1::2] - edges[::2]).tolist())
    runs = np.array(runs) if runs else np.array([0])
    return {"median_px": float(np.median(runs)), "p90_px": float(np.percentile(runs, 90)),
            "p99_px": float(np.percentile(runs, 99)), "longest_px": int(runs.max())}


def cut_window(bdir: Path, name: str, start: int, frames: int, cfg: dict, out: Path | None = None,
               only: set[str] | None = None) -> dict:
    """Into <bundle>/gen/windows/<name>/, or `out`; with `only`, just those files and window.json."""
    import cv2
    cc = cfg["controls"]
    out = out or bdir / "gen" / "windows" / name
    out.mkdir(parents=True, exist_ok=True)
    idx = list(range(start, start + frames))
    clip = bdir / "clip_0000"
    for folder in ("rgb", "segmentation", "edges"):
        if not (clip / folder / f"{folder}_{idx[-1]:04d}.png").exists():
            raise RuntimeError(f"{bdir}: frame {idx[-1]} is missing from {folder}/")
    near, far = depth_range(bdir, cfg)
    lo, hi = int(cc["canny_low"]), int(cc["canny_high"])
    pattern = dither_pattern((int(cfg["window"]["height"]), int(cfg["window"]["width"])), int(cfg["window"]["dither_seed"]))

    def png(folder):
        for i in idx:
            yield cv2.cvtColor(cv2.imread(str(clip / folder / f"{folder}_{i:04d}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth(smooth):
        for i in idx:
            d = np.load(bdir / "metric_depth" / f"depth_{i:04d}.npy")
            g = depth_grey_smooth(d, near, far, pattern) if smooth else depth_grey(d, near, far)
            yield np.repeat(g[:, :, None], 3, axis=2)

    def geometry_edges():
        for i in idx:
            g = depth_grey(np.load(bdir / "metric_depth" / f"depth_{i:04d}.npy"), near, far)
            seg = cv2.imread(str(clip / "segmentation" / f"segmentation_{i:04d}.png"), cv2.IMREAD_COLOR)
            e = cv2.Canny(g, lo, hi)
            for ch in range(3):
                e |= cv2.Canny(seg[:, :, ch], lo, hi)
            yield np.repeat(e[:, :, None], 3, axis=2)

    files = {"rgb.mp4": png("rgb"), "segmentation.mp4": png("segmentation"), "edges_rendered.mp4": png("edges"),
             "edges_geometry.mp4": geometry_edges(), "depth_control.mp4": depth(False),
             "depth_control_smooth.mp4": depth(True)}
    if only is not None:
        files = {k: v for k, v in files.items() if k in only}
    counts = {}
    for fname, gen in files.items():
        crf = int(cc["crf"]) if fname != "depth_control_smooth.mp4" else int(cfg["window"]["smooth_crf"])
        counts[fname] = write_video(gen, out / fname, cc["fps"], crf)
    meta = {"bundle": str(bdir), "bundle_sha256": bundle_hash(bdir), "name": name, "first_frame": start, "frames": frames,
            "depth_range_m": [near, far], "files": {k: video_info(out / k) for k in files}}
    (out / "window.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def cut_video(src: Path, start: int, count: int, dst: Path, cfg: dict) -> int:
    """Frames start .. start + count - 1 of a video, re-encoded like the other controls."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    for _ in range(start):                     # decode up to start: seeking in H.264 is not frame-exact
        if not cap.grab():
            raise RuntimeError(f"{src} ends before frame {start}")

    def frames():
        for _ in range(count):
            ok, f = cap.read()
            if not ok:
                raise RuntimeError(f"{src} ends before frame {start + count - 1}")
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    n = write_video(frames(), dst, cfg["controls"]["fps"], int(cfg["controls"]["crf"]))
    cap.release()
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--altitude", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    m = cut_window(bundle_dir(a.run_folder, a.altitude), a.name, a.start, a.frames, cfg)
    print(f"{a.name}: frames {m['first_frame']} to {m['first_frame'] + m['frames'] - 1}, "
          + ", ".join(f"{k} {v['frames']}" for k, v in m["files"].items()))
