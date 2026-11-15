"""Cut one window of consecutive frames out of a long clip, with every map that goes with it.

  python scripts/cut_window.py --location battery-shoreline --run run_006 --altitude 150 --start 120

Writes <bundle>/windows/w_<start>/ holding, for frames start .. start+frames-1 renumbered from 0:
the five PNG sequences and an MP4 of each, metric depth, semantic labels, camera files, the
matching slice of path.json and a window.json. Frames are hard-linked, not copied.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nyc_drone_sim.common import for_location, load_config, slugify  # noqa: E402
from nyc_drone_sim.postrender import MODALITIES, bundle_dir, ff_tools, probe  # noqa: E402


def cut(B: Path, start: int, frames: int, fps: float) -> Path:
    path = json.loads((B / "path.json").read_text())
    total = len(path["frames"])
    if start < 0 or start + frames > total:
        raise SystemExit(f"window {start}..{start + frames - 1} is outside the clip of {total} frames")
    out = B / "windows" / f"w_{start:04d}"
    for m in MODALITIES:
        (out / "clip_0000" / m).mkdir(parents=True, exist_ok=True)
    for d in ("metric_depth", "semantic", "camera"):
        (out / d).mkdir(exist_ok=True)

    def link(src: Path, dst: Path):
        if dst.exists():
            dst.unlink()
        os.link(src, dst)

    for k in range(frames):
        i = start + k
        for m in MODALITIES:
            link(B / "clip_0000" / m / f"{m}_{i:04d}.png", out / "clip_0000" / m / f"{m}_{k:04d}.png")
        link(B / "metric_depth" / f"depth_{i:04d}.npy", out / "metric_depth" / f"depth_{k:04d}.npy")
        link(B / "semantic" / f"semantic_{i:04d}.npz", out / "semantic" / f"semantic_{k:04d}.npz")
        link(B / "camera" / f"camera_{i:04d}.json", out / "camera" / f"camera_{k:04d}.json")
    for m in MODALITIES:
        subprocess.run([ff_tools()[0], "-y", "-v", "error", "-framerate", f"{fps:g}", "-i",
                        str(out / "clip_0000" / m / f"{m}_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", str(out / "clip_0000" / f"{m}.mp4")], check=True)
    t0 = path["frames"][start]["t"]
    sub = {**path, "frames": [{**f, "index": k, "t": f["t"] - t0, "source_index": start + k}
                              for k, f in enumerate(path["frames"][start:start + frames])],
           "clip": {"frames": frames, "fps": fps, "duration_s": frames / fps}}
    (out / "path.json").write_text(json.dumps(sub, indent=1) + "\n")
    info = {m: probe(out / "clip_0000" / f"{m}.mp4") for m in MODALITIES}
    (out / "window.json").write_text(json.dumps({"source_bundle": str(B), "start": start, "frames": frames, "fps": fps,
                                                 "mp4": info}, indent=1) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--altitude", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    frames, fps = int(cfg["clip"]["frames"]), float(cfg["clip"]["fps"])
    out = cut(bundle_dir(cfg, a.run, slugify(a.location), a.altitude), a.start, frames, fps)
    info = json.loads((out / "window.json").read_text())["mp4"]
    ok = all(v["frames"] == frames and abs(v["fps"] - fps) < 1e-6 for v in info.values())
    counts = {d: len(list((out / d).iterdir())) for d in ("metric_depth", "semantic", "camera")}
    print(f"{out}: {frames} frames from {a.start}; MP4s {'ok' if ok else 'WRONG'}: "
          f"{ {k: (v['frames'], v['fps']) for k, v in info.items()} }; files {counts}")
    sys.exit(0 if ok and all(c == frames for c in counts.values()) else 1)


if __name__ == "__main__":
    main()
