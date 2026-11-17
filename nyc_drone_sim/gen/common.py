"""Shared helpers for the video model. Imports nothing heavier than NumPy and PyYAML."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EDGE_OPTIONS = ("rendered", "none", "geometry")


def load_gen_config(path: str | None = None) -> dict:
    return yaml.safe_load((Path(path) if path else REPO_ROOT / "configs" / "gen.yaml").read_text())


def bundle_dir(run_folder: str | Path, altitude: int) -> Path:
    """<run folder>/<altitude>, e.g. runs/run_006/hudson-park-run/150."""
    p = Path(run_folder)
    return (p if p.is_absolute() else REPO_ROOT / p).resolve() / str(int(altitude))


def bundle_hash(bdir: Path) -> str:
    return hashlib.sha256((bdir / "bundle.json").read_bytes()).hexdigest()


def scene_name(bdir: Path) -> str:
    return json.loads((bdir / "bundle.json").read_text())["location"]


def inputs_dir(bdir: Path) -> Path:
    return bdir / "gen" / "inputs"


def model_dir(bdir: Path, cfg: dict) -> Path:
    return bdir / "gen" / cfg["model"]["name"]


def run_name(run: dict) -> str:
    return f"{run['weather']}_{run['edge']}"


def chunk_count(frames: int, cfg: dict) -> int:
    per, cond = int(cfg["sampling"]["chunk_frames"]), int(cfg["sampling"]["num_conditional_frames"])
    return 1 if frames <= per else 1 + math.ceil((frames - per) / (per - cond))


def chunk_starts(frames: int, cfg: dict) -> list[int]:
    """First output frame of every chunk after the first: the chunk boundaries."""
    per, cond = int(cfg["sampling"]["chunk_frames"]), int(cfg["sampling"]["num_conditional_frames"])
    return [per + k * (per - cond) for k in range(chunk_count(frames, cfg) - 1)]


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video(frames, path: Path, fps: float, crf: int) -> int:
    """Encode an iterable of HxWx3 uint8 RGB frames as H.264 yuv420p, like the simulator's MP4s."""
    proc, n = None, 0
    for f in frames:
        if proc is None:
            h, w = f.shape[:2]
            proc = subprocess.Popen(
                [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path)],
                stdin=subprocess.PIPE)
        proc.stdin.write(f.tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")
    return n


def video_info(path: Path) -> dict:
    """Frame count, size and rate of a video, by decoding it."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = 0
    while cap.grab():
        n += 1
    info = {"frames": n, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS))}
    cap.release()
    return info
