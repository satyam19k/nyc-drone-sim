"""Run the video model (Cosmos-Transfer2.5) on one or more spec files. Needs an 80 GB GPU.

Where the model is:
  model checkout   MODEL_REPO, else `model.repo_dir` in configs/gen.yaml
                   (a path relative to this repository, default external/cosmos-transfer2.5)
  its Python       MODEL_PYTHON, else <model.env_dir>/bin/python when that exists
                   (relative to this repository, default env_gen), else the Python running this

Two ways to call it:
  run_batch(specs, out_dir, cfg)   several short specs with one model load; each spec's "name"
                                   becomes <out_dir>/<name>.mp4, and a spec whose video already
                                   exists is skipped, so a batch that was cut off can be rerun
  run_clip(spec, final, cfg)       one spec into its own folder: the model writes into a
                                   temporary folder that is renamed to `final` only when out.mp4
                                   exists, so a failed run never looks finished; an existing
                                   folder is never overwritten (the new one gets _r2, _r3, ...)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import REPO_ROOT


def _in_repo(value: str) -> Path:
    p = Path(os.path.expanduser(value))
    return p if p.is_absolute() else REPO_ROOT / p


def model_repo(cfg: dict) -> Path:
    path = _in_repo(os.environ.get("MODEL_REPO") or cfg["model"]["repo_dir"])
    if not (path / "examples" / "inference.py").exists():
        raise RuntimeError(f"no model checkout at {path}: set MODEL_REPO or model.repo_dir in configs/gen.yaml")
    return path


def model_python(cfg: dict) -> str:
    if os.environ.get("MODEL_PYTHON"):
        return os.environ["MODEL_PYTHON"]
    env = _in_repo(cfg["model"].get("env_dir", "env_gen")) / "bin" / "python"
    return str(env) if env.exists() else sys.executable


def _commit(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _inference(specs: list[Path], out_dir: Path, cfg: dict, log: Path) -> int:
    repo = model_repo(cfg)
    cmd = [model_python(cfg), "examples/inference.py", "-i", *map(str, specs), "-o", str(out_dir)]
    with open(log, "a") as f:
        f.write(f"{time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT).returncode


def run_batch(specs: list[Path], out_dir: Path, cfg: dict, log: Path | None = None) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log or out_dir / "run.log"
    todo = [s for s in specs if not (out_dir / f"{json.loads(Path(s).read_text())['name']}.mp4").exists()]
    if not todo:
        return 0
    return _inference(todo, out_dir, cfg, log)


class _VramSampler:
    """Peak GPU memory in MiB, read every few seconds with nvidia-smi while the model runs."""

    def __init__(self, every_s: float = 5.0):
        self.peak, self.every_s, self._stop = None, every_s, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True)
            except OSError:                       # no nvidia-smi here: no samples
                return
            vals = [int(v) for v in r.stdout.split() if v.isdigit()]
            if vals:
                self.peak = max(self.peak or 0, max(vals))
            self._stop.wait(self.every_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)


def run_clip(spec: Path, final: Path, cfg: dict) -> Path | None:
    """Generate one clip; returns the finished folder, or None when the run failed."""
    tmp = final.with_name(final.name + ".tmp_" + time.strftime("%Y%m%d_%H%M%S"))
    tmp.mkdir(parents=True)
    t0 = time.time()
    with _VramSampler() as vram:
        rc = _inference([spec], tmp, cfg, tmp / "run.log")
    seconds = int(time.time() - t0)
    if rc != 0 or not (tmp / "out.mp4").exists() or (tmp / "out.mp4").stat().st_size == 0:
        tail = (tmp / "run.log").read_text(errors="replace").replace("\r", "\n").splitlines()[-15:]
        print(f"FAILED (exit {rc}); kept in {tmp}\n" + "\n".join(tail), flush=True)
        return None
    s = json.loads(spec.read_text())
    (tmp / "spec.json").write_text(spec.read_text())
    (tmp / "prompt.txt").write_text(s["prompt"] + "\n")
    (tmp / "seed.txt").write_text(f"{s['seed']}\n")
    bundle = Path(s["video_path"]).parents[1] / "bundle.json"
    if bundle.exists():
        (tmp / "bundle_hash.txt").write_text(hashlib.sha256(bundle.read_bytes()).hexdigest() + "\n")
    log = (tmp / "run.log").read_text(errors="replace")
    per = re.findall(r"Average time per chunk: ([0-9.]+)", log)
    chunks = re.findall(r"Generating chunk \d+/(\d+)", log)
    (tmp / "timing.json").write_text(json.dumps({
        "seconds": seconds, "seconds_per_chunk": float(per[-1]) if per else None,
        "chunks": int(chunks[-1]) if chunks else None, "peak_vram_mib": vram.peak}, indent=1) + "\n")
    (tmp / "model_commit.txt").write_text(_commit(model_repo(cfg)) + "\n")
    out, n = final, 2
    while out.exists():
        out, n = final.with_name(f"{final.name}_r{n}"), n + 1
    tmp.rename(out)
    print(f"done in {seconds} s -> {out}", flush=True)
    return out
