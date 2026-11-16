"""Unattended driver for one GPU job: self-test, then the run list, resumable and isolated.

  python -m nyc_drone_sim.gen.job --run-folder runs/run_006/hudson-park-run [--then runs/run_008/120-broadway]

Behaviour
- Step 0 checks the node (GPU memory, glibc, driver), runs the model repo's own example, then one
  93-frame chunk of the first run. Any failure stops the job and is written to the report.
- Each run is one process of the model's examples/inference.py. It writes into a temporary
  folder that is renamed only on success, so a killed run never looks finished.
- A run whose folder already exists is skipped. A failed run is logged and the next one starts.
- After every run a line is appended to reports/gen_job_<jobid>.md.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .common import REPO_ROOT, bundle_dir, bundle_hash, load_gen_config, model_dir, run_name, video_info
from .plan import spec_paths

JOB = os.environ.get("SLURM_JOB_ID", "local")
REPORT = REPO_ROOT / "reports" / f"gen_job_{JOB}.md"


def say(text: str) -> None:
    print(text, flush=True)
    with REPORT.open("a") as f:
        f.write(text + "\n")


def version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v))


def gpu_query(fields: str) -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        return []
    return [x.strip() for x in out[0].split(",")] if out else []


class VramSampler(threading.Thread):
    """Peak GPU memory in MiB, sampled every two seconds while a run is in progress."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.wait(2.0):
            try:
                self.peak = max(self.peak, int(gpu_query("memory.used")[0]))
            except Exception:
                pass


def node_check(cfg: dict) -> tuple[bool, str, str]:
    mc = cfg["model"]
    q = gpu_query("name,memory.total,driver_version")
    if len(q) < 3:
        return False, "no GPU visible to nvidia-smi", "unknown"
    name, mem_gb, driver = q[0], float(q[1]) / 1024, q[2]
    glibc = platform.libc_ver()[1]
    problems = []
    if mem_gb < float(mc["min_vram_gb"]):
        problems.append(f"GPU memory {mem_gb:.0f} GB is below the {mc['min_vram_gb']} GB this model needs")
    if version_tuple(glibc) < version_tuple(mc["min_glibc"]):
        problems.append(f"glibc {glibc} is older than {mc['min_glibc']}")
    if version_tuple(driver) < version_tuple(mc["min_driver"]):
        problems.append(f"NVIDIA driver {driver} is older than {mc['min_driver']}")
    desc = f"{name}, {mem_gb:.0f} GB, driver {driver}, glibc {glibc}, node {platform.node()}"
    return not problems, desc if not problems else desc + ": " + "; ".join(problems), name


def run_model(spec: Path, out_dir: Path, cfg: dict, log: Path, extra: list[str] | None = None) -> dict:
    """One process of the model's own inference script. Returns timing, peak memory and status."""
    mc = cfg["model"]
    cmd = [sys.executable, "examples/inference.py", "-i", str(spec), "-o", str(out_dir), *(extra or [])]
    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    with log.open("w") as f:
        rc = subprocess.run(cmd, cwd=mc["repo_dir"], stdout=f, stderr=subprocess.STDOUT).returncode
    sampler.stop.set()
    text = log.read_text(errors="replace")
    per_chunk = re.findall(r"Average time per chunk: ([0-9.]+)", text)
    chunks = [int(x) for x in re.findall(r"Generating chunk \d+/(\d+)", text)]
    return {"returncode": rc, "seconds": round(time.time() - t0, 1), "peak_vram_mib": sampler.peak,
            "seconds_per_chunk": float(per_chunk[-1]) if per_chunk else None,
            "chunks": chunks[-1] if chunks else None,
            "blocked_by_guardrail": "Guardrail blocked" in text or "GUARDRAIL BLOCKED" in text,
            "tail": "\n".join(text.strip().splitlines()[-25:])}


def finish_run(tmp: Path, final: Path, spec: Path, bdir: Path, cfg: dict, res: dict, gpu: str) -> None:
    spec_data = json.loads(spec.read_text())
    shutil.copy(spec, tmp / "spec.json")
    (tmp / "prompt.txt").write_text(spec_data["prompt"] + "\n")
    (tmp / "seed.txt").write_text(f"{spec_data['seed']}\n")
    (tmp / "model_commit.txt").write_text(cfg["model"]["commit"] + "\n")
    (tmp / "bundle_hash.txt").write_text(bundle_hash(bdir) + "\n")
    (tmp / "timing.json").write_text(json.dumps({k: v for k, v in res.items() if k != "tail"} | {
        "gpu": gpu, "job": JOB, "node": platform.node(), "output": video_info(tmp / "out.mp4")}, indent=1) + "\n")
    if final.exists():          # another job finished the same run first: keep its result
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.rename(final)


def one_run(number: int, run: dict, spec: Path, bdir: Path, cfg: dict, gpu: str,
            name: str | None = None, extra: list[str] | None = None) -> bool:
    final = model_dir(bdir, cfg) / (name or run_name(run))
    label = f"run {number} ({bdir.parent.name} {bdir.name} ft, {run['weather']}, edge {run['edge']})"
    if (final / "out.mp4").exists() and (final / "timing.json").exists():
        say(f"- {label}: already complete, skipped. `{final}`")
        return True
    tmp = final.with_name(final.name + f".tmp_{JOB}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        res = run_model(spec, tmp, cfg, tmp / "run.log", extra)
        ok = res["returncode"] == 0 and (tmp / "out.mp4").exists() and not res["blocked_by_guardrail"]
        if ok:
            finish_run(tmp, final, spec, bdir, cfg, res, gpu)
        vram = f"{res['peak_vram_mib'] / 1024:.1f} GB"
        per = f"{res['seconds_per_chunk']:.0f} s per chunk" if res["seconds_per_chunk"] else "time per chunk not reported"
        if ok:
            say(f"- {label}: **pass**. GPU {gpu}, {res['chunks']} chunks, {per}, {res['seconds']:.0f} s in all, "
                f"peak VRAM {vram}. `{final}`")
        else:
            why = "blocked by the model's guardrail" if res["blocked_by_guardrail"] else f"exit code {res['returncode']}"
            say(f"- {label}: **fail** ({why}). GPU {gpu}, {res['seconds']:.0f} s, peak VRAM {vram}. Log kept in `{tmp}`\n\n"
                f"```\n{res['tail']}\n```")
        return ok
    except Exception as e:                      # one broken run must not end the job
        say(f"- {label}: **fail** (driver error: {type(e).__name__}: {e}). `{tmp}`")
        return False


def self_test(run_folder: str, cfg: dict, gpu: str) -> bool:
    mc = cfg["model"]
    out = REPO_ROOT / "runs" / "gen_selftest" / JOB
    out.mkdir(parents=True, exist_ok=True)
    res = run_model(Path(mc["repo_dir"]) / mc["example_spec"], out / "example", cfg, out / "example.log")
    produced = sorted((out / "example").glob("*.mp4")) if (out / "example").exists() else []
    if res["returncode"] != 0 or not produced:
        say(f"- self-test, the model repo's own example: **fail** (exit code {res['returncode']}).\n\n```\n{res['tail']}\n```")
        return False
    per = f"{res['seconds_per_chunk']:.0f} s per chunk" if res["seconds_per_chunk"] else "time per chunk not reported"
    say(f"- self-test, the model repo's own example: **pass**. {per}, {res['seconds']:.0f} s in all, "
        f"peak VRAM {res['peak_vram_mib'] / 1024:.1f} GB. `{produced[0]}`")
    first = cfg["runs"][0]
    bdir = bundle_dir(run_folder, first["altitude"])
    number, run, spec = spec_paths(bdir, cfg)[0]
    frames = str(cfg["sampling"]["chunk_frames"])
    return one_run(0, run, spec, bdir, cfg, gpu, name=f"selftest_{run_name(run)}_{frames}f", extra=["--max-frames", frames])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--then", action="append", default=[], help="further run folders, run after the first")
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--self-test-only", action="store_true", help="node check and self-test, then stop")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    REPORT.parent.mkdir(exist_ok=True)
    say(f"# Video-model job {JOB}\n\nStarted {time.strftime('%Y-%m-%d %H:%M:%S')}. Model {cfg['model']['name']} "
        f"at commit {cfg['model']['commit'][:12]}.\n")
    ok, desc, gpu = node_check(cfg)
    say(f"- node check: {'**pass**' if ok else '**fail**'}. {desc}")
    if not ok:
        return 2
    if not a.skip_self_test and not self_test(a.run_folder, cfg, gpu):
        say("\nSelf-test failed, so no run was started.")
        return 3
    if a.self_test_only:
        say("\nSelf-test only: no run of the run list was started.")
        return 0
    failed = 0
    for folder in [a.run_folder, *a.then]:
        say(f"\n## {folder}\n")
        for alt in sorted({int(r["altitude"]) for r in cfg["runs"]}):
            if not (bundle_dir(folder, alt) / "bundle.json").exists():
                say(f"- {folder} {alt} ft: no bundle, skipped")
        todo = []
        for alt in {int(r["altitude"]) for r in cfg["runs"]}:
            bdir = bundle_dir(folder, alt)
            if (bdir / "bundle.json").exists():
                todo += [(n, run, spec, bdir) for n, run, spec in spec_paths(bdir, cfg)]
        for number, run, spec, bdir in sorted(todo, key=lambda x: x[0]):      # the run list's own order
            if not spec.exists():
                say(f"- run {number}: **fail** (spec file missing: `{spec}`)")
                failed += 1
                continue
            failed += not one_run(number, run, spec, bdir, cfg, gpu)
    say(f"\nFinished {time.strftime('%Y-%m-%d %H:%M:%S')}: {failed} run(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
