"""Unattended driver for one GPU job: self-test, then the run list, resumable and isolated.

  python -m nyc_drone_sim.gen.job --run-folder runs/run_006/hudson-park-run [--then runs/run_008/120-broadway]
  ... --self-test-only                 node check and self-test, then stop
  ... --max-frames 93 --batch          the run list on the first 93 frames only, one model load per control set

Behaviour
- Step 0 checks the node (GPU memory, glibc, driver), runs the model repo's own example, then one
  93-frame chunk of the first run on the full model, then the same chunk on the distilled edge
  model. A failure of the first two stops the job and is written to the report.
- Each run is one process of the model's examples/inference.py. It writes into a temporary
  folder that is renamed only on success, so a killed run never looks finished.
- A run whose folder already exists is skipped. A failed run is logged and the next one starts.
- With --batch, runs that use the same set of controls share one process, so the model is loaded
  once for all of them. Each finished run is still kept on its own, whatever happens to the rest.
- With --max-frames N only the first N frames are generated, into <weather>_<edge>_<N>f.
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


def parse_log(text: str) -> dict:
    """What the model's log says about one sample, or about a whole process."""
    per_chunk = re.findall(r"Average time per chunk: ([0-9.]+)", text)
    chunks = [int(x) for x in re.findall(r"Generating chunk \d+/(\d+)", text)]
    return {"seconds_per_chunk": float(per_chunk[-1]) if per_chunk else None,
            "chunks": chunks[-1] if chunks else None,
            "blocked_by_guardrail": "Guardrail blocked" in text or "GUARDRAIL BLOCKED" in text,
            # the colour render must not reach the model as a style reference: the model logs this
            # line when no image context is passed
            "image_context": "none" if "No image context provided" in text else "not confirmed in the log"}


def run_model(spec: Path | list[Path], out_dir: Path, cfg: dict, log: Path, extra: list[str] | None = None,
              env: dict | None = None) -> dict:
    """One process of the model's own inference script, for one spec or several (the model is then
    loaded once). Returns timing, peak memory and status."""
    mc = cfg["model"]
    specs = spec if isinstance(spec, list) else [spec]
    cmd = [sys.executable, "examples/inference.py", "-i", *[str(p) for p in specs], "-o", str(out_dir), *(extra or [])]
    sampler = VramSampler()
    sampler.start()
    t0 = time.time()
    with log.open("w") as f:
        rc = subprocess.run(cmd, cwd=mc["repo_dir"], stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, **(env or {})}).returncode
    sampler.stop.set()
    text = log.read_text(errors="replace")
    return {"returncode": rc, "seconds": round(time.time() - t0, 1), "peak_vram_mib": sampler.peak, **parse_log(text),
            "tail": "\n".join(text.strip().splitlines()[-25:]), "text": text}


def finish_run(tmp: Path, final: Path, spec: Path, bdir: Path, cfg: dict, res: dict, gpu: str) -> None:
    spec_data = json.loads(spec.read_text())
    shutil.copy(spec, tmp / "spec.json")
    (tmp / "prompt.txt").write_text(spec_data["prompt"] + "\n")
    (tmp / "seed.txt").write_text(f"{spec_data['seed']}\n")
    (tmp / "model_commit.txt").write_text(cfg["model"]["commit"] + "\n")
    (tmp / "bundle_hash.txt").write_text(bundle_hash(bdir) + "\n")
    (tmp / "timing.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("tail", "text")} | {
        "gpu": gpu, "job": JOB, "node": platform.node(), "output": video_info(tmp / "out.mp4")}, indent=1) + "\n")
    if final.exists():          # another job finished the same run first: keep its result
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        tmp.rename(final)


def describe(res: dict) -> str:
    per = f"{res['seconds_per_chunk']:.0f} s per chunk" if res["seconds_per_chunk"] else "time per chunk not reported"
    return f"{res['chunks']} chunks, {per}, image context {res['image_context']}"


def one_run(number: int, run: dict, spec: Path, bdir: Path, cfg: dict, gpu: str,
            name: str | None = None, extra: list[str] | None = None, env: dict | None = None,
            what: str = "") -> bool:
    final = model_dir(bdir, cfg) / (name or run_name(run))
    label = f"run {number} ({bdir.parent.name} {bdir.name} ft, {run['weather']}, edge {run['edge']}{what})"
    if (final / "out.mp4").exists() and (final / "timing.json").exists():
        say(f"- {label}: already complete, skipped. `{final}`")
        return True
    tmp = final.with_name(final.name + f".tmp_{JOB}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        res = run_model(spec, tmp, cfg, tmp / "run.log", extra, env)
        ok = res["returncode"] == 0 and (tmp / "out.mp4").exists() and not res["blocked_by_guardrail"]
        if ok:
            finish_run(tmp, final, spec, bdir, cfg, res, gpu)
        vram = f"{res['peak_vram_mib'] / 1024:.1f} GB"
        if ok:
            say(f"- {label}: **pass**. GPU {gpu}, {describe(res)}, {res['seconds']:.0f} s in all, "
                f"peak VRAM {vram}. `{final}`")
        else:
            why = "blocked by the model's guardrail" if res["blocked_by_guardrail"] else f"exit code {res['returncode']}"
            say(f"- {label}: **fail** ({why}). GPU {gpu}, {res['seconds']:.0f} s, peak VRAM {vram}. Log kept in `{tmp}`\n\n"
                f"```\n{res['tail']}\n```")
        return ok
    except Exception as e:                      # one broken run must not end the job
        say(f"- {label}: **fail** (driver error: {type(e).__name__}: {e}). `{tmp}`")
        return False


def control_set(spec: Path) -> tuple:
    data = json.loads(spec.read_text())
    return tuple(k for k in ("edge", "vis", "depth", "seg") if k in data)


def batch_runs(items: list, cfg: dict, gpu: str, suffix: str, extra: list[str]) -> int:
    """Runs of one bundle that use the same controls, in one process. Returns how many failed.

    The model takes several spec files and loads itself once. Only runs with the same set of
    controls are put together: across different sets it would weight the controls differently
    from a run on its own. Each sample's files are moved into its own folder as it is collected,
    so a batch that dies part-way still keeps the samples it finished.
    """
    bdir = items[0][3]
    pending = []
    for number, run, spec, _ in items:
        final = model_dir(bdir, cfg) / (run_name(run) + suffix)
        label = f"run {number} ({bdir.parent.name} {bdir.name} ft, {run['weather']}, edge {run['edge']}{suffix})"
        if (final / "out.mp4").exists() and (final / "timing.json").exists():
            say(f"- {label}: already complete, skipped. `{final}`")
        else:
            pending.append((number, run, spec, final, label, f"{number:02d}_{run_name(run)}"))
    if not pending:
        return 0
    work = model_dir(bdir, cfg) / f"batch.tmp_{JOB}"
    shutil.rmtree(work, ignore_errors=True)
    (work / "specs").mkdir(parents=True)
    named = []
    for number, run, spec, final, label, sample in pending:        # sample names must be unique in a batch
        data = json.loads(spec.read_text()) | {"name": sample}
        named.append(work / "specs" / f"{sample}.json")
        named[-1].write_text(json.dumps(data, indent=1) + "\n")
    try:
        res = run_model(named, work / "out", cfg, work / "run.log", extra)
    except Exception as e:
        say(f"- batch of {len(pending)} runs: **fail** (driver error: {type(e).__name__}: {e}). `{work}`")
        return len(pending)
    marks = [(m.start(), m.group(1)) for m in re.finditer(r"Processing sample (\S+)", res["text"])]
    parts = {name: res["text"][pos:(marks[i + 1][0] if i + 1 < len(marks) else len(res["text"]))]
             for i, (pos, name) in enumerate(marks)}
    say(f"- batch of {len(pending)} runs in one process: exit code {res['returncode']}, {res['seconds']:.0f} s in all, "
        f"peak VRAM {res['peak_vram_mib'] / 1024:.1f} GB, GPU {gpu}")
    failed = 0
    for number, run, spec, final, label, sample in pending:
        one = parse_log(parts.get(sample, ""))
        video = work / "out" / f"{sample}.mp4"
        if video.exists() and not one["blocked_by_guardrail"]:
            tmp = final.with_name(final.name + f".tmp_{JOB}")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            for src in sorted((work / "out").glob(f"{sample}*")):       # the video, its controls, its arguments
                src.rename(tmp / ("out" + src.name[len(sample):]))
            finish_run(tmp, final, spec, bdir, cfg, one | {"returncode": res["returncode"], "peak_vram_mib": res["peak_vram_mib"],
                                                           "seconds": None, "batch_seconds": res["seconds"],
                                                           "batch_size": len(pending)}, gpu)
            say(f"- {label}: **pass**. {describe(one)}. `{final}`")
        else:
            failed += 1
            why = "blocked by the model's guardrail" if one["blocked_by_guardrail"] else "no output video"
            say(f"- {label}: **fail** ({why}). Log kept in `{work}`\n\n```\n{res['tail']}\n```")
    if not failed:
        shutil.rmtree(work, ignore_errors=True)
    return failed


def distilled_spec(spec: Path) -> Path:
    """The same input through the distilled edge model: edge control only, four sampling steps."""
    data = json.loads(spec.read_text())
    if "edge" not in data:
        raise ValueError(f"{spec} has no edge control, which is the only control the distilled model takes")
    keep = {k: data[k] for k in ("name", "prompt", "video_path", "seed", "resolution", "edge")} | {"num_steps": 4}
    out = spec.with_name("selftest_distilled_edge.json")
    out.write_text(json.dumps(keep, indent=1) + "\n")
    return out


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
    ok = one_run(0, run, spec, bdir, cfg, gpu, name=f"selftest_{run_name(run)}_{frames}f", extra=["--max-frames", frames],
                 what=f", first {frames} frames, full model")
    # The distilled edge model on the same frames: 4 steps instead of the full model's, edge control
    # only, and one chunk at most. It is registered only with experimental checkpoints switched on.
    # Its result is reported; it does not decide whether the self-test passed.
    try:
        one_run(0, run, distilled_spec(spec), bdir, cfg, gpu, name=f"selftest_{run_name(run)}_distilled_{frames}f",
                extra=["--max-frames", frames, "--model", "edge/distilled"], env={"COSMOS_EXPERIMENTAL_CHECKPOINTS": "1"},
                what=f", first {frames} frames, distilled edge model")
    except Exception as e:
        say(f"- self-test, distilled edge model: **fail** (driver error: {type(e).__name__}: {e})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--then", action="append", default=[], help="further run folders, run after the first")
    ap.add_argument("--skip-self-test", action="store_true")
    ap.add_argument("--self-test-only", action="store_true", help="node check and self-test, then stop")
    ap.add_argument("--max-frames", type=int, default=None, help="generate only the first N frames of every run")
    ap.add_argument("--batch", action="store_true", help="one process, and one model load, per set of controls")
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
        suffix = f"_{a.max_frames}f" if a.max_frames else ""
        extra = ["--max-frames", str(a.max_frames)] if a.max_frames else []
        ready = []
        for number, run, spec, bdir in sorted(todo, key=lambda x: x[0]):      # the run list's own order
            if not spec.exists():
                say(f"- run {number}: **fail** (spec file missing: `{spec}`)")
                failed += 1
            else:
                ready.append((number, run, spec, bdir))
        if a.batch:
            groups: dict = {}
            for item in ready:
                groups.setdefault((item[3], control_set(item[2])), []).append(item)
            for group in groups.values():
                failed += batch_runs(group, cfg, gpu, suffix, extra)
        else:
            for number, run, spec, bdir in ready:
                failed += not one_run(number, run, spec, bdir, cfg, gpu, name=run_name(run) + suffix if suffix else None,
                                      extra=extra, what=suffix)
    say(f"\nFinished {time.strftime('%Y-%m-%d %H:%M:%S')}: {failed} run(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
