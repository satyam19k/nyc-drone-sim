"""One command from an address to drone videos.

    nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30

`run` does everything; `scene`, `render`, `generate` and `check` do one part each and take the
same options. The command only resolves options into a config file and calls the existing
modules in order; how scenes, paths and renders are computed is not changed here.

Everything a run produces is under --out:

    scene.usd  flight.json  report.json  side_by_side.mp4  config.yaml
    <altitude>/rgb.mp4 depth.mp4 segmentation.mp4 edges.mp4 camera_poses.json
    <altitude>/weather/<weather>.mp4     with --realism model
    work/      the intermediate files of every step
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .common import REPO_ROOT, load_config, slugify

STEPS = ("run", "scene", "render", "generate", "check")
VIDEOS = ("rgb", "depth", "segmentation", "edges")
# Measured on this project's hardware; used only for the --dry-run estimate.
RENDER_S_PER_FRAME = 1.25       # L40S, 1280x720, populated 800 m scene (measured 1.16 to 1.31)
RENDER_START_S = 70             # simulator start, scene load, warm-up and video writing, per altitude
SCENE_BUILD_MIN = (15, 60)      # CPU, 800 m scene; depends on the node's load


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nyc-drone", description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS, help="run = scene, render, generate (with --realism model) and check, in order")

    g = ap.add_argument_group("location")
    g.add_argument("--address", help="street address, intersection or place name in New York City")
    g.add_argument("--lat", type=float)
    g.add_argument("--lon", type=float)
    g.add_argument("--radius-m", type=float, help="scene radius around the location")

    g = ap.add_argument_group("flight")
    g.add_argument("--altitude-ft", type=int, nargs="+", help="one or more altitudes above ground")
    g.add_argument("--duration-s", type=float, help="clip length")
    g.add_argument("--speed-mps", type=float, help="ground speed for every altitude")
    g.add_argument("--path", choices=["street", "auto"], default="street",
                   help="street = follow street centre lines (any length); "
                        "auto = short flyover, orbit or hover-and-pan around the location, whichever fits")

    g = ap.add_argument_group("camera")
    g.add_argument("--pitch-deg", type=float, help="negative looks down")
    g.add_argument("--focal-mm", type=float)
    g.add_argument("--fov-deg", type=float, help="horizontal field of view, instead of --focal-mm")
    g.add_argument("--resolution", help="WIDTHxHEIGHT")
    g.add_argument("--fps", type=float)

    g = ap.add_argument_group("scene")
    g.add_argument("--seed", type=int)
    g.add_argument("--objects", choices=["on", "off"], help="vehicles, pedestrians, trees and street furniture")

    g = ap.add_argument_group("look and weather in the renderer")
    g.add_argument("--look", choices=["standard", "enhanced"], default="standard",
                   help="enhanced = sky light, rippled water, ground textures, wall and roof variation")
    g.add_argument("--sim-weather", choices=["clear", "fog_light", "fog_heavy", "wet"], default="clear",
                   help="weather drawn by the renderer itself")

    g = ap.add_argument_group("weather and realism from the video model")
    g.add_argument("--weather", nargs="+", choices=["clear", "fog", "rain", "snow"], default=["clear"],
                   help="one video per weather, made by the video model (needs --realism model)")
    g.add_argument("--model-setting", help="named setting from configs/gen_prompts_windows.yaml "
                                           "(default: the one configs/gen.yaml names for each weather)")
    g.add_argument("--realism", choices=["none", "model"], default="none",
                   help="model = restyle the render with the video model (needs an 80 GB GPU)")

    g = ap.add_argument_group("run")
    g.add_argument("--out", help="output folder (default runs/<location>)")
    g.add_argument("--config", help="YAML file whose values replace the defaults")
    g.add_argument("--dry-run", action="store_true", help="print what would run and stop")
    g.add_argument("--slurm", action="store_true", help="submit this command as a Slurm job")
    g.add_argument("--slurm-account")
    g.add_argument("--slurm-partition")
    g.add_argument("--slurm-time", default="02:00:00")
    g.add_argument("--sim-python", default=os.environ.get("NYC_DRONE_SIM_PYTHON", str(REPO_ROOT / "env_sim/bin/python")),
                   help="Python of the environment that holds Isaac Sim")
    g.add_argument("--offline", action="store_true", help="use only cached city data")
    return ap


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def speed_for(cfg: dict, alt_ft: int) -> float:
    """Configured speed, or a straight-line value between the configured altitudes."""
    table = sorted((int(k), float(v)) for k, v in cfg["speed_mps"].items())
    for a, v in table:
        if a == alt_ft:
            return v
    (a0, v0), (a1, v1) = table[0], table[-1]
    if a0 == a1:
        return v0
    return min(max(v0 + (v1 - v0) * (alt_ft - a0) / (a1 - a0), min(v0, v1)), max(v0, v1))


def resolve(a: argparse.Namespace) -> tuple[dict, str, str | None]:
    """(config, location text, path variant) from the defaults, --config and the options."""
    cfg = load_config()
    if a.config:
        cfg = merge(cfg, yaml.safe_load(Path(a.config).read_text()) or {})
    cfg = copy.deepcopy(cfg)
    if a.address and (a.lat is not None or a.lon is not None):
        raise SystemExit("give --address or --lat and --lon, not both")
    if a.address:
        location = a.address
    elif a.lat is not None and a.lon is not None:
        location = f"{a.lat:.6f},{a.lon:.6f}"
    else:
        raise SystemExit("give --address, or --lat and --lon")

    if a.radius_m:
        cfg["scene_radius_m"] = float(a.radius_m)
    if a.altitude_ft:
        cfg["altitudes_ft"] = list(a.altitude_ft)
    if a.seed is not None:
        cfg["seed"] = a.seed
    cam = cfg["camera"]
    if a.pitch_deg is not None:
        cam["pitch_deg"] = a.pitch_deg
    if a.focal_mm and a.fov_deg:
        raise SystemExit("give --focal-mm or --fov-deg, not both")
    if a.focal_mm:
        cam["focal_length_mm"] = a.focal_mm
    if a.fov_deg:
        cam["focal_length_mm"] = float(cam["horizontal_aperture_mm"]) / (2 * math.tan(math.radians(a.fov_deg) / 2))
    if a.resolution:
        w, h = (int(v) for v in a.resolution.lower().split("x"))
        cam["resolution"] = [w, h]
        cam["vertical_aperture_mm"] = float(cam["horizontal_aperture_mm"]) * h / w
    if a.fps:
        cfg["clip"]["fps"] = a.fps
    variant = "street" if a.path == "street" else None
    section = cfg["street"] if variant else cfg["clip"]
    if a.duration_s:
        section["frames"] = int(round(a.duration_s * float(cfg["clip"]["fps"])))
    cfg["speed_mps"] = {int(alt): float(a.speed_mps) if a.speed_mps else round(speed_for(cfg, int(alt)), 2)
                        for alt in cfg["altitudes_ft"]}
    if a.objects:
        cfg["populate"]["enabled"] = a.objects == "on"
    # recorded traffic and pedestrian motion must last as long as the clip
    cfg["populate"]["frames"] = max(int(section["frames"]), 1)

    out = Path(a.out) if a.out else REPO_ROOT / "runs" / slugify(location)
    out = out.resolve()
    cfg["output"] = {"scenes_dir": str(out / "work" / "scenes"), "runs_dir": str(out / "work")}
    cfg["_run"] = {"out": str(out), "location": location, "path": a.path}
    return cfg, location, variant


def frames_of(cfg: dict, variant: str | None) -> int:
    return int((cfg["street"] if variant else cfg["clip"])["frames"])


def geocode(cfg: dict, location: str, offline: bool) -> dict:
    from . import geo
    http = geo.HttpCache(REPO_ROOT / cfg["data"]["cache_dir"], offline=offline, timeout_s=float(cfg["data"]["http_timeout_s"]))
    return geo.geocode(location, cfg, http)


def dry_run(a, cfg: dict, location: str, variant: str | None) -> None:
    frames, fps = frames_of(cfg, variant), float(cfg["clip"]["fps"])
    try:
        loc = geocode(cfg, location, a.offline)
        where = f"{loc.get('label', location)}  lat {loc['lat']:.6f}, lon {loc['lon']:.6f}"
    except Exception as e:                                   # no network on this node, or no match
        where = f"{location}  (not resolved: {e})"
    alts = cfg["altitudes_ft"]
    render_min = len(alts) * (RENDER_START_S + frames * RENDER_S_PER_FRAME) / 60
    print(f"location     {where}")
    print(f"scene        circle of radius {cfg['scene_radius_m']:.0f} m, objects {'on' if cfg['populate']['enabled'] else 'off'}, seed {cfg['seed']}")
    print(f"flight       {a.path} path, altitudes {alts} ft, speeds {[cfg['speed_mps'][int(x)] for x in alts]} m/s")
    print(f"clip         {frames} frames at {fps:g} fps = {frames / fps:.1f} s, "
          f"{cfg['camera']['resolution'][0]}x{cfg['camera']['resolution'][1]}, "
          f"pitch {cfg['camera']['pitch_deg']} deg, focal {cfg['camera']['focal_length_mm']:.1f} mm")
    print(f"output       {cfg['_run']['out']}")
    print("steps")
    print(f"  scene      CPU only           about {SCENE_BUILD_MIN[0]} to {SCENE_BUILD_MIN[1]} min")
    print(f"  render     RTX GPU (L40S)     about {render_min:.0f} min for {len(alts)} altitude(s)")
    print("  check      CPU only           a few minutes")
    if a.realism == "model":
        print(f"  generate   80 GB GPU          video model, one video per weather and altitude: {', '.join(a.weather)}")
    print("nothing was run")


def call(cmd: list[str], log: Path, env: dict | None = None) -> float:
    t0 = time.time()
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    with open(log, "a") as f:
        f.write(f"\n### {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        tail = "".join(open(log, errors="replace").readlines()[-15:])
        raise SystemExit(f"step failed (exit {rc}); last lines of {log}:\n{tail}")
    return time.time() - t0


def module(py: str, name: str, cfg_path: Path, location: str, *extra: str) -> list[str]:
    return [py, "-m", f"nyc_drone_sim.{name}", "--config", str(cfg_path), "--location", location, *extra]


def step_scene(a, cfg, location, variant, cfg_path: Path, out: Path, timings: dict) -> None:
    log = out / "work" / "scene.log"
    built = out / "work" / "scene_built.yaml"
    if built.exists() and built.read_text() == cfg_path.read_text() and (out / "scene.usd").exists() and (out / "flight.json").exists():
        print("scene: already built with these options, reused", flush=True)
        return
    names = ["geo", "layers"] + (["populate"] if cfg["populate"]["enabled"] else []) + ["scene"]
    for name in names:
        extra = ("--offline",) if a.offline and name in ("geo", "layers") else ()
        timings[name] = call(module(sys.executable, name, cfg_path, location, *extra), log)
    timings["paths"] = call(module(sys.executable, "paths", cfg_path, location, *(("--variant", variant) if variant else ())), log)
    sdir = Path(cfg["output"]["scenes_dir"]) / slugify(location)
    shutil.copy(sdir / "scene.usdc", out / "scene.usd")
    suffix = f"_{variant}" if variant else ""
    flight = {"location": location, "path": a.path,
              "altitudes": {str(alt): json.loads((sdir / f"path_{int(alt)}{suffix}.json").read_text()) for alt in cfg["altitudes_ft"]}}
    (out / "flight.json").write_text(json.dumps(flight, indent=1) + "\n")
    shutil.copy(cfg_path, built)


def step_render(a, cfg, location, variant, cfg_path: Path, out: Path, timings: dict) -> None:
    if not Path(a.sim_python).exists():
        raise SystemExit(f"no Isaac Sim Python at {a.sim_python}; set --sim-python or NYC_DRONE_SIM_PYTHON")
    if not shutil.which("nvidia-smi"):
        raise SystemExit("the render step needs an RTX GPU and this machine has none; add --slurm to submit it as a job")
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("YES", "Y", "1", "TRUE"):
        raise SystemExit("Isaac Sim needs its licence accepted: read it, then set OMNI_KIT_ACCEPT_EULA=YES")
    log = out / "work" / "render.log"
    v = ("--variant", variant) if variant else ()
    slug = slugify(location)
    look = ("--look", a.look, "--sim-weather", a.sim_weather)
    for alt in cfg["altitudes_ft"]:
        timings[f"render_{alt}"] = call(module(a.sim_python, "sim.render", cfg_path, location,
                                               "--altitude", str(alt), "--run", "render", *v, *look), log)
    timings["postrender"] = call(module(sys.executable, "postrender", cfg_path, location, "--run", "render", *v), log)
    for alt in cfg["altitudes_ft"]:
        bdir = out / "work" / "render" / slug / str(alt)
        dst = out / str(alt)
        dst.mkdir(exist_ok=True)
        for name in VIDEOS:
            shutil.copy(bdir / "clip_0000" / f"{name}.mp4", dst / f"{name}.mp4")
        poses = [json.loads(p.read_text()) for p in sorted((bdir / "camera").glob("camera_*.json"))]
        (dst / "camera_poses.json").write_text(json.dumps(poses, indent=1) + "\n")
    side_by_side(out, cfg["altitudes_ft"])


def side_by_side(out: Path, alts: list) -> None:
    """All altitudes next to each other in one video; a copy of the only one if there is one."""
    from .postrender import ff_tools
    clips = [str(out / str(alt) / "rgb.mp4") for alt in alts]
    if len(clips) == 1:
        shutil.copy(clips[0], out / "side_by_side.mp4")
        return
    cmd = [ff_tools()[0], "-y", "-v", "error"]
    for c in clips:
        cmd += ["-i", c]
    cmd += ["-filter_complex", f"hstack=inputs={len(clips)}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out / "side_by_side.mp4")]
    subprocess.run(cmd, check=True)


def step_generate(a, cfg, location, out: Path, timings: dict) -> None:
    """The video model on the rendered clips: control videos, then one clip per weather, made from
    overlapping 93-frame windows (nyc_drone_sim.gen.makeclip). Runs in the model's Python."""
    from .gen.common import load_gen_config, model_dir
    from .gen.model import model_python, model_repo
    gcfg = load_gen_config()
    model_repo(gcfg)                                   # fails early, with a clear message, if missing
    py, log = model_python(gcfg), out / "work" / "generate.log"
    run_folder = out / "work" / "render" / slugify(location)
    method = gcfg["generate"]["method"]
    for alt in cfg["altitudes_ft"]:
        bdir = run_folder / str(alt)
        if not (bdir / "bundle.json").exists():
            raise SystemExit(f"no render at {bdir}: run the render step first")
        if not (bdir / "gen" / "inputs" / "inputs.json").exists():
            timings[f"controls_{alt}"] = call([py, "-m", "nyc_drone_sim.gen.controls", "--run-folder", str(run_folder),
                                               "--altitude", str(alt)], log)
        for w in a.weather:
            setting = a.model_setting or gcfg["generate"]["default_setting"][w]
            timings[f"generate_{alt}_{w}"] = call([py, "-m", "nyc_drone_sim.gen.makeclip", "--run-folder", str(run_folder),
                                                   "--altitude", str(alt), "--weather", w, "--setting", setting,
                                                   "--method", method], log)
            made = sorted((d for d in model_dir(bdir, gcfg).glob(f"{w}_{setting}_{method}*") if ".tmp_" not in d.name),
                          key=lambda d: d.stat().st_mtime)
            if not made or not (made[-1] / "out.mp4").exists():
                raise SystemExit(f"the video model left no out.mp4 for {w} at {alt} ft; see {log}")
            (out / str(alt) / "weather").mkdir(parents=True, exist_ok=True)
            shutil.copy(made[-1] / "out.mp4", out / str(alt) / "weather" / f"{w}.mp4")


def step_check(a, cfg, location, variant, cfg_path: Path, out: Path, timings: dict) -> dict:
    """Run the project's checks on this run. A failing check is reported, it does not stop the run."""
    log = out / "work" / "check.log"
    v = ["--variant", variant] if variant else []
    rendered = (out / "work" / "render" / slugify(location)).exists()
    todo = {"scene": ["checks/scene.py"], "path": ["checks/path.py", *v]}
    if rendered:
        todo["render"] = ["checks/render.py", "--run", "render", *v]
    results = {}
    for name, (script, *extra) in todo.items():
        cmd = [sys.executable, script, "--config", str(cfg_path), "--location", location, *extra]
        t0 = time.time()
        with open(log, "a") as f:
            f.write(f"\n### {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        results[name] = "passed" if rc == 0 else f"failed (exit {rc}), see {log}"
        timings[f"check_{name}"] = time.time() - t0
        print(f"check {name}: {results[name]}", flush=True)
    return results


def write_report(cfg, location, out: Path, timings: dict, checks: dict | None) -> None:
    sdir = Path(cfg["output"]["scenes_dir"]) / slugify(location)

    def read(name):
        p = sdir / name
        return json.loads(p.read_text()) if p.exists() else None

    loc, meta, sources = read("location.json") or {}, read("scene_meta.json") or {}, read("cache_manifest.json") or []
    placed = read("placements.json")
    counts = {}
    for o in (placed or {}).get("objects", []):
        counts[o["class"]] = counts.get(o["class"], 0) + 1
    prev = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    report = {
        "location": {"query": location, **{k: loc.get(k) for k in ("label", "lat", "lon", "borough", "scene_radius_m")}},
        "altitudes_ft": cfg["altitudes_ft"],
        "data_sources": sorted({s["url"].split("?")[0] for s in sources if isinstance(s, dict) and "url" in s}),
        "scene": {k: meta.get(k) for k in meta if k != "origin"},
        "objects": counts,
        "checks": checks if checks is not None else prev.get("checks"),
        "timings_s": {**prev.get("timings_s", {}), **{k: round(v, 1) for k, v in timings.items()}},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def submit(a, out: Path) -> None:
    args = [x for x in sys.argv[1:] if x != "--slurm"]
    script = out / "job.sbatch"
    script.write_text("\n".join([
        "#!/bin/bash",
        "#SBATCH --job-name=nyc-drone",
        *([f"#SBATCH --partition={a.slurm_partition}"] if a.slurm_partition else []),
        "#SBATCH --gres=gpu:1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=48G",
        f"#SBATCH --time={a.slurm_time}",
        f"#SBATCH --output={out}/slurm-%j.out",
        *([f"#SBATCH --account={a.slurm_account}"] if a.slurm_account else []),
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"{shlex.quote(sys.executable)} -m nyc_drone_sim.cli " + " ".join(shlex.quote(x) for x in args), ""]))
    print(subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True).stdout.strip())
    print(f"job script: {script}")


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    cfg, location, variant = resolve(a)
    if a.dry_run:
        dry_run(a, cfg, location, variant)
        return
    if a.weather != ["clear"] and a.realism != "model":
        raise SystemExit("--weather other than clear is made by the video model: add --realism model "
                         "(or use --sim-weather for the renderer's own fog and wet look)")
    out = Path(cfg["_run"]["out"])
    (out / "work").mkdir(parents=True, exist_ok=True)
    if a.slurm:
        submit(a, out)
        return
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.environ["NYC_DRONE_VALIDATION_DIR"] = str(out / "validation")     # check reports and pictures of this run
    timings: dict = {}
    checks = None
    try:
        if a.step in ("run", "scene"):
            step_scene(a, cfg, location, variant, cfg_path, out, timings)
        if a.step in ("run", "render"):
            step_render(a, cfg, location, variant, cfg_path, out, timings)
        if a.step == "generate" or (a.step == "run" and a.realism == "model"):
            step_generate(a, cfg, location, out, timings)
        if a.step in ("run", "check"):
            checks = step_check(a, cfg, location, variant, cfg_path, out, timings)
    finally:
        write_report(cfg, location, out, timings, checks)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
