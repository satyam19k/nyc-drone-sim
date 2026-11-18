"""One command from an address, or a route file, to drone videos. Everything runs on this machine.

    nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30
    nyc-drone route configs/routes/hudson-park-run.yaml
    nyc-drone reproduce canal-shore-street

`run` and `route` do everything; `reproduce` flies a route from configs/routes/ and compares its
camera poses with the saved ones; `scene`, `render`, `generate` and `check` do one part each and
take the same options. The command only resolves options into a config file and calls the
existing modules in order; how scenes, paths and renders are computed is not changed here.

Everything a run produces is under --out:

    scene.usd  flight.json  report.json  side_by_side.mp4  config.yaml
    <altitude>/rgb.mp4 depth.mp4 segmentation.mp4 edges.mp4 camera_poses.json
    <altitude>/weather/<weather>_<setting>_window_<start>.mp4             with --realism model
    <altitude>/weather/<weather>_<setting>_chain_<start>_<count>.mp4      with --method chain
        each beside its .spec.json (the model's input), .prompt.txt and .clip.json
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
from . import routes as routes_mod

COMMANDS = ("run", "route", "reproduce", "scene", "render", "generate", "check")
WEATHERS, NOT_YET = ("clear", "snow"), ("fog", "rain")
MODEL_SETTINGS = ("K2", "M2", "t4")
POSE_TOLERANCE = (0.05, 0.2)    # metres, degrees: the render check's own limits for camera pose
VIDEOS = ("rgb", "depth", "segmentation", "edges")
# Measured on this project's hardware; used only for the --dry-run estimate.
RENDER_S_PER_FRAME = 1.25       # L40S, 1280x720, populated 800 m scene (measured 1.16 to 1.31)
RENDER_START_S = 70             # simulator start, scene load, warm-up and video writing, per altitude
SCENE_BUILD_MIN = (15, 60)      # CPU, 800 m scene; depends on the node's load


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nyc-drone", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=COMMANDS,
                    help="run: scene, render, generate (with --realism model) and check, for --address; "
                         "route FILE: the same for a route file; reproduce NAME: route configs/routes/NAME.yaml "
                         "and compare the camera poses with NAME.expected_poses.json")
    ap.add_argument("target", nargs="?", help="the route file (route) or route name (reproduce)")

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

    g = ap.add_argument_group("look")
    g.add_argument("--look", choices=["standard", "enhanced"],
                   help="enhanced = sky light, rippled water, ground textures, wall and roof variation "
                        "(default standard, or the route file's look)")

    g = ap.add_argument_group("video model")
    g.add_argument("--realism", choices=["none", "model"], default="none",
                   help="model = restyle the render with the video model (needs an 80 GB GPU)")
    g.add_argument("--weather", nargs="+", default=["clear"],
                   help="clear or snow, one or more (fog and rain are not available yet)")
    g.add_argument("--method", choices=["window", "chain"], default="window",
                   help="window: one 93-frame window from --start (default); chain: --frames in one model run, "
                        "each 93-frame chunk continuing from the last frame of the one before")
    g.add_argument("--start", type=int, default=0, help="with --method window: first frame of the 93-frame window")
    g.add_argument("--frames", help="with --method chain: START:COUNT, at least 93 frames (default the whole clip)")
    g.add_argument("--model-setting", choices=MODEL_SETTINGS,
                   help="K2: depth 0.5, rendered edges 0.3, segmentation 0.5; M2: K2 plus the render's colours "
                        "(blur control 0.3, preset high), for renders made with --look enhanced; t4: depth 1.0, "
                        "geometry edges 1.0, no segmentation (default: K2 for clear, t4 for snow)")

    g = ap.add_argument_group("route and reproduce")
    g.add_argument("--poses-only", action="store_true",
                   help="fetch the buildings and plan the flight, compare poses (reproduce) and stop: CPU only")

    g = ap.add_argument_group("run")
    g.add_argument("--out", help="output folder (default runs/<location>)")
    g.add_argument("--config", help="YAML file whose values replace the defaults")
    g.add_argument("--dry-run", action="store_true", help="print what would run and stop")
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


def apply_camera(a, cfg: dict) -> None:
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
    if a.objects:
        cfg["populate"]["enabled"] = a.objects == "on"


def resolve(a: argparse.Namespace) -> tuple[dict, str, str | None]:
    """(config, location, path variant) from the defaults, --config, a route file and the options."""
    for w in a.weather:
        if w in NOT_YET:
            raise SystemExit(f"weather {w} is not available yet")
        if w not in WEATHERS:
            raise SystemExit(f"--weather {w}: choose from {', '.join(WEATHERS)}")
    model = a.realism == "model" or a.command == "generate"
    if not model and (a.weather != ["clear"] or a.method != "window" or a.frames or a.model_setting):
        raise SystemExit("--weather, --method, --frames and --model-setting are for the video model: add --realism model")
    if a.frames and a.method != "chain":
        raise SystemExit("--frames is for --method chain; --method window takes --start")
    cfg = load_config()
    if a.config:
        cfg = merge(cfg, yaml.safe_load(Path(a.config).read_text()) or {})
    cfg = copy.deepcopy(cfg)
    has_location = a.address or a.lat is not None or a.lon is not None
    if a.command in ("route", "reproduce") and not a.target:
        raise SystemExit(f"{a.command} needs a route file or the name of one in configs/routes/")
    if a.command == "run" and a.target:
        raise SystemExit("run takes --address, or --lat and --lon; fly a route file with: nyc-drone route FILE")
    if a.target and has_location:
        raise SystemExit("give a route or a location, not both")
    if a.command == "reproduce" and (a.method != "window" or a.frames):
        raise SystemExit("reproduce compares camera poses; --method and --frames are for generate and run")

    if a.target:                                                   # a route file
        fixed = [o for o in ("altitude_ft", "duration_s", "speed_mps", "radius_m", "seed") if getattr(a, o) not in (None, [])]
        if fixed:
            raise SystemExit("set " + ", ".join("--" + o.replace("_", "-") for o in fixed) + " in the route file instead")
        path = routes_mod.route_path(a.target)
        if not path.exists():
            raise SystemExit(f"no route file {path}")
        name, route, place = routes_mod.resolve_route(path, cfg, a.offline)
        cfg.setdefault("places", {})[name] = place
        cfg["altitudes_ft"] = [int(x) for x in route.get("altitudes_ft", cfg["altitudes_ft"])]
        cfg["seed"] = int(route.get("seed", cfg["seed"]))
        cfg["long"]["frames"] = int(route["frames"])
        if a.command == "reproduce" and any(getattr(a, o) is not None for o in ("pitch_deg", "focal_mm", "fov_deg", "resolution", "fps")):
            raise SystemExit("reproduce uses the camera the poses were saved with: leave the camera options out")
        apply_camera(a, cfg)
        location, variant, look, path_kind = name, "long", a.look or route.get("look", "standard"), "route"
    else:
        if a.address and (a.lat is not None or a.lon is not None):
            raise SystemExit("give --address or --lat and --lon, not both")
        if a.address:
            location = a.address
        elif a.lat is not None and a.lon is not None:
            location = f"{a.lat:.6f},{a.lon:.6f}"
        else:
            raise SystemExit("give --address, or --lat and --lon, or a route file")
        if a.radius_m:
            cfg["scene_radius_m"] = float(a.radius_m)
        if a.altitude_ft:
            cfg["altitudes_ft"] = list(a.altitude_ft)
        if a.seed is not None:
            cfg["seed"] = a.seed
        apply_camera(a, cfg)
        variant = "street" if a.path == "street" else None
        section = cfg["street"] if variant else cfg["clip"]
        if a.duration_s:
            section["frames"] = int(round(a.duration_s * float(cfg["clip"]["fps"])))
        cfg["speed_mps"] = {int(alt): float(a.speed_mps) if a.speed_mps else round(speed_for(cfg, int(alt)), 2)
                            for alt in cfg["altitudes_ft"]}
        # recorded traffic and pedestrian motion must last as long as the clip
        cfg["populate"]["frames"] = max(int(section["frames"]), 1)
        look, path_kind = a.look or "standard", a.path

    out = Path(a.out) if a.out else REPO_ROOT / "runs" / slugify(location)
    out = out.resolve()
    cfg["output"] = {"scenes_dir": str(out / "work" / "scenes"), "runs_dir": str(out / "work")}
    cfg["_run"] = {"out": str(out), "location": location, "path": path_kind, "look": look}
    return cfg, location, variant


def frames_of(cfg: dict, variant: str | None) -> int:
    return int((cfg[variant] if variant else cfg["clip"])["frames"])


def geocode(cfg: dict, location: str, offline: bool) -> dict:
    from . import geo
    http = geo.HttpCache(REPO_ROOT / cfg["data"]["cache_dir"], offline=offline, timeout_s=float(cfg["data"]["http_timeout_s"]))
    return geo.geocode(location, cfg, http)


def dry_run(a, cfg: dict, location: str, variant: str | None) -> None:
    frames, fps = frames_of(cfg, variant), float(cfg["clip"]["fps"])
    place = (cfg.get("places") or {}).get(location)
    if place is not None:
        where = f"route {location}  origin lat {place['origin'][0]:.6f}, lon {place['origin'][1]:.6f}"
    else:
        try:
            loc = geocode(cfg, location, a.offline)
            where = f"{loc.get('label', location)}  lat {loc['lat']:.6f}, lon {loc['lon']:.6f}"
        except Exception as e:                               # no network, or no match
            where = f"{location}  (not resolved: {e})"
    alts = cfg["altitudes_ft"]
    render_min = len(alts) * (RENDER_START_S + frames * RENDER_S_PER_FRAME) / 60
    print(f"location     {where}")
    extent = (f"corridor of {place['buffer_m']} m around {len(place['route_m'])} waypoints" if place is not None
              else f"circle of radius {cfg['scene_radius_m']:.0f} m")
    print(f"scene        {extent}, objects {'on' if cfg['populate']['enabled'] else 'off'}, seed {cfg['seed']}, look {cfg['_run']['look']}")
    speeds = ([place["speed_mps"]] * len(alts) if place is not None and place.get("sync_ground_track")
              else [cfg["speed_mps"][int(x)] for x in alts])
    print(f"flight       {cfg['_run']['path']} path, altitudes {alts} ft, speeds {speeds} m/s")
    print(f"clip         {frames} frames at {fps:g} fps = {frames / fps:.1f} s, "
          f"{cfg['camera']['resolution'][0]}x{cfg['camera']['resolution'][1]}, "
          f"pitch {cfg['camera']['pitch_deg']} deg, focal {cfg['camera']['focal_length_mm']:.1f} mm")
    print(f"output       {cfg['_run']['out']}")
    print("steps")
    print(f"  scene      CPU only           about {SCENE_BUILD_MIN[0]} to {SCENE_BUILD_MIN[1]} min")
    print(f"  render     RTX GPU (L40S)     about {render_min:.0f} min for {len(alts)} altitude(s)")
    if a.realism == "model":
        what = (f"one 93-frame window from frame {a.start}" if a.method == "window"
                else f"frames {a.frames} chained" if a.frames else "the whole clip chained")
        print(f"  generate   80 GB GPU          video model, {' and '.join(a.weather)}, {what}, per altitude")
    print("  check      CPU only           a few minutes")
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


def step_scene(a, cfg, location, variant, cfg_path: Path, out: Path, timings: dict, poses_only: bool = False) -> None:
    """Data, scene and flight paths. poses_only: the buildings and the flight paths, nothing else."""
    log = out / "work" / "scene.log"
    built = out / "work" / "scene_built.yaml"
    if built.exists() and built.read_text() == cfg_path.read_text() and (out / "scene.usd").exists() and (out / "flight.json").exists():
        print("scene: already built with these options, reused", flush=True)
        return
    names = ["geo"] if poses_only else ["geo", "layers"] + (["populate"] if cfg["populate"]["enabled"] else []) + ["scene"]
    for name in names:
        extra = ("--offline",) if a.offline and name in ("geo", "layers") else ()
        timings[name] = call(module(sys.executable, name, cfg_path, location, *extra), log)
    timings["paths"] = call(module(sys.executable, "paths", cfg_path, location, *(("--variant", variant) if variant else ())), log)
    sdir = Path(cfg["output"]["scenes_dir"]) / slugify(location)
    suffix = f"_{variant}" if variant else ""
    flight = {"location": location, "path": cfg["_run"]["path"],
              "altitudes": {str(alt): json.loads((sdir / f"path_{int(alt)}{suffix}.json").read_text()) for alt in cfg["altitudes_ft"]}}
    (out / "flight.json").write_text(json.dumps(flight, indent=1) + "\n")
    if not poses_only:
        shutil.copy(sdir / "scene.usdc", out / "scene.usd")
        shutil.copy(cfg_path, built)


def compare_with_expected(location: str, out: Path) -> dict:
    """reproduce: the planned camera poses against NAME.expected_poses.json."""
    flight = json.loads((out / "flight.json").read_text())
    res = routes_mod.compare_poses(routes_mod.poses_from_flight(flight), routes_mod.load_expected(location))
    worst_m = max((v.get("max_position_m", float("inf")) for v in res.values()), default=float("inf"))
    worst_deg = max((v.get("max_rotation_deg", float("inf")) for v in res.values()), default=float("inf"))
    for alt, v in res.items():
        if "error" in v:
            print(f"poses {alt} ft: {v['error']}", flush=True)
        else:
            print(f"poses {alt} ft: {v['frames']} frames, largest difference {v['max_position_m']:.6f} m "
                  f"(frame {v['frame_of_max_position']}) and {v['max_rotation_deg']:.6f} deg (frame {v['frame_of_max_rotation']})", flush=True)
    passed = worst_m <= POSE_TOLERANCE[0] and worst_deg <= POSE_TOLERANCE[1]
    print(f"largest difference over all frames: {worst_m:.6f} m, {worst_deg:.6f} deg -> "
          f"{'matches' if passed else 'DOES NOT MATCH'} (limits {POSE_TOLERANCE[0]} m, {POSE_TOLERANCE[1]} deg)", flush=True)
    return {"per_altitude": res, "largest_position_m": worst_m, "largest_rotation_deg": worst_deg,
            "limits": {"position_m": POSE_TOLERANCE[0], "rotation_deg": POSE_TOLERANCE[1]}, "matches": passed}


INSTALL = "see the Install section of README.md"


def check_simulator(a) -> None:
    """Stop at once, before any work, when the render step could not run here."""
    if not shutil.which("nvidia-smi"):
        raise SystemExit("the render step needs an NVIDIA RTX GPU and its driver, and this machine has none; " + INSTALL)
    if not Path(a.sim_python).exists():
        raise SystemExit(f"Isaac Sim 6.2 is not installed: no Python at {a.sim_python}. Install it with\n"
                         "  uv venv env_sim --python 3.12\n"
                         '  uv pip install --python env_sim/bin/python "isaacsim[all,extscache]==6.2.0" --extra-index-url https://pypi.nvidia.com\n'
                         "or point --sim-python / NYC_DRONE_SIM_PYTHON at an existing one; " + INSTALL)
    probe = subprocess.run([a.sim_python, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('isaacsim') else 1)"])
    if probe.returncode != 0:
        raise SystemExit(f"{a.sim_python} has no isaacsim package: install Isaac Sim 6.2 into it; " + INSTALL)
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in ("YES", "Y", "1", "TRUE"):
        raise SystemExit("Isaac Sim needs its licence accepted: read it, then set OMNI_KIT_ACCEPT_EULA=YES")


def check_model() -> None:
    """Stop at once when the video model's checkout or environment is missing."""
    from .gen.common import load_gen_config
    from .gen.model import model_python, model_repo
    gcfg = load_gen_config()
    try:
        repo = model_repo(gcfg)
    except RuntimeError as e:
        raise SystemExit(f"{e}. Clone Cosmos-Transfer2.5 at commit {gcfg['model']['commit'][:12]} and create its "
                         "environment; " + INSTALL)
    py = model_python(gcfg)
    probe = subprocess.run([py, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('cosmos_transfer2') "
                                      "and importlib.util.find_spec('cv2') else 1)"], cwd=repo)
    if probe.returncode != 0:
        raise SystemExit(f"the video model's environment is missing or incomplete ({py} cannot import cosmos_transfer2 "
                         "and cv2): create it with uv sync in the model checkout, then set MODEL_PYTHON; " + INSTALL)


def fetch_models_if_missing(cfg: dict) -> None:
    """Vehicle, tree and pedestrian models: downloaded and converted once, the first time a scene needs them."""
    if not cfg["populate"]["enabled"] or (REPO_ROOT / cfg["assets"]["library_dir"] / "manifest.json").exists():
        return
    print("models: vehicle, tree and pedestrian models are missing; downloading them (about 0.6 GB, once)", flush=True)
    env = {**os.environ, "SKIP_CONVERT": "1"}
    if subprocess.run(["bash", str(REPO_ROOT / "scripts" / "fetch_assets.sh")], cwd=REPO_ROOT, env=env).returncode != 0:
        raise SystemExit("downloading the models failed; run scripts/fetch_assets.sh by hand to see why")
    if subprocess.run([sys.executable, "-m", "nyc_drone_sim.assets"], cwd=REPO_ROOT).returncode != 0:
        raise SystemExit("converting the models to USD failed; run python -m nyc_drone_sim.assets to see why")


def step_render(a, cfg, location, variant, cfg_path: Path, out: Path, timings: dict) -> None:
    check_simulator(a)
    log = out / "work" / "render.log"
    v = ("--variant", variant) if variant else ()
    slug = slugify(location)
    look = ("--look", cfg["_run"]["look"])
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
    """The video model on the rendered clips, for each --weather. By default one 93-frame window from
    --start; --method chain restyles --frames (default the whole clip) in one model run. Runs in the
    model's Python. Each clip is copied to <altitude>/weather/ beside its spec, prompt and clip.json."""
    from .gen.common import load_gen_config, model_dir
    from .gen.makeclip import frame_range, look_warning, prompts
    from .gen.model import model_python
    check_model()
    gcfg = load_gen_config()
    settings = prompts()["settings"]
    py, log = model_python(gcfg), out / "work" / "generate.log"
    run_folder = out / "work" / "render" / slugify(location)
    for alt in cfg["altitudes_ft"]:
        bdir = run_folder / str(alt)
        if not (bdir / "bundle.json").exists():
            raise SystemExit(f"no render at {bdir}: run the render step first")
        if a.method == "chain":
            start, count = frame_range(a.frames, int(json.loads((bdir / "bundle.json").read_text())["clip"]["frames"]))
            tag = f"chain_{start:04d}_{count:04d}"
        else:
            tag = f"window_{a.start:04d}"
        if not (bdir / "gen" / "inputs" / "inputs.json").exists():
            timings[f"controls_{alt}"] = call([py, "-m", "nyc_drone_sim.gen.controls", "--run-folder", str(run_folder),
                                               "--altitude", str(alt)], log)
        for weather in a.weather:
            setting = a.model_setting or gcfg["generate"]["default_setting"][weather]
            warn = look_warning(bdir, setting, settings[setting])
            if warn:
                print(f"{alt} ft: {warn}", flush=True)
            cmd = [py, "-m", "nyc_drone_sim.gen.makeclip", "--run-folder", str(run_folder), "--altitude", str(alt),
                   "--weather", weather, "--setting", setting, "--method", a.method]
            cmd += ["--start", str(a.start)] if a.method == "window" else ["--frames", f"{start}:{count}"]
            timings[f"generate_{alt}_{weather}"] = call(cmd, log)
            made = sorted((d for d in model_dir(bdir, gcfg).glob(f"{weather}_{setting}_{tag}*") if ".tmp_" not in d.name),
                          key=lambda d: d.stat().st_mtime)
            if not made or not (made[-1] / "out.mp4").exists():
                raise SystemExit(f"the video model left no out.mp4 at {alt} ft, {weather}; see {log}")
            export_clip(made[-1], out / str(alt) / "weather", f"{weather}_{setting}_{tag}")


def export_clip(made: Path, dest: Path, stem: str) -> None:
    """The generated clip and what it was made from: <stem>.mp4, .spec.json, .prompt.txt, .clip.json."""
    dest.mkdir(parents=True, exist_ok=True)
    for src, suffix in (("out.mp4", ".mp4"), ("spec.json", ".spec.json"), ("prompt.txt", ".prompt.txt"), ("clip.json", ".clip.json")):
        if (made / src).exists():
            shutil.copy(made / src, dest / f"{stem}{suffix}")


def step_check(a, cfg, location, variant, cfg_path: Path, out: Path, timings: dict) -> dict:
    """Run the project's checks on this run. A failing check does not stop the other checks; the
    command then exits 1 (see exit_code)."""
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


def write_report(cfg, location, out: Path, timings: dict, checks: dict | None, poses: dict | None = None) -> None:
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
        "reproduced_poses": poses if poses is not None else prev.get("reproduced_poses"),
        "timings_s": {**prev.get("timings_s", {}), **{k: round(v, 1) for k, v in timings.items()}},
    }
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    cfg, location, variant = resolve(a)
    if a.dry_run:
        dry_run(a, cfg, location, variant)
        return
    cmd = a.command
    full = cmd in ("run", "route", "reproduce") and not a.poses_only
    # stop at once, before any work, when a later step could not run on this machine
    if full or cmd == "render":
        check_simulator(a)
    if (full and a.realism == "model") or cmd == "generate":
        check_model()
    if full or cmd == "scene":
        fetch_models_if_missing(cfg)
    out = Path(cfg["_run"]["out"])
    (out / "work").mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.environ["NYC_DRONE_VALIDATION_DIR"] = str(out / "validation")     # check reports and pictures of this run
    timings: dict = {}
    checks, poses = None, None
    try:
        if full or cmd == "scene" or (a.poses_only and cmd in ("route", "reproduce")):
            step_scene(a, cfg, location, variant, cfg_path, out, timings, poses_only=a.poses_only)
        if cmd == "reproduce":
            poses = compare_with_expected(location, out)
        if full or cmd == "render":
            step_render(a, cfg, location, variant, cfg_path, out, timings)
        if (full and a.realism == "model") or cmd == "generate":
            step_generate(a, cfg, location, out, timings)
        if full or cmd == "check":
            checks = step_check(a, cfg, location, variant, cfg_path, out, timings)
    finally:
        write_report(cfg, location, out, timings, checks, poses)
    print(f"done: {out}")
    sys.exit(exit_code(checks, poses))


def exit_code(checks: dict | None, poses: dict | None) -> int:
    """1 when a check failed or the poses do not match the saved ones; warnings do not count."""
    failed = [name for name, result in (checks or {}).items() if result != "passed"]
    if failed:
        print(f"checks failed: {', '.join(failed)}", flush=True)
    return 1 if failed or (poses is not None and not poses["matches"]) else 0


if __name__ == "__main__":
    main()
