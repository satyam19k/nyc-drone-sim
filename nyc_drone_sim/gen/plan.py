"""Write one spec file per run of the run list, validate them without a GPU, and print the plan.

  python -m nyc_drone_sim.gen.plan --run-folder runs/run_006/hudson-park-run            write and check
  python -m nyc_drone_sim.gen.plan --run-folder runs/run_006/hudson-park-run --check    check only

Specs go to <bundle>/gen/<model>/specs/NN_<weather>_<edge>.json. Every control is passed as a
file: a control without a path would be computed by the model from the colour video. The blur
("vis") control is never used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import (EDGE_OPTIONS, bundle_dir, bundle_hash, chunk_count, chunk_starts, inputs_dir,
                     load_gen_config, model_dir, run_name, scene_name, video_info)


def prompt_for(scene: str, weather: str, cfg: dict) -> str:
    sentence = cfg["scenes"].get(scene, cfg["scenes"]["default"])
    return " ".join(sentence.split()) + " " + cfg["weather"][weather]


def make_spec(bdir: Path, run: dict, cfg: dict) -> dict:
    if run["edge"] not in EDGE_OPTIONS:
        raise ValueError(f"unknown edge option {run['edge']!r}")
    sc, w = cfg["sampling"], cfg["sampling"]["control_weight"]
    clip, inp = bdir / "clip_0000", inputs_dir(bdir)
    spec = {
        "name": "out",
        "prompt": prompt_for(scene_name(bdir), run["weather"], cfg),
        "video_path": str(clip / "rgb.mp4"),
        "seed": int(sc["seed"]), "guidance": int(sc["guidance"]), "num_steps": int(sc["num_steps"]),
        "resolution": str(sc["resolution"]),
        "num_video_frames_per_chunk": int(sc["chunk_frames"]),
        "num_conditional_frames": int(sc["num_conditional_frames"]),
        "depth": {"control_path": str(inp / "depth_control.mp4"), "control_weight": float(w["depth"])},
        "seg": {"control_path": str(clip / "segmentation.mp4"), "control_weight": float(w["seg"])},
    }
    if run["edge"] != "none":
        edge = clip / "edges.mp4" if run["edge"] == "rendered" else inp / "edges_geometry.mp4"
        spec["edge"] = {"control_path": str(edge), "control_weight": float(w["edge"])}
    return spec


def spec_paths(bdir: Path, cfg: dict) -> list[tuple[int, dict, Path]]:
    alt = int(bdir.name)
    out = []
    for i, run in enumerate(cfg["runs"], start=1):
        if int(run["altitude"]) == alt:
            out.append((i, run, model_dir(bdir, cfg) / "specs" / f"{i:02d}_{run_name(run)}.json"))
    return out


def write_specs(run_folder: str, cfg: dict) -> list[Path]:
    written = []
    for alt in sorted({int(r["altitude"]) for r in cfg["runs"]}):
        bdir = bundle_dir(run_folder, alt)
        for _, run, path in spec_paths(bdir, cfg):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(make_spec(bdir, run, cfg), indent=1) + "\n")
            written.append(path)
    return written


def check_spec(path: Path, cfg: dict, cache: dict) -> list[str]:
    """Load one spec, open every file it names and compare frame counts and sizes."""
    problems = []
    spec = json.loads(path.read_text())
    if "vis" in spec:
        problems.append("uses the blur control")
    files = {"video": spec["video_path"]}
    for key in ("depth", "edge", "seg"):
        if key in spec:
            if not spec[key].get("control_path"):
                problems.append(f"{key} has no control file, the model would compute it from the colour video")
            else:
                files[key] = spec[key]["control_path"]
    infos = {}
    for key, f in files.items():
        try:
            infos[key] = cache[f] if f in cache else cache.setdefault(f, video_info(Path(f)))
        except Exception as e:
            problems.append(f"{key}: {e}")
    ref = infos.get("video")
    for key, i in infos.items():
        if ref and (i["frames"], i["width"], i["height"]) != (ref["frames"], ref["width"], ref["height"]):
            problems.append(f"{key} is {i['frames']} frames {i['width']}x{i['height']}, the input video is "
                            f"{ref['frames']} frames {ref['width']}x{ref['height']}")
        if abs(i["fps"] - float(cfg["controls"]["fps"])) > 1e-6:
            problems.append(f"{key} is {i['fps']} fps")
    try:                                    # the model's own validation, when its package is importable
        from cosmos_transfer2.config import InferenceArguments
        InferenceArguments.model_validate(spec)
    except ImportError:
        problems.append("model package not importable here, its own spec validation was not run")
    except Exception as e:
        problems.append(f"model validation: {e}")
    return problems


def plan(run_folder: str, cfg: dict, check: bool = True) -> int:
    bad, cache, rows = 0, {}, []
    est = cfg["estimate_s_per_chunk"]
    for alt in sorted({int(r["altitude"]) for r in cfg["runs"]}):
        bdir = bundle_dir(run_folder, alt)
        for i, run, path in spec_paths(bdir, cfg):
            problems = check_spec(path, cfg, cache) if check else []
            frames = cache.get(json.loads(path.read_text())["video_path"], {}).get("frames", 0)
            chunks = chunk_count(frames, cfg)
            rows.append((i, alt, run["weather"], run["edge"], frames, chunks, problems, bdir))
            bad += bool(problems)
    print(f"plan for {run_folder} (bundle hashes: " +
          ", ".join(f"{b.name} ft {bundle_hash(b)[:12]}" for b in sorted({r[7] for r in rows})) + ")")
    print(f"{'#':>2} {'alt':>4} {'weather':<6} {'edge':<9} {'frames':>6} {'chunks':>6}  " +
          "  ".join(f"{k} min" for k in est) + "  status")
    total = {k: 0.0 for k in est}
    for i, alt, weather, edge, frames, chunks, problems, _ in sorted(rows):
        mins = {k: (chunks * float(v) + float(cfg["model_load_s"])) / 60 for k, v in est.items()}
        for k in est:
            total[k] += mins[k]
        print(f"{i:>2} {alt:>4} {weather:<6} {edge:<9} {frames:>6} {chunks:>6}  " +
              "  ".join(f"{mins[k]:>{len(k) + 4}.0f}" for k in est) + "  " + ("ok" if not problems else "; ".join(problems)))
    print("total hours: " + ", ".join(f"{k} {v / 60:.1f}" for k, v in total.items()))
    if rows:
        print(f"chunk boundaries at output frames {chunk_starts(rows[0][4], cfg)} "
              f"(overlap {cfg['sampling']['num_conditional_frames']} frame)")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--check", action="store_true", help="validate existing specs, write nothing")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    if not a.check:
        write_specs(a.run_folder, cfg)
    sys.exit(1 if plan(a.run_folder, cfg) else 0)
