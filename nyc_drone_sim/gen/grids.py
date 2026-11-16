"""Write the spec files of a tuning grid from its definition file, and validate them. CPU only.

  python -m nyc_drone_sim.gen.grids --grid configs/gen_grid_clear.yaml

Each run uses one window cut by nyc_drone_sim.gen.windows. Specs go to
<bundle>/gen/<model>/<grid_name>/specs/<run>.json and are named after the run, so one process of
the model can generate the whole grid with a single model load.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .common import REPO_ROOT, bundle_dir, load_gen_config, model_dir, video_info


def default_negative() -> str:
    from cosmos_transfer2.config import DEFAULT_NEGATIVE_PROMPT
    return DEFAULT_NEGATIVE_PROMPT


def build(grid_file: str, cfg: dict) -> list[Path]:
    g = yaml.safe_load((REPO_ROOT / grid_file if not Path(grid_file).is_absolute() else Path(grid_file)).read_text())
    bdir = bundle_dir(g["run_folder"], g["altitude"])
    out = model_dir(bdir, cfg) / g["grid_name"] / "specs"
    out.mkdir(parents=True, exist_ok=True)
    sc = cfg["sampling"]
    written = []
    for r in g["runs"]:
        w = bdir / "gen" / "windows" / r["window"]
        # a run may name its own weather, seed and edge input; the grid's values are the defaults
        weather = " ".join(g["weather"][r.get("weather", "clear")].split())
        spec = {"name": r["name"], "prompt": " ".join(g["scenes"][r["window"]].split()) + " " + weather,
                "video_path": str(w / "rgb.mp4"), "seed": int(r.get("seed", g["seed"])), "guidance": int(r["guidance"]),
                "num_steps": int(g["num_steps"]), "resolution": str(sc["resolution"]),
                "num_video_frames_per_chunk": int(sc["chunk_frames"]), "num_conditional_frames": int(sc["num_conditional_frames"])}
        if r.get("negative_extra"):
            spec["negative_prompt"] = default_negative().rstrip() + " " + g["negative_extra"]
        depth_w, edge_w, seg_w = r["weights"]
        if depth_w is not None:
            spec["depth"] = {"control_path": str(w / ("depth_control_smooth.mp4" if r.get("smooth_depth") else "depth_control.mp4")),
                             "control_weight": float(depth_w)}
        if edge_w is not None:
            edge_file = {"rendered": "edges_rendered.mp4", "geometry": "edges_geometry.mp4"}.get(r.get("edge"), g["edge_file"])
            spec["edge"] = {"control_path": str(w / edge_file), "control_weight": float(edge_w)}
        if seg_w is not None:
            spec["seg"] = {"control_path": str(w / "segmentation.mp4"), "control_weight": float(seg_w)}
        p = out / f"{r['name']}.json"
        p.write_text(json.dumps(spec, indent=1) + "\n")
        written.append(p)
    return written


def check(paths: list[Path]) -> int:
    from cosmos_transfer2.config import InferenceArguments
    bad = 0
    for p in paths:
        s = json.loads(p.read_text())
        problems = []
        if "vis" in s:
            problems.append("uses the blur control")
        ref = video_info(Path(s["video_path"]))
        for k in ("depth", "edge", "seg"):
            if k in s:
                if not s[k].get("control_path"):
                    problems.append(f"{k} has no file")
                    continue
                i = video_info(Path(s[k]["control_path"]))
                if (i["frames"], i["width"], i["height"]) != (ref["frames"], ref["width"], ref["height"]):
                    problems.append(f"{k} does not match the input video")
        try:
            InferenceArguments.model_validate(s)
        except Exception as e:
            problems.append(f"model validation: {e}")
        used = ", ".join(f"{k} {s[k]['control_weight']}" for k in ("depth", "edge", "seg") if k in s)
        print(f"{s['name']:4s} {ref['frames']} frames, guidance {s['guidance']}, {used:38s} prompt {len(s['prompt'].split())} words"
              f"{', extended negative prompt' if 'negative_prompt' in s else ''}: {'ok' if not problems else '; '.join(problems)}")
        bad += bool(problems)
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    paths = build(a.grid, load_gen_config(a.config))
    print("specs in", paths[0].parent)
    sys.exit(1 if check(paths) else 0)
