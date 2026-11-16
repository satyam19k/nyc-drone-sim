"""Write the spec for one full clip from a chosen setting. CPU only.

  python -m nyc_drone_sim.gen.fullclip --run-folder runs/run_009/canal-shore-street --altitude 150 \
      --weather clear --setting '{"name": "K2", "guidance": 3, "weights": [0.5, 0.3, 0.5]}'

The setting gives guidance and the weights for depth / edge / segmentation; a weight of null leaves
that control out of the spec, so the model never computes it from the colour video. Optional keys:
negative_extra (true adds the extra negative prompt), smooth_depth (true uses the dithered depth
control), edge ("rendered" or "geometry"; clear weather defaults to rendered).
The spec is written to <bundle>/gen/<model>/full_specs/<weather>_<setting name>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .common import REPO_ROOT, bundle_dir, inputs_dir, load_gen_config, model_dir, scene_name, video_info


def full_spec(bdir: Path, weather: str, setting: dict, cfg: dict) -> dict:
    p = yaml.safe_load((REPO_ROOT / "configs" / "gen_prompts.yaml").read_text())
    sc = cfg["sampling"]
    clip, inp = bdir / "clip_0000", inputs_dir(bdir)
    spec = {"name": "out",
            "prompt": " ".join(p["scenes"][scene_name(bdir)].split()) + " " + " ".join(p["weather"][weather].split()),
            "video_path": str(clip / "rgb.mp4"), "seed": int(sc["seed"]), "guidance": int(setting["guidance"]),
            "num_steps": int(sc["num_steps"]), "resolution": str(sc["resolution"]),
            "num_video_frames_per_chunk": int(sc["chunk_frames"]), "num_conditional_frames": int(sc["num_conditional_frames"])}
    if setting.get("negative_extra"):
        from cosmos_transfer2.config import DEFAULT_NEGATIVE_PROMPT
        spec["negative_prompt"] = DEFAULT_NEGATIVE_PROMPT.rstrip() + " " + p["negative_extra"]
    d, e, s = setting["weights"]
    if d is not None:
        spec["depth"] = {"control_path": str(inp / ("depth_control_smooth.mp4" if setting.get("smooth_depth") else "depth_control.mp4")),
                         "control_weight": float(d)}
    if e is not None:
        kind = setting.get("edge", "rendered" if weather == "clear" else "geometry")
        spec["edge"] = {"control_path": str(clip / "edges.mp4" if kind == "rendered" else inp / "edges_geometry.mp4"),
                        "control_weight": float(e)}
    if s is not None:
        spec["seg"] = {"control_path": str(clip / "segmentation.mp4"), "control_weight": float(s)}
    return spec


def write_and_check(bdir: Path, weather: str, setting: dict, cfg: dict) -> Path:
    spec = full_spec(bdir, weather, setting, cfg)
    ref = video_info(Path(spec["video_path"]))
    for k in ("depth", "edge", "seg"):
        if k in spec:
            i = video_info(Path(spec[k]["control_path"]))
            if (i["frames"], i["width"], i["height"]) != (ref["frames"], ref["width"], ref["height"]):
                raise RuntimeError(f"{k} control does not match the input video: {i} against {ref}")
    from cosmos_transfer2.config import InferenceArguments
    InferenceArguments.model_validate(spec)
    out = model_dir(bdir, cfg) / "full_specs" / f"{weather}_{setting['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--altitude", type=int, required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--setting", required=True, help="JSON: name, guidance, weights [depth, edge, seg], ...")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    b = bundle_dir(a.run_folder, a.altitude)
    path = write_and_check(b, a.weather, json.loads(a.setting), cfg)
    n = video_info(b / "clip_0000" / "rgb.mp4")["frames"]
    print(f"{path}  ({n} frames)")
