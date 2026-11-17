"""Per-piece prompts for a chained clip, and the spec that uses them. CPU only.

  python -m nyc_drone_sim.gen.pieces --run-folder runs/run_009/canal-shore-street --altitude 150 \\
      --weather clear --setting K2 --out <folder> [--max-frames 185]

The model generates a long clip in 93-frame chunks, each starting from the last frame of the one
before. For each chunk (a piece) the share of every class over its frames is measured from the
semantic frames; configs/gen_prompts_pieces.yaml turns the classes above its threshold into
sentences, always with the same wording. Writes <out>/spec.json (the model's own format; its
prompt is the first piece's), <out>/chunk_prompts.json (one prompt per chunk, for
nyc_drone_sim.gen.chunk_prompt_inference) and <out>/pieces.json (frames, shares and prompt of
every piece).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .common import REPO_ROOT, bundle_dir, inputs_dir, load_gen_config, video_info

FRAME_STEP = 4          # every 4th frame is enough to measure shares over a 93-frame piece


def piece_ranges(n: int, cfg: dict) -> list[tuple[int, int]]:
    """(first, last) frame of every chunk the model will generate for an n-frame clip."""
    per, cond = int(cfg["sampling"]["chunk_frames"]), int(cfg["sampling"]["num_conditional_frames"])
    out, first = [], 0
    while True:
        last = min(first + per, n) - 1
        out.append((first, last))
        if last >= n - 1:
            return out
        first = last + 1 - cond


def class_shares(bdir: Path, first: int, last: int) -> dict[str, float]:
    from nyc_drone_sim.postrender import id_to_class
    names = id_to_class(bdir)
    tot, cnt = {}, 0
    for i in range(first, last + 1, FRAME_STEP):
        ids, c = np.unique(np.load(bdir / "semantic" / f"semantic_{i:04d}.npz")["data"], return_counts=True)
        for k, v in zip(ids.tolist(), c.tolist()):
            name = names.get(int(k), "unlabelled")
            tot[name] = tot.get(name, 0) + v
        cnt += int(c.sum())
    return {k: v / cnt for k, v in sorted(tot.items(), key=lambda x: -x[1])}


def piece_prompt(shares: dict, weather: str, pp: dict, wp: dict) -> str:
    """Opening, one fixed sentence per class above the threshold, then the weather paragraph."""
    lo = float(pp["min_share"])
    words = [pp["opening"]] + [s for c, s in pp["classes"].items() if shares.get(c, 0.0) > lo]
    key = "with_water" if shares.get("water", 0.0) > lo else "without_water"
    text = " ".join(" ".join(w.split()) for w in words + [wp["weather"][weather][key]])
    if "water tank" in text.lower():
        raise RuntimeError("a prompt mentions water tanks")
    return text


def build(bdir: Path, weather: str, setting_name: str, cfg: dict, out: Path, max_frames: int | None = None) -> dict:
    from .makeclip import base_spec
    pp = yaml.safe_load((REPO_ROOT / "configs" / "gen_prompts_pieces.yaml").read_text())
    wp = yaml.safe_load((REPO_ROOT / "configs" / "gen_prompts_windows.yaml").read_text())
    setting = wp["settings"][setting_name]
    clip, inp = bdir / "clip_0000", inputs_dir(bdir)
    n = video_info(clip / "rgb.mp4")["frames"]
    if max_frames:
        n = min(n, int(max_frames))
    pieces = []
    for first, last in piece_ranges(n, cfg):
        shares = class_shares(bdir, first, last)
        pieces.append({"first": first, "last": last, "prompt": piece_prompt(shares, weather, pp, wp),
                       "shares_percent": {k: round(100 * v, 1) for k, v in shares.items() if v >= 0.005}})
    edge = clip / "edges.mp4" if setting["edge"] == "rendered" else inp / "edges_geometry.mp4"
    spec = base_spec(setting, cfg, "out", pieces[0]["prompt"], clip / "rgb.mp4", inp / "depth_control.mp4",
                     edge, clip / "segmentation.mp4")
    if max_frames:
        spec["max_frames"] = int(n)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    (out / "chunk_prompts.json").write_text(json.dumps([p["prompt"] for p in pieces], indent=1) + "\n")
    (out / "pieces.json").write_text(json.dumps({"bundle": str(bdir), "weather": weather, "setting": setting_name,
                                                 "frames": n, "pieces": pieces}, indent=1) + "\n")
    return {"frames": n, "pieces": len(pieces)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--altitude", type=int, required=True)
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--setting", default="K2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None, help="only the first N frames (a short test)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    r = build(bundle_dir(a.run_folder, a.altitude), a.weather, a.setting, load_gen_config(a.config), Path(a.out), a.max_frames)
    print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
