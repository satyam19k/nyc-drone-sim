"""One sheet for a tuning grid: the simulator row on top of each window, then a row per run with
three frames, its structure score and its ripple score. Also writes the scores as JSON.

  python scripts/gen_grid_sheet.py --grid configs/gen_grid_clear.yaml \
      --other-clip runs/run_006/hudson-park-run/150
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nyc_drone_sim.gen.common import bundle_dir, load_gen_config, model_dir  # noqa: E402
from nyc_drone_sim.gen.scores import FLAT_CLASSES, ripple_score, structure_score  # noqa: E402
from nyc_drone_sim.postrender import id_to_class  # noqa: E402

SHOW = (10, 46, 82)            # frames of the window drawn on the sheet
SCORE = (6, 21, 36, 51, 66, 81)   # frames of the window that are scored


_CACHE: dict = {}


def frame(path: Path, k: int):
    """Frame k of a video, decoded in order from the start. Seeking by frame number is not
    reliable for these H.264 files, so each video is read once and its frames are kept."""
    key = str(path)
    if key not in _CACHE:
        cap = cv2.VideoCapture(key)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        _CACHE.clear()          # keep one video in memory at a time
        _CACHE[key] = frames
    if k >= len(_CACHE[key]):
        raise RuntimeError(f"cannot read frame {k} of {path}: it has {len(_CACHE[key])} frames")
    return _CACHE[key][k]


def label(im, text):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(im, text, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--other-clip", required=True, help="bundle of a different clip, for the mismatched baseline")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    g = yaml.safe_load((REPO / a.grid).read_text())
    bdir = bundle_dir(g["run_folder"], g["altitude"])
    out = model_dir(bdir, cfg) / g["grid_name"]
    other = REPO / a.other_clip
    ids = id_to_class(bdir)
    flat_ids = [k for k, c in ids.items() if c in FLAT_CLASSES]
    # land only: the prompt asks for ripples on the river, so water is scored separately
    land_ids = [k for k, c in ids.items() if c in FLAT_CLASSES and c != "water"]
    rows, scores = [], {}
    for window in dict.fromkeys(r["window"] for r in g["runs"]):
        w = bdir / "gen" / "windows" / window
        first = json.loads((w / "window.json").read_text())["first_frame"]

        def sim_edges(k):
            return cv2.imread(str(bdir / "clip_0000" / "edges" / f"edges_{first + k:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127

        def other_edges(k):
            return cv2.imread(str(other / "clip_0000" / "edges" / f"edges_{k:04d}.png"), cv2.IMREAD_GRAYSCALE) > 127

        def flat(k, which=None):
            return np.isin(np.load(bdir / "semantic" / f"semantic_{first + k:04d}.npz")["data"], which or flat_ids)

        sim = [frame(w / "rgb.mp4", k) for k in SCORE]
        base_ripple = float(np.nanmean([ripple_score(f, flat(k, land_ids), sim_edges(k)) for f, k in zip(sim, SCORE)]))
        rows.append(np.hstack([label(frame(w / "rgb.mp4", k), f"simulator, {window} window, frame {first + k}"
                                     + (f"   land ripple {base_ripple:.2f}" if i == 0 else "")) for i, k in enumerate(SHOW)]))
        for r in (r for r in g["runs"] if r["window"] == window):
            mp4 = out / f"{r['name']}.mp4"
            if not mp4.exists():
                scores[r["name"]] = {"missing": True}
                rows.append(np.hstack([label(np.zeros((720, 1280, 3), np.uint8), f"{r['name']}: no output")] * len(SHOW)))
                continue
            ss = [structure_score(frame(mp4, k), sim_edges(k), other_edges(k)) for k in SCORE]
            rp = [ripple_score(frame(mp4, k), flat(k), sim_edges(k)) for k in SCORE]
            rl = [ripple_score(frame(mp4, k), flat(k, land_ids), sim_edges(k)) for k in SCORE]
            s = {"window": window, "structure": float(np.mean([x["score"] for x in ss])),
                 "structure_min": float(np.min([x["score"] for x in ss])),
                 "agreement": float(np.mean([x["agreement"] for x in ss])), "baseline": float(np.mean([x["baseline"] for x in ss])),
                 "ripple_land": float(np.nanmean(rl)), "ripple_all_flat": float(np.nanmean(rp)), "simulator_ripple_land": base_ripple, "guidance": r["guidance"], "weights": r["weights"],
                 "negative_extra": bool(r.get("negative_extra")), "smooth_depth": bool(r.get("smooth_depth"))}
            scores[r["name"]] = s
            wtxt = "/".join("none" if v is None else f"{v:g}" for v in r["weights"])
            tag = (f"{r['name']}  g{r['guidance']}  d/e/s {wtxt}" + ("  +neg" if r.get("negative_extra") else "")
                   + ("  smooth depth" if r.get("smooth_depth") else "") + f"   structure {s['structure']:.2f}   land ripple {s['ripple_land']:.2f}")
            rows.append(np.hstack([label(frame(mp4, k), tag if i == 0 else f"{r['name']}, frame {first + k}") for i, k in enumerate(SHOW)]))
    sheet = cv2.resize(np.vstack(rows), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    png = REPO / "evidence" / f"gen_{g['grid_name']}_{bdir.parent.name}_{bdir.name}.png"
    cv2.imwrite(str(png), sheet)
    (out / "scores.json").write_text(json.dumps(scores, indent=1) + "\n")
    print(f"{'run':4s} {'window':9s} {'structure':>9s} {'min':>5s} {'land ripple':>11s} {'with water':>10s}")
    for n, s in scores.items():
        if s.get("missing"):
            print(f"{n:4s} no output")
        else:
            print(f"{n:4s} {s['window']:9s} {s['structure']:9.2f} {s['structure_min']:5.2f} {s['ripple_land']:11.2f} {s['ripple_all_flat']:10.2f}")
    print("sheet:", png)


if __name__ == "__main__":
    main()
