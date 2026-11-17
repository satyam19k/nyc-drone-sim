"""Measure the long clips of a place: class shares over time, contact sheets, marking stability.

  python scripts/long_clip_report.py --location battery-shoreline --run run_006

For each altitude it writes under docs/validation/images/:
  timeline_<place>_<alt>.png  class shares predicted from the ground footprint, beside the shares
                              measured in the rendered segmentation
  sheet8_<place>_<alt>.png    eight evenly spaced frames as RGB, depth and segmentation
and docs/validation/reports/long_clips_<place>.json with the numbers, including a frame-by-frame test of whether
road markings flicker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import shapely
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nyc_drone_sim.postrender import id_to_class
from nyc_drone_sim import scout  # noqa: E402
from nyc_drone_sim.common import altitude_m, for_location, load_config, scene_dir, slugify  # noqa: E402
from nyc_drone_sim.paths import Pinhole  # noqa: E402
from nyc_drone_sim.postrender import bundle_dir  # noqa: E402

MEASURED = {"water": ("water",), "park": ("park", "lawn", "path"), "road": ("road", "parking", "road_marking"),
            "walkway": ("sidewalk", "median", "plaza", "pier", "boardwalk"), "building": ("building",),
            "tree": ("vegetation",), "vehicle": ("vehicle",), "person": ("person",), "furniture": ("furniture",)}
COLORS = {**scout.COLORS, "tree": "#1f6f2f", "vehicle": "#e0b000", "person": "#d040a0", "furniture": "#30c0c0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. the view name, so runs do not overwrite each other")
    ap.add_argument("--altitude", type=int, action="append", help="limit to this altitude; repeatable")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    sdir = scene_dir(cfg, slug)
    cam = Pinhole(cfg["camera"])
    fps = float(cfg["clip"]["fps"])
    U = scout.class_unions(sdir)
    for g in U.values():
        shapely.prepare(g)
    off = scout.not_at_grade(sdir, float(cfg["scout"]["nongrade_buffer_m"]), U["road"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = {"place": a.location, "run": a.run, "altitudes": {}}
    for alt in (a.altitude or [int(x) for x in cfg["altitudes_ft"]]):
        B = bundle_dir(cfg, a.run, slug, alt)
        b = json.loads((B / "bundle.json").read_text())
        path = json.loads((B / "path.json").read_text())
        n = len(path["frames"])
        xy = np.array([f["position"][:2] for f in path["frames"]])
        yaw = np.array([f["yaw_deg"] for f in path["frames"]])
        pred = scout.footprint_shares(xy, yaw, altitude_m(alt), cam, U, off, 8)
        ids = {c: k for k, c in id_to_class(B).items()}
        frames = list(range(0, n, 4))
        meas = {k: [] for k in MEASURED}
        meas["other"] = []
        marks = []
        for i in range(n):
            sem = np.load(B / "semantic" / f"semantic_{i:04d}.npz")["data"]
            marks.append(int((sem == ids.get("road_marking", -1)).sum()))
            if i % 4:
                continue
            cnt = np.bincount(sem.ravel(), minlength=max(ids.values()) + 1) / sem.size
            tot = 0.0
            for k, cs in MEASURED.items():
                v = float(sum(cnt[ids[c]] for c in cs if c in ids))
                meas[k].append(v)
                tot += v
            meas["other"].append(max(0.0, 1 - tot))
        summ = {k: {"mean": round(float(np.mean(v)), 4), "max": round(float(np.max(v)), 4)} for k, v in meas.items()}
        psum = {k: {"mean": round(float(np.mean(pred[k])), 4), "max": round(float(np.max(pred[k])), 4)} for k in scout.ORDER}

        # road markings, every frame: a flicker shows as a count that jumps away from both neighbours
        m = np.array(marks, dtype=np.float64)
        mid = (m[:-2] + m[2:]) / 2
        jump = np.abs(m[1:-1] - mid) / np.maximum(mid, 200.0)
        shimmer = {"frames": n, "marking_pixels_min": int(m.min()), "marking_pixels_max": int(m.max()),
                   "median_jump": round(float(np.median(jump)), 4), "p99_jump": round(float(np.percentile(jump, 99)), 4),
                   "max_jump": round(float(jump.max()), 4), "frame_of_max": int(jump.argmax() + 1),
                   "frames_over_5pct": int((jump > 0.05).sum())}

        fig, ax = plt.subplots(1, 2, figsize=(15, 3.6), dpi=110)
        scout.stack(ax[0], pred, fps, f"{a.location}{' ' + a.tag if a.tag else ''}, {alt} ft: predicted from the ground footprint (buildings mean {psum['building']['mean']:.0%})")
        x = np.array(frames) / fps
        keys = ("water", "park", "road", "walkway", "building", "tree", "vehicle", "person", "furniture", "other")
        ax[1].stackplot(x, *[np.array(meas[k]) * 100 for k in keys], colors=[COLORS[k] for k in keys], labels=keys)
        ax[1].axhline(10, color="k", lw=0.5, ls=":")
        ax[1].set_ylim(0, 100)
        ax[1].set_xlim(x[0], x[-1])
        ax[1].set_title(f"measured in the rendered segmentation (buildings mean {summ['building']['mean']:.0%})", fontsize=8)
        ax[1].legend(fontsize=6, ncol=5, loc="upper center")
        for q in ax:
            q.set_xlabel("time (s)", fontsize=8)
            q.set_ylabel("share of frame (%)", fontsize=8)
        out = REPO / "docs" / "validation" / "images" / f"timeline_{slug}_{alt}{tag}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

        pick = [int(round(v)) for v in np.linspace(0, n - 1, 8)]
        W, H = 426, 240
        sheet = Image.new("RGB", (8 * (W + 4), 3 * (H + 4) + 18), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for c, i in enumerate(pick):
            d.text((c * (W + 4) + 4, 2), f"frame {i} ({i / fps:.1f} s)", fill=(0, 0, 0))
            for r, mname in enumerate(("rgb", "depth", "segmentation")):
                im = Image.open(B / "clip_0000" / mname / f"{mname}_{i:04d}.png").convert("RGB").resize((W, H), Image.LANCZOS)
                sheet.paste(im, (c * (W + 4), 18 + r * (H + 4)))
        sout = REPO / "docs" / "validation" / "images" / f"sheet8_{slug}_{alt}{tag}.png"
        sheet.save(sout)
        report["altitudes"][alt] = {"video": str(B / "clip_0000" / "rgb.mp4"), "frames": n, "predicted": psum, "measured": summ,
                                    "measured_over_10pct": [k for k in ("water", "park", "road", "building") if summ[k]["max"] >= 0.10],
                                    "road_marking_stability": shimmer,
                                    "timeline": str(out.relative_to(REPO)), "sheet": str(sout.relative_to(REPO))}
        print(f"{alt} ft: measured max {({k: format(summ[k]['max'], '.0%') for k in ('water', 'park', 'road', 'building')})}, "
              f"buildings mean {summ['building']['mean']:.0%}; objects mean tree {summ['tree']['mean']:.1%}, vehicle "
              f"{summ['vehicle']['mean']:.2%}, person {summ['person']['mean']:.2%}; markings {shimmer}")
    (REPO / "docs" / "validation" / "reports" / f"long_clips_{slug}{tag}.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
