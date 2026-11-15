"""Checks for stage 5: rendered clip bundles, one per altitude."""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys

import numpy as np
from PIL import Image

from _checks import REPO_ROOT, CheckList

from nyc_drone_sim.common import altitude_m, clip_settings, load_config, slugify, variant_suffix
from nyc_drone_sim.postrender import MODALITIES, bundle_dir, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, choices=["street"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    slug = slugify(a.location)
    g = CheckList(5, slug, a.variant)
    n, fps = clip_settings(cfg, a.variant)
    W, H = (int(v) for v in cfg["camera"]["resolution"])
    alts = [int(x) for x in cfg["altitudes_ft"]]
    B = {k: bundle_dir(cfg, a.run, slug, k) for k in alts}
    bundle = {k: json.loads((B[k] / "bundle.json").read_text()) for k in alts}
    path = {k: json.loads((B[k] / "path.json").read_text()) for k in alts}

    def label_ids(k):
        return {v["class"]: int(i) for i, v in bundle[k]["semantic_info"]["idToLabels"].items()}

    @functools.lru_cache(maxsize=200)    # bounded, so long clips do not fill memory
    def sem(k, i):
        return np.load(B[k] / "semantic" / f"semantic_{i:04d}.npz")["data"]

    def g51():
        det, ok = [], True
        for k in alts:
            for m in MODALITIES:
                files = sorted((B[k] / "clip_0000" / m).glob("*.png"))
                sizes = {Image.open(f).size for f in files}
                v = probe(B[k] / "clip_0000" / f"{m}.mp4")
                good = (len(files) == n and sizes == {(W, H)} and v is not None and v["frames"] == n
                        and abs(v["fps"] - fps) < 1e-6 and (v["width"], v["height"]) == (W, H))
                ok &= good
                if not good:
                    det.append(f"{k} ft {m}: {len(files)} PNGs {sizes}, mp4 {v}")
            det.append(f"{k} ft: 5 folders x {n} PNGs at {W}x{H}; 5 MP4s at {n} frames, {fps:g} fps "
                       f"(re-encoded: {[m for m, r in bundle[k].get('mp4_check', {}).items() if r['re_encoded']] or 'none'})")
        return ok, "; ".join(det)

    def g52():
        det, ok = [], True
        for k in alts:
            worst_p = worst_r = 0.0
            for i, fr in enumerate(path[k]["frames"]):
                c = json.loads((B[k] / "camera" / f"camera_{i:04d}.json").read_text())["camera_params"]
                view = np.array(c["cameraViewTransform"], dtype=np.float64).reshape(4, 4)   # row-vector world->camera
                c2w = np.linalg.inv(view).T                                             # column-vector camera->world
                want = np.array(fr["camera_to_world"])
                worst_p = max(worst_p, float(np.linalg.norm(c2w[:3, 3] - want[:3, 3])))
                cosang = (np.trace(want[:3, :3].T @ c2w[:3, :3]) - 1) / 2
                worst_r = max(worst_r, math.degrees(math.acos(min(1.0, max(-1.0, cosang)))))
            ok &= worst_p <= 0.05 and worst_r <= 0.2
            det.append(f"{k} ft: max position error {worst_p * 100:.4f} cm, max orientation error {worst_r:.5f} deg over {n} frames")
        return ok, "; ".join(det)

    def g53():
        det, ok, any_ground = [], True, True
        for k in alts:
            gid = label_ids(k)["ground"]
            alt = altitude_m(k)
            errs, depths = [], []
            for i in range(n):
                if sem(k, i)[H // 2, W // 2] != gid:
                    continue
                d = np.load(B[k] / "metric_depth" / f"depth_{i:04d}.npy")
                c = float(np.mean(d[H // 2 - 1:H // 2 + 1, W // 2 - 1:W // 2 + 1]))
                depths.append(c)
                errs.append(abs(c * math.sin(math.radians(-cfg["camera"]["pitch_deg"])) - alt) / alt)
            if errs:
                ok &= max(errs) <= 0.01
                det.append(f"{k} ft: {len(errs)} frames with ground at centre, centre depth {np.mean(depths):.2f} m "
                           f"(expected {alt / math.sin(math.radians(45)):.1f}), worst altitude error {max(errs):.3%}")
            else:
                any_ground = False
                det.append(f"{k} ft: no frame has ground at the image centre, so this clip gives nothing to test")
        return ok, "; ".join(det), not any_ground

    def g54():
        det, ok = [], True
        for k in alts:
            ids = label_ids(k)
            empty = sky = 0
            for i in range(n):
                d = np.load(B[k] / "metric_depth" / f"depth_{i:04d}.npy")
                empty += int((~np.isfinite(d) | (d <= 0)).sum())
                sky += int((sem(k, i) == ids["BACKGROUND"]).sum())
            tot = n * W * H
            ok &= empty / tot < 0.01 and sky == 0
            det.append(f"{k} ft: empty metric-depth pixels {empty / tot:.4%}, sky/background pixels {sky}")
        return ok, "; ".join(det)

    def g55():
        det, ok = [], True
        for k in alts:
            prev, diffs = None, []
            for i in range(n):
                im = np.asarray(Image.open(B[k] / "clip_0000" / "rgb" / f"rgb_{i:04d}.png").convert("RGB"), dtype=np.int16)
                if prev is not None:
                    diffs.append(float(np.abs(im - prev).mean()))
                prev = im
            diffs = np.array(diffs)
            lim = diffs.mean() + 4 * diffs.std()
            ok &= diffs.min() > 0 and diffs.max() <= lim
            det.append(f"{k} ft: frame-to-frame mean abs difference min {diffs.min():.3f}, mean {diffs.mean():.3f}, "
                       f"max {diffs.max():.3f} (4-sigma limit {lim:.3f})")
        return ok, "; ".join(det)

    def g56():
        det, flag = [], False
        for k in alts:
            bid = label_ids(k)["building"]
            share = np.array([(sem(k, i) == bid).mean() for i in range(n)])
            out = int(((share < 0.05) | (share > 0.95)).sum())
            flag |= out > 0
            det.append(f"{k} ft: building share per frame min {share.min():.1%}, mean {share.mean():.1%}, max {share.max():.1%}; "
                       f"{out} frames outside 5-95%")
        return True, "; ".join(det) + ("; flagged for review" if flag else ""), flag

    def g57():
        det, ok = [], True
        for k in alts:
            a0 = np.asarray(Image.open(B[k] / "clip_0000" / "rgb" / "rgb_0000.png").convert("RGB"), dtype=np.float64)
            b0 = np.load(B[k] / "rerender_frame0.npz")["rgb"][..., :3].astype(np.float64)
            mse = float(((a0 - b0) ** 2).mean())
            psnr = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
            ok &= psnr >= 35
            det.append(f"{k} ft: PSNR of second frame-0 render against the clip's frame 0 = {psnr:.2f} dB")
        return ok, "; ".join(det)

    def g59():
        det, ok = [], True
        need = float(cfg["render"]["depth_min_grey_range"])
        for k in alts:
            lo, hi = [], []
            for i in range(0, n, 4):
                im = np.asarray(Image.open(B[k] / "clip_0000" / "depth" / f"depth_{i:04d}.png").convert("L"))
                lo.append(np.percentile(im, 1))
                hi.append(np.percentile(im, 99))
            rng = float(np.mean(hi) - np.mean(lo))
            ok &= rng >= need
            det.append(f"{k} ft: colourised depth grey levels p1 {np.mean(lo):.0f} to p99 {np.mean(hi):.0f} (range {rng:.0f}, need >= {need:g})")
        return ok, "; ".join(det)

    def g58():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        det, ok = [], True
        cols = ("rgb", "depth", "segmentation", "edges")
        for k in alts:
            # the default clip uses the configured frames; other variants use four evenly spaced ones
            fr = ([int(v) for v in cfg["render"]["evidence_frames"]] if not a.variant
                  else [int(round(v)) for v in np.linspace(0, n - 1, 4)])
            fig, ax = plt.subplots(len(fr), len(cols), figsize=(4.6 * len(cols), 2.7 * len(fr)), dpi=110)
            for r, i in enumerate(fr):
                for c, m in enumerate(cols):
                    ax[r, c].imshow(Image.open(B[k] / "clip_0000" / m / f"{m}_{i:04d}.png"), cmap="gray")
                    ax[r, c].axis("off")
                    ax[r, c].set_title(f"{m}, frame {i}", fontsize=9)
            fig.suptitle(f"{slug} at {k} ft: {bundle[k]['path_type']} {bundle[k]['path_params']}", fontsize=11)
            out = REPO_ROOT / "evidence" / f"stage_5_{slug}_{k}{variant_suffix(a.variant)}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            ok &= out.stat().st_size > 10_000
            det.append(f"{out.relative_to(REPO_ROOT)} ({out.stat().st_size} bytes)")
        return ok, "; ".join(det)

    for gid, fn in [("G5.1", g51), ("G5.2", g52), ("G5.3", g53), ("G5.4", g54), ("G5.5", g55),
                    ("G5.6", g56), ("G5.7", g57), ("G5.8", g58), ("G5.9", g59)]:
        g.run(gid, fn)
    for k in alts:
        print(f"RGB video {k} ft: {B[k] / 'clip_0000' / 'rgb.mp4'}")
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
