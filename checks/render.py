"""Render checks: rendered clip bundles, one per altitude."""
from __future__ import annotations

import argparse
import functools
import json
import math
import sys

import numpy as np
from PIL import Image

from _checks import CheckList, IMAGES, shown

from nyc_drone_sim.common import for_location, altitude_m, clip_settings, load_config, slugify, variant_suffix
from nyc_drone_sim.postrender import MODALITIES, bundle_dir, id_to_class, probe


COLOUR_PSNR_DB = 35.0      # render-7: colour below this is a warning, not a failure
GEOMETRY_SHARE = 0.999     # render-7: share of pixels whose depth and class must agree


def colour_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def rerender_agreement(depth_a, depth_b, sem_a, classes_a: dict, sem_b, classes_b: dict) -> tuple[float, float]:
    """Share of pixels whose depth agrees to 1% (or is empty sky in both), and share whose class
    name agrees. Classes are compared by name: the two renders may number them differently."""
    da, db = np.asarray(depth_a, np.float64).squeeze(), np.asarray(depth_b, np.float64).squeeze()
    fin_a, fin_b = np.isfinite(da) & (da > 0), np.isfinite(db) & (db > 0)
    both = fin_a & fin_b
    close = np.zeros(da.shape, bool)
    close[both] = np.abs(db[both] - da[both]) <= 0.01 * da[both]
    same_depth = close | (~fin_a & ~fin_b)

    def names(arr, classes):
        arr = np.asarray(arr).squeeze()
        ids, inv = np.unique(arr, return_inverse=True)
        return np.array([classes.get(int(i), f"id {int(i)}") for i in ids], dtype=object)[inv].reshape(arr.shape)

    same_class = names(sem_a, classes_a) == names(sem_b, classes_b)
    return float(same_depth.mean()), float(same_class.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, choices=["street", "long"])
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    g = CheckList("render", slug, a.variant, run=a.run)
    n, fps = clip_settings(cfg, a.variant)
    W, H = (int(v) for v in cfg["camera"]["resolution"])
    alts = [int(x) for x in cfg["altitudes_ft"]]
    flat = set(cfg["scene"]["flat_classes"])
    B = {k: bundle_dir(cfg, a.run, slug, k) for k in alts}
    bundle = {k: json.loads((B[k] / "bundle.json").read_text()) for k in alts}
    path = {k: json.loads((B[k] / "path.json").read_text()) for k in alts}

    def label_ids(k):
        return {c: i for i, c in id_to_class(B[k]).items()}

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
        min_frames = int(cfg["render"]["centre_depth_min_frames"])
        for k in alts:
            # ground-level classes: plain ground, and in a populated scene road, sidewalk and so on
            gids = {v for c, v in label_ids(k).items() if c in flat}
            alt = altitude_m(k)
            errs, depths = [], []
            block = (slice(H // 2 - 1, H // 2 + 1), slice(W // 2 - 1, W // 2 + 1))
            for i in range(n):
                # the depth is averaged over the centre block, so every pixel of the block must be
                # ground: a vehicle or pole covering part of it would be averaged in
                if not all(int(v) in gids for v in sem(k, i)[block].ravel()):
                    continue
                d = np.load(B[k] / "metric_depth" / f"depth_{i:04d}.npy")
                c = float(np.mean(d[block]))
                depths.append(c)
                errs.append(abs(c * math.sin(math.radians(-cfg["camera"]["pitch_deg"])) - alt) / alt)
            if len(errs) >= min_frames:
                ok &= max(errs) <= 0.01
                det.append(f"{k} ft: {len(errs)} frames with ground over the whole centre block, centre depth {np.mean(depths):.2f} m "
                           f"(expected {alt / math.sin(math.radians(45)):.1f}), worst altitude error {max(errs):.3%}")
            else:
                any_ground = False
                det.append(f"{k} ft: not evaluated: {len(errs)} frames have ground over the whole centre block, "
                           f"at least {min_frames} are needed")
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
            others = {c: v for c, v in label_ids(k).items() if c not in ("building", "BACKGROUND", "UNLABELLED")}
            if len(others) > 1:
                tot = {c: 0.0 for c in others}
                step = max(1, n // 24)
                for i in range(0, n, step):
                    ids, cnt = np.unique(sem(k, i), return_counts=True)
                    look = dict(zip(ids.tolist(), cnt.tolist()))
                    for c, v in others.items():
                        tot[c] += look.get(v, 0) / (W * H)
                m = len(range(0, n, step))
                det.append(f"{k} ft mean share of other classes: " + ", ".join(f"{c} {tot[c] / m:.2%}" for c in sorted(tot)))
        return True, "; ".join(det) + ("; flagged for review" if flag else ""), flag

    def g57():
        # Geometry must match: depth and classes of the second frame-0 render against the clip's
        # frame 0. Colour may differ slightly, because the real-time renderer smooths each image
        # over the frames before it; below COLOUR_PSNR_DB it is a warning with its value.
        det, ok, warn, saved = [], True, False, 0
        id2c = {k: id_to_class(B[k]) for k in alts}
        for k in alts:
            z = np.load(B[k] / "rerender_frame0.npz")
            a0 = np.asarray(Image.open(B[k] / "clip_0000" / "rgb" / "rgb_0000.png").convert("RGB"), dtype=np.float64)
            psnr = colour_psnr(a0, z["rgb"][..., :3].astype(np.float64))
            warn |= psnr < COLOUR_PSNR_DB
            colour = f"colour PSNR {psnr:.2f} dB" + (f" (below {COLOUR_PSNR_DB} dB: warning)" if psnr < COLOUR_PSNR_DB else "")
            if "depth" not in z.files or "semantic" not in z.files:
                det.append(f"{k} ft: {colour}; depth and classes were not saved by this render")
                continue
            saved += 1
            labels = {int(i): (v.get("class") if isinstance(v, dict) else str(v))
                      for i, v in json.loads(str(z["labels"])).items()}
            d_share, c_share = rerender_agreement(np.load(B[k] / "metric_depth" / "depth_0000.npy"), z["depth"],
                                                  sem(k, 0), id2c[k], z["semantic"], labels)
            good = d_share >= GEOMETRY_SHARE and c_share >= GEOMETRY_SHARE
            ok &= good
            det.append(f"{k} ft: depth within 1% on {d_share:.3%} of pixels, same class on {c_share:.3%} "
                       f"(at least {GEOMETRY_SHARE:.1%} needed); {colour}")
        if saved == 0:
            return None, "; ".join(det)
        return ok, "; ".join(det), warn

    def g59():
        # The depth picture must use the grey range its scene offers. That range depends on the
        # scene: bare flat ground seen at this pitch spans far fewer grey levels than a street
        # of tall buildings. So the range expected for each frame is computed from its metric
        # depth through the configured colour scale (log, near = white, far = black, applied to
        # distance from the camera), and the picture must reach a set share of it.
        det, ok = [], True
        rc, cc = cfg["render"], cfg["camera"]
        share = float(rc["depth_grey_range_share"])
        near, far = float(rc["depth_colorize_near_m"]), float(rc["depth_colorize_far_m"])
        th = float(cc["horizontal_aperture_mm"]) / 2 / float(cc["focal_length_mm"])
        tv = float(cc["vertical_aperture_mm"]) / 2 / float(cc["focal_length_mm"])
        u = (2 * (np.arange(W) + 0.5) / W - 1)[None, :]
        v = (1 - 2 * (np.arange(H) + 0.5) / H)[:, None]
        ray = np.sqrt(1 + (u * th) ** 2 + (v * tv) ** 2)     # distance from camera per metre of axis depth
        for k in alts:
            lo, hi, want = [], [], []
            for i in range(0, n, 4):
                im = np.asarray(Image.open(B[k] / "clip_0000" / "depth" / f"depth_{i:04d}.png").convert("L"))
                lo.append(np.percentile(im, 1))
                hi.append(np.percentile(im, 99))
                dist = np.load(B[k] / "metric_depth" / f"depth_{i:04d}.npy") * ray
                grey = 255 * (1 - np.log(np.clip(dist, near, far) / near) / math.log(far / near))
                want.append(np.percentile(grey, 99) - np.percentile(grey, 1))
            rng, expect = float(np.mean(hi) - np.mean(lo)), float(np.mean(want))
            ok &= rng >= share * expect
            det.append(f"{k} ft: colourised depth grey levels p1 {np.mean(lo):.0f} to p99 {np.mean(hi):.0f} (range {rng:.0f}); "
                       f"the scene's metric depth gives {expect:.0f} on the {near:g}-{far:g} m scale, need >= {share:.0%} of it")
        return ok, "; ".join(det)

    def g510():
        # Camera height from every ground pixel: metric depth along the optical axis times the
        # downward slope of that pixel's ray. Independent of what happens to sit at the centre.
        det, ok, thin = [], True, False
        cc = cfg["camera"]
        tv = float(cc["vertical_aperture_mm"]) / 2 / float(cc["focal_length_mm"])
        pitch = math.radians(-float(cc["pitch_deg"]))
        v = (1 - 2 * (np.arange(H) + 0.5) / H)[:, None]
        slope = math.sin(pitch) - v * tv * math.cos(pitch)      # metres of height per metre of axis depth
        step, min_frames = int(cfg["render"]["ground_height_frame_step"]), int(cfg["render"]["centre_depth_min_frames"])
        for k in alts:
            gids = [v_ for c, v_ in label_ids(k).items() if c in flat and c != "road_marking"]   # paint sits 2 cm up
            alt = altitude_m(k)
            med, within, used = [], 0, 0
            for i in range(0, n, step):
                mask = np.isin(sem(k, i), gids)
                if not mask.any():
                    continue
                h = (np.load(B[k] / "metric_depth" / f"depth_{i:04d}.npy") * slope)[mask]
                med.append(float(np.median(h)))
                within += int((np.abs(h - alt) <= 0.01 * alt).sum())
                used += int(h.size)
            if len(med) < min_frames:
                thin = True
                det.append(f"{k} ft: not evaluated: {len(med)} sampled frames show ground, at least {min_frames} are needed")
                continue
            worst = max(abs(m - alt) for m in med) / alt
            ok &= worst <= 0.01
            det.append(f"{k} ft: camera height from {used / 1e6:.1f} M ground pixels in {len(med)} frames: per-frame median "
                       f"{min(med):.3f} to {max(med):.3f} m (altitude {alt:g} m, worst error {worst:.3%}, limit 1%); "
                       f"{within / used:.2%} of the pixels are within 1%")
        return ok, "; ".join(det), thin

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
            out = IMAGES / f"render_{slug}_{a.run}_{k}{variant_suffix(a.variant)}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            ok &= out.stat().st_size > 10_000
            det.append(f"{shown(out)} ({out.stat().st_size} bytes)")
        return ok, "; ".join(det)

    for gid, fn in [("render-1", g51), ("render-2", g52), ("render-3", g53), ("render-4", g54), ("render-5", g55),
                    ("render-6", g56), ("render-7", g57), ("render-8", g58), ("render-9", g59),
                    ("render-10", g510)]:
        g.run(gid, fn)
    for k in alts:
        print(f"RGB video {k} ft: {B[k] / 'clip_0000' / 'rgb.mp4'}")
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
