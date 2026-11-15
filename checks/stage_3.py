"""Checks for stage 3: scene loaded, lit and labelled in Isaac Sim. Reads the probe outputs."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from _checks import REPO_ROOT, CheckList

from nyc_drone_sim.common import for_location, load_config, scene_dir, slugify

UNLABELLED = {"BACKGROUND", "UNLABELLED", ""}


def label_of(info: dict, sid: int) -> str:
    v = info["idToLabels"].get(str(int(sid)), {})
    return v.get("class", "") if isinstance(v, dict) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    pdir = scene_dir(cfg, slug) / "probe"
    g = CheckList(3, slug)
    res = json.loads((pdir / "probe_result.json").read_text())
    data = {}
    for pr in res["probes"]:
        z = np.load(pdir / pr["file"])
        data[pr["name"]] = {"rgb": z["rgb"][..., :3], "depth": z["depth"], "seg": z["seg"], "meta": pr}

    def centre_depth(d):
        h, w = d.shape[:2]
        return float(np.mean(d[h // 2 - 1:h // 2 + 1, w // 2 - 1:w // 2 + 1]))

    def depth_check(name, tol):
        def fn():
            p = data[name]
            want, got = p["meta"]["expect_depth_m"], centre_depth(p["depth"])
            return abs(got - want) <= tol, (f"{name}: centre depth {got:.4f} m, expected {want:.2f} m +/- {tol} "
                                            f"(annotator {res['depth_annotator']}, stage metersPerUnit {res['stage_meters_per_unit']})")
        return fn

    names = [pr["name"] for pr in res["probes"]]
    flat = set(cfg["scene"]["flat_classes"])
    ground = [n for n in names if n.startswith("ground")]

    def g34():
        det, ok = [], True
        for n in names:
            p = data[n]
            seg, info = p["seg"], p["meta"]["seg_info"]
            h, w = seg.shape[:2]
            c = label_of(info, seg[h // 2, w // 2])
            ids, cnt = np.unique(seg, return_counts=True)
            un = sum(int(k) for i, k in zip(ids, cnt) if label_of(info, i) in UNLABELLED) / seg.size
            want = p["meta"]["expect_label"]
            # in a populated scene the ground under the camera may be road, sidewalk and so on;
            # any class that lies on the ground plane counts, objects and buildings do not
            good = (c in flat if want == "ground" else c == want) and un < 0.01
            ok &= good
            shares = {label_of(info, i) or "?": round(int(k) / seg.size, 4) for i, k in zip(ids, cnt)}
            det.append(f"{n}: centre '{c}' (expected '{want}'" + (", any ground-level class" if want == "ground" else "")
                       + f"), unlabelled {un:.2%}, shares {shares}")
        return ok, "; ".join(det)

    def g35():
        det, ok = [], True
        for n in names:
            rgb = data[n]["rgb"].astype(np.float64)
            mean = rgb.mean()
            clipped = float(((rgb.max(-1) >= 255) | (rgb.max(-1) <= 0)).mean())
            ok &= 30 <= mean <= 220 and clipped < 0.01
            det.append(f"{n}: mean {mean:.1f}, clipped {clipped:.2%}")
        return ok, "; ".join(det)

    def g36():
        share = res["vram_peak_mib"] / res["vram_total_mib"]
        rtx = any(k in res["gpu_name"] for k in ("L40S", "RTX"))
        return share < 0.8 and rtx, (f"peak {res['vram_peak_mib']:.0f} MiB of {res['vram_total_mib']:.0f} MiB "
                                     f"({share:.1%}) on {res['gpu_name']}; scene load {res['scene_load_s']:.1f} s")

    def g37():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(names), 3, figsize=(18, 3.6 * len(names)), dpi=110)
        for r, n in enumerate(names):
            p = data[n]
            d = np.where(np.isfinite(p["depth"]) & (p["depth"] > 0), p["depth"], np.nan)
            ax[r, 0].imshow(p["rgb"])
            ax[r, 0].set_title(f"{n}: RGB, camera at {[round(v, 2) for v in p['meta']['position']]}")
            im = ax[r, 1].imshow(d, cmap="turbo")
            fig.colorbar(im, ax=ax[r, 1], shrink=0.8, label="m")
            ax[r, 1].set_title(f"metric depth: centre {centre_depth(p['depth']):.3f} m (expected {p['meta']['expect_depth_m']:.2f})")
            info = p["meta"]["seg_info"]
            ids = np.unique(p["seg"])
            lut = {int(i): k for k, i in enumerate(ids)}
            segc = np.vectorize(lut.get)(p["seg"])
            ax[r, 2].imshow(segc, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
            ax[r, 2].set_title("semantic: " + ", ".join(f"{k}={label_of(info, i) or '?'}" for i, k in lut.items()))
            for c in range(3):
                ax[r, c].plot(p["depth"].shape[1] / 2, p["depth"].shape[0] / 2, "w+", ms=14, mew=2)
                ax[r, c].axis("off")
        out = REPO_ROOT / "evidence" / f"stage_3_{slug}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out.stat().st_size > 10_000, f"{out.relative_to(REPO_ROOT)} ({out.stat().st_size} bytes)"

    g.run("G3.1", depth_check(ground[0], 0.25))
    g.run("G3.2", depth_check(ground[1], 0.5))
    g.run("G3.3", depth_check("roof", 0.5))
    for gid, fn in [("G3.4", g34), ("G3.5", g35), ("G3.6", g36), ("G3.7", g37)]:
        g.run(gid, fn)
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
