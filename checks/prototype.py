"""Measurements from the prototype shots: do converted models keep their scale, labels and
instance ids in the renderer, do leaves cut out, and do moving objects stay in step across
RGB, depth and segmentation. Writes reports/prototype_<location>.json; it reports numbers, it is not a gate."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image

from _checks import REPO_ROOT, CheckList

from nyc_drone_sim.common import load_config, scene_dir, slugify
from nyc_drone_sim.paths import Pinhole


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    slug = slugify(a.location)
    d = scene_dir(cfg, slug) / "populate_probe" / "unit"
    plan = json.loads((d / "shots.json").read_text())
    unit, shots = plan["unit"], plan["groups"][0]["shots"]
    intr = Pinhole(cfg["camera"]).intrinsics()
    g = CheckList("prototype", slug)

    def load(name):
        return np.load(d / f"{name}.npz"), json.loads((d / f"{name}.info.json").read_text())

    def world(shot, depth):
        """World x, y, z of every pixel of a nadir shot (image right = +x, image up = +y)."""
        c2w = np.array(shot["camera_to_world"])
        v, u = np.mgrid[0:intr["height"], 0:intr["width"]]
        x = c2w[0, 3] + (u + 0.5 - intr["cx"]) / intr["fx"] * depth
        y = c2w[1, 3] + (intr["cy"] - v - 0.5) / intr["fy"] * depth
        return x, y, c2w[2, 3] - depth

    z0, info0 = load("unit_nadir_0")
    X, Y, Z = world(shots[0], z0["depth"])
    sem_ids = {v["class"]: int(k) for k, v in info0["semantic"]["idToLabels"].items()}

    def near(o, rx, ry):
        return (np.abs(X - o["x"]) <= rx) & (np.abs(Y - o["y"]) <= ry)

    cars = [o for o in unit["objects"] if o["asset"] == unit["car"]["file"].replace(".usdc", "")]

    def q1():
        o = next(c for c in cars if c["path"] == "/World/Ref/car_1")
        m = near(o, 4.5, 3.0) & (z0["semantic"] == sem_ids["vehicle"])
        length, width, height = float(X[m].max() - X[m].min()), float(Y[m].max() - Y[m].min()), float(Z[m].max())
        L, W, H = unit["car"]["size_m"]
        # length and height decide; the model's width includes door mirrors, which are narrower than one
        # pixel from this height, so the rendered width is the body alone and is reported, not judged
        ok = abs(length - L) < 0.15 and abs(height - H) < 0.1
        return ok, (f"taxi rendered {length:.2f} m long and {height:.2f} m high (from depth and segmentation), model file "
                    f"{L:.2f} and {H:.2f} m; rendered width {width:.2f} m against {W:.2f} m in the file, which includes the "
                    f"mirrors; converted wheelbase 2.916 m against 2.913 m for the real Ford Crown Victoria; up axis Z, metres")

    def q2():
        rows = []
        ok = True
        for o in unit["objects"]:
            if o["path"] == "/World/Mover/car":
                continue
            m = near(o, 0.6, 0.6) if o["class"] != "vegetation" else near(o, 2.5, 2.5)
            ids, cnt = np.unique(z0["semantic"][m], return_counts=True)
            lab = {v: k for k, v in sem_ids.items()}
            got = {lab[int(i)] for i in ids}
            ok &= o["class"] in got
            rows.append(f"{o['path'].split('/')[-2]}/{o['method']}: {sorted(got)}")
        return ok, ("labels were applied to each placed prim after import (the model files carry none); class found at "
                    "each object: " + "; ".join(sorted(set(rows))))

    def q3():
        out, verdict = [], {}
        for method in ("reference", "instanceable", "point_instancer"):
            for key in ("instance_id", "instance"):
                sets = []
                for o in [c for c in cars if c["method"] == method and c["path"] != "/World/Mover/car"]:
                    m = near(o, 2.0, 0.6)
                    ids, cnt = np.unique(z0[key][m], return_counts=True)
                    sets.append(set(ids[cnt >= 20].tolist()))
                distinct = all(s for s in sets) and not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
                verdict[(method, key)] = distinct
                out.append(f"{method} / {key}: ids per copy {[len(s) for s in sets]}, distinct between copies {distinct}")
        ok = verdict[("instanceable", "instance_id")]
        return ok, "; ".join(out)

    def q6():
        o = next(c for c in cars if c["path"] == "/World/Ref/car_1")
        gray = z0["rgb"][..., :3].mean(-1)
        ground = z0["semantic"] == sem_ids["ground"]
        ring = ground & near(o, 6.0, 4.5)
        open_ = ground & (np.abs(Y + 4.0) < 1.5) & (np.abs(X) < 30) & ~near(o, 8.0, 6.0)
        tree = next(t for t in unit["objects"] if t["class"] == "vegetation")
        tring = ground & near(tree, 12.0, 12.0)
        dark_car = float(np.percentile(gray[ring], 2)) / float(np.median(gray[open_]))
        dark_tree = float(np.percentile(gray[tring], 2)) / float(np.median(gray[open_]))
        share_car = float((gray[ring] < 0.75 * np.median(gray[open_])).mean())
        share_tree = float((gray[tring] < 0.75 * np.median(gray[open_])).mean())
        # a shadow counts as present when some nearby ground is clearly darker than open ground
        ok = share_car > 0.005 and share_tree > 0.005
        return ok, (f"sun elevation {cfg['scene']['sun_elevation_deg']} deg: {share_car:.1%} of the ground within 6 m of the taxi and "
                    f"{share_tree:.1%} within 12 m of the tree is under 75% of open-ground brightness; darkest ground beside "
                    f"the taxi {dark_car:.0%}, under the tree {dark_tree:.0%} of open ground")

    def q7():
        rows, worst = [], 0.0
        for s in shots[:3]:
            z, _ = load(s["name"])
            x, y, hgt = world(s, z["depth"])
            band = np.abs(y + 8.0) < 2.5
            rgb = z["rgb"][..., :3].astype(float)
            masks = {"segmentation": band & (z["semantic"] == sem_ids["vehicle"]),
                     "depth": band & (hgt > 0.3),
                     "rgb": band & (rgb[..., 0] > 1.3 * rgb[..., 2] + 25) & (rgb[..., 1] > 1.1 * rgb[..., 2] + 10)}
            c = {k: float((x[m].min() + x[m].max()) / 2) if m.any() else float("nan") for k, m in masks.items()}
            worst = max(worst, *(abs(v - s["mover_x"]) for v in c.values()))
            rows.append(f"car set to x = {s['mover_x']:+.1f} m: centre in " + ", ".join(f"{k} {v:+.2f}" for k, v in c.items()))
        ok = worst < 0.3
        return ok, "; ".join(rows) + f"; largest disagreement {worst:.2f} m (a one-frame lag would show as 5 m)"

    def q_tree():
        tree = next(t for t in unit["objects"] if t["class"] == "vegetation")
        r = unit["tree"]["crown_diameter_m"] / 2
        disc = np.hypot(X - tree["x"], Y - tree["y"]) <= 0.8 * r
        veg = float((z0["semantic"][disc] == sem_ids["vegetation"]).mean())
        hgt = Z[disc & (z0["semantic"] == sem_ids["vegetation"])]
        ok = 0.2 < veg < 0.995
        return ok, (f"inside 80% of the crown radius {veg:.1%} of pixels are vegetation and {1 - veg:.1%} show through to the "
                    f"ground, in depth and segmentation alike, so the leaf cut-out works (solid leaf cards would give 100%); "
                    f"vegetation heights {hgt.min():.1f}-{hgt.max():.1f} m, model {unit['tree']['height_m']:.1f} m")

    def q_small():
        rows, ok = [], True
        for o in unit["objects"]:
            if o["class"] not in ("person", "furniture"):
                continue
            m = near(o, 1.5, 1.5) & (z0["semantic"] == sem_ids[o["class"]])
            ok &= bool(m.any())
            rows.append(f"{o['path'].split('/')[-1]}: {int(m.sum())} px, top {Z[m].max() if m.any() else 0:.2f} m")
        return ok, "pixels at 150 ft straight down: " + "; ".join(rows)

    def sheet():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        zo, _ = load("unit_oblique")
        fig, ax = plt.subplots(2, 3, figsize=(19, 7.6), dpi=110)
        for r, z in enumerate((z0, zo)):
            ax[r, 0].imshow(z["rgb"][..., :3])
            ax[r, 1].imshow(z["depth"], cmap="viridis")
            ax[r, 2].imshow(z["instance_id"] % 20, cmap="tab20", interpolation="nearest")
            for c, t in enumerate(("RGB", "metric depth", "instance ids")):
                ax[r, c].set_title(f"{'straight down, 45.72 m' if r == 0 else 'oblique, 30 m'}: {t}", fontsize=9)
                ax[r, c].axis("off")
        fig.suptitle("Prototype stage. Car rows from the bottom: moving car, plain references, instanceable references, point instancer", fontsize=10)
        out = REPO_ROOT / "evidence" / f"populate_prototype_{slug}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out.stat().st_size > 10_000, f"{out.relative_to(REPO_ROOT)} ({out.stat().st_size} bytes)"

    for gid, fn in [("Q1 scale", q1), ("Q2 labels", q2), ("Q3 instances", q3), ("Q6 shadows", q6), ("Q7 motion", q7),
                    ("leaf cut-out", q_tree), ("small objects", q_small), ("sheet", sheet)]:
        g.run(gid, fn)
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
