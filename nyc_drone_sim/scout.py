"""Score candidate camera routes from the scene's own layers, before rendering. CPU only.

For each candidate and altitude the camera is flown for one long clip. At sampled frames the
ground footprint of the view is intersected with the water, park, road, walkway and building
polygons, giving each class's share of the footprint over time. Roads that are not at ground
level (bridges, viaducts, ramps, underpasses) are reported as their own share, because the
scene flattens them and they should stay out of frame.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import shapely
from shapely.geometry import LineString, shape
from shapely.ops import unary_union

from .common import REPO_ROOT, altitude_m, for_location, load_config, scene_dir, slugify
from .paths import Obstacles, Pinhole, camera_to_world, frame_times, frustum_ground, route_track

log = logging.getLogger("scout")
GROUPS = {"water": ("water",), "park": ("park", "lawn", "path"), "road": ("road", "parking"),
          "walkway": ("sidewalk", "median", "plaza", "pier", "boardwalk")}
GRADE = 13      # centre-line level code for ground level
ORDER = ("water", "park", "road", "walkway", "building", "other", "not_at_grade")


def class_unions(sdir) -> dict:
    by = {}
    for f in json.loads((sdir / "surfaces.geojson").read_text())["features"]:
        by.setdefault(f["properties"]["class"], []).append(shape(f["geometry"]))
    U = {g: unary_union([p for c in cs for p in by.get(c, [])]) for g, cs in GROUPS.items()}
    U["building"] = unary_union([shape(f["geometry"]) for f in json.loads((sdir / "buildings.geojson").read_text())["features"]])
    return U


def not_at_grade(sdir, buffer_m: float, road):
    def off(p) -> bool:
        a, b = p.get("from_level") or GRADE, p.get("to_level") or GRADE
        # above ground level anywhere, or a ramp between ground level and a tunnel. A segment
        # wholly below ground level is a tunnel under the surface streets and is not drawn.
        return max(a, b) > GRADE or (min(a, b) < GRADE and max(a, b) == GRADE)
    lines = [LineString(np.asarray(f["geometry"]["coordinates"])[:, :2])
             for f in json.loads((sdir / "centerlines.geojson").read_text())["features"] if off(f["properties"])]
    polys = [shape(f["geometry"]) for f in json.loads((sdir / "elevated.geojson").read_text())["features"]]
    mask = unary_union([ln.buffer(buffer_m) for ln in lines] + polys)
    # only road surface that the scene actually draws counts: a tunnel under a park has no
    # roadbed polygon, shows nothing from above, and is not a problem
    return mask.intersection(road)


def footprint_shares(xy, yaw, alt: float, cam: Pinhole, U: dict, off_grade, every: int) -> dict:
    idx = np.unique(np.append(np.arange(0, len(xy), every), len(xy) - 1))
    out = {k: [] for k in ORDER}
    for i in idx:
        quad = shapely.Polygon(frustum_ground(camera_to_world([xy[i, 0], xy[i, 1], alt], float(yaw[i]), cam.pitch_deg), cam))
        a = quad.area
        # buildings cover whatever surface lies under them
        b = quad.intersection(U["building"])
        out["building"].append(b.area / a)
        rest = quad.difference(U["building"]) if b.area > 0 else quad
        tot = b.area
        for g in GROUPS:
            v = rest.intersection(U[g]).area
            out[g].append(v / a)
            tot += v
        out["other"].append(max(0.0, 1 - tot / a))
        out["not_at_grade"].append(quad.intersection(off_grade).area / a)
    return {"frames": idx.tolist(), **{k: [round(float(v), 4) for v in vs] for k, vs in out.items()}}


def score(cfg: dict) -> dict:
    sc = cfg["scout"]
    pcfg = for_location(cfg, sc["place"])
    sdir = scene_dir(cfg, slugify(sc["place"]))
    U = class_unions(sdir)
    for g in U.values():
        shapely.prepare(g)
    off = not_at_grade(sdir, float(sc["nongrade_buffer_m"]), U["road"])
    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    cam = Pinhole(cfg["camera"])
    t = frame_times(cfg, "long")
    obs = {int(a): Obstacles(feats, altitude_m(a), pcfg) for a in cfg["altitudes_ft"]}
    res = {}
    for name, c in sc["candidates"].items():
        res[name] = {"kind": c["kind"], "route_m": c["route_m"], "altitudes": {}}
        for a in cfg["altitudes_ft"]:
            a = int(a)
            xy, yaw = route_track(c["route_m"], float(cfg["speed_mps"][a]), t, float(c["focus"][a]), float(c["yaw_offset_deg"]), cfg)
            sh = footprint_shares(xy, yaw, altitude_m(a), cam, U, off, int(sc["sample_every"]))
            blocked = int(sum(obs[a].is_blocked(p) for p in xy))
            summary = {k: {"mean": round(float(np.mean(sh[k])), 4), "max": round(float(np.max(sh[k])), 4)} for k in ORDER}
            res[name]["altitudes"][a] = {"track": xy[::8].round(1).tolist(), "yaw_deg": yaw[::8].round(1).tolist(),
                                         "frames_inside_clearance": blocked,
                                         "clearance_m": round(float(min(obs[a].clearance(p) for p in xy)), 1),
                                         "shares": sh, "summary": summary,
                                         "classes_over_10pct": [k for k in ("water", "park", "road", "building") if summary[k]["max"] >= 0.10]}
            log.info("%s %d ft: %s; buildings mean %.0f%%; not at grade max %.1f%%; frames inside clearance %d",
                     name, a, {k: f"{summary[k]['max']:.0%}" for k in ("water", "park", "road", "building")},
                     100 * summary["building"]["mean"], 100 * summary["not_at_grade"]["max"], blocked)
    out = REPO_ROOT / "reports" / "scout.json"
    out.write_text(json.dumps({"place": sc["place"], "clip_frames": len(t), "candidates": res}, indent=1) + "\n")
    return res


COLORS = {"water": "#4a90c2", "park": "#5aa55a", "road": "#444444", "walkway": "#b9b3a6", "building": "#c8553d",
          "other": "#dddddd", "not_at_grade": "#d100d1"}


def stack(ax, sh: dict, fps: float, title: str):
    x = np.array(sh["frames"]) / fps
    keys = ("water", "park", "road", "walkway", "building", "other")
    ax.stackplot(x, *[np.array(sh[k]) * 100 for k in keys], colors=[COLORS[k] for k in keys], labels=keys)
    if "not_at_grade" in sh:
        ax.plot(x, np.array(sh["not_at_grade"]) * 100, color=COLORS["not_at_grade"], lw=1.5, label="not at grade")
    ax.axhline(10, color="k", lw=0.5, ls=":")
    ax.set_ylim(0, 100)
    ax.set_xlim(x[0], x[-1])
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def plot(cfg: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    sc = cfg["scout"]
    res = json.loads((REPO_ROOT / "reports" / "scout.json").read_text())["candidates"]
    sdir = scene_dir(cfg, slugify(sc["place"]))
    U = class_unions(sdir)
    fps = float(cfg["clip"]["fps"])
    n = len(res)
    fig = plt.figure(figsize=(22, 2.4 * n + 1), dpi=100)
    gs = fig.add_gridspec(n, 4, width_ratios=[2.6, 1, 1, 0.02])
    ax = fig.add_subplot(gs[:, 0])

    def polys(g):
        return [np.asarray(p.exterior.coords) for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]
    # road first: its outer rings enclose everything else, and holes are not drawn
    for k in ("water", "road", "walkway", "park", "building"):
        ax.add_collection(PolyCollection(polys(U[k]), facecolors=COLORS[k], edgecolors="none"))
    off = not_at_grade(sdir, float(sc["nongrade_buffer_m"]), U["road"])
    ax.add_collection(PolyCollection(polys(off), facecolors=COLORS["not_at_grade"], edgecolors="none"))
    tab = plt.cm.tab10
    for i, (name, c) in enumerate(res.items()):
        for a, ls in (("500", "--"), ("150", "-")):
            tr = np.array(c["altitudes"][a]["track"])
            ax.plot(tr[:, 0], tr[:, 1], ls, color=tab(i), lw=3 if a == "150" else 1.8)
        tr = np.array(c["altitudes"]["150"]["track"])
        ax.annotate("", xy=tr[-1], xytext=tr[-4], arrowprops=dict(arrowstyle="->", color=tab(i), lw=3))
        ax.text(tr[0, 0], tr[0, 1], f" {i + 1} {name}", color="k", fontsize=9, weight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=tab(i)))
    ax.set_xlim(-900, 500)
    ax.set_ylim(-1300, 1500)
    ax.set_aspect("equal")
    ax.set_title("Candidate camera tracks (solid: 150 ft clip, dashed: 500 ft clip). Magenta: road not at ground level", fontsize=10)
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    for i, (name, c) in enumerate(res.items()):
        for j, a in enumerate(("150", "500")):
            axx = fig.add_subplot(gs[i, 1 + j])
            s = c["altitudes"][a]["summary"]
            stack(axx, c["altitudes"][a]["shares"], fps,
                  f"{i + 1} {name}, {a} ft: buildings mean {s['building']['mean']:.0%}\nover 10%: "
                  f"{', '.join(c['altitudes'][a]['classes_over_10pct'])}")
            if i == 0 and j == 1:
                axx.legend(fontsize=6, loc="upper right", ncol=2)
            if i == n - 1:
                axx.set_xlabel("time (s)", fontsize=8)
    out = REPO_ROOT / "evidence" / f"scout_{slugify(sc['place'])}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score candidate camera routes")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config)
    score(cfg)
    log.info("figure %s", plot(cfg))


if __name__ == "__main__":
    main()
