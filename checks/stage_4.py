"""Checks for stage 4: drone camera paths. Geometry only."""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
from pxr import Usd, UsdGeom
from shapely.geometry import Point, shape

from _checks import REPO_ROOT, CheckList

from nyc_drone_sim.common import extent_polygon, for_location, altitude_m, clip_settings, load_config, scene_dir, slugify, variant_suffix
from nyc_drone_sim.paths import path_file


def point_triangles_distance(p, a, b, c):
    """Distance from point p to each triangle (a, b, c); closest-point regions after Ericson."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = p - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = p - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        inside = a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]
        e_ab = a + ab * (d1 / (d1 - d3))[:, None]
        e_ac = a + ac * (d2 / (d2 - d6))[:, None]
        e_bc = b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]
    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
    ]
    choices = [a, b, e_ab, c, e_ac, e_bc]
    closest = inside
    for cond, ch in reversed(list(zip(conds, choices))):
        closest = np.where(cond[:, None], ch, closest)
    return np.linalg.norm(closest - p, axis=1)


def corner_hits(m, intr):
    th = intr["width"] / (2 * intr["fx"])
    tv = intr["height"] / (2 * intr["fy"])
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        d = m[:3, :3] @ np.array([sx * th, sy * tv, -1.0])
        if d[2] >= -1e-9:
            return None
        out.append(m[:2, 3] + (-m[2, 3] / d[2]) * d[:2])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, choices=["street", "long"])
    ap.add_argument("--view", default=None)
    ap.add_argument("--altitude", type=int, action="append")
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    sdir = scene_dir(cfg, slug)
    g = CheckList(4, slug, "_".join(v for v in (a.variant, a.view) if v) or None)
    if a.altitude:
        cfg = {**cfg, "altitudes_ft": a.altitude}
    want_frames, want_fps = clip_settings(cfg, a.variant)
    R = float(cfg["scene_radius_m"])
    corridor = extent_polygon(cfg) if cfg.get("corridor") else None
    alts = [altitude_m(x) for x in cfg["altitudes_ft"]]
    paths = {int(x): json.loads(path_file(sdir, x, a.variant, a.view).read_text()) for x in cfg["altitudes_ft"]}
    M = {k: np.array([f["camera_to_world"] for f in p["frames"]]) for k, p in paths.items()}
    P = {k: np.array([f["position"] for f in p["frames"]]) for k, p in paths.items()}
    T = {k: np.array([f["t"] for f in p["frames"]]) for k, p in paths.items()}

    def g41():
        det = []
        for k in paths:
            want = altitude_m(k)
            dz = np.abs(P[k][:, 2] - want).max()
            dm = np.abs(M[k][:, :3, 3] - P[k]).max()
            rot = M[k][:, :3, :3]
            orth = np.abs(rot @ rot.transpose(0, 2, 1) - np.eye(3)).max()
            pitch = np.degrees(np.arcsin(-rot[:, 2, 2]))   # forward = -Z column
            pe = np.abs(pitch - cfg["camera"]["pitch_deg"]).max()
            ok = dz <= 0.01 and dm <= 1e-9 and orth <= 1e-9 and pe <= 1e-6 and any(abs(want - x) < 1e-9 for x in alts)
            det.append((ok, f"{k} ft: max |z - {want:.2f}| = {dz:.1e} m; matrix translation = position; rotation orthonormal; pitch {pitch.mean():.3f} deg"))
        return all(o for o, _ in det), "; ".join(d for _, d in det)

    def g42():
        stage = Usd.Stage.Open(str(sdir / "scene.usdc"))
        A, B, C = [], [], []
        for prim in stage.GetPrimAtPath("/World/Buildings").GetChildren():
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            A.append(pts[idx[:, 0]]); B.append(pts[idx[:, 1]]); C.append(pts[idx[:, 2]])
        A, B, C = np.vstack(A), np.vstack(B), np.vstack(C)
        need = float(cfg["paths"]["min_surface_distance_m"])
        # placed models (trees, lamps, vehicles, people): distance to each one's bounding box,
        # which is never more than the distance to the model itself
        boxes = []
        objects = stage.GetPrimAtPath("/World/Objects")
        if objects.IsValid():
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            for group in objects.GetChildren():
                for prim in group.GetChildren():
                    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    boxes.append((group.GetName(), *r.GetMin(), *r.GetMax()))
        lo = np.array([b[1:4] for b in boxes], dtype=np.float64).reshape(-1, 3)
        hi = np.array([b[4:7] for b in boxes], dtype=np.float64).reshape(-1, 3)
        det, ok = [], True
        for k in paths:
            pts = np.unique(P[k], axis=0)
            dmin = min(point_triangles_distance(p, A, B, C).min() for p in pts)
            ok &= dmin >= need
            det.append(f"{k} ft: nearest building surface {dmin:.2f} m")
            if len(boxes):
                d = np.linalg.norm(np.maximum(np.maximum(lo[None] - pts[:, None], pts[:, None] - hi[None]), 0.0), axis=2)
                j = int(np.argmin(d.min(0)))
                ok &= float(d.min()) >= need
                det.append(f"nearest placed model {d.min():.2f} m ({boxes[j][0]}, top at {hi[j][2]:.1f} m)")
        return ok, (f"from {len(A)} USD triangles and {len(boxes)} placed-model bounding boxes; " + "; ".join(det)
                    + f"; required >= {need:g} m")

    def g43():
        det, ok = [], True
        for k, p in paths.items():
            dt = np.diff(T[k])
            sp = np.linalg.norm(np.diff(P[k], axis=0), axis=1) / dt
            yaw = np.degrees(np.unwrap(np.arctan2(-M[k][:, 1, 2], -M[k][:, 0, 2])))
            yr = np.abs(np.diff(yaw)) / dt
            if p["path_type"] == "hover_pan":
                s_ok = sp.max() == 0
                s_txt = "stationary (hover_pan)"
            elif p.get("sync_ground_track"):
                # The constant-speed rule applies to the point the image centre follows on the
                # ground, which both altitudes share. The camera sits altitude / tan(pitch) behind
                # that point, so in the turn it swings and its own speed cannot stay constant.
                G = ground_points(k)
                gs = np.linalg.norm(np.diff(G, axis=0), axis=1) / dt
                s_ok = (gs.max() - gs.min()) / gs.mean() <= 0.02 and abs(gs.mean() - cfg["speed_mps"][k]) / cfg["speed_mps"][k] <= 0.02
                s_txt = (f"image-centre ground point speed {gs.min():.4f}-{gs.max():.4f} m/s (config {cfg['speed_mps'][k]}); "
                         f"camera speed {sp.min():.2f}-{sp.max():.2f} m/s (not held constant: it swings through the turn)")
            else:
                s_ok = (sp.max() - sp.min()) / sp.mean() <= 0.02 and abs(sp.mean() - cfg["speed_mps"][k]) / cfg["speed_mps"][k] <= 0.02
                s_txt = f"speed {sp.min():.4f}-{sp.max():.4f} m/s (config {cfg['speed_mps'][k]})"
            ok &= s_ok and yr.max() <= cfg["paths"]["max_yaw_rate_deg_s"]
            det.append(f"{k} ft: {s_txt}, max yaw rate {yr.max():.2f} deg/s (limit {cfg['paths']['max_yaw_rate_deg_s']:g})")
        return ok, "; ".join(det)

    def ground_points(k):
        """Where the image-centre ray meets the ground, per frame, from the saved camera matrices."""
        pos, fwd = M[k][:, :3, 3], -M[k][:, :3, 2]
        return (pos + fwd * (-pos[:, 2] / fwd[:, 2])[:, None])[:, :2]

    def g49():
        # Both altitudes must look at the same ground point in the same frame.
        if not all(p.get("sync_ground_track") for p in paths.values()):
            return None, "not evaluated: this clip does not share a ground track between altitudes"
        if len(paths) < 2:
            return None, "not evaluated: only one altitude was checked"
        ks = sorted(paths)
        ref = ground_points(ks[0])
        det, ok = [], True
        for k in ks[1:]:
            d = np.linalg.norm(ground_points(k) - ref, axis=1)
            ok &= d.max() <= 1.0
            det.append(f"image-centre ground points at {ks[0]} ft and {k} ft: largest separation {d.max():.4f} m "
                       f"over {len(d)} frames (limit 1 m)")
        saved = max(float(np.linalg.norm(ground_points(k) - np.array([f["ground_point"] for f in paths[k]["frames"]]), axis=1).max()) for k in ks)
        ok &= saved <= 0.01
        return ok, "; ".join(det) + f"; against the planned track: {saved:.4f} m"

    def g44():
        det, ok = [], True
        for k, p in paths.items():
            n, fps = len(p["frames"]), 1.0 / np.diff(T[k]).mean()
            dur = n / fps
            good = (n == want_frames and abs(fps - want_fps) < 1e-9
                    and np.allclose(np.diff(T[k]), 1 / want_fps)
                    and abs(dur - want_frames / want_fps) < 1e-9
                    and abs(p["clip"]["duration_s"] - dur) < 1e-9)
            ok &= good
            det.append(f"{k} ft: {n} frames at {fps:.3f} fps, {dur:.4f} s")
        return ok, "; ".join(det)

    def g45():
        det, ok = [], True
        for k, p in paths.items():
            far, outside = 0.0, 0.0
            for m in M[k]:
                h = corner_hits(m, p["intrinsics"])
                if h is None:
                    far = outside = float("inf")
                    break
                far = max(far, float(np.hypot(h[:, 0], h[:, 1]).max()))
                if corridor is not None:
                    outside = max(outside, max(corridor.distance(Point(*c)) for c in h))
            if corridor is None:
                ok &= far <= R
                det.append(f"{k} ft: farthest corner ground hit {far:.1f} m from origin")
            else:
                # a corridor scene: the corners must land inside the corridor itself, a stricter
                # test than the circle that encloses it
                ok &= outside == 0.0
                det.append(f"{k} ft: farthest corner ground hit {far:.1f} m from origin, corners outside the corridor by at most {outside:.1f} m")
        return ok, "; ".join(det) + (f"; limit {R:g} m" if corridor is None else "; limit: inside the scene corridor")

    def target_share(k):
        intr = paths[k]["intrinsics"]
        n = 0
        for m in M[k]:
            pc = np.linalg.inv(m) @ np.array([0, 0, 0, 1.0])
            if pc[2] < 0:
                px = intr["cx"] + intr["fx"] * pc[0] / -pc[2]
                py = intr["cy"] - intr["fy"] * pc[1] / -pc[2]
                n += 0 <= px <= intr["width"] and 0 <= py <= intr["height"]
        return n / len(M[k])

    def g46():
        det, ok = [], True
        for k, p in paths.items():
            s = target_share(k)
            if p["path_type"] in ("flyover", "orbit"):
                ok &= s >= 0.8
                det.append(f"{k} ft ({p['path_type']}): target in image in {s:.1%} of frames")
            else:
                det.append(f"{k} ft ({p['path_type']}): not applicable; target in image in {s:.1%} of frames")
        return ok, "; ".join(det)

    def g47():
        det, ok = [], True
        for k, p in paths.items():
            i = p["intrinsics"]
            h = math.degrees(2 * math.atan(i["width"] / (2 * i["fx"])))
            v = math.degrees(2 * math.atan(i["height"] / (2 * i["fy"])))
            ok &= abs(h - 73.74) <= 0.05 and abs(v - 45.75) <= 0.05
            det.append(f"{k} ft: {h:.3f} x {v:.3f} deg")
        return ok, "; ".join(det)

    def g48():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
        fig, axes = plt.subplots(1, len(paths), figsize=(11 * len(paths), 11), dpi=110)
        for ax, (k, p) in zip(np.atleast_1d(axes), paths.items()):
            alt = altitude_m(k)
            low, tall = [], []
            for f in feats:
                for poly in f["geometry"]["coordinates"]:
                    (tall if f["properties"]["height_m"] > alt else low).append(np.asarray(poly[0]))
            ax.add_collection(PolyCollection(low, facecolors="0.8", edgecolors="0.5", linewidths=0.2))
            ax.add_collection(PolyCollection(tall, facecolors="tomato", edgecolors="darkred", linewidths=0.3))
            t = np.linspace(0, 2 * np.pi, 300)
            ax.plot(R * np.cos(t), R * np.sin(t), "k--", lw=0.8)
            for m, col, lab in ((M[k][0], "tab:green", "first frame ground footprint"),
                                (M[k][-1], "tab:blue", "last frame ground footprint")):
                h = corner_hits(m, p["intrinsics"])
                ax.fill(h[:, 0], h[:, 1], facecolor=col, alpha=0.18, edgecolor=col, lw=1.5, label=lab)
                f = -m[:2, 2] / np.linalg.norm(m[:2, 2])
                ax.annotate("", xy=m[:2, 3] + 40 * f, xytext=m[:2, 3], arrowprops=dict(arrowstyle="->", color=col, lw=2))
            ax.plot(P[k][:, 0], P[k][:, 1], "-", color="magenta", lw=3, label=f"camera path ({p['path_type']})")
            ax.plot(P[k][0, 0], P[k][0, 1], "o", color="tab:green", ms=7, mec="k")
            ax.plot(P[k][-1, 0], P[k][-1, 1], "s", color="tab:blue", ms=7, mec="k")
            ax.plot(0, 0, "*", color="gold", ms=18, mec="k", label="target")
            lim = max(350.0, float(np.abs(P[k][:, :2]).max()) + 100)
            lim = max(lim, max(float(np.abs(corner_hits(m, p["intrinsics"])).max()) for m in (M[k][0], M[k][-1])) + 30)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            params = ", ".join(f"{a_}={[round(x) for x in v] if isinstance(v, list) else round(v, 1) if isinstance(v, float) else v}"
                               for a_, v in p["path_params"].items())
            ax.set_title(f"{k} ft ({alt:.2f} m): {p['path_type']}, {params}\n"
                         f"red = taller than the altitude ({p['taller_count_within_300m']} of "
                         f"{p['buildings_within_300m']} within 300 m)")
            ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)")
            ax.legend(loc="upper right")
        out = REPO_ROOT / "evidence" / f"stage_4_{slug}{variant_suffix(a.variant, a.view)}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out.stat().st_size > 10_000, f"{out.relative_to(REPO_ROOT)} ({out.stat().st_size} bytes)"

    for gid, fn in [("G4.1", g41), ("G4.2", g42), ("G4.3", g43), ("G4.4", g44),
                    ("G4.5", g45), ("G4.6", g46), ("G4.7", g47), ("G4.8", g48), ("G4.9", g49)]:
        g.run(gid, fn)
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
