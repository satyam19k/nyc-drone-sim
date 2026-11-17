"""Scene checks: open data to USD. Values under test are read back from scene.usdc."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import shapely
from pxr import Usd, UsdGeom, UsdSemantics, UsdValidation
from shapely.geometry import shape

from _checks import REPO_ROOT, CheckList, IMAGES

from nyc_drone_sim.common import for_location, load_config, scene_dir, slugify


def mesh_arrays(prim):
    m = UsdGeom.Mesh(prim)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
    assert (cnt == 3).all(), "non-triangle face"
    return pts, idx.reshape(-1, 3)


def signed_volume(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def raster_usd(meshes, half, res=1.0):
    """Top-down raster by pixel-centre sampling of upward-facing triangles. Returns height map
    (NaN where empty)."""
    n = int(round(2 * half / res))
    img = np.full((n, n), np.nan)
    for pts, tris in meshes:
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        nz = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        for t in np.nonzero(nz > 1e-9)[0]:
            A, B, C = a[t], b[t], c[t]
            x0 = max(int(np.floor((min(A[0], B[0], C[0]) + half) / res)), 0)
            x1 = min(int(np.ceil((max(A[0], B[0], C[0]) + half) / res)), n)
            y0 = max(int(np.floor((min(A[1], B[1], C[1]) + half) / res)), 0)
            y1 = min(int(np.ceil((max(A[1], B[1], C[1]) + half) / res)), n)
            if x1 <= x0 or y1 <= y0:
                continue
            xs = (np.arange(x0, x1) + 0.5) * res - half
            ys = (np.arange(y0, y1) + 0.5) * res - half
            X, Y = np.meshgrid(xs, ys)
            w0 = (B[0] - A[0]) * (Y - A[1]) - (B[1] - A[1]) * (X - A[0])
            w1 = (C[0] - B[0]) * (Y - B[1]) - (C[1] - B[1]) * (X - B[0])
            w2 = (A[0] - C[0]) * (Y - C[1]) - (A[1] - C[1]) * (X - C[0])
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            sub = img[y0:y1, x0:x1]
            sub[inside] = np.fmax(sub[inside], A[2])
    return img


def raster_footprints(geoms, heights, half, res=1.0):
    n = int(round(2 * half / res))
    img = np.full((n, n), np.nan)
    for g, h in zip(geoms, heights):
        minx, miny, maxx, maxy = g.bounds
        x0 = max(int(np.floor((minx + half) / res)), 0)
        x1 = min(int(np.ceil((maxx + half) / res)), n)
        y0 = max(int(np.floor((miny + half) / res)), 0)
        y1 = min(int(np.ceil((maxy + half) / res)), n)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = (np.arange(x0, x1) + 0.5) * res - half
        ys = (np.arange(y0, y1) + 0.5) * res - half
        X, Y = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(g, X, Y)
        sub = img[y0:y1, x0:x1]
        sub[inside] = np.fmax(sub[inside], h)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    sdir = scene_dir(cfg, slug)
    g = CheckList("scene", slug)
    R = float(cfg["scene_radius_m"])

    usd_path = sdir / "scene.usdc"
    stage = Usd.Stage.Open(str(usd_path))
    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    s1 = {f"B_{f['properties']['doitt_id']}": f for f in feats}
    bprims = [p for p in stage.GetPrimAtPath("/World/Buildings").GetChildren() if p.IsA(UsdGeom.Mesh)]
    arrays = {p.GetName(): mesh_arrays(p) for p in bprims}
    state = {}

    def g21():
        mpu, up = UsdGeom.GetStageMetersPerUnit(stage), UsdGeom.GetStageUpAxis(stage)
        return mpu == 1.0 and up == UsdGeom.Tokens.z, f"metersPerUnit={mpu}, upAxis={up}"

    def g22():
        reg = UsdValidation.ValidationRegistry()
        validators = reg.GetOrLoadAllValidators()
        errs = UsdValidation.ValidationContext(validators).Validate(stage)
        by = {}
        for e in errs:
            by.setdefault(e.GetType().name, []).append(e.GetErrorAsString())
        n_err = len(by.get("Error", []))
        return n_err == 0, (f"UsdValidation (successor of UsdUtils.ComplianceChecker, which usd-core "
                            f"{'.'.join(map(str, Usd.GetVersion()))} no longer ships): {len(validators)} validators, "
                            f"{n_err} errors, {len(by.get('Warn', []))} warnings"
                            + (f"; first errors: {by['Error'][:3]}" if n_err else "")
                            + (f"; first warnings: {by['Warn'][:2]}" if by.get("Warn") else ""))

    def g23():
        kept = json.loads((sdir / "location.json").read_text())["stats"]["kept"]
        return len(bprims) == kept == len(feats), f"USD building meshes {len(bprims)}, data step kept {kept}"

    def g24():
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        worst_lo = worst_hi = 0.0
        for p in bprims:
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            h = p.GetAttribute("height_m").Get()
            worst_lo = max(worst_lo, abs(r.GetMin()[2]))
            worst_hi = max(worst_hi, abs(r.GetMax()[2] - h))
        return worst_lo <= 0.01 and worst_hi <= 0.01, (
            f"over {len(bprims)} meshes: max |zmin - 0| = {worst_lo:.2e} m, max |zmax - height_m| = {worst_hi:.2e} m")

    def g25():
        rng = np.random.default_rng(cfg["seed"])
        names = sorted(arrays)
        pick = rng.choice(len(names), size=min(50, len(names)), replace=False)
        worst, worst_name, neg = 0.0, None, 0
        for i in pick:
            nm = names[i]
            pts, tris = arrays[nm]
            vol = signed_volume(pts, tris)
            ref = shape(s1[nm]["geometry"]).area * stage.GetPrimAtPath(f"/World/Buildings/{nm}").GetAttribute("height_m").Get()
            rel = abs(vol - ref) / ref
            neg += vol < 0
            if rel > worst:
                worst, worst_name = rel, nm
        # also report the whole set, not only the 50 sampled
        all_rel = [abs(signed_volume(*arrays[nm]) - shape(s1[nm]["geometry"]).area * s1[nm]["properties"]["height_m"])
                   / (shape(s1[nm]["geometry"]).area * s1[nm]["properties"]["height_m"]) for nm in names]
        return worst <= 0.01 and neg == 0, (
            f"50 random meshes (seed {cfg['seed']}): worst relative volume error {worst:.2e} ({worst_name}), "
            f"negative volumes {neg}; over all {len(names)} meshes worst {max(all_rel):.2e}")

    def g26():
        half = R + 150
        u = raster_usd(arrays.values(), half)
        f = raster_footprints([shape(x["geometry"]) for x in feats],
                              [x["properties"]["height_m"] for x in feats], half)
        state["u"], state["f"], state["half"] = u, f, half
        um, fm = ~np.isnan(u), ~np.isnan(f)
        iou = (um & fm).sum() / (um | fm).sum()
        state["iou"] = iou
        return iou >= 0.98, f"top-down IoU at 1 m/px = {iou:.5f} (USD {um.sum()} px, footprints {fm.sum()} px)"

    def g27():
        q = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default())
        bad, seen = [], {}
        for p in stage.Traverse():
            if not p.IsA(UsdGeom.Mesh):
                continue
            labels = list(q.ComputeUniqueInheritedLabels(p))
            if len(labels) != 1:
                bad.append((str(p.GetPath()), labels))
            else:
                seen[labels[0]] = seen.get(labels[0], 0) + 1
        ok = not bad and seen.get("building") == len(bprims) and seen.get("ground") == 1
        return ok, f"class labels resolved per mesh: {seen}; meshes without exactly one label: {len(bad)} {bad[:3]}"

    def g28():
        # every mesh the renderer draws, including each placed copy of a model; a populated scene
        # stores each model once and draws it many times, so both numbers are given
        def count(prims):
            return sum(len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or []) for p in prims if p.IsA(UsdGeom.Mesh))
        own = count(stage.Traverse())                      # meshes authored once in the scene itself
        per_model = {proto.GetPath(): count(Usd.PrimRange(proto)) for proto in stage.GetPrototypes()}
        copies = [per_model[p.GetPrototype().GetPath()] for p in stage.Traverse() if p.IsInstance()]
        drawn, stored = own + sum(copies), own + sum(per_model.values())
        mb = usd_path.stat().st_size / 1e6
        return True, (f"{drawn} triangles drawn ({stored} stored once, the rest are repeated copies of placed models), "
                      f"scene.usdc {mb:.2f} MB"), drawn > 5_000_000

    def g29():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        u, f, half = state["u"], state["f"], state["half"]
        ext = [-half, half, -half, half]
        vmax = np.nanpercentile(f, 99)
        fig, ax = plt.subplots(1, 3, figsize=(21, 7.5), dpi=110)
        ax[0].imshow(u, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
        ax[0].set_title(f"scene.usdc top-down raster (roof z, m) - {len(bprims)} meshes")
        ax[1].imshow(f, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax)
        ax[1].set_title("footprints from the data step (height_m)")
        um, fm = ~np.isnan(u), ~np.isnan(f)
        diff = np.zeros(u.shape + (3,)) + 1.0
        diff[um & fm] = [0.8, 0.8, 0.8]
        diff[um & ~fm] = [1, 0, 0]
        diff[~um & fm] = [0, 0, 1]
        ax[2].imshow(diff, origin="lower", extent=ext)
        ax[2].set_title(f"agreement (grey), USD only (red), data step only (blue) - IoU {state['iou']:.4f}")
        for x in ax:
            x.plot(0, 0, "r*", ms=12, mec="k")
            x.set_xlabel("x east (m)")
        ax[0].set_ylabel("y north (m)")
        out = IMAGES / f"scene_{slug}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out.stat().st_size > 10_000, f"{out.relative_to(REPO_ROOT)} ({out.stat().st_size} bytes)"

    for gid, fn in [("scene-1", g21), ("scene-2", g22), ("scene-3", g23), ("scene-4", g24), ("scene-5", g25),
                    ("scene-6", g26), ("scene-7", g27), ("scene-8", g28), ("scene-9", g29)]:
        g.run(gid, fn)
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
