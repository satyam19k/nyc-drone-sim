"""Data checks: location to open data."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape

from _checks import REPO_ROOT, CheckList, IMAGES, shown

from nyc_drone_sim import geo
from nyc_drone_sim.common import extent_polygon, for_location, FT_TO_M, load_config, scene_dir, slugify

FIXTURE = {"query": "120 Broadway", "bin": "1001026", "lon": -74.0105, "lat": 40.7082}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default=FIXTURE["query"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = for_location(load_config(a.config), a.location)
    slug = slugify(a.location)
    sdir = scene_dir(cfg, slug)
    g = CheckList("data", slug)
    R = float(cfg["scene_radius_m"])

    feats = json.loads((sdir / "buildings.geojson").read_text())["features"]
    loc = json.loads((sdir / "location.json").read_text())
    geoms = [shape(f["geometry"]) for f in feats]
    props = [f["properties"] for f in feats]
    http = geo.HttpCache(REPO_ROOT / cfg["data"]["cache_dir"], offline=True)

    def g11():
        fx = geo.geocode(FIXTURE["query"], cfg, http)
        ok = (fx["bin"] == FIXTURE["bin"] and abs(fx["lon"] - FIXTURE["lon"]) < 1e-3
              and abs(fx["lat"] - FIXTURE["lat"]) < 1e-3)
        if slug != slugify(FIXTURE["query"]):
            return ok, f"fixture geocode bin={fx['bin']} lon={fx['lon']} lat={fx['lat']}; footprint part skipped (location is not the fixture)"
        d = [gm.distance(Point(0, 0)) for gm, p in zip(geoms, props) if p["bin"] == FIXTURE["bin"]]
        ok = ok and bool(d) and min(d) <= 100
        return ok, f"bin={fx['bin']} lon={fx['lon']} lat={fx['lat']}; footprint(s) with that bin: {len(d)}, nearest {min(d) if d else None:.2f} m from point"

    def g12():
        raw, count = geo.fetch_buildings(loc["where"], cfg, http)
        return len(raw) == count, f"rows fetched {len(raw)}, server count(*) {count}"

    def g13():
        raw, _ = geo.fetch_buildings(loc["where"], cfg, http)
        missing = {k: sum(1 for f in raw if k not in f["properties"]) for k in geo.REQUIRED_FIELDS}
        no_geom = sum(1 for f in raw if not f.get("geometry"))
        st = loc["stats"]
        eligible = st["fetched"] - st.get("dropped_feature_code", 0) - st.get("duplicate_objectid", 0)
        bad = st.get("invalid_after_repair", 0) + st.get("null_geometry", 0)
        share = 1 - bad / eligible
        out_valid = all(gm.is_valid and not gm.is_empty for gm in geoms)
        ok = not any(missing.values()) and no_geom == 0 and share >= 0.99 and out_valid
        return ok, (f"missing-field rows {missing}; null geometries {no_geom}; valid after repair "
                    f"{share:.4%} ({st.get('repaired', 0)} repaired); all {len(geoms)} output geometries valid: {out_valid}")

    def g14():
        real = [p for p in props if not p["height_imputed"]]
        imp = [p for p in props if p["height_imputed"]]
        err = max(abs(p["height_m"] - FT_TO_M * p["height_ft_raw"]) for p in real)
        imp_ok = all((not p["height_ft_raw"]) and p["height_m"] >= cfg["imputation"]["floor_m"] for p in imp)
        h = np.array([p["height_m"] for p in props])
        med, mx = float(np.median(h)), float(h.max())
        ok = err <= 1e-6 and imp_ok and 3 <= med <= 100 and mx < 500
        return ok, (f"max |height_m - 0.3048*height_ft_raw| = {err:.2e} over {len(real)} measured rows; "
                    f"{len(imp)} imputed rows have no raw height and are >= floor: {imp_ok}; "
                    f"median {med:.2f} m, max {mx:.2f} m")

    def g15():
        far = max(float(np.hypot(*np.asarray(pg.exterior.coords).T).max())
                  for gm in geoms for pg in gm.geoms)
        if not cfg.get("corridor"):
            return far <= R + 150, f"farthest vertex {far:.1f} m, limit {R + 150:g} m"
        # a corridor scene: measure from the corridor itself, not from the circle that encloses it
        ext = extent_polygon(cfg)
        out = max(ext.distance(Point(*c)) for gm in geoms for pg in gm.geoms for c in pg.exterior.coords)
        return out <= 150, f"farthest vertex {out:.1f} m outside the scene corridor, limit 150 m ({far:.1f} m from the origin)"

    def g16():
        share = sum(p["height_imputed"] for p in props) / len(props)
        return share <= 0.40, f"imputed share {share:.2%} ({sum(p['height_imputed'] for p in props)} of {len(props)})", share > 0.10

    def g17():
        with tempfile.TemporaryDirectory() as td:
            geo.fetch_location(a.location, cfg, out_dir=Path(td), offline=True, evidence=False)
            a_b = (Path(td) / "buildings.geojson").read_bytes()
        b_b = (sdir / "buildings.geojson").read_bytes()
        return a_b == b_b, f"offline rerun from cache byte-identical: {a_b == b_b} ({len(b_b)} bytes)"

    def g19():
        # The fetch must return footprints that straddle the query box, and the box must hold
        # the whole scene circle. within_box alone drops straddling footprints.
        from pyproj import Transformer
        b = loc["bbox"]
        fwd = Transformer.from_crs("EPSG:4326", loc["projection"], always_xy=True)
        xs, ys = fwd.transform([b["west"], b["east"], b["west"], b["east"]],
                               [b["south"], b["south"], b["north"], b["north"]])
        inner = min(min(abs(v) for v in xs), min(abs(v) for v in ys))
        raw, _ = geo.fetch_buildings(loc["where"], cfg, http)
        ok = loc["where"].startswith("intersects(") and inner >= R and len(raw) >= loc["within_box_count"]
        return ok, (f"filter is intersects over a box reaching at least {inner:.0f} m from the target (radius {R:g} m); "
                    f"{len(raw)} rows against {loc['within_box_count']} from within_box, so "
                    f"{len(raw) - loc['within_box_count']} straddling footprints were recovered")

    def g18():
        p = IMAGES / f"data_{slug}.png"
        return p.exists() and p.stat().st_size > 10_000, f"{shown(p)} ({p.stat().st_size} bytes)"

    for gid, fn in [("data-1", g11), ("data-2", g12), ("data-3", g13), ("data-4", g14),
                    ("data-5", g15), ("data-6", g16), ("data-7", g17), ("data-8", g18), ("data-9", g19)]:
        g.run(gid, fn)
    sys.exit(g.finish())


if __name__ == "__main__":
    main()
