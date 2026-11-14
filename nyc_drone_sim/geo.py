"""Stage 1: location to open data. CPU only; must not import isaacsim.

geocode -> bounding box -> paged fetch of NYC BUILDING (5zhs-2jue) -> local metres
-> height normalisation and imputation -> buildings.geojson.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.geometry.polygon import orient

from .common import FT_TO_M, REPO_ROOT, load_config, scene_dir, slugify

log = logging.getLogger("stage1")

REQUIRED_FIELDS = [
    "height_roof", "ground_elevation", "feature_code", "bin", "doitt_id",
    "construction_year", "last_status_type", "base_bbl", "mappluto_bbl", "geom_source",
]
LATLON_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


# ---------------------------------------------------------------------------
# HTTP with a content-addressed cache
# ---------------------------------------------------------------------------
class HttpCache:
    """Caches every response body under data/cache/ with URL, UTC time and SHA-256.

    A cache hit never touches the network, so reruns work offline.
    """

    def __init__(self, cache_dir: Path, offline: bool = False, timeout_s: float = 120):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.timeout_s = timeout_s
        self.used: list[dict] = []

    @staticmethod
    def full_url(url: str, params: dict | None) -> str:
        return requests.Request("GET", url, params=params).prepare().url

    def get(self, url: str, params: dict | None = None) -> bytes:
        full = self.full_url(url, params)
        key = hashlib.sha256(full.encode()).hexdigest()
        body_p = self.dir / f"{key}.body"
        meta_p = self.dir / f"{key}.meta.json"
        if body_p.exists() and meta_p.exists():
            body = body_p.read_bytes()
            meta = json.loads(meta_p.read_text())
            if hashlib.sha256(body).hexdigest() != meta["sha256"]:
                raise RuntimeError(f"cache corrupt for {full}: sha256 mismatch")
        else:
            if self.offline:
                raise RuntimeError(f"offline and not cached: {full}")
            log.info("GET %s", full)
            r = requests.get(full, timeout=self.timeout_s)
            r.raise_for_status()
            body = r.content
            meta = {
                "url": full,
                "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(body).hexdigest(),
                "bytes": len(body),
            }
            body_p.write_bytes(body)
            meta_p.write_text(json.dumps(meta, indent=1, sort_keys=True))
        self.used.append({**meta, "cache_file": str(body_p.relative_to(REPO_ROOT))})
        return body

    def get_json(self, url: str, params: dict | None = None):
        return json.loads(self.get(url, params))


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
def geocode(text: str, cfg: dict, http: HttpCache) -> dict:
    m = LATLON_RE.match(text)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (40.4 < lat < 41.0 and -74.3 < lon < -73.6):
            raise ValueError(f"{text!r} parsed as lat={lat}, lon={lon}, which is not in NYC")
        return {"query": text, "source": "latlon", "lat": lat, "lon": lon,
                "label": text, "borough": None, "bin": None, "bbl": None}

    data = http.get_json(cfg["data"]["geosearch_url"], {"text": text, "size": 1})
    feats = data.get("features") or []
    if not feats:
        raise ValueError(f"GeoSearch returned no feature for {text!r}")
    f = feats[0]
    props = f.get("properties", {})
    if not props.get("borough"):
        raise ValueError(f"GeoSearch result for {text!r} has empty borough: {props.get('label')}")
    lon, lat = f["geometry"]["coordinates"]  # GeoJSON order: [lon, lat]
    pad = (props.get("addendum") or {}).get("pad") or {}
    return {"query": text, "source": "geosearch", "lat": float(lat), "lon": float(lon),
            "label": props.get("label"), "borough": props.get("borough"),
            "bin": pad.get("bin"), "bbl": pad.get("bbl")}


def local_proj(lat: float, lon: float) -> str:
    return f"+proj=aeqd +lat_0={lat:.9f} +lon_0={lon:.9f} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def bbox_latlon(lat: float, lon: float, half_m: float) -> dict:
    inv = Transformer.from_crs(local_proj(lat, lon), "EPSG:4326", always_xy=True)
    xs = np.array([-half_m, 0, half_m, half_m, half_m, 0, -half_m, -half_m])
    ys = np.array([-half_m, -half_m, -half_m, 0, half_m, half_m, half_m, 0])
    lons, lats = inv.transform(xs, ys)
    return {"north": float(np.max(lats)), "south": float(np.min(lats)),
            "west": float(np.min(lons)), "east": float(np.max(lons))}


# ---------------------------------------------------------------------------
# Building fetch
# ---------------------------------------------------------------------------
def within_box_clause(b: dict) -> str:
    # Socrata within_box takes latitude first: (north, west, south, east)
    return f"within_box(the_geom, {b['north']:.7f}, {b['west']:.7f}, {b['south']:.7f}, {b['east']:.7f})"


def fetch_buildings(where: str, cfg: dict, http: HttpCache) -> tuple[list[dict], int]:
    base = cfg["data"]["buildings_url"]
    n = int(cfg["data"]["page_size"])
    count = int(http.get_json(base + ".json", {"$select": "count(*)", "$where": where})[0]["count"])
    feats: list[dict] = []
    offset = 0
    while True:
        page = http.get_json(base + ".geojson", {"$where": where, "$order": "objectid",
                                                 "$limit": n, "$offset": offset})
        if page.get("type") != "FeatureCollection":
            raise RuntimeError(f"unexpected response shape: {str(page)[:200]}")
        got = page.get("features", [])
        feats.extend(got)
        if len(got) < n:
            break
        offset += n
    return feats, count


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _float_or_none(v):
    if v is None or v == "":
        return None
    return float(v)


def _polygonal(g) -> MultiPolygon | None:
    if g.is_empty:
        return None
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, MultiPolygon):
        return g
    polys = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    flat = []
    for p in polys:
        flat.extend(p.geoms if isinstance(p, MultiPolygon) else [p])
    flat = [p for p in flat if not p.is_empty and p.area > 0]
    return MultiPolygon(flat) if flat else None


def normalise(raw: list[dict], loc: dict, cfg: dict) -> tuple[list[dict], dict]:
    fwd = Transformer.from_crs("EPSG:4326", local_proj(loc["lat"], loc["lon"]), always_xy=True)
    R = float(cfg["scene_radius_m"])
    grid = 10.0 ** -int(cfg["data"]["coord_decimals"])
    circle = Point(0, 0).buffer(R, quad_segs=256)
    drop = {int(c) for c in cfg["data"]["drop_feature_codes"]}
    keep_codes = {int(c) for c in cfg["data"]["keep_feature_codes"]}

    stats = Counter()
    code_counts_all, code_counts_kept = Counter(), Counter()
    fields_seen = Counter()
    seen_obj = set()
    rows = []
    for f in raw:
        p = f.get("properties") or {}
        for k in REQUIRED_FIELDS:
            if k in p:
                fields_seen[k] += 1
        oid = int(p["objectid"])
        if oid in seen_obj:
            stats["duplicate_objectid"] += 1
            continue
        seen_obj.add(oid)
        code = int(p["feature_code"])
        code_counts_all[code] += 1
        if code in drop:
            stats["dropped_feature_code"] += 1
            continue
        if code not in keep_codes:
            raise RuntimeError(f"unknown feature_code {code} (objectid {oid}); schema differs from brief")
        if f.get("geometry") is None:
            stats["null_geometry"] += 1
            continue
        g = shapely.transform(
            shape(f["geometry"]),
            lambda a: np.column_stack(fwd.transform(a[:, 0], a[:, 1])))
        valid_before = g.is_valid
        stats["valid_before"] += int(valid_before)
        if not valid_before:
            g = shapely.make_valid(g)
            stats["repaired"] += 1
        g = _polygonal(g)
        if g is not None:
            g = _polygonal(shapely.set_precision(g, grid))
        if g is None or not g.is_valid:
            stats["invalid_after_repair"] += 1
            continue
        if not g.intersects(circle):
            stats["outside_circle"] += 1
            continue
        g = MultiPolygon([orient(pg, sign=1.0) for pg in g.geoms])
        h_ft = _float_or_none(p.get("height_roof"))
        code_counts_kept[code] += 1
        rows.append({
            "geom": g,
            "props": {
                "objectid": oid,
                "doitt_id": int(p["doitt_id"]),
                "bin": p.get("bin"),
                "feature_code": code,
                "height_ft_raw": h_ft,
                "ground_elevation_ft": _float_or_none(p.get("ground_elevation")),
                "construction_year": p.get("construction_year"),
                "last_status_type": p.get("last_status_type"),
                "base_bbl": p.get("base_bbl"),
                "mappluto_bbl": p.get("mappluto_bbl"),
                "geom_source": p.get("geom_source"),
                "geom_repaired": not valid_before,
            },
        })

    # Heights: feet above ground -> metres. 0 or NULL means unknown -> impute.
    known = [r for r in rows if r["props"]["height_ft_raw"] and r["props"]["height_ft_raw"] > 0]
    for r in known:
        r["props"]["height_m"] = r["props"]["height_ft_raw"] * FT_TO_M
        r["props"]["height_imputed"] = False
    unknown = [r for r in rows if not (r["props"]["height_ft_raw"] and r["props"]["height_ft_raw"] > 0)]
    rad = float(cfg["imputation"]["neighbour_radius_m"])
    floor = float(cfg["imputation"]["floor_m"])
    if unknown:
        if known:
            kc = np.array([[r["geom"].centroid.x, r["geom"].centroid.y] for r in known])
            kh = np.array([r["props"]["height_m"] for r in known])
            tree = cKDTree(kc)
        for r in unknown:
            c = r["geom"].centroid
            idx = tree.query_ball_point([c.x, c.y], rad) if known else []
            if idx:
                h = float(np.median(kh[sorted(idx)]))
                src = f"median_{len(idx)}_within_{rad:g}m"
            else:
                h = floor
                src = "floor_no_neighbours"
            r["props"]["height_m"] = max(floor, h)
            r["props"]["height_imputed"] = True
            r["props"]["imputation_source"] = src

    rows.sort(key=lambda r: (r["props"]["doitt_id"], r["props"]["objectid"]))
    stats.update({
        "fetched": len(raw),
        "kept": len(rows),
        "imputed": len(unknown),
        "imputed_share": (len(unknown) / len(rows)) if rows else 0.0,
    })
    report = {
        "stats": dict(stats),
        "feature_codes_fetched": {str(k): v for k, v in sorted(code_counts_all.items())},
        "feature_codes_kept": {str(k): v for k, v in sorted(code_counts_kept.items())},
        "required_fields_seen": {k: fields_seen[k] for k in REQUIRED_FIELDS},
    }
    return rows, report


def to_geojson_bytes(rows: list[dict]) -> bytes:
    lines = ['{"type":"FeatureCollection","crs_local":"aeqd metres, x east, y north","features":[']
    for i, r in enumerate(rows):
        feat = {"type": "Feature", "properties": r["props"], "geometry": mapping(r["geom"])}
        s = json.dumps(feat, sort_keys=True, separators=(",", ":"))
        lines.append(s + ("," if i < len(rows) - 1 else ""))
    lines.append("]}")
    return ("\n".join(lines) + "\n").encode()


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def plot_footprints(rows: list[dict], loc: dict, R: float, out_png: Path, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    polys, heights = [], []
    for r in rows:
        for pg in r["geom"].geoms:
            polys.append(np.asarray(pg.exterior.coords))
            heights.append(r["props"]["height_m"])
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    pc = PolyCollection(polys, array=np.asarray(heights), cmap="viridis", linewidths=0.1,
                        edgecolors="k")
    pc.set_clim(0, np.percentile(heights, 99) if heights else 1)
    ax.add_collection(pc)
    imp = [np.asarray(pg.exterior.coords) for r in rows if r["props"]["height_imputed"]
           for pg in r["geom"].geoms]
    if imp:
        ax.add_collection(PolyCollection(imp, facecolors="none", edgecolors="red", linewidths=0.4))
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * np.cos(t), R * np.sin(t), "r--", lw=1, label=f"scene radius {R:g} m")
    ax.plot(0, 0, marker="*", color="red", ms=16, mec="k", label="target")
    x0, y0 = -R * 0.95, -R * 0.95
    ax.plot([x0, x0 + 100], [y0, y0], "k-", lw=4)
    ax.text(x0 + 50, y0 + 15, "100 m", ha="center")
    ax.set_xlim(-R * 1.05, R * 1.05)
    ax.set_ylim(-R * 1.05, R * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    fig.colorbar(pc, ax=ax, shrink=0.7, label="height (m); red outline = imputed")
    ax.legend(loc="upper right")
    ax.set_title(title or f"{loc.get('label')} - {len(rows)} buildings")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_stage1(location: str, cfg: dict, out_dir: Path | None = None, offline: bool = False,
               evidence: bool = True) -> Path:
    slug = slugify(location)
    out = out_dir or scene_dir(cfg, slug)
    out.mkdir(parents=True, exist_ok=True)
    http = HttpCache(REPO_ROOT / cfg["data"]["cache_dir"], offline=offline,
                     timeout_s=cfg["data"]["http_timeout_s"])
    loc = geocode(location, cfg, http)
    R = float(cfg["scene_radius_m"])
    box = bbox_latlon(loc["lat"], loc["lon"], R + float(cfg["data"]["bbox_margin_m"]))
    where = within_box_clause(box)
    raw, server_count = fetch_buildings(where, cfg, http)
    rows, report = normalise(raw, loc, cfg)
    for code, n in report["feature_codes_kept"].items():
        log.info("feature_code %s: %d kept", code, n)

    (out / "buildings.geojson").write_bytes(to_geojson_bytes(rows))
    location_json = {
        **loc, "slug": slug, "projection": local_proj(loc["lat"], loc["lon"]),
        "scene_radius_m": R, "bbox": box, "where": where,
        "server_count": server_count, "rows_fetched": len(raw), **report,
    }
    (out / "location.json").write_text(json.dumps(location_json, indent=1, sort_keys=True) + "\n")
    (out / "cache_manifest.json").write_text(json.dumps(http.used, indent=1) + "\n")
    if evidence:
        plot_footprints(rows, loc, R, REPO_ROOT / "evidence" / f"stage_1_{slug}.png")
    log.info("stage 1 done: %d fetched, %d kept, %d imputed -> %s",
             len(raw), len(rows), report["stats"]["imputed"], out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 1: location to open data")
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_stage1(a.location, load_config(a.config), offline=a.offline)


if __name__ == "__main__":
    main()
