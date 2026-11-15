"""Street-level layers from NYC Open Data, in the same local frame as the buildings. CPU only.

Fetches ground surfaces (roadbed, sidewalk, median, plaza, park, parking lot, water), street
centre lines, tree points and street-furniture points for the location that Stage 1 already
resolved. Every layer is projected with the projection string stored in location.json, so it
cannot drift from the buildings. Uses the same response cache as Stage 1.
"""
from __future__ import annotations

import argparse
import json
import logging
import re

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Point, Polygon, mapping, shape
from shapely.ops import unary_union

from .common import REPO_ROOT, extent_polygon, for_location, load_config, scene_dir, slugify
from .geo import HttpCache

log = logging.getLogger("layers")


def box_wkt(b: dict) -> str:
    w, e, s, n = (f"{b[k]:.7f}" for k in ("west", "east", "south", "north"))
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def fetch(dataset: str, where: str, cfg: dict, http: HttpCache) -> list[dict]:
    """All rows of one dataset matching `where`, as GeoJSON features, paged to exhaustion."""
    base = cfg["layers"]["resource_url"] + dataset
    n = int(cfg["data"]["page_size"])
    count = int(http.get_json(base + ".json", {"$select": "count(*)", "$where": where})[0]["count"])
    feats, offset = [], 0
    while True:
        page = http.get_json(base + ".geojson", {"$where": where, "$order": ":id", "$limit": n, "$offset": offset})
        if page.get("type") != "FeatureCollection":
            raise RuntimeError(f"{dataset}: unexpected response shape: {str(page)[:200]}")
        got = page.get("features", [])
        feats.extend(got)
        if len(got) < n:
            break
        offset += n
    if len(feats) != count:
        raise RuntimeError(f"{dataset}: fetched {len(feats)} rows, server count {count}")
    return feats


class Frame:
    """Projection and scene extent shared by every layer.

    Some city layers are published in NAD83 coordinates while the building footprints are in
    WGS84; at New York the two differ by about one metre. Layers marked `datum: nad83` in the
    config get the standard datum transformation first, then the same local projection as
    everything else.
    """

    def __init__(self, loc: dict, cfg: dict):
        self.fwd = Transformer.from_crs("EPSG:4326", loc["projection"], always_xy=True)
        self.circle = extent_polygon(cfg)
        self.radius = float(cfg["scene_radius_m"])
        src, dst = cfg["layers"]["nad83_to_wgs84"]
        self.datum = Transformer.from_crs(src, dst, always_xy=True)

    def local(self, geom, datum: str | None = None):
        def f(a):
            lon, lat = a[:, 0], a[:, 1]
            if datum == "nad83":
                lon, lat = self.datum.transform(lon, lat)
            return np.column_stack(self.fwd.transform(lon, lat))
        return shapely.transform(geom, f)


def polygons(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) for p in polygons(p)]


def clean(g, min_area: float):
    g = shapely.make_valid(g)
    ps = [p for p in polygons(g) if p.area >= min_area]
    return MultiPolygon(ps) if ps else None


def round_geom(g, nd: int):
    return shapely.set_precision(g, 10.0 ** -nd)


def park_interior(poly: Polygon, pk: dict) -> tuple[list[Polygon], list[Polygon]]:
    """Split one park polygon into lawn and paths by rule: no city layer maps park paths here.

    Paths are a walk just inside the park edge plus straight walks from an interior point to the
    edge, at evenly spaced bearings whose rotation depends only on the park's own position.
    """
    if poly.area < float(pk["min_area_m2"]):
        return [poly], []
    w = float(pk["path_width_m"])
    inner = poly.buffer(-float(pk["edge_inset_m"]))
    parts = []
    if not inner.is_empty:
        parts.append(inner.boundary.buffer(w / 2))
    c = poly.representative_point()
    reach = float(np.hypot(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]))
    n = int(pk["spokes"])
    turn = (abs(c.x) * 7.0 + abs(c.y) * 13.0) % (360.0 / n)
    for k in range(n):
        a = np.radians(turn + k * 360.0 / n)
        parts.append(shapely.LineString([(c.x, c.y), (c.x + reach * np.cos(a), c.y + reach * np.sin(a))]).buffer(w / 2))
    paths = unary_union(parts).intersection(poly)
    return polygons(poly.difference(paths)), polygons(paths)


def grade_masks(lines: list[dict], cfg: dict):
    """Buffers around centre lines that are off ground level, and around those at ground level."""
    grade = int(cfg["populate"]["ground_level_code"])
    r = float(cfg["layers"]["off_grade_buffer_m"])
    off, on = [], []
    for f in lines:
        p = f["properties"]
        ln = shapely.LineString(np.asarray(f["geometry"]["coordinates"])[:, :2]).buffer(r)
        a, b = p.get("from_level") or grade, p.get("to_level") or grade
        (on if a == grade and b == grade else off).append(ln)
    return (unary_union(off) if off else None), (unary_union(on) if on else None)


def build_surfaces(cfg: dict, loc: dict, http: HttpCache, frame: Frame, buildings=(), lines=()) -> tuple[list[dict], dict]:
    """Non-overlapping surface polygons, clipped to the scene circle.

    Where two source layers overlap, the class listed first in `layers.surface_order` wins.
    """
    lc = cfg["layers"]
    where = "intersects({geom}, '%s')" % box_wkt(loc["bbox"])
    raw: dict[str, list] = {}
    stats = {}
    covered = unary_union(list(buildings)) if len(buildings) else None
    if covered is not None:
        shapely.prepare(covered)
    off_grade, at_grade = grade_masks(list(lines), cfg) if len(lines) else (None, None)
    for spec in lc["surfaces"]:
        feats = fetch(spec["dataset"], where.format(geom=spec.get("geom", "the_geom")), cfg, http)
        keep, n_covered, n_off, a_off = [], 0, 0, 0.0
        for f in feats:
            if f.get("geometry") is None:
                continue
            if spec.get("where_field") and str(f["properties"].get(spec["where_field"])) not in [str(v) for v in spec["where_in"]]:
                continue
            g = clean(frame.local(shape(f["geometry"]), spec.get("datum")).intersection(frame.circle), float(lc["min_area_m2"]))
            if g is None:
                continue
            # a piece lying almost wholly under a building is roadway or walkway inside a structure
            # (a garage deck, a terminal concourse): it cannot be seen and is not street surface
            if covered is not None and spec["class"] != "water":
                parts = [p for p in polygons(g) if p.intersection(covered).area < float(lc["covered_share_drop"]) * p.area]
                n_covered += len(polygons(g)) - len(parts)
                g = MultiPolygon(parts) if parts else None
            # a road piece that follows a tunnel, viaduct or ramp centre line and no street at
            # ground level is that structure's roadway, mapped although it is under or over the
            # ground: the flat scene must not draw it as a street
            if g is not None and spec["class"] == "road" and off_grade is not None:
                parts = []
                for p in polygons(g):
                    share_off = p.intersection(off_grade).area / p.area
                    share_on = p.intersection(at_grade).area / p.area if at_grade is not None else 0.0
                    if share_off >= float(lc["off_grade_share_drop"]) and share_on < float(lc["at_grade_share_keep"]):
                        n_off += 1
                        a_off += p.area
                    else:
                        parts.append(p)
                g = MultiPolygon(parts) if parts else None
            if g is not None:
                keep.append(g)
        raw.setdefault(spec["class"], []).extend(keep)
        stats[spec["dataset"]] = {"class": spec["class"], "rows": len(feats), "kept": len(keep), "pieces_under_buildings_dropped": n_covered,
                                  "off_grade_pieces_dropped": n_off, "off_grade_area_dropped_m2": round(a_off, 1)}
    # The city's water polygon ends at the city line in mid-river. Whatever lies beyond the land,
    # out to the edge of the scene, is open water too: close the land mass over its small gaps
    # and call the rest of the extent water.
    if raw.get("water"):
        land = unary_union([g for c, gs in raw.items() if c != "water" for g in gs] + list(buildings))
        closed = land.buffer(float(lc["water_fill_close_m"])).buffer(-float(lc["water_fill_close_m"]))
        fill = clean(frame.circle.difference(closed).difference(unary_union(raw["water"])), float(lc["water_fill_min_m2"]))
        if fill is not None:
            near = [p for p in polygons(fill) if p.distance(unary_union(raw["water"])) < 1.0]
            raw["water"].extend(near)
            stats["water_fill_m2"] = round(sum(p.area for p in near), 1)
    out, taken = [], None
    raw_union = {}
    for cls in lc["surface_order"]:
        if not raw.get(cls):
            continue
        u = unary_union(raw[cls])
        raw_union[cls] = u
        g = u if taken is None else u.difference(taken)
        taken = u if taken is None else unary_union([taken, u])
        g = clean(round_geom(g, int(cfg["data"]["coord_decimals"])), float(lc["min_area_m2"]))
        if g is None:
            continue
        ordered = sorted(polygons(g), key=lambda q: (round(q.centroid.x, 3), round(q.centroid.y, 3), round(q.area, 3)))
        groups = {cls: ordered}
        if cls == "park":
            # a park becomes lawn and paths
            groups = {"lawn": [], "path": []}
            for p in ordered:
                lawn, path = park_interior(p, lc["park"])
                groups["lawn"] += [q for q in lawn if q.area >= float(lc["min_area_m2"])]
                groups["path"] += [q for q in path if q.area >= float(lc["min_area_m2"])]
        for name, ps in groups.items():
            for k, p in enumerate(ps):
                p = shapely.make_valid(round_geom(p, int(cfg["data"]["coord_decimals"])))
                for q in polygons(p):
                    out.append({"type": "Feature", "properties": {"class": name, "id": f"{name}_{k:04d}", "area_m2": round(q.area, 2)},
                                "geometry": mapping(q)})
            stats.setdefault("area_m2", {})[name] = round(sum(p.area for p in ps), 1)
    stats["raw_area_m2"] = {k: round(v.area, 1) for k, v in raw_union.items()}
    return out, stats, raw_union


def build_centerlines(cfg: dict, loc: dict, http: HttpCache, frame: Frame) -> list[dict]:
    lc = cfg["layers"]
    feats = fetch(lc["centerline"]["dataset"], "intersects(the_geom, '%s')" % box_wkt(loc["bbox"]), cfg, http)
    out = []
    for f in feats:
        p = f["properties"]
        if f.get("geometry") is None:
            continue
        g = frame.local(shape(f["geometry"]), lc["centerline"].get("datum"))
        for line in getattr(g, "geoms", [g]):
            if line.length < 1.0 or not line.intersects(frame.circle):
                continue

            def num(k):
                try:
                    return int(float(p.get(k)))
                except (TypeError, ValueError):
                    return None
            out.append({"type": "Feature", "geometry": mapping(round_geom(line, int(cfg["data"]["coord_decimals"]))),
                        "properties": {"physicalid": p.get("physicalid"), "street": p.get("full_street_name"),
                                       "rw_type": num("rw_type"), "trafdir": p.get("trafdir"),
                                       "travel_lanes": num("number_travel_lanes"), "park_lanes": num("number_park_lanes"),
                                       "total_lanes": num("number_total_lanes"), "posted_speed_mph": num("posted_speed"),
                                       "street_width_ft": num("streetwidth"), "status": p.get("status"),
                                       "from_level": num("from_level_code"), "to_level": num("to_level_code")}})
    out.sort(key=lambda f: (str(f["properties"]["physicalid"]), f["geometry"]["coordinates"][0]))
    return out


def build_trees(cfg: dict, loc: dict, http: HttpCache, frame: Frame) -> list[dict]:
    tc = cfg["layers"]["trees"]
    feats = fetch(tc["dataset"], "intersects(%s, '%s')" % (tc["geom"], box_wkt(loc["bbox"])), cfg, http)
    out = []
    for f in feats:
        p = f["properties"]
        if f.get("geometry") is None or p.get("tpstructure") not in tc["keep_structure"] \
                or p.get("tpcondition") in tc["drop_condition"]:
            continue
        pt = frame.local(shape(f["geometry"]), tc.get("datum"))
        if not frame.circle.contains(pt):
            continue
        try:
            dbh_in = float(p.get("dbh"))
        except (TypeError, ValueError):
            dbh_in = None
        out.append({"id": str(p.get("objectid")), "x": round(pt.x, 3), "y": round(pt.y, 3),
                    "dbh_in": dbh_in, "species": p.get("genusspecies") or ""})
    out.sort(key=lambda t: t["id"])
    return out


def build_furniture(cfg: dict, loc: dict, http: HttpCache, frame: Frame) -> list[dict]:
    b = loc["bbox"]
    out = []
    for spec in cfg["layers"]["furniture"]:
        if spec.get("geom"):
            where = "intersects(%s, '%s')" % (spec["geom"], box_wkt(b))
        else:
            where = (f"latitude between {b['south']:.7f} and {b['north']:.7f} and "
                     f"longitude between {b['west']:.7f} and {b['east']:.7f}")
        if spec.get("where"):
            where += f" and {spec['where']}"
        for f in fetch(spec["dataset"], where, cfg, http):
            p = f["properties"]
            if f.get("geometry") is not None:
                g = shape(f["geometry"])
                g = g if g.geom_type == "Point" else g.centroid
            else:
                g = Point(float(p["longitude"]), float(p["latitude"]))
            pt = frame.local(g, spec.get("datum"))
            if frame.circle.contains(pt):
                ident = next((str(p[k]) for k in spec["id_fields"] if p.get(k) is not None), "")
                out.append({"kind": spec["kind"], "id": ident, "x": round(pt.x, 3), "y": round(pt.y, 3)})
    out.sort(key=lambda r: (r["kind"], r["id"], r["x"], r["y"]))
    return out


def run_layers(location: str, cfg: dict, offline: bool = False) -> dict:
    slug = slugify(location)
    cfg = for_location(cfg, location)
    sdir = scene_dir(cfg, slug)
    loc = json.loads((sdir / "location.json").read_text())
    http = HttpCache(REPO_ROOT / cfg["data"]["cache_dir"], offline=offline, timeout_s=float(cfg["data"]["http_timeout_s"]))
    frame = Frame(loc, cfg)
    blds = [shape(f["geometry"]) for f in json.loads((sdir / "buildings.geojson").read_text())["features"]]
    lines = build_centerlines(cfg, loc, http, frame)
    surfaces, stats, raw_union = build_surfaces(cfg, loc, http, frame, blds, lines)
    trees = build_trees(cfg, loc, http, frame)
    furniture = build_furniture(cfg, loc, http, frame)
    # structures that carry roads above the ground: bridges, viaducts, overpasses. The scene is
    # flat, so these are kept as a mask to steer cameras away from, not built
    elevated = []
    for spec in cfg["layers"].get("elevated", []):
        for f in fetch(spec["dataset"], "intersects(the_geom, '%s')" % box_wkt(loc["bbox"]), cfg, http):
            if f.get("geometry") is None:
                continue
            g = clean(frame.local(shape(f["geometry"]), spec.get("datum")).intersection(frame.circle), 1.0)
            if g is not None:
                elevated.append({"type": "Feature", "geometry": mapping(round_geom(g, 3)),
                                 "properties": {"name": f["properties"].get("name"), "feat_code": f["properties"].get("feat_code")}})

    # alignment evidence: how much raw road and sidewalk area falls on building footprints
    bu = unary_union(blds)
    overlap = {k: round(raw_union[k].intersection(bu).area, 1) for k in ("road", "sidewalk") if k in raw_union}

    def dump(name, obj):
        (sdir / name).write_text(json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n")

    dump("surfaces.geojson", {"type": "FeatureCollection", "features": surfaces})
    dump("centerlines.geojson", {"type": "FeatureCollection", "features": lines})
    dump("trees.json", trees)
    dump("furniture.json", furniture)
    dump("elevated.geojson", {"type": "FeatureCollection", "features": elevated})
    meta = {"location": slug, "projection": loc["projection"], "scene_radius_m": frame.radius,
            "surfaces": stats, "centerlines": len(lines), "trees": len(trees), "elevated_structures": len(elevated),
            "furniture": {k: sum(1 for r in furniture if r["kind"] == k) for k in sorted({r["kind"] for r in furniture})},
            "building_area_m2": round(bu.area, 1), "raw_overlap_with_buildings_m2": overlap,
            "fetched_utc": sorted({u["fetched_utc"][:10] for u in http.used}),
            "requests": len(http.used)}
    (sdir / "layers_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    log.info("surfaces %s; %d centre lines, %d trees, furniture %s", stats.get("area_m2"), len(lines), len(trees), meta["furniture"])
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch street-level layers for a location")
    ap.add_argument("--location", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run_layers(a.location, load_config(a.config), offline=a.offline)


if __name__ == "__main__":
    main()
