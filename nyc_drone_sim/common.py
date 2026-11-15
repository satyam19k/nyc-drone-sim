"""Shared helpers: config loading, paths, slugs. CPU only."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
FT_TO_M = 0.3048


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    if not s:
        raise ValueError(f"cannot build a location slug from {text!r}")
    return s


def scene_dir(cfg: dict, location_slug: str) -> Path:
    return REPO_ROOT / cfg["output"]["scenes_dir"] / location_slug


def altitude_m(alt_ft: float) -> float:
    return float(alt_ft) * FT_TO_M


def clip_settings(cfg: dict, variant: str | None = None) -> tuple[int, float]:
    """(frames, fps) for the default clip or for a named path variant such as "street"."""
    frames = cfg[variant]["frames"] if variant else cfg["clip"]["frames"]
    return int(frames), float(cfg["clip"]["fps"])


def variant_suffix(variant: str | None, view: str | None = None) -> str:
    """File-name suffix for a path variant, and for one named camera view of a place."""
    return (f"_{variant}" if variant else "") + (f"_{view}" if view else "")


def view_spec(cfg: dict, view: str | None) -> dict:
    """Route, view offset and focus of a place's main route, or of one of its extra `views`."""
    place = cfg["place"]
    if not view:
        return place
    if view not in (place.get("views") or {}):
        raise KeyError(f"place {place['name']} has no view named {view!r}")
    return {**place, **place["views"][view]}


def for_location(cfg: dict, location: str) -> dict:
    """Config for one location. A name listed under `places` gets its own origin and extent.

    A place is either a circle (`radius_m`) or a corridor: `route_m`, a camera ground track in
    local metres from the origin, buffered by `buffer_m`. `scene_radius_m` then becomes the
    radius of the smallest origin-centred circle that holds the extent, so code that only needs
    a bound keeps working; code that clips uses `extent_polygon`.
    """
    place = (cfg.get("places") or {}).get(location)
    if place is None:
        return cfg
    out = dict(cfg)
    out["place"] = {**place, "name": location}
    if "motion_frames" in place:      # recorded motion only needs to last as long as the place's clips
        out["populate"] = {**cfg["populate"], "frames": int(place["motion_frames"])}
    if "route_m" in place:
        # No geometry library here: this also runs inside the renderer's Python, which has none.
        # The farthest point of a buffered polyline from the origin is its farthest vertex plus
        # the buffer.
        out["scene_radius_m"] = max((x * x + y * y) ** 0.5 for x, y in place["route_m"]) + float(place["buffer_m"])
        out["corridor"] = True
    else:
        out["scene_radius_m"] = float(place["radius_m"])
    return out


def location_query(cfg: dict, location: str) -> str:
    """What to geocode: a place's origin as "lat,lon", or the location text itself."""
    place = cfg.get("place")
    return f"{place['origin'][0]:.7f},{place['origin'][1]:.7f}" if place else location


def extent_polygon(cfg: dict):
    """The scene extent in local metres: the place's corridor, or the scene circle."""
    from shapely.geometry import LineString, Point
    if cfg.get("corridor"):
        place = cfg["place"]
        return LineString(place["route_m"]).buffer(float(place["buffer_m"]), quad_segs=16)
    return Point(0, 0).buffer(float(cfg["scene_radius_m"]), quad_segs=256)
