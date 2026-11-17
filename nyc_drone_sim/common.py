"""Shared helpers: config loading, paths, slugs. CPU only."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
FT_TO_M = 0.3048


def validation_dir() -> Path:
    """Where check reports and pictures go: docs/validation, or NYC_DRONE_VALIDATION_DIR when set
    (the nyc-drone command sets it to the run's own folder, so a run does not change the repository)."""
    import os
    return Path(os.environ.get("NYC_DRONE_VALIDATION_DIR") or REPO_ROOT / "docs" / "validation")


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


ROUTES_DIR = REPO_ROOT / "configs" / "routes"
ROUTE_KEYS = ("altitudes_ft", "seed", "speed_mps", "sync_ground_track", "max_yaw_rate_deg_s", "frames",
              "motion_frames", "buffer_m", "yaw_offset_deg", "focus", "look")


def route_place(route: dict) -> dict:
    """A route file (configs/routes/<name>.yaml) as a place. The origin must be [lat, lon] and the
    waypoints in metres here; nyc_drone_sim.routes.resolve_route converts an address origin and
    waypoints_lonlat first."""
    origin = route["origin"]
    if isinstance(origin, str) or "waypoints_m" not in route:
        raise ValueError("this route has an address origin or waypoints_lonlat: "
                         "fly it with nyc-drone route, which converts them")
    place = {"origin": [float(origin[0]), float(origin[1])], "route_m": route["waypoints_m"],
             **{k: route[k] for k in ROUTE_KEYS if k in route}}
    place.setdefault("focus", {int(a): 0.5 for a in route.get("altitudes_ft", [150, 500])})
    place.setdefault("yaw_offset_deg", 0.0)
    if route.get("views"):
        place["views"] = {name: {"route_m": v["waypoints_m"], **{k: x for k, x in v.items() if k != "waypoints_m"}}
                          for name, v in route["views"].items()}
    return place


def for_location(cfg: dict, location: str) -> dict:
    """Config for one location. A name listed under `places`, or with a route file in
    configs/routes/, gets its own origin and extent.

    A place is either a circle (`radius_m`) or a corridor: `route_m`, a camera ground track in
    local metres from the origin, buffered by `buffer_m`. `scene_radius_m` then becomes the
    radius of the smallest origin-centred circle that holds the extent, so code that only needs
    a bound keeps working; code that clips uses `extent_polygon`.
    """
    place = (cfg.get("places") or {}).get(location)
    if place is None and (ROUTES_DIR / f"{location}.yaml").exists():
        place = route_place(yaml.safe_load((ROUTES_DIR / f"{location}.yaml").read_text()))
    if place is None:
        return cfg
    out = dict(cfg)
    out["place"] = {**place, "name": location}
    if "altitudes_ft" in place:
        out["altitudes_ft"] = [int(a) for a in place["altitudes_ft"]]
    if "seed" in place:
        out["seed"] = int(place["seed"])
    if "speed_mps" in place and not place.get("sync_ground_track"):
        sp = place["speed_mps"]
        out["speed_mps"] = ({int(k): float(v) for k, v in sp.items()} if isinstance(sp, dict)
                            else {int(a): float(sp) for a in out["altitudes_ft"]})
    if "motion_frames" in place:      # recorded motion only needs to last as long as the place's clips
        out["populate"] = {**cfg["populate"], "frames": int(place["motion_frames"])}
    if "frames" in place:             # a place may set the length of its own long clip
        out["long"] = {**cfg["long"], "frames": int(place["frames"])}
    if place.get("sync_ground_track"):
        # One ground speed and one yaw-rate limit for every altitude: the route is the ground
        # track of the image-centre point, and each altitude's camera sits behind it.
        out["speed_mps"] = {k: float(place["speed_mps"]) for k in cfg["speed_mps"]}
        out["paths"] = {**cfg["paths"], "max_yaw_rate_deg_s": float(place["max_yaw_rate_deg_s"])}
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
