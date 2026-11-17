"""Route files: a camera route over New York City, flown by `nyc-drone route FILE`. CPU only.

A route file is YAML (examples in configs/routes/):

    origin: [40.725359, -74.011585]   # [latitude, longitude], or an address in quotes
    altitudes_ft: [150, 500]
    speed_mps: 8                      # one value, or one per altitude: {150: 8, 500: 12}
    seed: 2026
    look: standard                    # or enhanced
    sync_ground_track: true           # both altitudes share one ground track of the image centre
    frames: 768                       # clip length at 16 fps
    waypoints_m: [[x, y], ...]        # metres east and north of the origin
    # or waypoints_lonlat: [[longitude, latitude], ...]

Optional: motion_frames (length of recorded traffic, default frames), buffer_m (scene extent
around the route, default 768), max_yaw_rate_deg_s, and, without sync_ground_track, focus
({altitude: 0..1}, where along the route the clip is centred) and yaw_offset_deg.

`<name>.expected_poses.json` beside a route file holds the camera poses of an earlier render of
it; `nyc-drone reproduce <name>` compares a new flight with them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .common import ROUTES_DIR, route_place



def route_path(name_or_file: str) -> Path:
    p = Path(name_or_file)
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p
    return ROUTES_DIR / f"{name_or_file}.yaml"


def resolve_route(path: Path, cfg: dict, offline: bool = False) -> tuple[str, dict, dict]:
    """(name, route with [lat, lon] origin and waypoints in metres, place for the config)."""
    route = yaml.safe_load(Path(path).read_text())
    name = Path(path).stem
    if isinstance(route.get("origin"), str):
        from . import geo
        http = geo.HttpCache(geo.REPO_ROOT / cfg["data"]["cache_dir"], offline=offline,
                             timeout_s=float(cfg["data"]["http_timeout_s"]))
        loc = geo.geocode(route["origin"], cfg, http)
        route["origin_address"] = route["origin"]
        route["origin"] = [loc["lat"], loc["lon"]]
    if "waypoints_m" not in route:
        if "waypoints_lonlat" not in route:
            raise SystemExit(f"{path}: give waypoints_m or waypoints_lonlat")
        route["waypoints_m"] = lonlat_to_m(route["waypoints_lonlat"], *route["origin"])
    route.setdefault("frames", 480)
    route.setdefault("motion_frames", route["frames"])
    route.setdefault("buffer_m", 768)
    return name, route, route_place(route)


def lonlat_to_m(points, lat0: float, lon0: float) -> list[list[float]]:
    """Longitude/latitude to metres east and north of the origin, with the scene's own projection."""
    from pyproj import Transformer
    from .geo import local_proj
    tr = Transformer.from_crs("EPSG:4326", local_proj(lat0, lon0), always_xy=True)
    xs, ys = tr.transform([float(p[0]) for p in points], [float(p[1]) for p in points])
    return [[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs, ys)]


def poses_from_flight(flight: dict) -> dict:
    """{altitude: [3x4 camera-to-world per frame]} from flight.json or a path record."""
    return {str(alt): [np.asarray(f["camera_to_world"], dtype=float)[:3].tolist() for f in p["frames"]]
            for alt, p in flight["altitudes"].items()}


def compare_poses(new: dict, expected: dict) -> dict:
    """Largest position difference (m) and rotation difference (degrees) per altitude."""
    out = {}
    for alt, exp in expected["poses"].items():
        if alt not in new:
            out[alt] = {"error": "altitude missing from the new flight"}
            continue
        a, b = np.asarray(new[alt], dtype=float), np.asarray(exp, dtype=float)
        if a.shape != b.shape:
            out[alt] = {"error": f"{len(a)} frames, expected {len(b)}"}
            continue
        dpos = np.linalg.norm(a[:, :, 3] - b[:, :, 3], axis=1)
        # rotation angle between the two orientations, from |R_a - R_b| = 2 sqrt(2) sin(angle / 2);
        # unlike arccos of the trace this stays accurate for tiny angles
        fro = np.linalg.norm(a[:, :, :3] - b[:, :, :3], axis=(1, 2))
        ang = np.degrees(2 * np.arcsin(np.clip(fro / (2 * np.sqrt(2)), 0, 1)))
        out[alt] = {"frames": int(len(a)), "max_position_m": float(dpos.max()), "frame_of_max_position": int(dpos.argmax()),
                    "max_rotation_deg": float(ang.max()), "frame_of_max_rotation": int(ang.argmax())}
    return out


def load_expected(name: str) -> dict:
    p = ROUTES_DIR / f"{name}.expected_poses.json"
    if not p.exists():
        raise SystemExit(f"no {p.name} in configs/routes/: nothing to compare with")
    return json.loads(p.read_text())
