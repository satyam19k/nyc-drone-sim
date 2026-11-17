"""Route files: loading, conversion, and that the saved routes still give the saved camera poses.
The flight paths are planned without buildings, which only veto a path, never move it."""
import pytest

from nyc_drone_sim import paths, routes
from nyc_drone_sim.common import ROUTES_DIR, for_location, load_config

NAMES = sorted(p.stem for p in ROUTES_DIR.glob("*.yaml"))


def test_routes_exist():
    assert {"canal-shore-street", "hudson-park-run"} <= set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_route_reproduces_saved_poses(name):
    cfg = for_location(load_config(), name)
    expected = routes.load_expected(name)
    flight = {"altitudes": {str(a): paths.plan_route([], a, cfg) for a in cfg["altitudes_ft"]}}
    res = routes.compare_poses(routes.poses_from_flight(flight), expected)
    for alt, r in res.items():
        assert "error" not in r, (alt, r)
        assert r["max_position_m"] < 1e-5 and r["max_rotation_deg"] < 1e-3, (alt, r)


def test_route_settings_reach_the_config():
    cfg = for_location(load_config(), "hudson-park-run")
    assert cfg["altitudes_ft"] == [150, 500] and cfg["speed_mps"] == {150: 8.0, 500: 12.0}
    assert cfg["long"]["frames"] == 480 and cfg["populate"]["frames"] == 480
    cfg = for_location(load_config(), "canal-shore-street")
    assert cfg["speed_mps"] == {150: 8.0, 500: 8.0} and cfg["long"]["frames"] == 768


def test_lonlat_waypoints_become_metres(tmp_path):
    f = tmp_path / "r.yaml"
    f.write_text("origin: [40.72, -74.01]\nwaypoints_lonlat: [[-74.01, 40.72], [-74.0, 40.72]]\n")
    name, route, place = routes.resolve_route(f, load_config())
    (x0, y0), (x1, y1) = place["route_m"]
    assert name == "r" and abs(x0) < 1e-6 and abs(y0) < 1e-6
    assert 840 < x1 < 845 and abs(y1) < 2          # 0.01 degree of longitude at 40.72 N is about 843 m


def test_pose_comparison_sees_a_shift():
    pose = [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7]]
    moved = [[1, 0, 0, 5], [0, 1, 0, 6.5], [0, 0, 1, 7]]
    r = routes.compare_poses({"150": [moved]}, {"poses": {"150": [pose]}})["150"]
    assert r["max_position_m"] == pytest.approx(0.5) and r["max_rotation_deg"] == pytest.approx(0)
