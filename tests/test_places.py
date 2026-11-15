import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from nyc_drone_sim.common import extent_polygon, for_location, load_config
from nyc_drone_sim.layers import park_interior
from nyc_drone_sim.paths import frame_times, route_track


def test_default_location_is_a_circle():
    cfg = load_config()
    out = for_location(cfg, "120 Broadway")
    assert out is cfg
    ext = extent_polygon(out)
    assert ext.area == pytest.approx(np.pi * cfg["scene_radius_m"] ** 2, rel=1e-3)


def test_corridor_place_holds_its_route_and_bounding_radius():
    cfg = for_location(load_config(), "battery-shoreline")
    ext = extent_polygon(cfg)
    place = cfg["place"]
    for x, y in place["route_m"]:
        assert ext.contains(Point(x, y))
        assert ext.exterior.distance(Point(x, y)) == pytest.approx(place["buffer_m"], rel=0.01)
    # the bounding radius is the farthest route vertex plus the buffer; the polygon, drawn with
    # short straight segments, reaches almost as far and never farther
    far = max(np.hypot(x, y) for x, y in ext.exterior.coords)
    assert far <= cfg["scene_radius_m"] + 1e-6
    assert cfg["scene_radius_m"] - far < 5.0
    assert cfg["populate"]["frames"] == place["motion_frames"]


def test_route_track_speed_window_and_view_offset():
    cfg = load_config()
    t = frame_times(cfg, "long")
    xy, yaw = route_track([[0, 0], [1000, 0]], 8.0, t, 0.5, 25.0, cfg)
    step = np.hypot(*np.diff(xy, axis=0).T) * cfg["clip"]["fps"]
    assert step == pytest.approx(8.0, abs=1e-6)
    assert (xy[0, 0] + xy[-1, 0]) / 2 == pytest.approx(500.0, abs=0.5)      # centred on the focus
    assert yaw == pytest.approx(25.0, abs=1e-6)                            # heading east, looking 25 deg left
    with pytest.raises(RuntimeError):
        route_track([[0, 0], [100, 0]], 8.0, t, 0.5, 0.0, cfg)             # route shorter than the clip


def test_park_interior_partitions_the_park():
    park = Polygon([(0, 0), (120, 0), (120, 80), (0, 80)])
    pk = {"min_area_m2": 1500, "path_width_m": 3.0, "edge_inset_m": 6.0, "spokes": 5}
    lawn, path = park_interior(park, pk)
    assert sum(p.area for p in lawn) + sum(p.area for p in path) == pytest.approx(park.area, rel=1e-6)
    assert 0.05 < sum(p.area for p in path) / park.area < 0.4
    small_lawn, small_path = park_interior(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]), pk)
    assert not small_path and small_lawn[0].area == pytest.approx(400.0)
