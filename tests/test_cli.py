"""Option handling of the nyc-drone command, without network, GPU or rendering."""
import pytest

from nyc_drone_sim import cli


def resolve(*args, command="run"):
    return cli.resolve(cli.build_parser().parse_args([command, *args]))


def test_lat_lon_duration_and_altitudes(tmp_path):
    cfg, location, variant = resolve("--lat", "40.7", "--lon", "-74.0", "--altitude-ft", "150", "300",
                                      "--duration-s", "20", "--out", str(tmp_path))
    assert location == "40.700000,-74.000000" and variant == "street"
    assert cfg["street"]["frames"] == 320 and cfg["populate"]["frames"] == 320
    assert cfg["altitudes_ft"] == [150, 300]
    assert cfg["speed_mps"][150] == 8.0 and 8.0 < cfg["speed_mps"][300] < 12.0


def test_field_of_view_sets_focal_length(tmp_path):
    cfg, _, _ = resolve("--address", "x", "--fov-deg", "73.74", "--out", str(tmp_path))
    assert cfg["camera"]["focal_length_mm"] == pytest.approx(24.0, abs=0.01)


def test_outputs_stay_under_out(tmp_path):
    cfg, _, _ = resolve("--address", "x", "--out", str(tmp_path))
    assert cfg["output"]["runs_dir"].startswith(str(tmp_path))


def test_address_and_coordinates_together_are_refused(tmp_path):
    with pytest.raises(SystemExit):
        resolve("--address", "x", "--lat", "1", "--lon", "2", "--out", str(tmp_path))


def test_only_clear_weather(tmp_path):
    with pytest.raises(SystemExit, match="weather is not available yet"):
        resolve("--address", "x", "--weather", "snow", "--out", str(tmp_path))


def test_route_file_sets_the_flight(tmp_path):
    cfg, location, variant = resolve("canal-shore-street", "--out", str(tmp_path), command="reproduce")
    assert location == "canal-shore-street" and variant == "long"
    assert cfg["places"][location]["sync_ground_track"] and cfg["long"]["frames"] == 768


def test_route_refuses_flight_options(tmp_path):
    with pytest.raises(SystemExit, match="route file"):
        resolve("hudson-park-run", "--altitude-ft", "300", "--out", str(tmp_path), command="route")


def test_no_slurm_option():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["run", "--address", "x", "--slurm"])
