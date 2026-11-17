"""Option handling of the nyc-drone command, without network, GPU or rendering."""
import pytest

from nyc_drone_sim import cli


def resolve(*args):
    return cli.resolve(cli.build_parser().parse_args(["run", *args]))


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
