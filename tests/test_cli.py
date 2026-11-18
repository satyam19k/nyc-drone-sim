"""Option handling of the nyc-drone command, without network, GPU or rendering."""
import json

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


def test_rain_is_not_available(tmp_path):
    with pytest.raises(SystemExit, match="weather rain is not available yet"):
        resolve("--address", "x", "--realism", "model", "--weather", "rain", "--out", str(tmp_path))
    resolve("--address", "x", "--realism", "model", "--weather", "snow", "--out", str(tmp_path))


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


def test_sim_fog_brightness_per_altitude(tmp_path):
    cfg, _, _ = resolve("--address", "x", "--altitude-ft", "150", "500", "--sim-weather", "fog", "--out", str(tmp_path))
    assert cfg["_run"]["sim_weather"] == "fog"
    assert [cli.fog_brightness(cfg, x) for x in (100, 150, 325, 500, 800)] == [10.0, 10.0, 6.0, 2.0, 2.0]
    args = cli.sim_weather_args(cfg, 500)
    assert args[:2] == ["--sim-weather", "fog_heavy"]
    assert json.loads(args[3]) == {"weather": {"fog_heavy": {"color_intensity": 2.0}}}
    assert cli.sim_weather_args(resolve("--address", "x", "--out", str(tmp_path))[0], 150) == []


def test_sim_fog_reaches_the_renderer(tmp_path, monkeypatch):
    a = cli.build_parser().parse_args(["route", "canal-shore-street", "--sim-weather", "fog", "--out", str(tmp_path)])
    cfg, location, variant = cli.resolve(a)
    cmds = []

    class Stop(Exception):
        pass

    def fake_call(cmd, log):
        cmds.append(cmd)
        if len(cmds) == len(cfg["altitudes_ft"]):
            raise Stop
        return 1.0
    monkeypatch.setattr(cli, "call", fake_call)
    monkeypatch.setattr(cli, "check_simulator", lambda a: None)
    (tmp_path / "work").mkdir()
    with pytest.raises(Stop):
        cli.step_render(a, cfg, location, variant, tmp_path / "config.yaml", tmp_path, {})
    for alt, cmd in zip(cfg["altitudes_ft"], cmds):
        assert cmd[cmd.index("--sim-weather") + 1] == "fog_heavy"
        assert json.loads(cmd[cmd.index("--appearance") + 1])["weather"]["fog_heavy"]["color_intensity"] == {150: 10.0, 500: 2.0}[alt]
        assert cmd[cmd.index("--look") + 1] == "standard"


@pytest.mark.parametrize("args, command, message", [
    (["--look", "enhanced"], "run", "not combined with --look enhanced"),
    (["--realism", "model"], "run", "leave --realism model out"),
    ([], "generate", "for run, route, reproduce and render"),
    ([], "check", "for run, route, reproduce and render"),
])
def test_sim_fog_refused(tmp_path, args, command, message):
    with pytest.raises(SystemExit, match=message):
        resolve("--address", "x", "--sim-weather", "fog", "--out", str(tmp_path), *args, command=command)
