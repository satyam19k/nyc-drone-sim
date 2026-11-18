"""CPU only: the generate step with a stand-in for the video model. Checks the spec of each setting,
the frame range, the prompt and its thresholds, the guide video of M2, --specs-only, and what is
kept beside each clip. Decoding and encoding video needs OpenCV, which the CPU environment does not
have, so frame counts are given and cut videos are stand-in files."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from nyc_drone_sim import cli
from nyc_drone_sim.gen import makeclip, windows
from nyc_drone_sim.gen.common import load_gen_config
from nyc_drone_sim.gen.prompt import class_config, class_prompt

FRAMES = 200
LABELS = {"1": {"class": "water"}, "2": {"class": "road"}, "3": {"class": "building"},
          "4": {"class": "sidewalk"}, "5": {"class": "person"}}
CONTROLS = ("rgb.mp4", "depth_control.mp4", "edges_rendered.mp4", "edges_geometry.mp4", "segmentation.mp4")

# a stand-in for the model's examples/inference.py: writes <out>/<name>.mp4 for every spec
STAND_IN = '''
import json, sys
args = sys.argv[1:]
specs, out = args[args.index("-i") + 1:args.index("-o")], args[args.index("-o") + 1]
for s in specs:
    name = json.load(open(s))["name"]
    open(f"{out}/{name}.mp4", "wb").write(b"stand-in video")
'''


def pose(i: int, shift: float = 0.0) -> dict:
    m = np.eye(4)
    m[:3, 3] = [i * 0.5 + shift, -200.0, 45.72]
    return {"index": i, "requested_camera_to_world": m.tolist()}


def make_bundle(root: Path, water: bool, look: str = "standard") -> Path:
    """A 200-frame bundle with only what the generate step reads: bundle.json, semantic frames and
    camera poses. With water, the left 30% of every frame is water; one person is 0.1%."""
    bdir = root / "run" / "tiny" / "150"
    (bdir / "semantic").mkdir(parents=True)
    (bdir / "camera").mkdir()
    (bdir / "bundle.json").write_text(json.dumps({
        "location": "tiny", "clip": {"frames": FRAMES}, "appearance": {"weather": "clear", "look": look},
        "semantic_info": {"idToLabels": LABELS}}))
    sem = np.full((10, 100), 3, np.int32)          # buildings
    sem[:, 30:60] = 2                              # road
    sem[:, 60:80] = 4                              # sidewalk
    sem[0, 90] = 5                                 # one person: 0.1% of the pixels
    if water:
        sem[:, :30] = 1
    for i in range(FRAMES):
        np.savez_compressed(bdir / "semantic" / f"semantic_{i:04d}.npz", data=sem)
        (bdir / "camera" / f"camera_{i:04d}.json").write_text(json.dumps(pose(i)))
    return bdir


def cut(bdir: Path, name: str) -> Path:
    """Controls already cut in the bundle, as cut_window leaves them (stand-in contents)."""
    w = bdir / "gen" / "windows" / name
    w.mkdir(parents=True)
    (w / "window.json").write_text("{}")
    for f in CONTROLS:
        (w / f).write_bytes(f"bundle {f}".encode())
    return w


def make_guide(root: Path, shift: float = 0.0, frames: int = FRAMES) -> Path:
    """An enhanced render of the same flight: rgb.mp4 with camera_poses.json beside it."""
    g = root / "enhanced" / "150"
    g.mkdir(parents=True)
    (g / "rgb.mp4").write_bytes(b"enhanced colours")
    (g / "camera_poses.json").write_text(json.dumps([pose(i, shift) for i in range(frames)]))
    return g / "rgb.mp4"


@pytest.fixture
def gcfg():
    return load_gen_config()


@pytest.fixture
def fake_video(monkeypatch):
    """Frame counts without decoding, and a cut that records what it was asked for."""
    cuts = []

    def cut_video(src, start, count, dst, cfg):
        cuts.append((Path(src), start, count))
        Path(dst).write_bytes(f"{Path(src).read_bytes().decode()} {start}:{count}".encode())
        return count
    monkeypatch.setattr(makeclip, "video_info", lambda p: {"frames": FRAMES if "short" not in str(p) else FRAMES - 1})
    monkeypatch.setattr(windows, "cut_video", cut_video)
    return cuts


@pytest.fixture
def stand_in_model(tmp_path, monkeypatch):
    repo = tmp_path / "model"
    (repo / "examples").mkdir(parents=True)
    (repo / "examples" / "inference.py").write_text(STAND_IN)
    monkeypatch.setenv("MODEL_REPO", str(repo))
    monkeypatch.setenv("MODEL_PYTHON", sys.executable)
    monkeypatch.setattr(makeclip, "describe_chain", lambda *a: None)       # decodes the video: needs cv2
    monkeypatch.setattr(makeclip, "SUMMARY", tmp_path / "gen_summary.md")


@pytest.mark.parametrize("setting, depth, edge, seg, vis, edge_file", [
    ("K2", 0.5, 0.3, 0.5, None, "edges_rendered.mp4"),
    ("M2", 0.5, 0.3, 0.5, {"control_weight": 0.3, "preset_blur_strength": "high"}, "edges_rendered.mp4"),
    ("t4", 1.0, 1.0, None, None, "edges_geometry.mp4"),
])
def test_spec_of_each_setting(tmp_path, gcfg, fake_video, setting, depth, edge, seg, vis, edge_file):
    bdir = make_bundle(tmp_path, water=True)
    cut(bdir, "chain_0000_0200")
    out = makeclip.write_specs(bdir, "clear", setting, gcfg, "chain", tmp_path / "specs")
    spec, info = json.loads((out / "spec.json").read_text()), json.loads((out / "clip.json").read_text())
    assert spec["depth"] == {"control_path": "depth_control.mp4", "control_weight": depth}
    assert spec["edge"] == {"control_path": edge_file, "control_weight": edge}
    assert spec.get("seg") == (None if seg is None else {"control_path": "segmentation.mp4", "control_weight": seg})
    assert spec.get("vis") == vis                  # no control_path: the model blurs the input video
    assert spec["video_path"] == "rgb.mp4"
    assert spec["num_conditional_frames"] == 1 and spec["num_video_frames_per_chunk"] == 93
    assert spec["seed"] == gcfg["sampling"]["seed"] and spec["prompt"] == info["prompt"]
    assert (info["start"], info["frames"]) == (0, FRAMES)
    assert (info["look_warning"] is not None) == (setting == "M2")      # the render used the standard look
    # only the files the spec uses, plus spec, prompt and clip.json
    used = {"rgb.mp4", "depth_control.mp4", edge_file} | ({"segmentation.mp4"} if seg else set())
    assert {p.name for p in out.iterdir()} == used | {"spec.json", "prompt.txt", "clip.json"}


def test_m2_does_not_warn_on_an_enhanced_render(tmp_path, gcfg, fake_video):
    bdir = make_bundle(tmp_path, water=True, look="enhanced")
    assert makeclip.describe(bdir, "clear", "M2", gcfg, "chain", 0, FRAMES)[1]["look_warning"] is None


def test_frame_range():
    assert makeclip.frame_range(None, FRAMES) == (0, FRAMES)
    assert makeclip.frame_range("100:93", FRAMES) == (100, 93)
    for bad in ("150:93", "0:50", "-1:93", "a:b", "100"):
        with pytest.raises(SystemExit):
            makeclip.frame_range(bad, FRAMES)


def test_prompt_with_and_without_water(tmp_path, gcfg):
    pc = class_config()
    wet = makeclip.describe(make_bundle(tmp_path / "wet", water=True), "clear", "K2", gcfg, "chain", 100, 93)[1]
    dry = makeclip.describe(make_bundle(tmp_path / "dry", water=False), "clear", "K2", gcfg, "chain", 100, 93)[1]
    assert wet["described"] == ["water", "road", "sidewalk_or_plaza", "buildings", "people"]
    assert dry["described"] == ["road", "sidewalk_or_plaza", "buildings", "people"]
    assert pc["classes"]["water"]["sentence"] in wet["prompt"] and "river is deep blue-green" in wet["prompt"]
    assert "river" not in dry["prompt"].lower()
    # the same view gets the same words: only the water sentence and the weather's river part differ
    assert dry["prompt"].startswith(pc["opening"]) and pc["classes"]["road"]["sentence"] in dry["prompt"]
    snow, _ = class_prompt({"water": 0.0, "road": 0.5}, "snow", pc, makeclip.prompts())
    assert "fresh white snow" in snow and "river" not in snow.lower()


@pytest.mark.parametrize("shares, described", [
    ({"road": 0.5, "lawn": 0.011, "vehicle": 0.0031, "person": 0.00051}, ["road", "lawn", "vehicles", "people"]),
    ({"road": 0.5, "lawn": 0.009, "vehicle": 0.0029, "person": 0.00049}, ["road"]),
    ({"road": 0.5, "water": 0.029, "vegetation": 0.031, "building": 0.029}, ["road", "trees"]),
])
def test_threshold_of_each_class(shares, described):
    """water, road, sidewalk or plaza, buildings and trees 3%; lawn 1%; vehicles 0.3%; people 0.05%."""
    assert class_prompt(shares, "clear", class_config(), makeclip.prompts())[1] == described


def test_chain_with_a_stand_in_model(tmp_path, gcfg, fake_video, stand_in_model):
    bdir = make_bundle(tmp_path, water=True)
    w = cut(bdir, "chain_0100_0093")
    assert makeclip.run_request(bdir, "clear", "M2", gcfg, "chain", frames="100:93") == 0
    out = bdir / "gen" / gcfg["model"]["name"] / "clear_M2_chain_0100_0093"
    spec = json.loads((out / "spec.json").read_text())
    clip = json.loads((out / "clip.json").read_text())
    assert (out / "out.mp4").read_bytes() == b"stand-in video"
    assert spec["video_path"] == str(w / "rgb.mp4") and spec["vis"]["preset_blur_strength"] == "high"
    assert (out / "prompt.txt").read_text().strip() == spec["prompt"] == clip["prompt"]
    assert (clip["start"], clip["frames"], clip["num_conditional_frames"]) == (100, 93, 1)
    assert "standard look" in clip["look_warning"]


def test_guide_video_replaces_the_colour_source(tmp_path, gcfg, fake_video, stand_in_model):
    bdir = make_bundle(tmp_path, water=True)
    w = cut(bdir, "chain_0100_0093")
    guide = make_guide(tmp_path)
    assert makeclip.run_request(bdir, "clear", "M2", gcfg, "chain", frames="100:93", guide=str(guide)) == 0
    out = bdir / "gen" / gcfg["model"]["name"] / "clear_M2_chain_0100_0093"
    spec, clip = json.loads((out / "spec.json").read_text()), json.loads((out / "clip.json").read_text())
    assert fake_video == [(guide, 100, 93)]                                 # the guide, cut to the range
    assert Path(spec["video_path"]).parent == w and Path(spec["video_path"]).read_bytes() == b"enhanced colours 100:93"
    assert spec["depth"]["control_path"] == str(w / "depth_control.mp4")    # the geometry still comes from the render
    assert clip["guide_video"]["path"] == str(guide) and clip["guide_video"]["pose_check"]["max_position_m"] == 0.0
    assert clip["look_warning"] is None
    assert Path(spec["video_path"]).name == "guide_" + clip["guide_video"]["sha256"][:12] + ".mp4"
    # the whole clip: the guide itself, no cut
    cut(bdir, "chain_0000_0200")
    assert makeclip.run_request(bdir, "clear", "M2", gcfg, "chain", guide=str(guide)) == 0
    spec = json.loads((bdir / "gen" / gcfg["model"]["name"] / "clear_M2_chain_0000_0200" / "spec.json").read_text())
    assert spec["video_path"] == str(guide) and len(fake_video) == 1


@pytest.mark.parametrize("setting, shift, frames, message", [
    ("K2", 0.0, FRAMES, "M2"),                       # only M2 has a colour guide
    ("M2", 0.10, FRAMES, "camera poses differ"),     # 10 cm: over the 5 cm limit
    ("M2", 0.0, FRAMES - 1, "frames"),               # another flight
])
def test_guide_video_is_checked(tmp_path, gcfg, fake_video, setting, shift, frames, message):
    bdir = make_bundle(tmp_path, water=True)
    guide = make_guide(tmp_path, shift=shift, frames=frames)
    if frames != FRAMES:
        guide = guide.rename(guide.with_name("short.mp4"))                  # fake_video gives it 199 frames
    with pytest.raises(SystemExit, match=message):
        makeclip.write_specs(bdir, "clear", setting, gcfg, "chain", tmp_path / "specs", guide=str(guide))


def test_specs_only_reuses_built_controls_and_can_be_moved(tmp_path, gcfg, fake_video):
    bdir = make_bundle(tmp_path, water=True)
    cut(bdir, "chain_0100_0093")
    guide = make_guide(tmp_path)
    out = makeclip.write_specs(bdir, "clear", "M2", gcfg, "chain", tmp_path / "specs", frames="100:93", guide=str(guide))
    clip = json.loads((out / "clip.json").read_text())
    assert clip["controls"].startswith("copied from") and clip["specs_only"]
    assert (out / "depth_control.mp4").read_bytes() == b"bundle depth_control.mp4"
    assert (out / "guide.mp4").read_bytes() == b"enhanced colours 100:93" and not (out / "rgb.mp4").exists()
    # another machine: every path in the spec is relative to spec.json and resolves there
    moved = shutil.copytree(out, tmp_path / "elsewhere" / "clip")
    shutil.rmtree(tmp_path / "run")
    spec = json.loads((moved / "spec.json").read_text())
    paths = [spec["video_path"]] + [spec[k]["control_path"] for k in ("depth", "edge", "seg")]
    assert all(not Path(p).is_absolute() and (moved / p).is_file() for p in paths)
    assert spec["video_path"] == "guide.mp4" and "control_path" not in spec["vis"]
    # a new folder each time
    with pytest.raises(SystemExit, match="not empty"):
        makeclip.write_specs(make_bundle(tmp_path, water=True), "clear", "K2", gcfg, "chain", out)


def test_specs_only_cuts_controls_when_none_are_built(tmp_path, gcfg, fake_video, monkeypatch):
    bdir = make_bundle(tmp_path, water=False)
    calls = []

    def cut_window(bdir_, name, start, count, cfg, out=None, only=None):
        calls.append((name, start, count, out, set(only)))
        for f in only:
            (out / f).write_bytes(b"cut")
    monkeypatch.setattr(windows, "cut_window", cut_window)
    out = makeclip.write_specs(bdir, "snow", "t4", gcfg, "window", tmp_path / "specs", start=40)
    assert calls == [("one_0040", 40, 93, out, {"rgb.mp4", "depth_control.mp4", "edges_geometry.mp4"})]
    assert not (bdir / "gen").exists()                                       # nothing written into the bundle
    assert json.loads((out / "clip.json").read_text())["controls"] == "cut from the bundle's frames"


def setup_run(tmp_path: Path) -> tuple[Path, Path]:
    out = tmp_path / "out"
    run_folder = out / "work" / "render" / "tiny"
    run_folder.parent.mkdir(parents=True)
    make_bundle(tmp_path, water=False)
    (tmp_path / "run" / "tiny").rename(run_folder)
    return out, run_folder / "150"


def test_generate_step_keeps_spec_and_prompt_beside_each_clip(tmp_path, gcfg, monkeypatch):
    out, bdir = setup_run(tmp_path)
    cmds = []

    def fake_call(cmd, log):
        """What makeclip leaves behind, for the arguments it was given."""
        cmds.append(cmd)
        arg = dict(zip(cmd[3::2], cmd[4::2]))
        start, count = (int(v) for v in arg["--frames"].split(":"))
        made = bdir / "gen" / gcfg["model"]["name"] / f"{arg['--weather']}_{arg['--setting']}_chain_{start:04d}_{count:04d}"
        made.mkdir(parents=True)
        for f in ("out.mp4", "spec.json", "prompt.txt", "clip.json"):
            (made / f).write_text(f"{arg['--weather']} {f}")
        return 1.0

    monkeypatch.setattr(cli, "call", fake_call)
    monkeypatch.setattr(cli, "check_model", lambda: None)
    a = cli.build_parser().parse_args(["generate", "--lat", "40.7", "--lon", "-74", "--altitude-ft", "150",
                                       "--weather", "clear", "snow", "--method", "chain", "--frames", "100:93"])
    cli.step_generate(a, {"altitudes_ft": [150]}, "tiny", out, {})

    assert [c[c.index("--setting") + 1] for c in cmds] == ["K2", "t4"]       # the default for each weather
    assert all(c[c.index("--frames") + 1] == "100:93" for c in cmds)
    names = sorted(p.name for p in (out / "150" / "weather").iterdir())
    for stem in ("clear_K2_chain_0100_0093", "snow_t4_chain_0100_0093"):
        assert [n for n in names if n.startswith(stem)] == [f"{stem}{s}" for s in (".clip.json", ".mp4", ".prompt.txt", ".spec.json")]


def test_generate_step_specs_only_with_a_guide(tmp_path, monkeypatch):
    out, bdir = setup_run(tmp_path)
    guide, cmds = make_guide(tmp_path), []

    def fake_call(cmd, log):
        cmds.append(cmd)
        dest = Path(cmd[cmd.index("--specs-only") + 1])
        dest.mkdir(parents=True)
        (dest / "spec.json").write_text("{}")
        return 1.0

    def no_model():
        raise AssertionError("--specs-only must not need the model")
    monkeypatch.setattr(cli, "call", fake_call)
    monkeypatch.setattr(cli, "check_model", no_model)
    a = cli.build_parser().parse_args(["generate", "--lat", "40.7", "--lon", "-74", "--altitude-ft", "150",
                                       "--method", "chain", "--model-setting", "M2", "--guide-video", str(guide),
                                       "--specs-only", str(tmp_path / "specs")])
    cli.step_generate(a, {"altitudes_ft": [150]}, "tiny", out, {})
    (cmd,) = cmds
    assert cmd[cmd.index("--guide-video") + 1] == str(guide)
    assert cmd[cmd.index("--specs-only") + 1] == str(tmp_path / "specs" / "150" / "clear_M2_chain_0000_0200")
    assert not (out / "150" / "weather").exists()


@pytest.mark.parametrize("args, message", [
    (["--weather", "fog"], "not available yet"),
    (["--weather", "rain"], "not available yet"),
    (["--weather", "hail"], "choose from"),
    (["--frames", "0:93"], "--method chain"),
    (["--guide-video", "g.mp4"], "M2"),
    (["--model-setting", "M2", "--guide-poses", "p.json"], "one per --guide-video"),
])
def test_generate_options_that_are_refused(tmp_path, args, message):
    a = cli.build_parser().parse_args(["generate", "--lat", "40.7", "--lon", "-74", "--out", str(tmp_path), *args])
    with pytest.raises(SystemExit) as e:
        cli.resolve(a)
    assert message in str(e.value)


def test_model_options_need_the_model(tmp_path):
    for extra in (["--weather", "snow"], ["--specs-only", str(tmp_path / "s")]):
        a = cli.build_parser().parse_args(["run", "--lat", "40.7", "--lon", "-74", "--out", str(tmp_path), *extra])
        with pytest.raises(SystemExit, match="--realism model"):
            cli.resolve(a)
    a = cli.build_parser().parse_args(["run", "--lat", "40.7", "--lon", "-74", "--out", str(tmp_path), "--realism", "model",
                                       "--weather", "snow", "--method", "chain", "--model-setting", "M2"])
    assert cli.resolve(a)[1] == "40.700000,-74.000000"
