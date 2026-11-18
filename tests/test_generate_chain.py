"""CPU only: the chained generate step with a stand-in for the video model. Checks the spec of each
setting, the frame range, the prompt with and without water, and what is kept beside each clip."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from nyc_drone_sim import cli
from nyc_drone_sim.gen import makeclip
from nyc_drone_sim.gen.common import load_gen_config
from nyc_drone_sim.gen.prompt import class_config, class_prompt

FRAMES = 200
LABELS = {"1": {"class": "water"}, "2": {"class": "road"}, "3": {"class": "building"},
          "4": {"class": "sidewalk"}, "5": {"class": "person"}}

# a stand-in for the model's examples/inference.py: writes <out>/<name>.mp4 for every spec
STAND_IN = '''
import json, sys
args = sys.argv[1:]
specs, out = args[args.index("-i") + 1:args.index("-o")], args[args.index("-o") + 1]
for s in specs:
    name = json.load(open(s))["name"]
    open(f"{out}/{name}.mp4", "wb").write(b"stand-in video")
    print("Generating chunk 1/2")
'''


def make_bundle(root: Path, water: bool, look: str = "standard") -> Path:
    """A 200-frame bundle with only what the generate step reads: bundle.json and semantic frames.
    With water, the left 30% of every frame is water; people are 1% of every frame."""
    bdir = root / "run" / "tiny" / "150"
    (bdir / "semantic").mkdir(parents=True)
    (bdir / "bundle.json").write_text(json.dumps({
        "location": "tiny", "clip": {"frames": FRAMES}, "appearance": {"weather": "clear", "look": look},
        "semantic_info": {"idToLabels": LABELS}}))
    sem = np.full((10, 100), 3, np.int32)          # buildings
    sem[:, 30:60] = 2                              # road
    sem[:, 60:80] = 4                              # sidewalk
    sem[0, 90] = 5                                 # one person: 0.1%, under the 3% threshold
    if water:
        sem[:, :30] = 1
    for i in range(FRAMES):
        np.savez_compressed(bdir / "semantic" / f"semantic_{i:04d}.npz", data=sem)
    return bdir


def cut(bdir: Path, start: int, count: int) -> Path:
    """The controls of a chain, as cut_window would leave them (contents are not read here)."""
    w = bdir / "gen" / "windows" / f"chain_{start:04d}_{count:04d}"
    w.mkdir(parents=True)
    (w / "window.json").write_text("{}")
    return w


@pytest.fixture
def gcfg():
    return load_gen_config()


@pytest.mark.parametrize("setting, depth, edge, seg, vis, edge_file", [
    ("K2", 0.5, 0.3, 0.5, None, "edges_rendered.mp4"),
    ("M2", 0.5, 0.3, 0.5, {"control_weight": 0.3, "preset_blur_strength": "high"}, "edges_rendered.mp4"),
    ("t4", 1.0, 1.0, None, None, "edges_geometry.mp4"),
])
def test_spec_of_each_setting(tmp_path, gcfg, setting, depth, edge, seg, vis, edge_file):
    bdir = make_bundle(tmp_path, water=True)
    w = cut(bdir, 0, FRAMES)
    spec, info = makeclip.chain_spec(bdir, "clear", setting, gcfg, 0, FRAMES)
    assert spec["depth"] == {"control_path": str(w / "depth_control.mp4"), "control_weight": depth}
    assert spec["edge"] == {"control_path": str(w / edge_file), "control_weight": edge}
    assert spec.get("seg") == (None if seg is None else {"control_path": str(w / "segmentation.mp4"), "control_weight": seg})
    assert spec.get("vis") == vis                  # no control_path: the model blurs the input video
    assert spec["video_path"] == str(w / "rgb.mp4")
    assert spec["num_conditional_frames"] == 1 and spec["num_video_frames_per_chunk"] == 93
    assert spec["seed"] == gcfg["sampling"]["seed"] and spec["prompt"] == info["prompt"]
    assert (info["look_warning"] is not None) == (setting == "M2")      # the render used the standard look


def test_m2_does_not_warn_on_an_enhanced_render(tmp_path, gcfg):
    bdir = make_bundle(tmp_path, water=True, look="enhanced")
    cut(bdir, 0, FRAMES)
    assert makeclip.chain_spec(bdir, "clear", "M2", gcfg, 0, FRAMES)[1]["look_warning"] is None


def test_frame_range():
    assert makeclip.frame_range(None, FRAMES) == (0, FRAMES)
    assert makeclip.frame_range("100:93", FRAMES) == (100, 93)
    for bad in ("150:93", "0:50", "-1:93", "a:b", "100"):
        with pytest.raises(SystemExit):
            makeclip.frame_range(bad, FRAMES)


def test_prompt_with_and_without_water(tmp_path, gcfg):
    pc = class_config()
    wet_dir, dry_dir = make_bundle(tmp_path / "wet", water=True), make_bundle(tmp_path / "dry", water=False)
    for d in (wet_dir, dry_dir):
        cut(d, 100, 93)
    wet = makeclip.chain_spec(wet_dir, "clear", "K2", gcfg, 100, 93)[1]
    dry = makeclip.chain_spec(dry_dir, "clear", "K2", gcfg, 100, 93)[1]
    assert wet["described"] == ["water", "road", "sidewalk_or_plaza", "buildings"]
    assert dry["described"] == ["road", "sidewalk_or_plaza", "buildings"]
    assert pc["classes"]["water"]["sentence"] in wet["prompt"] and "river is deep blue-green" in wet["prompt"]
    assert "river" not in dry["prompt"].lower()
    assert pc["classes"]["people"]["sentence"] not in wet["prompt"]          # 0.1% of the pixels
    # the same view gets the same words: only the water sentence and the weather's river part differ
    assert dry["prompt"].startswith(pc["opening"]) and pc["classes"]["road"]["sentence"] in dry["prompt"]
    snow, _ = class_prompt({"water": 0.0, "road": 0.5}, "snow", pc, makeclip.prompts())
    assert "fresh white snow" in snow and "river" not in snow.lower()


def test_chain_with_a_stand_in_model(tmp_path, gcfg, monkeypatch):
    bdir = make_bundle(tmp_path, water=True)
    cut(bdir, 100, 93)
    repo = tmp_path / "model"
    (repo / "examples").mkdir(parents=True)
    (repo / "examples" / "inference.py").write_text(STAND_IN)
    monkeypatch.setenv("MODEL_REPO", str(repo))
    monkeypatch.setenv("MODEL_PYTHON", sys.executable)
    monkeypatch.setattr(makeclip, "video_info", lambda p: {"frames": FRAMES})
    monkeypatch.setattr(makeclip, "describe_chain", lambda *a: None)       # decodes the video: needs cv2
    monkeypatch.setattr(makeclip, "SUMMARY", tmp_path / "gen_summary.md")

    assert makeclip.run_chain(bdir, "clear", "M2", gcfg, frames="100:93") == 0

    out = bdir / "gen" / gcfg["model"]["name"] / "clear_M2_chain_0100_0093"
    spec = json.loads((out / "spec.json").read_text())
    clip = json.loads((out / "clip.json").read_text())
    assert (out / "out.mp4").read_bytes() == b"stand-in video"
    assert spec["video_path"].endswith("chain_0100_0093/rgb.mp4") and spec["vis"]["preset_blur_strength"] == "high"
    assert (out / "prompt.txt").read_text().strip() == spec["prompt"] == clip["prompt"]
    assert (clip["start"], clip["frames"], clip["num_conditional_frames"]) == (100, 93, 1)
    assert "standard look" in clip["look_warning"]


def test_generate_step_keeps_spec_and_prompt_beside_each_clip(tmp_path, gcfg, monkeypatch):
    out = tmp_path / "out"
    run_folder = out / "work" / "render" / "tiny"
    run_folder.parent.mkdir(parents=True)
    make_bundle(tmp_path, water=False)
    (tmp_path / "run" / "tiny").rename(run_folder)
    bdir = run_folder / "150"
    (bdir / "gen" / "inputs").mkdir(parents=True)
    (bdir / "gen" / "inputs" / "inputs.json").write_text("{}")
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


@pytest.mark.parametrize("args, message", [
    (["--weather", "fog"], "not available yet"),
    (["--weather", "rain"], "not available yet"),
    (["--weather", "hail"], "choose from"),
    (["--frames", "0:93"], "--method chain"),
])
def test_generate_options_that_are_refused(tmp_path, args, message):
    a = cli.build_parser().parse_args(["generate", "--lat", "40.7", "--lon", "-74", "--out", str(tmp_path), *args])
    with pytest.raises(SystemExit) as e:
        cli.resolve(a)
    assert message in str(e.value)


def test_model_options_need_the_model(tmp_path):
    a = cli.build_parser().parse_args(["run", "--lat", "40.7", "--lon", "-74", "--out", str(tmp_path), "--weather", "snow"])
    with pytest.raises(SystemExit, match="--realism model"):
        cli.resolve(a)
    a = cli.build_parser().parse_args(["run", "--lat", "40.7", "--lon", "-74", "--out", str(tmp_path), "--realism", "model", "--weather", "snow",
                                       "--method", "chain", "--model-setting", "M2"])
    assert cli.resolve(a)[1] == "40.700000,-74.000000"
