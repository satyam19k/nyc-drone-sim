"""The video-model job driver, exercised against a stand-in for the model's inference script.

The stand-in takes the same arguments, writes the files the real script writes and prints the log
lines the driver reads, so the driver's bookkeeping is tested without a GPU: which command is
built, where results land, what a rerun skips, and what a blocked sample leaves behind.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from nyc_drone_sim.gen import job

FAKE_INFERENCE = r'''
import json, os, sys
from pathlib import Path
a = sys.argv[1:]
specs = []
i = a.index("-i") + 1
while i < len(a) and not a[i].startswith("-"):
    specs.append(a[i]); i += 1
out = Path(a[a.index("-o") + 1]); out.mkdir(parents=True, exist_ok=True)
calls = Path(os.environ["FAKE_CALLS"])
calls.write_text(calls.read_text() + json.dumps({"argv": a, "experimental": os.environ.get("COSMOS_EXPERIMENTAL_CHECKPOINTS")}) + "\n")
for n, s in enumerate(specs):
    d = json.loads(Path(s).read_text())
    print(f"[{n + 1}/{len(specs)}] Processing sample {d['name']}")
    print("No image context provided.")
    print("Generating chunk 1/1")
    if "BLOCK" in d.get("prompt", ""):
        print("Guardrail blocked generation. Prompt: x")
        continue
    (out / f"{d['name']}.mp4").write_bytes(b"video")
    (out / f"{d['name']}_control_edge.mp4").write_bytes(b"control")
    (out / f"{d['name']}.json").write_text("{}")
    print(f"Average time per chunk: {100 + n}.5")
'''


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo = tmp_path / "model"
    (repo / "examples").mkdir(parents=True)
    (repo / "examples" / "inference.py").write_text(FAKE_INFERENCE)
    (repo / "example").mkdir()
    (repo / "example" / "spec.json").write_text(json.dumps({"name": "robot", "prompt": "a robot"}))
    calls = tmp_path / "calls.jsonl"
    calls.write_text("")
    monkeypatch.setenv("FAKE_CALLS", str(calls))
    bdir = tmp_path / "runs" / "run_x" / "scene" / "150"
    bdir.mkdir(parents=True)
    (bdir / "bundle.json").write_text(json.dumps({"location": "scene"}))
    cfg = {"model": {"name": "m", "repo_dir": str(repo), "commit": "c" * 40, "example_spec": "example/spec.json",
                     "min_vram_gb": 70, "min_glibc": "2.35", "min_driver": "570.124.06"},
           "sampling": {"chunk_frames": 93},
           "runs": [{"altitude": 150, "weather": w, "edge": "rendered" if w == "clear" else "geometry"}
                    for w in ("clear", "rain", "snow", "fog")]}
    specs = bdir / "gen" / "m" / "specs"
    specs.mkdir(parents=True)
    for i, r in enumerate(cfg["runs"], start=1):
        (specs / f"{i:02d}_{r['weather']}_{r['edge']}.json").write_text(json.dumps(
            {"name": "out", "prompt": f"scene, {r['weather']}", "video_path": "rgb.mp4", "seed": 2026, "guidance": 3,
             "num_steps": 35, "resolution": "720", "edge": {"control_path": "e.mp4"}, "depth": {"control_path": "d.mp4"},
             "seg": {"control_path": "s.mp4"}}))
    monkeypatch.setattr(job, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(job, "REPORT", tmp_path / "report.md")
    monkeypatch.setattr(job, "video_info", lambda p: {"frames": 93})
    monkeypatch.setattr(job, "gpu_query", lambda fields: ["0"] if fields == "memory.used" else ["Test GPU", "81920", "575.57.08"])
    monkeypatch.setattr(job.platform, "libc_ver", lambda: ("glibc", "2.39"))
    return {"cfg": cfg, "bdir": bdir, "folder": str(bdir.parent), "calls": calls, "tmp": tmp_path,
            "out": bdir / "gen" / "m", "specs": specs}


def calls_of(world):
    return [json.loads(line) for line in world["calls"].read_text().splitlines()]


def test_self_test_runs_example_full_and_distilled(world):
    assert job.self_test(world["folder"], world["cfg"], "Test GPU")
    c = calls_of(world)
    assert len(c) == 3
    assert "--max-frames" not in c[0]["argv"] and "--model" not in c[0]["argv"]        # the example, unmodified
    assert c[1]["argv"][-2:] == ["--max-frames", "93"] and c[1]["experimental"] is None
    assert c[2]["argv"][-4:] == ["--max-frames", "93", "--model", "edge/distilled"] and c[2]["experimental"] == "1"
    full, dist = world["out"] / "selftest_clear_rendered_93f", world["out"] / "selftest_clear_rendered_distilled_93f"
    for d in (full, dist):
        assert (d / "out.mp4").exists() and json.loads((d / "timing.json").read_text())["image_context"] == "none"
    spec = json.loads((dist / "spec.json").read_text())
    assert spec["num_steps"] == 4 and "depth" not in spec and "seg" not in spec and spec["edge"] == {"control_path": "e.mp4"}
    assert json.loads((full / "spec.json").read_text())["num_steps"] == 35
    report = (world["tmp"] / "report.md").read_text()
    assert report.count("**pass**") == 3 and "image context none" in report


def run_main(world, monkeypatch, *args):
    monkeypatch.setattr(job, "load_gen_config", lambda path=None: world["cfg"])
    monkeypatch.setattr(sys, "argv", ["job", "--run-folder", world["folder"], "--skip-self-test", *args])
    return job.main()


def test_batch_window_loads_the_model_once_and_keeps_each_run(world, monkeypatch):
    assert run_main(world, monkeypatch, "--max-frames", "93", "--batch") == 0
    c = calls_of(world)
    assert len(c) == 1 and c[0]["argv"].count("--max-frames") == 1
    assert len([x for x in c[0]["argv"] if x.endswith(".json")]) == 4
    for i, name in enumerate(("clear_rendered", "rain_geometry", "snow_geometry", "fog_geometry")):
        d = world["out"] / f"{name}_93f"
        assert (d / "out.mp4").read_bytes() == b"video" and (d / "out_control_edge.mp4").exists()
        t = json.loads((d / "timing.json").read_text())
        assert t["seconds_per_chunk"] == 100.5 + i and t["batch_size"] == 4 and t["image_context"] == "none"
        assert json.loads((d / "spec.json").read_text())["name"] == "out"
        assert (d / "prompt.txt").read_text().strip().endswith(name.split("_")[0])
    assert not list(world["out"].glob("batch.tmp_*"))
    assert run_main(world, monkeypatch, "--max-frames", "93", "--batch") == 0          # a rerun starts nothing
    assert len(calls_of(world)) == 1
    assert not (world["out"] / "clear_rendered").exists()                              # full-length names stay free


def test_batch_keeps_finished_runs_when_one_is_blocked(world, monkeypatch):
    spec = world["specs"] / "03_snow_geometry.json"
    spec.write_text(json.dumps(json.loads(spec.read_text()) | {"prompt": "BLOCK this"}))
    assert run_main(world, monkeypatch, "--max-frames", "93", "--batch") == 1
    done = sorted(p.name for p in world["out"].iterdir() if (p / "out.mp4").exists())
    assert done == ["clear_rendered_93f", "fog_geometry_93f", "rain_geometry_93f"]
    assert "blocked by the model's guardrail" in (world["tmp"] / "report.md").read_text()
    spec.write_text(json.dumps(json.loads(spec.read_text()) | {"prompt": "scene, snow"}))
    assert run_main(world, monkeypatch, "--max-frames", "93", "--batch") == 0          # only the missing run is redone
    assert len([x for x in calls_of(world)[-1]["argv"] if x.endswith(".json")]) == 1
    assert (world["out"] / "snow_geometry_93f" / "out.mp4").exists()


def test_runs_with_different_controls_are_not_batched_together(world, monkeypatch):
    spec = world["specs"] / "04_fog_geometry.json"
    data = json.loads(spec.read_text())
    del data["edge"]
    spec.write_text(json.dumps(data))
    assert run_main(world, monkeypatch, "--batch") == 0
    sizes = sorted(len([x for x in c["argv"] if x.endswith(".json")]) for c in calls_of(world))
    assert sizes == [1, 3]
    assert (world["out"] / "fog_geometry" / "out.mp4").exists()                        # no suffix without --max-frames


def test_single_runs_still_work_one_process_each(world, monkeypatch):
    assert run_main(world, monkeypatch) == 0
    assert len(calls_of(world)) == 4 and (world["out"] / "rain_geometry" / "timing.json").exists()
