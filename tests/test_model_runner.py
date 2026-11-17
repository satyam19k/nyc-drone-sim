"""The Python model runner: where it finds the model, batch skipping, and the clip folder rules.
A stand-in examples/inference.py takes the model's place, so no GPU or weights are needed."""
import json
import sys
from pathlib import Path

import pytest

from nyc_drone_sim.gen import model

FAKE = """
import argparse, json, pathlib, sys
ap = argparse.ArgumentParser()
ap.add_argument("-i", nargs="+")
ap.add_argument("-o")
a = ap.parse_args()
out = pathlib.Path(a.o)
out.mkdir(parents=True, exist_ok=True)
for spec in a.i:
    name = json.loads(pathlib.Path(spec).read_text())["name"]
    if name == "broken":
        sys.exit(3)
    (out / f"{name}.mp4").write_bytes(b"video")
    print("Generating chunk 1/1")
print("Average time per chunk: 1.5")
"""


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "model"
    (r / "examples").mkdir(parents=True)
    (r / "examples" / "inference.py").write_text(FAKE)
    monkeypatch.setenv("MODEL_REPO", str(r))
    monkeypatch.setenv("MODEL_PYTHON", sys.executable)
    return r


def spec(tmp_path: Path, name: str) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"name": name, "prompt": "p", "seed": 1, "video_path": str(tmp_path / "x" / "clip_0000" / "rgb.mp4")}))
    return p


CFG = {"model": {"repo_dir": "does/not/exist", "env_dir": "does/not/exist"}}


def test_environment_variable_wins_over_config(repo):
    assert model.model_repo(CFG) == repo
    assert model.model_python(CFG) == sys.executable


def test_missing_checkout_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MODEL_REPO", raising=False)
    with pytest.raises(RuntimeError, match="MODEL_REPO"):
        model.model_repo(CFG)


def test_batch_skips_videos_that_exist(repo, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.mp4").write_bytes(b"old")
    assert model.run_batch([spec(tmp_path, "a"), spec(tmp_path, "b")], out, CFG) == 0
    assert (out / "a.mp4").read_bytes() == b"old"
    assert (out / "b.mp4").read_bytes() == b"video"


def test_clip_folder_appears_only_on_success_and_is_never_overwritten(repo, tmp_path):
    final = tmp_path / "gen" / "clear_K2_chain"
    s = spec(tmp_path, "out")
    first = model.run_clip(s, final, CFG)
    second = model.run_clip(s, final, CFG)
    assert first == final and second == final.with_name("clear_K2_chain_r2")
    timing = json.loads((first / "timing.json").read_text())
    assert timing["chunks"] == 1 and timing["seconds_per_chunk"] == 1.5
    assert (first / "prompt.txt").read_text() == "p\n"
    assert model.run_clip(spec(tmp_path, "broken"), tmp_path / "gen" / "failed", CFG) is None
    assert not (tmp_path / "gen" / "failed").exists()
