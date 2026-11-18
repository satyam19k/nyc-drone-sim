"""render-7 (geometry must match, colour only warns) and the nyc-drone exit code. CPU only."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "checks"))

import render as render_check  # noqa: E402  (checks/render.py)

from nyc_drone_sim import cli  # noqa: E402


def _frame():
    rng = np.random.default_rng(0)
    depth = rng.uniform(20, 400, (36, 64)).astype(np.float32)
    depth[:4] = np.inf                                   # sky
    sem = np.where(rng.random((36, 64)) < 0.5, 3, 7).astype(np.uint32)
    return depth, sem


def test_render7_identical_geometry_matches_even_with_other_class_ids():
    depth, sem = _frame()
    # the second render numbers the same classes differently
    renumbered = np.where(sem == 3, 11, 12).astype(np.uint32)
    d, c = render_check.rerender_agreement(depth, depth * 1.001, sem, {3: "building", 7: "road"},
                                           renumbered, {11: "building", 12: "road"})
    assert d == 1.0 and c == 1.0


def test_render7_changed_geometry_does_not_match():
    depth, sem = _frame()
    moved = depth.copy()
    moved[10:20] *= 1.2
    swapped = np.where(sem == 3, 7, 3).astype(np.uint32)
    d, c = render_check.rerender_agreement(depth, moved, sem, {3: "building", 7: "road"}, swapped, {3: "building", 7: "road"})
    assert d < render_check.GEOMETRY_SHARE and c < render_check.GEOMETRY_SHARE


def test_render7_colour_psnr():
    a = np.full((8, 8, 3), 100.0)
    assert render_check.colour_psnr(a, a) == float("inf")
    assert render_check.colour_psnr(a, a + 8) == pytest.approx(30.07, abs=0.01)   # below 35 dB: a warning


@pytest.mark.parametrize("checks, poses, code", [
    ({"scene": "passed", "path": "passed", "render": "passed"}, None, 0),            # warnings leave a check passed
    ({"scene": "passed", "path": "passed", "render": "failed (exit 1), see check.log"}, None, 1),
    ({"scene": "failed (exit 1), see check.log"}, None, 1),
    (None, {"matches": True}, 0),
    (None, {"matches": False}, 1),
    ({"scene": "passed"}, {"matches": False}, 1),
])
def test_exit_code(checks, poses, code):
    assert cli.exit_code(checks, poses) == code


def test_run_exits_1_when_a_check_fails_and_0_when_all_pass(tmp_path, monkeypatch):
    for name in ("check_simulator", "check_model", "fetch_models_if_missing", "step_scene", "step_render",
                 "write_report"):
        monkeypatch.setattr(cli, name, lambda *a, **k: None)
    args = ["run", "--lat", "40.7", "--lon", "-74.0", "--out", str(tmp_path / "r")]
    for result, code in (("passed", 0), ("failed (exit 1), see check.log", 1)):
        monkeypatch.setattr(cli, "step_check", lambda *a, r=result, **k: {"scene": "passed", "render": r})
        with pytest.raises(SystemExit) as e:
            cli.main(args)
        assert e.value.code == code
