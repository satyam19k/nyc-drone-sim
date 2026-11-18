"""CPU only: frame i of a render made with --frames START:COUNT is path frame START + i."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from nyc_drone_sim.postrender import path_frames


def bundle(tmp_path: Path, window) -> Path:
    frames = [{"index": i, "camera_to_world": [[1, 0, 0, i], [0, 1, 0, 0], [0, 0, 1, 45], [0, 0, 0, 1]]} for i in range(768)]
    (tmp_path / "path.json").write_text(json.dumps({"frames": frames}))
    (tmp_path / "bundle.json").write_text(json.dumps({"frame_window": window}))
    return tmp_path


def test_window_render_starts_at_its_first_path_frame(tmp_path):
    fr = path_frames(bundle(tmp_path, "598:93"))
    assert len(fr) == 93 and [fr[0]["index"], fr[92]["index"]] == [598, 690]
    assert fr[10]["camera_to_world"][0][3] == 608


@pytest.mark.parametrize("window", ["all", None])
def test_whole_clip_is_the_whole_path(tmp_path, window):
    assert len(path_frames(bundle(tmp_path, window))) == 768


def test_window_outside_the_path_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="outside"):
        path_frames(bundle(tmp_path, "700:93"))
