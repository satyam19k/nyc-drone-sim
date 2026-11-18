"""CPU only: nyc-drone join on small made-up videos, with the real ffmpeg."""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

try:                                          # the CPU environment brings ffmpeg in this package
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

from nyc_drone_sim import cli
from nyc_drone_sim.gen.common import write_video
from nyc_drone_sim.gen.join import frames, probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
W, H = 64, 32


def video(path: Path, values: list[int]) -> Path:
    """Flat grey frames, one value per frame."""
    write_video((np.full((H, W, 3), v, np.uint8) for v in values), path, 16, 0)
    return path


def means(path: Path) -> list[float]:
    return [float(f.mean()) for f in frames(path, *probe(path)[:2])]


def test_join_cross_fades_the_shared_frames(tmp_path):
    a = video(tmp_path / "a.mp4", [40] * 20)
    b = video(tmp_path / "b.mp4", [200] * 30)
    with pytest.raises(SystemExit) as e:
        cli.main(["join", str(a), str(b), "--overlap", "4", "--out", str(tmp_path / "c.mp4")])
    assert e.value.code == 0
    m = means(tmp_path / "c.mp4")
    assert len(m) == 20 + 30 - 4
    # H.264 in yuv420p moves a flat grey by up to 2 levels
    assert m[:16] == pytest.approx([40] * 16, abs=3) and m[20:] == pytest.approx([200] * 26, abs=3)
    # overlap frame k: (1 - w) 40 + w 200, w = (k + 1) / 5
    assert m[16:20] == pytest.approx([40 + 160 * (k + 1) / 5 for k in range(4)], abs=3)
    info = json.loads((tmp_path / "c.join.json").read_text())
    assert (info["frames_a"], info["frames_b"], info["frames"]) == (20, 30, 46)


def test_join_checks_the_frame_ranges(tmp_path):
    a = video(tmp_path / "a.mp4", [40] * 20)
    b = video(tmp_path / "b.mp4", [200] * 30)
    (tmp_path / "a.clip.json").write_text(json.dumps({"start": 100, "frames": 20}))
    (tmp_path / "b.clip.json").write_text(json.dumps({"start": 110, "frames": 30}))
    with pytest.raises(SystemExit, match="must start at 116"):
        cli.main(["join", str(a), str(b), "--overlap", "4", "--out", str(tmp_path / "c.mp4")])
    (tmp_path / "b.clip.json").write_text(json.dumps({"start": 116, "frames": 30}))
    with pytest.raises(SystemExit):
        cli.main(["join", str(a), str(b), "--overlap", "4", "--out", str(tmp_path / "c.mp4")])
    assert json.loads((tmp_path / "c.join.json").read_text())["flight_frames"] == [100, 145]


def test_join_refuses_what_does_not_fit(tmp_path):
    a = video(tmp_path / "a.mp4", [40] * 3)
    b = video(tmp_path / "b.mp4", [200] * 30)
    with pytest.raises(SystemExit, match="fewer than --overlap"):
        cli.main(["join", str(a), str(b), "--overlap", "4", "--out", str(tmp_path / "c.mp4")])
    (tmp_path / "d.mp4").write_bytes(b"x")
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["join", str(b), str(b), "--overlap", "4", "--out", str(tmp_path / "d.mp4")])
