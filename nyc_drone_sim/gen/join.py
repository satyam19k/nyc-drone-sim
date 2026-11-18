"""Join two generated chains that share frames, with a linear cross-fade over the shared frames.

  nyc-drone join A.mp4 B.mp4 --overlap 16 --out C.mp4

The last `overlap` frames of A and the first `overlap` frames of B show the same frames of the
flight. Over them the picture fades linearly from A to B (frame k of the overlap is
(1 - w) A + w B with w = (k + 1) / (overlap + 1)), so neither clip is cut. C has
frames(A) + frames(B) - overlap frames, which is checked after writing. When A and B have the
.clip.json that generate keeps beside each clip, their frame ranges are checked too: B must start
`overlap` frames before A ends. Writes C.join.json beside C. Needs only ffmpeg and NumPy; the
frames are streamed, so long clips do not have to fit in memory.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from .common import ffmpeg_exe, write_video


def probe(path: Path) -> tuple[int, int, float]:
    """(width, height, fps) of the first video stream, from ffmpeg's own description."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err)
    f = re.search(r"([\d.]+) fps", err)
    if not m or not f:
        raise SystemExit(f"{path}: not a video ffmpeg can read")
    return int(m.group(1)), int(m.group(2)), float(f.group(1))


def frames(path: Path, w: int, h: int):
    """The decoded frames of a video, as HxWx3 uint8 RGB, one at a time."""
    proc = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed reading {path}")


def clip_range(video: Path) -> tuple[int, int] | None:
    """(start, frames) from the .clip.json kept beside a generated clip, when there is one."""
    side = video.with_name(video.stem + ".clip.json")
    if not side.exists():
        side = video.parent / "clip.json"
    if not side.exists():
        return None
    info = json.loads(side.read_text())
    return int(info["start"]), int(info["frames"])


def join(a: Path, b: Path, overlap: int, out: Path, crf: int = 14) -> dict:
    if out.exists():
        raise SystemExit(f"--out {out} exists; give a new file")
    if overlap < 1:
        raise SystemExit("--overlap: at least 1 frame")
    (wa, ha, fa), (wb, hb, fb) = probe(a), probe(b)
    if (wa, ha) != (wb, hb) or abs(fa - fb) > 1e-3:
        raise SystemExit(f"{a} is {wa}x{ha} at {fa:g} fps, {b} is {wb}x{hb} at {fb:g} fps: they must match")
    ra, rb = clip_range(a), clip_range(b)
    if ra and rb and rb[0] != ra[0] + ra[1] - overlap:
        raise SystemExit(f"{a.name} covers frames {ra[0]} to {ra[0] + ra[1] - 1} and {b.name} starts at {rb[0]}: "
                         f"with --overlap {overlap} it must start at {ra[0] + ra[1] - overlap}")
    count = {"a": 0, "b": 0}
    alpha = np.arange(1, overlap + 1, dtype=np.float32) / (overlap + 1)

    def joined():
        tail = []                                   # the last `overlap` frames of A, held back
        for f in frames(a, wa, ha):
            count["a"] += 1
            tail.append(f)
            if len(tail) > overlap:
                yield tail.pop(0)
        if len(tail) < overlap:
            raise SystemExit(f"{a} has {count['a']} frames, fewer than --overlap {overlap}")
        for f in frames(b, wb, hb):
            k = count["b"]
            count["b"] += 1
            if k < overlap:
                mix = (1 - alpha[k]) * tail[k].astype(np.float32) + alpha[k] * f.astype(np.float32)
                yield np.clip(np.rint(mix), 0, 255).astype(np.uint8)
            else:
                yield f
        if count["b"] < overlap:
            raise SystemExit(f"{b} has {count['b']} frames, fewer than --overlap {overlap}")

    out.parent.mkdir(parents=True, exist_ok=True)
    n = write_video(joined(), out, fa, crf)
    expected = count["a"] + count["b"] - overlap
    if n != expected:
        raise SystemExit(f"{out}: wrote {n} frames, expected {count['a']} + {count['b']} - {overlap} = {expected}")
    info = {"a": str(a.resolve()), "b": str(b.resolve()), "overlap": overlap, "frames_a": count["a"], "frames_b": count["b"],
            "frames": n, "fps": fa, "cross_fade": "linear, w = (k + 1) / (overlap + 1) for overlap frame k"}
    if ra and rb:
        info["flight_frames"] = [ra[0], rb[0] + rb[1] - 1]
    out.with_name(out.stem + ".join.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"{out}: {count['a']} + {count['b']} - {overlap} = {n} frames", flush=True)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nyc-drone join", description="Join two chains that share frames, with a cross-fade.")
    ap.add_argument("a", help="the first clip")
    ap.add_argument("b", help="the second clip, starting --overlap frames before the first ends")
    ap.add_argument("--overlap", type=int, required=True, help="frames the two clips share")
    ap.add_argument("--out", required=True, help="the joined clip (a new file)")
    a = ap.parse_args(argv)
    join(Path(a.a), Path(a.b), a.overlap, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
