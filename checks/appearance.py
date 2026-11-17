"""Check that a render with a weather or look flag kept the geometry: its metric depth and its
semantic classes must match a reference render of the same frames. CPU only.

  python checks/appearance.py --render runs/<run>/<location>/<altitude> \\
      --reference runs/<clear run>/<location>/<altitude> [--offset FIRST_FRAME]

--offset is the reference frame that the render's frame 0 corresponds to (a window of a clip).
Passes when, over all compared frames, at least `min_share` of the pixels have a depth within
`depth_rel_tol` of the reference and at least `min_share` have the same semantic class.
Writes <render>/appearance_check.json and exits 0 on a pass, 1 on a fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEPTH_REL_TOL = 0.01      # a pixel's depth matches when it is within 1% of the reference
MIN_SHARE = 0.99          # share of pixels that must match, for depth and for class


def labels(bdir: Path) -> dict[int, str]:
    """Semantic id to class name: the bundle's map plus ids named afterwards from the scene."""
    info = json.loads((bdir / "bundle.json").read_text())["semantic_info"]
    out = {int(k): v["class"] for k, v in info["idToLabels"].items()}
    f = bdir / "semantic_labels.json"
    if f.exists():
        out.update({int(k): v for k, v in json.loads(f.read_text()).get("added", {}).items()})
    return out


def classes(bdir: Path, index: int, names: dict[int, str], table: dict[str, int]) -> np.ndarray:
    data = np.load(bdir / "semantic" / f"semantic_{index:04d}.npz")["data"]
    lut = np.zeros(max(max(names), int(data.max())) + 1, dtype=np.int32)
    for k in range(len(lut)):
        lut[k] = table.setdefault(names.get(k, f"unnamed id {k} in {bdir}"), len(table))
    return lut[data]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--step", type=int, default=4, help="compare every this-many-th frame")
    a = ap.parse_args()
    new, ref = Path(a.render), Path(a.reference)
    n = len(list((new / "metric_depth").glob("depth_*.npy")))
    if n == 0:
        print("no frames in", new)
        return 1
    ln, lr, table = labels(new), labels(ref), {}
    depth_ok = class_ok = total = 0
    worst = 0.0
    for i in range(0, n, a.step):
        dn = np.load(new / "metric_depth" / f"depth_{i:04d}.npy")
        dr = np.load(ref / "metric_depth" / f"depth_{i + a.offset:04d}.npy")
        rel = np.abs(dn - dr) / np.maximum(dr, 1e-6)
        depth_ok += int((rel <= DEPTH_REL_TOL).sum())
        worst = max(worst, float(np.median(rel)))
        class_ok += int((classes(new, i, ln, table) == classes(ref, i + a.offset, lr, table)).sum())
        total += dn.size
    result = {"render": str(new), "reference": str(ref), "offset": a.offset, "frames_compared": len(range(0, n, a.step)),
              "depth_share_within_1pct": depth_ok / total, "worst_frame_median_depth_error": worst,
              "class_share_equal": class_ok / total, "min_share": MIN_SHARE}
    result["passed"] = result["depth_share_within_1pct"] >= MIN_SHARE and result["class_share_equal"] >= MIN_SHARE
    (new / "appearance_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
