"""Check every chunk join of a generated full clip, and draw the frames either side of each join.

  python scripts/gen_seam_check.py --clip <output folder with out.mp4> [--sheet evidence/name.png]

A join is the first frame of a chunk after the first: frames 93, 185, 277, ... for 93-frame chunks
that overlap by one frame. For each join the change from the frame before is compared with the
clip's own frame-to-frame changes. A join is flagged when its change is more than 4 standard
deviations above the clip mean, or more than twice the larger of its two neighbours. Also reports
colour drift from chunk to chunk. Writes seams.json in the clip folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--chunk", type=int, default=93)
    ap.add_argument("--overlap", type=int, default=1)
    a = ap.parse_args()
    clip = Path(a.clip)
    cap = cv2.VideoCapture(str(clip / "out.mp4"))
    small = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small.append(cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA).astype(np.int16))
    n = len(small)
    d = np.array([np.abs(small[i + 1] - small[i]).mean() for i in range(n - 1)])
    joins = list(range(a.chunk, n, a.chunk - a.overlap))
    mean, sd = float(d.mean()), float(d.std())
    out = {"frames": n, "mean_change": mean, "sd_change": sd, "limit_4sd": mean + 4 * sd, "joins": []}
    for j in joins:
        v = float(d[j - 1])
        nb = max(float(d[j - 2]), float(d[j])) if j < n - 1 else float(d[j - 2])
        flagged = v > mean + 4 * sd or v > 2 * nb
        out["joins"].append({"frame": j, "change": v, "sd_above_mean": (v - mean) / sd if sd else 0.0,
                             "neighbours": [float(d[j - 2]), float(d[j]) if j < n - 1 else None], "flagged": bool(flagged)})
    b = [0] + joins + [n]
    out["chunk_mean_bgr"] = [np.round(np.mean([f.mean(axis=(0, 1)) for f in small[b[k]:b[k + 1]]], axis=0), 1).tolist()
                             for k in range(len(b) - 1)]
    out["flagged_joins"] = [x["frame"] for x in out["joins"] if x["flagged"]]
    (clip / "seams.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{clip.name}: {n} frames, change mean {mean:.2f} sd {sd:.2f}")
    for x in out["joins"]:
        print(f"  join into frame {x['frame']}: {x['change']:.2f} ({x['sd_above_mean']:+.1f} sd){'  FLAGGED' if x['flagged'] else ''}")
    print("  chunk mean blue,green,red:", out["chunk_mean_bgr"])
    if a.sheet:
        cap = cv2.VideoCapture(str(clip / "out.mp4"))
        rows = []
        for j in joins:
            row = []
            for k in (j - 2, j - 1, j, j + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(k, n - 1))
                ok, f = cap.read()
                f = f.copy()
                cv2.rectangle(f, (0, 0), (f.shape[1], 44), (0, 0, 0), -1)
                note = "  last of chunk" if k == j - 1 else "  first of next chunk" if k == j else ""
                cv2.putText(f, f"frame {k}{note}", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                row.append(f)
            rows.append(np.hstack(row))
        cv2.imwrite(a.sheet, cv2.resize(np.vstack(rows), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
        print("  sheet:", a.sheet)
    return 1 if out["flagged_joins"] else 0


if __name__ == "__main__":
    sys.exit(main())
