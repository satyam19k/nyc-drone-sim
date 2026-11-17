"""Build the before-and-after comparisons for the populated scene.

  python scripts/populate_evidence.py --before run_003 --after run_005

Writes under docs/validation/images/: a before/after still and a side-by-side clip per altitude, a contact
sheet (RGB, depth, segmentation) of one populated frame per altitude, and a close crop at the
lower altitude. The two runs must share camera paths; the script refuses to compare otherwise.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nyc_drone_sim.common import load_config, scene_dir, slugify  # noqa: E402
from nyc_drone_sim.postrender import bundle_dir, ff_tools  # noqa: E402


def labelled(im: Image.Image, text: str) -> Image.Image:
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 12 + 8 * len(text), 26], fill=(0, 0, 0))
    d.text((6, 6), text, fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="120 Broadway")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--frame", type=int, default=46)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    slug = slugify(a.location)
    ev = REPO / "docs" / "validation" / "images"
    out = []
    for alt in cfg["altitudes_ft"]:
        alt = int(alt)
        B, A = bundle_dir(cfg, a.before, slug, alt), bundle_dir(cfg, a.after, slug, alt)
        pb, pa = (json.loads((d / "path.json").read_text())["frames"] for d in (B, A))
        same = len(pb) == len(pa) and np.allclose([f["camera_to_world"] for f in pb], [f["camera_to_world"] for f in pa], atol=1e-9)
        if not same:
            raise SystemExit(f"{alt} ft: the two runs do not share a camera path")
        f = a.frame
        b = labelled(Image.open(B / "clip_0000" / "rgb" / f"rgb_{f:04d}.png"), f"before: buildings only, {alt} ft, frame {f}")
        p = labelled(Image.open(A / "clip_0000" / "rgb" / f"rgb_{f:04d}.png"), f"after: populated, {alt} ft, frame {f}")
        W, H = b.size
        pair = Image.new("RGB", (2 * W + 8, H), (255, 255, 255))
        pair.paste(b, (0, 0))
        pair.paste(p, (W + 8, 0))
        dst = ev / f"populate_before_after_{slug}_{alt}.png"
        pair.resize((pair.width // 2, pair.height // 2), Image.LANCZOS).save(dst)
        out.append(dst)

        sheet = Image.new("RGB", (3 * W + 16, H), (255, 255, 255))
        for k, m in enumerate(("rgb", "depth", "segmentation")):
            sheet.paste(labelled(Image.open(A / "clip_0000" / m / f"{m}_{f:04d}.png"), f"{m}, {alt} ft, frame {f}"), (k * (W + 8), 0))
        dst = ev / f"populate_sheet_{slug}_{alt}.png"
        sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(dst)
        out.append(dst)

        dst = ev / f"populate_before_after_{slug}_{alt}.mp4"
        subprocess.run([ff_tools()[0], "-y", "-v", "error", "-i", str(B / "clip_0000" / "rgb.mp4"), "-i",
                        str(A / "clip_0000" / "rgb.mp4"), "-filter_complex", "[0:v][1:v]hstack=inputs=2,scale=1920:-2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)], check=True)
        out.append(dst)

    # close crop from the dedicated oblique shot at the lower altitude
    pdir = scene_dir(cfg, slug) / "populate_probe" / "scene"
    shot = pdir / "scene_close.npz"
    if shot.exists():
        rgb = Image.fromarray(np.load(shot)["rgb"][..., :3])
        W, H = rgb.size
        cw, ch = 520, 300
        crop = rgb.crop((W // 2 - cw // 2, H // 2 - ch // 2, W // 2 + cw // 2, H // 2 + ch // 2))
        both = Image.new("RGB", (W, H + ch * 2 + 8), (255, 255, 255))
        both.paste(labelled(rgb, "150 ft, pitch -45 deg: full frame"), (0, 0))
        both.paste(labelled(crop.resize((cw * 2, ch * 2), Image.LANCZOS), "centre crop, enlarged 2x"), ((W - cw * 2) // 2, H + 8))
        dst = ev / f"populate_close_{slug}.png"
        both.save(dst)
        out.append(dst)
    for p in out:
        print(f"{p.relative_to(REPO)} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
