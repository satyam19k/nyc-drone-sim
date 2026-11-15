"""Stage 5 post-step, CPU: check each MP4 with ffprobe and re-encode from the PNG frames with
ffmpeg if its frame count or rate differs from the config (or the writer skipped encoding)."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from .common import for_location, REPO_ROOT, clip_settings, load_config, slugify

log = logging.getLogger("postrender")
MODALITIES = ("rgb", "depth", "segmentation", "shaded_seg", "edges")


def ff_tools() -> tuple[str, str]:
    import shutil
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return shutil.which("ffmpeg"), shutil.which("ffprobe")
    from static_ffmpeg import run
    return run.get_or_fetch_platform_executables_else_raise()


def probe(mp4: Path) -> dict | None:
    if not mp4.exists():
        return None
    out = subprocess.run([ff_tools()[1], "-v", "error", "-select_streams", "v:0", "-count_frames",
                          "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
                          "-of", "json", str(mp4)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = (int(v) for v in s["avg_frame_rate"].split("/"))
    return {"codec": s["codec_name"], "width": int(s["width"]), "height": int(s["height"]),
            "fps": num / den if den else 0.0, "frames": int(s["nb_read_frames"])}


def bundle_dir(cfg: dict, run_id: str, slug: str, alt_ft) -> Path:
    return REPO_ROOT / cfg["output"]["runs_dir"] / run_id / slug / str(int(alt_ft))


def fix_bundle(bdir: Path, cfg: dict, variant: str | None = None) -> dict:
    frames, fps = clip_settings(cfg, variant)
    report = {}
    for k in MODALITIES:
        mp4 = bdir / "clip_0000" / f"{k}.mp4"
        info = probe(mp4)
        ok = info is not None and info["frames"] == frames and abs(info["fps"] - fps) < 1e-6
        if not ok:
            log.info("%s: %s -> re-encoding from PNG frames", mp4, info)
            subprocess.run([ff_tools()[0], "-y", "-v", "error", "-framerate", f"{fps:g}", "-i",
                            str(bdir / "clip_0000" / k / f"{k}_%04d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", str(mp4)], check=True)
            info = probe(mp4)
        report[k] = {**info, "re_encoded": not ok}
    b = json.loads((bdir / "bundle.json").read_text())
    b["mp4_check"] = report
    (bdir / "bundle.json").write_text(json.dumps(b, indent=1) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variant", default=None, choices=["street", "long"])
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = for_location(load_config(a.config), a.location)
    for alt in cfg["altitudes_ft"]:
        r = fix_bundle(bundle_dir(cfg, a.run, slugify(a.location), alt), cfg, a.variant)
        log.info("%s ft: %s", alt, {k: (v["frames"], v["fps"], "re-encoded" if v["re_encoded"] else "as written") for k, v in r.items()})
