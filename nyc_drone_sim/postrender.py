"""Stage 5 post-step, CPU: check each MP4 with ffprobe and re-encode from the PNG frames with
ffmpeg if its frame count or rate differs from the config (or the writer skipped encoding)."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from .common import for_location, REPO_ROOT, clip_settings, load_config, scene_dir, slugify

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


def id_to_class(bdir: Path) -> dict[int, str]:
    """Class of every semantic id in a bundle: the map saved by the renderer, completed by
    semantic_labels.json when complete_labels() has written one."""
    b = json.loads((bdir / "bundle.json").read_text())
    ids = {int(i): v["class"] for i, v in b["semantic_info"]["idToLabels"].items()}
    extra = bdir / "semantic_labels.json"
    if extra.exists():
        ids.update({int(i): c for i, c in json.loads(extra.read_text())["added"].items()})
    return ids


def complete_labels(bdir: Path, sdir: Path, cfg: dict, every: int = 16) -> dict:
    """Name the semantic ids that appear in a clip's frames but not in its saved label map.

    Bundles rendered before the renderer kept the union of every frame's labels only list the
    classes visible in the last frame. The missing ids are named from the scene itself: pixels of
    the id are projected to the world with the frame's metric depth and saved camera pose, and
    looked up in the scene's surface polygons, building footprints and placed objects. The
    result goes in semantic_labels.json beside bundle.json, which is left untouched.
    """
    import numpy as np
    import shapely
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    b = json.loads((bdir / "bundle.json").read_text())
    known = {int(i): v["class"] for i, v in b["semantic_info"]["idToLabels"].items()}
    path = json.loads((bdir / "path.json").read_text())
    K = path["intrinsics"]
    n = b["clip"]["frames"]
    samples: dict[int, list] = {}
    for i in range(0, n, every):
        sem = np.load(bdir / "semantic" / f"semantic_{i:04d}.npz")["data"]
        new = [int(k) for k in np.unique(sem) if int(k) not in known]
        if not new:
            continue
        d = np.load(bdir / "metric_depth" / f"depth_{i:04d}.npy")
        M = np.array(path["frames"][i]["camera_to_world"])
        for k in new:
            v, u = np.nonzero(sem == k)
            pick = np.linspace(0, len(u) - 1, min(len(u), 200)).astype(int)
            u, v = u[pick], v[pick]
            z = d[v, u]
            cam = np.column_stack([(u + 0.5 - K["cx"]) / K["fx"] * z, -(v + 0.5 - K["cy"]) / K["fy"] * z, -z, np.ones(len(z))])
            samples.setdefault(k, []).append((M @ cam.T).T[:, :3])
    if not samples:
        return {"added": {}, "evidence": {}}
    surf = json.loads((sdir / "surfaces.geojson").read_text())["features"]
    geoms = [shape(f["geometry"]) for f in surf]
    classes = [f["properties"]["class"] for f in surf]
    tree = STRtree(geoms)
    placed = json.loads((sdir / "placements.json").read_text())["objects"] if (sdir / "placements.json").exists() else []
    static = [o for o in placed if o.get("motion", "static") == "static"]
    otree = STRtree(shapely.points(np.array([[o["x"], o["y"]] for o in static]))) if static else None
    flat = float(cfg["populate"].get("label_flat_height_m", 0.15)) if "populate" in cfg else 0.15
    added, evidence = {}, {}
    for k, parts in samples.items():
        pts = np.vstack(parts)
        votes: dict[str, int] = {}
        for x, y, z in pts:
            p = shapely.Point(x, y)
            if z <= flat:
                hit = [classes[j] for j in tree.query(p, predicate="intersects")]
                c = hit[0] if hit else "ground"
            elif otree is not None:
                j = int(otree.nearest(p))
                c = static[j]["class"] if p.distance(shapely.Point(static[j]["x"], static[j]["y"])) < 4.0 else "unknown"
            else:
                c = "unknown"
            votes[c] = votes.get(c, 0) + 1
        best = max(votes, key=votes.get)
        share = votes[best] / len(pts)
        evidence[k] = {"class": best, "share_of_pixels": round(share, 4), "pixels_tested": len(pts), "votes": votes}
        if best != "unknown" and share >= 0.9:
            added[k] = best
    out = {"added": {str(k): c for k, c in sorted(added.items())}, "evidence": {str(k): e for k, e in sorted(evidence.items())},
           "method": "pixels projected to the world with metric depth and the saved camera pose, looked up in the scene's "
                     "surfaces and placed objects; an id is named only when at least 90% of its tested pixels agree"}
    (bdir / "semantic_labels.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


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
        bdir = bundle_dir(cfg, a.run, slugify(a.location), alt)
        r = fix_bundle(bdir, cfg, a.variant)
        added = complete_labels(bdir, scene_dir(cfg, slugify(a.location)), cfg)["added"]
        if added:
            log.info("%s ft: semantic ids missing from the saved label map, named from the scene: %s", alt, added)
        log.info("%s ft: %s", alt, {k: (v["frames"], v["fps"], "re-encoded" if v["re_encoded"] else "as written") for k, v in r.items()})
