"""Generate one full clip with the video model, by one of three methods. Needs a GPU.

  python -m nyc_drone_sim.gen.makeclip --run-folder runs/run_009/canal-shore-street --altitude 150 \
      --weather clear --setting K2 --method windows|pieces|chain

chain    One run of the model over the whole clip. The model works in 93-frame chunks and starts
         each chunk from the last frame of the one before, so a fault in one chunk carries into
         the next.
windows  Overlapping 93-frame windows, each generated alone from its own controls and its own
         prompt, with no frame taken from another window. The windows are then joined on the CPU:
         once with a linear cross-fade over each overlap (out.mp4, out_crossfade.mp4) and once with
         a hard cut at the middle of each overlap (out_hardcut.mp4). Neighbours whose colours
         differ in the overlap by more than a threshold are colour-matched before the cross-fade.
pieces   Like windows, with longer independent pieces (--piece-frames, 185 = two chained chunks).

Either way the result goes to <bundle>/gen/<model>/<weather>_<setting>_<method>/ with an 8-frame
sheet under docs/validation/images/ and a section appended to runs/logs/gen_summary.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from .model import run_batch, run_clip
from .common import REPO_ROOT, bundle_dir, bundle_hash, inputs_dir, load_gen_config, model_dir, scene_name, video_info, write_video

WINDOW, OVERLAP = 93, 16
MATCH_THRESHOLD = 5.0          # grey levels: larger colour differences in an overlap are matched
SUMMARY = REPO_ROOT / "runs" / "logs" / "gen_summary.md"      # a run log, not part of the repository


def prompts() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "gen_prompts_windows.yaml").read_text())


def window_starts(n: int, piece: int = WINDOW) -> list[int]:
    """Starts of overlapping pieces; the last one ends on the clip's last frame."""
    s = list(range(0, n - piece + 1, piece - OVERLAP))
    if s[-1] + piece < n:
        last = n - piece
        if last - s[-1] < OVERLAP:          # the regular last window would nearly repeat it
            s[-1] = last
        else:
            s.append(last)
    return s


def water_share(bdir: Path, start: int, frames: int) -> float:
    from nyc_drone_sim.postrender import id_to_class
    ids = [k for k, c in id_to_class(bdir).items() if c == "water"]
    if not ids:
        return 0.0
    shares = []
    for i in range(start, start + frames, 8):
        s = np.load(bdir / "semantic" / f"semantic_{i:04d}.npz")["data"]
        shares.append(float(np.isin(s, ids).mean()))
    return float(np.mean(shares))


def scene_legs(bdir: Path, p: dict) -> list[dict]:
    """The scene's own prompt legs, or the generic city description for a place without any."""
    return p["scenes"].get(scene_name(bdir)) or p["scenes"]["default"]


def prompt_for(bdir: Path, weather: str, start: int, frames: int, p: dict) -> tuple[str, bool]:
    legs = scene_legs(bdir, p)
    mid = start + frames // 2
    leg = next(l for l in legs if mid < int(l["to_frame"]))
    wet = water_share(bdir, start, frames) >= float(p["water_share_min"])
    key = "with_water" if wet else "without_water"
    text = " ".join(leg[key].split()) + " " + " ".join(p["weather"][weather][key].split())
    if "water tank" in text.lower():
        raise RuntimeError("a prompt mentions water tanks")
    return text, wet


def base_spec(setting: dict, cfg: dict, name: str, prompt: str, video: Path, depth: Path, edge: Path, seg: Path) -> dict:
    sc = cfg["sampling"]
    spec = {"name": name, "prompt": prompt, "video_path": str(video), "seed": int(sc["seed"]),
            "guidance": int(setting["guidance"]), "num_steps": int(sc["num_steps"]), "resolution": str(sc["resolution"]),
            "num_video_frames_per_chunk": int(sc["chunk_frames"]), "num_conditional_frames": int(sc["num_conditional_frames"])}
    d, e, s = setting["weights"]
    if d is not None:
        spec["depth"] = {"control_path": str(depth), "control_weight": float(d)}
    if e is not None:
        spec["edge"] = {"control_path": str(edge), "control_weight": float(e)}
    if s is not None:
        spec["seg"] = {"control_path": str(seg), "control_weight": float(s)}
    return spec


def read_video(path: Path) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def chunk_brightness(video: np.ndarray, bounds: list[int]) -> list[float]:
    return [float(video[a:b:4].mean()) for a, b in zip(bounds[:-1], bounds[1:])]


def sheet(bdir: Path, video: np.ndarray, title: str, out_png: Path) -> None:
    import cv2
    n = len(video)
    idx = [int(round(v)) for v in np.linspace(0, n - 1, 8)]
    sim = read_video(bdir / "clip_0000" / "rgb.mp4")

    def lab(im, t):
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), (640, 360), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (640, 26), (0, 0, 0), -1)
        cv2.putText(im, t, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return im
    rows = []
    for part in (idx[:4], idx[4:]):
        rows.append(np.hstack([lab(sim[i], f"simulator {i} ({i / 16:.0f} s)") for i in part]))
        rows.append(np.hstack([lab(video[i], f"{title} {i}") for i in part]))
    out_png.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack(rows))


def summarise(lines: list[str]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a") as f:
        f.write("\n".join(lines) + "\n\n")
    print("\n".join(lines))


def unique_dir(path: Path) -> Path:
    out, n = path, 2
    while out.exists():
        out = path.with_name(f"{path.name}_r{n}")
        n += 1
    return out


def join_windows(work: Path, starts: list[int], piece: int, n: int, fps: float) -> list[dict] | None:
    """Join the generated pieces in <work>/windows into three clips, two pieces in memory at a time:
      out_hardcut.mp4            raw pieces, cut at the middle of each overlap
      out_crossfade.mp4          raw pieces, linear cross-fade over each overlap (also out.mp4)
      out_crossfade_matched.mp4  the same, after a local colour match: on each side of a join the
                                 piece is shifted toward the two pieces' common mean colour, at
                                 full strength inside the overlap and fading to nothing over the
                                 same number of frames beyond it
    Each piece is only ever adjusted from its own raw frames and its raw neighbours, so a
    correction cannot build up along the clip. Returns the join measurements, or None when a
    joined clip has the wrong length."""
    def load(st):
        return read_video(work / "windows" / f"w_{st:04d}.mp4").astype(np.float32)

    ovs = [a_ + piece - b_ for a_, b_ in zip(starts[:-1], starts[1:])]
    # colour of each side of each join, from the raw pieces
    tails, heads = [], []
    for k, st in enumerate(starts):
        w = load(st)
        heads.append(w[:ovs[k - 1]].mean(axis=(0, 1, 2)) if k > 0 else None)
        tails.append(w[piece - ovs[k]:].mean(axis=(0, 1, 2)) if k < len(ovs) else None)
    joins = []
    for k, ov in enumerate(ovs):
        d = np.abs(tails[k] - heads[k + 1])
        joins.append({"frames": [starts[k + 1], starts[k] + piece - 1], "overlap": ov, "colour_difference": float(d.max()),
                      "colour_difference_per_channel_rgb": [float(v) for v in d]})

    def corrected(k: int) -> np.ndarray:
        """Piece k with the local colour match toward each neighbour."""
        w = load(starts[k])
        shift = np.zeros((piece, 3), np.float32)
        if k > 0:                                   # head: full shift in the overlap, fading out after it
            ov = ovs[k - 1]
            delta = (tails[k - 1] + heads[k]) / 2 - heads[k]
            ramp = np.concatenate([np.ones(ov), np.linspace(1, 0, ov + 2)[1:-1]])[:piece]
            shift[:len(ramp)] += ramp[:, None] * delta
        if k < len(ovs):                            # tail: fading in before the overlap, full inside it
            ov = ovs[k]
            delta = (tails[k] + heads[k + 1]) / 2 - tails[k]
            ramp = np.concatenate([np.linspace(0, 1, ov + 2)[1:-1], np.ones(ov)])[-piece:]
            shift[piece - len(ramp):] += ramp[:, None] * delta
        return np.clip(w + shift[:, None, None, :], 0, 255)

    def joined(fade: bool, match: bool):
        get = corrected if match else (lambda k: load(starts[k]))
        prev, skip = get(0), 0
        for k, ov in enumerate(ovs):
            cur = get(k + 1)
            if fade:
                for f in prev[skip:piece - ov]:
                    yield f
                alpha = (np.arange(1, ov + 1, dtype=np.float32) / (ov + 1))[:, None, None, None]
                for f in prev[piece - ov:] * (1 - alpha) + cur[:ov] * alpha:
                    yield f
                skip = ov
            else:
                cut = ov // 2
                for f in prev[skip:piece - ov + cut]:
                    yield f
                skip = cut
            prev = cur
        for f in prev[skip:]:
            yield f

    counts = []
    for name, fade, match in (("out_hardcut.mp4", False, False), ("out_crossfade.mp4", True, False), ("out_crossfade_matched.mp4", True, True)):
        counts.append(write_video((np.clip(f, 0, 255).astype(np.uint8) for f in joined(fade, match)), work / name, fps, 14))
    if any(c != n for c in counts):
        return None
    import shutil
    shutil.copy(work / "out_crossfade.mp4", work / "out.mp4")
    return joins


def run_windows(bdir: Path, weather: str, setting_name: str, cfg: dict, piece: int = WINDOW) -> int:
    """piece = 93: independent single-chunk windows. piece = 185: independent pieces of two
    chunks each, chained inside a piece by the model and independent between pieces."""
    p = prompts()
    setting = p["settings"][setting_name]
    n = video_info(bdir / "clip_0000" / "rgb.mp4")["frames"]
    starts = window_starts(n, piece)
    method = "windows" if piece == WINDOW else f"pieces{piece}"
    prefix = "ov" if piece == WINDOW else f"pc{piece}"
    final = unique_dir(model_dir(bdir, cfg) / f"{weather}_{setting_name}_{method}")
    work = final.with_name(final.name + ".tmp_" + time.strftime("%Y%m%d_%H%M%S"))
    (work / "specs").mkdir(parents=True)
    edge_file = "edges_rendered.mp4" if setting["edge"] == "rendered" else "edges_geometry.mp4"
    specs, meta = [], []
    for s in starts:
        w = bdir / "gen" / "windows" / f"{prefix}_{s:04d}"
        if not (w / "window.json").exists():
            from .windows import cut_window
            cut_window(bdir, f"{prefix}_{s:04d}", s, piece, cfg)
        prompt, wet = prompt_for(bdir, weather, s, piece, p)
        spec = base_spec(setting, cfg, f"w_{s:04d}", prompt, w / "rgb.mp4", w / "depth_control.mp4", w / edge_file, w / "segmentation.mp4")
        path = work / "specs" / f"w_{s:04d}.json"
        path.write_text(json.dumps(spec, indent=1) + "\n")
        specs.append(path)
        meta.append({"start": s, "water_in_frame": wet, "prompt": prompt})
    t0 = time.time()
    rc = run_batch(specs, work / "windows", cfg, log=work / "run.log")
    seconds = time.time() - t0
    missing = [s for s in starts if not (work / "windows" / f"w_{s:04d}.mp4").exists()]
    if rc != 0 or missing:
        summarise([f"## FAILED: {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, window method",
                   f"- exit code {rc}; windows missing: {missing}; kept in `{work}`"])
        return 1
    # ---- join on the CPU
    fps = float(cfg["controls"]["fps"])
    joins = join_windows(work, starts, piece, n, fps)
    if joins is None:
        summarise([f"## FAILED: {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, {method}",
                   f"- a joined clip does not have {n} frames; kept in `{work}`"])
        return 1
    cross = read_video(work / "out.mp4")
    bright = chunk_brightness(cross, starts + [n])
    info = {"method": method, "piece_frames": piece, "scene": scene_name(bdir), "altitude_ft": int(bdir.name), "weather": weather, "setting": setting_name,
            "setting_values": setting, "seed": int(cfg["sampling"]["seed"]), "windows": meta, "starts": starts, "joins": joins,
            "brightness_per_window": bright, "seconds": round(seconds, 1), "join_version": 2, "model_commit": cfg["model"]["commit"],
            "bundle_sha256": bundle_hash(bdir), "frames": int(n)}
    (work / "clip.json").write_text(json.dumps(info, indent=1) + "\n")
    work.rename(final)
    png = REPO_ROOT / "docs" / "validation" / "images" / f"gen_full_{scene_name(bdir)}_{bdir.name}_{weather}_{setting_name}_{method}.png"
    sheet(bdir, cross, f"{weather} {setting_name} {method}", png)
    change = (bright[-1] - bright[0]) / bright[0]
    summarise([f"## {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, {'independent windows' if piece == WINDOW else 'independent pieces of two chunks'} ({len(starts)} pieces, {n} frames)",
               f"- cross-fade: `{final / 'out_crossfade.mp4'}`", f"- cross-fade with local colour matching: `{final / 'out_crossfade_matched.mp4'}`",
               f"- hard cut: `{final / 'out_hardcut.mp4'}`", f"- sheet: `{png}`",
               f"- generation time {seconds / 60:.0f} min; brightness first window {bright[0]:.0f}, last {bright[-1]:.0f} ({change:+.0%})",
               "- joins (frames, colour difference between the raw neighbours in the overlap): "
               + "; ".join(f"{j['frames'][0]}-{j['frames'][1]}: {j['colour_difference']:.1f}" for j in joins),
               f"- windows with water in frame: {[m['start'] for m in meta if m['water_in_frame']]}"])
    return 0


def run_chain(bdir: Path, weather: str, setting_name: str, cfg: dict, cond: int = 1) -> int:
    p = prompts()
    setting = p["settings"][setting_name]
    n = video_info(bdir / "clip_0000" / "rgb.mp4")["frames"]
    legs = scene_legs(bdir, p)
    wet = water_share(bdir, 0, n) >= float(p["water_share_min"])
    key = "with_water" if wet else "without_water"
    # one prompt for the whole clip: every leg's text, then the weather
    prompt = " ".join(" ".join(l[key].split()) for l in legs) + " " + " ".join(p["weather"][weather][key].split())
    clip, inp = bdir / "clip_0000", inputs_dir(bdir)
    edge = clip / "edges.mp4" if setting["edge"] == "rendered" else inp / "edges_geometry.mp4"
    spec = base_spec(setting, cfg, "out", prompt, clip / "rgb.mp4", inp / "depth_control.mp4", edge, clip / "segmentation.mp4")
    spec["num_conditional_frames"] = int(cond)      # frames of the previous chunk each chunk starts from
    tag = "chain" if cond == 1 else f"chain_cond{cond}"
    sdir = model_dir(bdir, cfg) / "full_specs"
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{weather}_{setting_name}_{tag}.json"
    spath.write_text(json.dumps(spec, indent=1) + "\n")
    final = model_dir(bdir, cfg) / f"{weather}_{setting_name}_{tag}"
    out = run_clip(spath, final, cfg)
    if out is None:
        summarise([f"## FAILED: {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, chained"])
        return 1
    video = read_video(out / "out.mp4")
    bounds = [0] + list(range(WINDOW, n, WINDOW - cond)) + [n]
    bright = chunk_brightness(video, bounds)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "gen_seam_check.py"), "--clip", str(out), "--overlap", str(cond)])
    seams = json.loads((out / "seams.json").read_text()) if (out / "seams.json").exists() else {}
    png = REPO_ROOT / "docs" / "validation" / "images" / f"gen_full_{scene_name(bdir)}_{bdir.name}_{weather}_{setting_name}_{tag}.png"
    sheet(bdir, video, f"{weather} {setting_name} chained", png)
    change = (bright[-1] - bright[0]) / bright[0]
    summarise([f"## {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, chained with {cond} conditioning frame(s) ({len(bounds) - 1} chunks, {n} frames)",
               f"- video: `{out / 'out.mp4'}`", f"- sheet: `{png}`",
               f"- brightness first chunk {bright[0]:.0f}, last {bright[-1]:.0f} ({change:+.0%}); joins flagged: {seams.get('flagged_joins', 'not checked')}"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--altitude", type=int, required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--method", required=True, choices=["windows", "pieces", "chain"])
    ap.add_argument("--piece-frames", type=int, default=185, help="with --method pieces: frames per independent piece")
    ap.add_argument("--cond-frames", type=int, default=1, choices=[1, 2], help="with --method chain: frames of the previous chunk each chunk starts from")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    method = a.method
    b = bundle_dir(a.run_folder, a.altitude)
    print(f"{time.strftime('%F %T')} {scene_name(b)} {a.altitude} ft {a.weather} {a.setting}: method {method}", flush=True)
    if method == "windows":
        return run_windows(b, a.weather, a.setting, cfg)
    if method == "pieces":
        return run_windows(b, a.weather, a.setting, cfg, piece=a.piece_frames)
    return run_chain(b, a.weather, a.setting, cfg, cond=a.cond_frames)


if __name__ == "__main__":
    sys.exit(main())
