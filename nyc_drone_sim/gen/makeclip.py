"""Generate a clip with the video model. Needs an 80 GB GPU.

  python -m nyc_drone_sim.gen.makeclip --run-folder runs/run_009/canal-shore-street --altitude 150 \
      --weather clear --setting K2 --method window|chain|windows|pieces [--start 0] [--frames 0:768]

window   One 93-frame window from --start, the model's own clip length: the default of the
         generate step.
chain    One run of the model over --frames START:COUNT (default the whole clip). The model works
         in 93-frame chunks and starts each chunk from the last frame of the one before
         (num_conditional_frames 1), so a fault in one chunk carries into the next. One prompt for
         the chain, built from the classes in view over those frames (nyc_drone_sim.gen.prompt).
         The spec, prompt.txt and clip.json are kept beside out.mp4.

The two methods below are experimental.
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
from .common import REPO_ROOT, bundle_dir, bundle_hash, load_gen_config, model_dir, scene_name, video_info, write_video

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
    if setting.get("vis"):
        # no control_path: the model blurs the input video itself, so its colours guide the result
        spec["vis"] = {"control_weight": float(setting["vis"]["weight"]), "preset_blur_strength": str(setting["vis"]["blur"])}
    return spec


def look_warning(bdir: Path, setting_name: str, setting: dict) -> str | None:
    """A setting with the blur control copies the render's colours; say so when the render used the
    standard look."""
    if not setting.get("vis"):
        return None
    look = (json.loads((bdir / "bundle.json").read_text()).get("appearance") or {}).get("look", "standard")
    if look == "enhanced":
        return None
    return (f"warning: setting {setting_name} takes its colours from the render, which used the {look} look; "
            "it is meant for renders made with --look enhanced")


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


def sheet(bdir: Path, video: np.ndarray, title: str, out_png: Path, start: int = 0) -> None:
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
        rows.append(np.hstack([lab(sim[start + i], f"simulator {start + i} ({(start + i) / 16:.0f} s)") for i in part]))
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


def run_window(bdir: Path, weather: str, setting_name: str, cfg: dict, start: int = 0) -> int:
    """One 93-frame window from `start`, generated alone from its own controls and prompt."""
    p = prompts()
    setting = p["settings"][setting_name]
    n = video_info(bdir / "clip_0000" / "rgb.mp4")["frames"]
    if start < 0 or start + WINDOW > n:
        raise SystemExit(f"--start {start}: a {WINDOW}-frame window must fit in the clip's {n} frames (0 to {n - WINDOW})")
    name = f"one_{start:04d}"
    w = bdir / "gen" / "windows" / name
    if not (w / "window.json").exists():
        from .windows import cut_window
        cut_window(bdir, name, start, WINDOW, cfg)
    prompt, wet = prompt_for(bdir, weather, start, WINDOW, p)
    edge_file = "edges_rendered.mp4" if setting["edge"] == "rendered" else "edges_geometry.mp4"
    final = unique_dir(model_dir(bdir, cfg) / f"{weather}_{setting_name}_window_{start:04d}")
    work = final.with_name(final.name + ".tmp_" + time.strftime("%Y%m%d_%H%M%S"))
    work.mkdir(parents=True)
    spec = base_spec(setting, cfg, "out", prompt, w / "rgb.mp4", w / "depth_control.mp4", w / edge_file, w / "segmentation.mp4")
    (work / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    (work / "prompt.txt").write_text(prompt + "\n")
    warn = look_warning(bdir, setting_name, setting)
    if warn:
        print(warn, flush=True)
    t0 = time.time()
    rc = run_batch([work / "spec.json"], work, cfg, log=work / "run.log")
    if rc != 0 or not (work / "out.mp4").exists():
        summarise([f"## FAILED: {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, window from frame {start}",
                   f"- exit code {rc}; kept in `{work}`"])
        return 1
    (work / "clip.json").write_text(json.dumps({
        "method": "window", "start": start, "frames": WINDOW, "scene": scene_name(bdir), "altitude_ft": int(bdir.name),
        "weather": weather, "setting": setting_name, "setting_values": setting, "seed": int(cfg["sampling"]["seed"]),
        "prompt": prompt, "water_in_frame": wet, "look_warning": warn, "seconds": round(time.time() - t0, 1),
        "model_commit": cfg["model"]["commit"], "bundle_sha256": bundle_hash(bdir)}, indent=1) + "\n")
    work.rename(final)
    summarise([f"## {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, frames {start} to {start + WINDOW - 1}",
               f"- video: `{final / 'out.mp4'}`", f"- generation time {(time.time() - t0) / 60:.0f} min"])
    return 0


CHAIN_COND = 1         # frames of the previous chunk each chunk of a chain starts from


def frame_range(frames: str | None, n: int, min_frames: int = WINDOW) -> tuple[int, int]:
    """(start, count) from START:COUNT, or the whole clip when not given."""
    if not frames:
        start, count = 0, n
    else:
        try:
            start, count = (int(v) for v in frames.split(":"))
        except ValueError:
            raise SystemExit(f"--frames {frames}: give START:COUNT, e.g. 0:768") from None
    if start < 0 or count < min_frames or start + count > n:
        raise SystemExit(f"--frames {start}:{count}: at least {min_frames} frames, inside the clip's {n} frames (0 to {n - 1})")
    return start, count


def chain_spec(bdir: Path, weather: str, setting_name: str, cfg: dict, start: int, count: int) -> tuple[dict, dict]:
    """The model's spec for frames start .. start + count - 1 in one chained run, and what it was
    made from. The controls are cut from the bundle's own frames for that range."""
    from .prompt import class_config, class_prompt, class_shares
    p = prompts()
    setting = p["settings"][setting_name]
    name = f"chain_{start:04d}_{count:04d}"
    w = bdir / "gen" / "windows" / name
    if not (w / "window.json").exists():
        from .windows import cut_window
        cut_window(bdir, name, start, count, cfg)
    shares = class_shares(bdir, start, count)
    prompt, groups = class_prompt(shares, weather, class_config(), p)
    edge_file = "edges_rendered.mp4" if setting["edge"] == "rendered" else "edges_geometry.mp4"
    spec = base_spec(setting, cfg, "out", prompt, w / "rgb.mp4", w / "depth_control.mp4", w / edge_file, w / "segmentation.mp4")
    spec["num_conditional_frames"] = CHAIN_COND
    info = {"method": "chain", "start": start, "frames": count, "scene": scene_name(bdir), "altitude_ft": int(bdir.name),
            "weather": weather, "setting": setting_name, "setting_values": setting, "seed": int(cfg["sampling"]["seed"]),
            "num_conditional_frames": CHAIN_COND, "prompt": prompt, "described": groups,
            "shares_percent": {k: round(100 * v, 1) for k, v in shares.items() if v >= 0.005},
            "look_warning": look_warning(bdir, setting_name, setting),
            "model_commit": cfg["model"]["commit"], "bundle_sha256": bundle_hash(bdir)}
    return spec, info


def run_chain(bdir: Path, weather: str, setting_name: str, cfg: dict, frames: str | None = None) -> int:
    """One run of the model over frames START:COUNT (default the whole clip), in its own 93-frame
    chunks, each starting from the last frame of the one before; one prompt for the chain, built
    from what is in view over those frames. The spec, the prompt and clip.json are kept beside
    out.mp4, so the clip can be made again exactly."""
    n = video_info(bdir / "clip_0000" / "rgb.mp4")["frames"]
    start, count = frame_range(frames, n)
    spec, info = chain_spec(bdir, weather, setting_name, cfg, start, count)
    if info["look_warning"]:
        print(info["look_warning"], flush=True)
    tag = f"chain_{start:04d}_{count:04d}"
    sdir = model_dir(bdir, cfg) / "full_specs"
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{weather}_{setting_name}_{tag}.json"
    spath.write_text(json.dumps(spec, indent=1) + "\n")
    out = run_clip(spath, model_dir(bdir, cfg) / f"{weather}_{setting_name}_{tag}", cfg)
    if out is None:
        summarise([f"## FAILED: {scene_name(bdir)} {bdir.name} ft, {weather}, {setting_name}, chained, frames {start}:{count}"])
        return 1
    (out / "clip.json").write_text(json.dumps(info, indent=1) + "\n")
    describe_chain(bdir, out, info)
    return 0


def describe_chain(bdir: Path, out: Path, info: dict) -> None:
    """Brightness per chunk, the join check and an 8-frame sheet beside the clip; a log entry."""
    start, n = info["start"], info["frames"]
    video = read_video(out / "out.mp4")
    bounds = [0] + list(range(WINDOW, n, WINDOW - CHAIN_COND)) + [n]
    bright = chunk_brightness(video, bounds)
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "gen_seam_check.py"), "--clip", str(out), "--overlap", str(CHAIN_COND)])
    seams = json.loads((out / "seams.json").read_text()) if (out / "seams.json").exists() else {}
    sheet(bdir, video, f"{info['weather']} {info['setting']} chained", out / "sheet.png", start=start)
    change = (bright[-1] - bright[0]) / bright[0]
    summarise([f"## {info['scene']} {bdir.name} ft, {info['weather']}, {info['setting']}, chained, frames {start} to {start + n - 1} ({len(bounds) - 1} chunks)",
               f"- video: `{out / 'out.mp4'}`", f"- sheet: `{out / 'sheet.png'}`",
               f"- brightness first chunk {bright[0]:.0f}, last {bright[-1]:.0f} ({change:+.0%}); joins flagged: {seams.get('flagged_joins', 'not checked')}"]
              + ([f"- {info['look_warning']}"] if info["look_warning"] else []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--altitude", type=int, required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--method", required=True, choices=["window", "windows", "pieces", "chain"],
                    help="window: one 93-frame window from --start; the others (whole clip) are experimental")
    ap.add_argument("--start", type=int, default=0, help="with --method window: its first frame")
    ap.add_argument("--piece-frames", type=int, default=185, help="with --method pieces: frames per independent piece")
    ap.add_argument("--frames", default=None, help="with --method chain: START:COUNT (default the whole clip)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_gen_config(a.config)
    method = a.method
    b = bundle_dir(a.run_folder, a.altitude)
    print(f"{time.strftime('%F %T')} {scene_name(b)} {a.altitude} ft {a.weather} {a.setting}: method {method}", flush=True)
    if method == "window":
        return run_window(b, a.weather, a.setting, cfg, start=a.start)
    if method == "windows":
        return run_windows(b, a.weather, a.setting, cfg)
    if method == "pieces":
        return run_windows(b, a.weather, a.setting, cfg, piece=a.piece_frames)
    return run_chain(b, a.weather, a.setting, cfg, frames=a.frames)


if __name__ == "__main__":
    sys.exit(main())
