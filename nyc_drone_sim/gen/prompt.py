"""The prompt of a chained clip, from what is in view over its frames. Needs only NumPy and PyYAML.

The share of every class over the frames is measured from the bundle's semantic frames;
configs/gen_prompts_classes.yaml turns each group of classes above its own threshold into a fixed
sentence, so the same view always gets the same words, at any address. The weather paragraph of
configs/gen_prompts_windows.yaml follows, in its river version only when water is in view.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .common import REPO_ROOT

FRAME_STEP = 4          # every 4th frame is enough to measure shares over a stretch of frames


def class_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "gen_prompts_classes.yaml").read_text())


def class_shares(bdir: Path, first: int, count: int) -> dict[str, float]:
    """Share of the pixels of each class over frames first .. first + count - 1, largest first."""
    from nyc_drone_sim.postrender import id_to_class
    names = id_to_class(bdir)
    tot, cnt = {}, 0
    for i in range(first, first + count, FRAME_STEP):
        ids, c = np.unique(np.load(bdir / "semantic" / f"semantic_{i:04d}.npz")["data"], return_counts=True)
        for k, v in zip(ids.tolist(), c.tolist()):
            name = names.get(int(k), "unlabelled")
            tot[name] = tot.get(name, 0) + v
        cnt += int(c.sum())
    return {k: v / cnt for k, v in sorted(tot.items(), key=lambda x: -x[1])}


def group_shares(shares: dict[str, float], pc: dict) -> dict[str, float]:
    """Share of each sentence's group of classes."""
    return {g: sum(shares.get(c, 0.0) for c in v["counts"]) for g, v in pc["classes"].items()}


def class_prompt(shares: dict[str, float], weather: str, pc: dict, wp: dict) -> tuple[str, list[str]]:
    """(prompt, groups described): the opening, one fixed sentence per group above its threshold,
    then the weather paragraph."""
    groups = [g for g, s in group_shares(shares, pc).items() if s > float(pc["classes"][g]["min_share"])]
    key = "with_water" if "water" in groups else "without_water"
    words = [pc["opening"], *(pc["classes"][g]["sentence"] for g in groups), wp["weather"][weather][key]]
    text = " ".join(" ".join(w.split()) for w in words)
    if "water tank" in text.lower():
        raise RuntimeError("a prompt mentions water tanks")
    return text, groups
