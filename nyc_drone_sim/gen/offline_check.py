"""Prove, without a GPU and with the network switched off, that every weight file the inference
path asks for is already in the local cache.

  HF_HUB_OFFLINE=1 UV_OFFLINE=1 python -m nyc_drone_sim.gen.offline_check          resolve and load
  ... --resolve-only                                                              resolve paths only
  ... --config other.yaml                                                         another machine's paths

With COSMOS_EXPERIMENTAL_CHECKPOINTS=1 the distilled edge checkpoint is checked as well.

It uses the model's own lookup functions, so the paths are found the same way the job finds them:
the four control checkpoints (multi-control loads all four), the video tokenizer, the text
encoder, and the guardrail models. Guardrail models are then loaded on the CPU and run once.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .common import load_gen_config
from .model import model_repo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolve-only", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("UV_OFFLINE") != "1":
        print("set HF_HUB_OFFLINE=1 and UV_OFFLINE=1: this check only means something offline")
        return 2
    cfg = load_gen_config(a.config)
    repo = str(model_repo(cfg))
    os.chdir(repo)
    sys.path.insert(0, repo)
    from cosmos_transfer2.config import CONTROL_KEYS, MODEL_CHECKPOINTS, ModelKey, ModelVariant
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT
    from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path

    failed = 0

    def step(name, fn):
        nonlocal failed
        try:
            out = fn()
            print(f"[ok]   {name}: {out}", flush=True)
        except BaseException as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)

    def size(p):
        p = Path(p)
        n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
        return f"{p} ({n / 1e9:.2f} GB)"

    for key in CONTROL_KEYS:           # the multi-control model loads every branch
        ck = MODEL_CHECKPOINTS[ModelKey(variant=key)]
        step(f"control checkpoint {key}", lambda ck=ck: size(get_checkpoint_path(ck.s3.uri)))
    distilled = ModelKey(variant=ModelVariant.EDGE, distilled=True)
    if distilled in MODEL_CHECKPOINTS:   # registered only with COSMOS_EXPERIMENTAL_CHECKPOINTS=1
        dk = MODEL_CHECKPOINTS[distilled]
        step("distilled edge checkpoint", lambda: size(get_checkpoint_path(dk.s3.uri)))
    step("video tokenizer (Wan2.1 VAE)", lambda: size(get_checkpoint_path("685afcaa-4de2-42fe-b7b9-69f7a2dee4d8")))
    step("text encoder (Cosmos-Reason1)", lambda: size(get_checkpoint_path("cb3e3ffa-7b08-4c34-822d-61c7aa31a14f")))
    step("guardrail files", lambda: size(GUARDRAIL1_CHECKPOINT.download()))
    if not a.resolve_only:
        from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common import presets

        def text():
            runner = presets.create_text_guardrail_runner(offload_model_to_cpu=True)
            return f"loaded on CPU; a harmless prompt is judged safe: {presets.run_text_guardrail('Aerial view of a city waterfront.', runner)}"

        def video():
            # The video guardrail moves its image encoder to the GPU whatever it is told, so on a
            # CPU node its two weight sources are read directly instead: same files, same lookup.
            from transformers import SiglipModel, SiglipProcessor
            model_id = "google/siglip-so400m-patch14-384"
            SiglipProcessor.from_pretrained(model_id)
            n = sum(p.numel() for p in SiglipModel.from_pretrained(model_id).parameters())
            root = Path(GUARDRAIL1_CHECKPOINT.download())
            files = [root / "video_content_safety_filter" / "safety_filter.pt", root / "face_blur_filter" / "Resnet50_Final.pth"]
            for f in files:
                torch.load(f, map_location="cpu", weights_only=False)
            return f"image encoder read on CPU ({n / 1e6:.0f} M parameters); {', '.join(f.name for f in files)} read"

        import torch
        step("text guardrail (blocklist, Qwen3Guard)", text)
        step("video guardrail weights (SigLIP filter, face blur)", video)
        def processor():
            from transformers import AutoProcessor
            d = get_checkpoint_path("s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/Qwen/Qwen2.5-VL-7B-Instruct/")
            return f"{type(AutoProcessor.from_pretrained(d)).__name__} from {d}"

        step("text encoder's processor", processor)
        step("video tokenizer read", lambda: f"{len(torch.load(get_checkpoint_path('685afcaa-4de2-42fe-b7b9-69f7a2dee4d8'), map_location='cpu', weights_only=False))} entries")
        for key in CONTROL_KEYS:
            ck = MODEL_CHECKPOINTS[ModelKey(variant=key)]
            step(f"read checkpoint {key}",
                 lambda ck=ck: f"{len(torch.load(get_checkpoint_path(ck.s3.uri), map_location='cpu', weights_only=False))} entries")
        if distilled in MODEL_CHECKPOINTS:
            step("read distilled edge checkpoint",
                 lambda: f"{len(torch.load(get_checkpoint_path(dk.s3.uri), map_location='cpu', weights_only=False))} entries")
    print(f"{'all found offline' if not failed else str(failed) + ' item(s) missing or unreadable'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
