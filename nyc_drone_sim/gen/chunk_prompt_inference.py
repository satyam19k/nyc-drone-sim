"""Run the model's own examples/inference.py with one prompt per chunk. Needs the model's GPU.

  cd <model checkout> && PYTHONPATH=<this repository> python -m nyc_drone_sim.gen.chunk_prompt_inference \\
      --chunk-prompts <chunk_prompts.json> -i <spec.json> -o <output folder>

The model's pipeline accepts a list of prompts, one per chunk, but its spec file holds one string.
This runs the unchanged inference script after wrapping the pipeline's generate call: when the
prompt it receives is the first prompt of the list (the spec's own prompt), the whole list is
passed instead. The model's text guardrail checks the spec's prompt, which is the first piece's.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def main() -> None:
    args = sys.argv[1:]
    k = args.index("--chunk-prompts")
    prompts = json.loads(Path(args[k + 1]).read_text())
    del args[k:k + 2]
    from cosmos_transfer2._src.transfer2.inference import inference_pipeline as ip

    original = ip.ControlVideo2WorldInference.generate_img2world

    def generate(self, *a, prompt=None, **kw):
        if prompt == prompts[0]:
            print(f"chunk prompts: {len(prompts)} prompts, one per chunk", flush=True)
            prompt = list(prompts)
        return original(self, *a, prompt=prompt, **kw)

    ip.ControlVideo2WorldInference.generate_img2world = generate
    sys.argv = ["examples/inference.py", *args]
    runpy.run_path("examples/inference.py", run_name="__main__")


if __name__ == "__main__":
    main()
