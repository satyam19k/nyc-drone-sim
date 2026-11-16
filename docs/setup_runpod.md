# Setting up on a rented GPU machine (RunPod)

How to bring this project up from a clean clone on a RunPod pod. It was developed on a Slurm
cluster; nothing below has been run on RunPod yet, so treat each step as "do, then verify", and
stop at the first verification that fails.

Rules that apply on any machine are in the project brief kept outside the repository: one author
on every commit, no generated-by footers, never relax a check threshold, never invent an API
name. Ask the owner for that brief before changing code.

## 1. Choose the pod

The two halves of the project need different GPUs.

| Half | Needs | Suitable pods | Not suitable |
| --- | --- | --- | --- |
| Simulator (Isaac Sim 6.2) | An RTX card with ray-tracing cores. The scenes peak near 3 GB of VRAM. | L40S, RTX A6000, RTX 4090, RTX 6000 Ada | A100, H100, H200: no ray-tracing cores |
| Video model (Cosmos-Transfer2.5-2B) | At least 70 GB of VRAM (the model's guide states 65.4 GB). | A100 80 GB, H100 80 GB, H200 | 48 GB cards and smaller |

No single common card does both. Either use two pods that share one network volume, or render
on one machine and copy the clip bundles to the other (section 6).

Other requirements: Linux x86-64, glibc 2.35 or newer (Ubuntu 22.04 or newer), NVIDIA driver
570.124.06 or newer for the video model. Check with `ldd --version` and `nvidia-smi`.

Disk: put everything on the pod's persistent volume (`/workspace` by default), not the container
disk, which is lost when the pod stops. Budget about 12 GB for the video model's environment,
52 GB for its weights, 20 GB for the simulator's environment and its caches, and 3 to 8 GB per
rendered clip bundle.

## 2. System packages and the clone

On RunPod you are root inside a container, so `apt` works:

```bash
apt update && apt -y install git git-lfs curl ffmpeg libgl1 libglib2.0-0 libx11-6 libxext6 tree
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

export WS=/workspace
git clone git@github.com:satyam19k/nyc-drone-sim.git $WS/nyc-drone-sim
cd $WS/nyc-drone-sim
git config user.name          # must print satyam19k before any commit; do not change user.email
```

`libgl1` matters: the video model's OpenCV build links against `libGL.so.1`. On the cluster this
had to be supplied by hand; with `apt` it is one package.

Keep every cache on the volume. Put these in `~/.bashrc`:

```bash
export WS=/workspace
export UV_CACHE_DIR=$WS/cache/uv
export UV_PYTHON_INSTALL_DIR=$WS/cache/uv-python
export UV_TOOL_DIR=$WS/cache/uv-tools
export UV_LINK_MODE=copy
export HF_HOME=$WS/.huggingface
export OMNI_KIT_ACCEPT_EULA=YES
```

## 3. CPU environment (both halves use it)

Fetching city data, building scenes, planning paths and every check run here. No GPU needed.

```bash
cd $WS/nyc-drone-sim
uv venv env_cpu --python 3.12
uv pip install --python env_cpu/bin/python \
    shapely pyproj usd-core numpy scipy pillow matplotlib pyyaml requests mapbox_earcut \
    static-ffmpeg pytest
env_cpu/bin/python -m pytest -q tests          # verify: all tests pass
```

Versions on the development machine, if a newer release misbehaves: shapely 2.2.0, pyproj 3.8.0,
usd-core 26.11, numpy 2.5.4, scipy 1.18.1, pillow 13.0.0, matplotlib 3.11.2.

Model library for vehicles, trees and pedestrians (0.6 GB, public, no token):

```bash
scripts/fetch_assets.sh
env_cpu/bin/python -m nyc_drone_sim.assets     # converts the downloads to data/assets/usd
```

Verify: `data/assets/usd/manifest.json` exists. Do not add the `peds24/bodies` folder: its licence
forbids building datasets with it.

## 4. Simulator environment (RTX pod only)

```bash
cd $WS/nyc-drone-sim
uv venv env_sim --python 3.12
uv pip install --python env_sim/bin/python "isaacsim[all,extscache]==6.2.0" \
    --extra-index-url https://pypi.nvidia.com
```

The exact command used on the development machine was not recorded; the line above is the
documented pip route for Isaac Sim, and the result there was isaacsim 6.2.0.0 with the
`extscache` packages. Confirm against the Isaac Sim 6.2 install page if it fails to resolve.

The simulator's Python has no shapely and must not need it. Code under `nyc_drone_sim/sim/` and
`nyc_drone_sim/common.py` may only import what `env_sim` has.

Verify with the depth and label probe. The first launch builds a shader cache and can take tens
of minutes; do not interrupt it.

```bash
L="120 Broadway"
for m in geo layers populate scene probes shotlist; do env_cpu/bin/python -m nyc_drone_sim.$m --location "$L"; done
env_cpu/bin/python -m nyc_drone_sim.paths --location "$L"
source env_sim/bin/activate
python -m nyc_drone_sim.sim.probe --config configs/default.yaml --location "$L" --run setup
deactivate
env_cpu/bin/python checks/stage_3.py --location "$L"
```

Pass means the probe read 45.72 m, 152.4 m and 50 m at the centre pixel. A scene in the wrong
units fails here, which is the point of the step.

Then render and check a clip:

```bash
source env_sim/bin/activate
for A in 150 500; do python -m nyc_drone_sim.sim.render --config configs/default.yaml --location "$L" --altitude $A --run run_pod; done
deactivate
env_cpu/bin/python -m nyc_drone_sim.postrender --location "$L" --run run_pod
env_cpu/bin/python checks/stage_5.py --location "$L" --run run_pod
```

For the named waterfront places add `--variant long` to `paths`, `render`, `postrender` and the
checks, for example `--location canal-shore-street --variant long`.

There is no Slurm on a pod. `slurm/batch.sbatch` is only a loop over a text file of commands;
run the commands directly, or under `nohup`/`tmux` so a dropped connection does not kill them.
If a render script fails it prints its traceback and exits non-zero. Confirm every step by its
output files, not by the exit of the shell.

## 5. Video-model environment (80 GB pod only)

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git $WS/src/cosmos-transfer2.5
cd $WS/src/cosmos-transfer2.5
git checkout 2ff49d0717af02057ae79bc75c00fbff9da1b4e7
git lfs pull
UV_PROJECT_ENVIRONMENT=$WS/nyc-drone-sim/env_gen uv sync --extra=cu128 --python 3.10
```

`--python 3.10` is required: uv otherwise picks a newer Python for which the model's pinned
wheels do not exist.

### Weights (52 GB, needs the owner's Hugging Face account)

Three repositories are gated and each needs its licence accepted on its own page by the account
owner: `nvidia/Cosmos-Guardrail1`, `nvidia/Cosmos-Transfer2.5-2B`, `nvidia/Cosmos-Predict2.5-2B`.
Ask the owner to log in on the pod with `uvx hf auth login`. Never print, log or commit the
token.

```bash
dl() { uvx "hf>=1.3.5" download "$@" --quiet; }
T=nvidia/Cosmos-Transfer2.5-2B
dl nvidia/Cosmos-Guardrail1 --repo-type model --revision d6d4bfa899a71454a700907664f3e88f503950cf --include "*"
dl $T --repo-type model --revision b67b64abda3801a9aceddbff2bdb86126c06db74 general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt
dl $T --repo-type model --revision dea7737ca29dd8d9086413c6dc5724b8250a0bb4 general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt
dl $T --repo-type model --revision eb5325b77d358944da58a690157dd2b8071bbf85 general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt
dl $T --repo-type model --revision 23057a4167b89de89a4a397fdbf3887994d115eb general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt
dl nvidia/Cosmos-Predict2.5-2B --repo-type model --revision f176dc95b4a70f53ce01c4b302851595e7322b00 tokenizer.pth
dl nvidia/Cosmos-Reason1-7B --repo-type model --revision 3210bec0495fdc7a8d3dbb8d58da5711eab4b423 --include "*"
dl Qwen/Qwen3Guard-Gen-0.6B
dl google/siglip-so400m-patch14-384
```

All four control checkpoints are needed even when a run uses fewer controls: the multi-control
model loads every branch. "Access denied ... requires approval" means that repository's licence
has not been accepted yet; stop and tell the owner its exact name.

The model finds its weights by running `uvx "hf>=1.3.5" download` itself. The `uvx` inside
`env_gen` is a different uv version from the one in `~/.local/bin` and keeps a separate tool
cache, so warm both once while online:

```bash
$WS/nyc-drone-sim/env_gen/bin/uvx "hf>=1.3.5" version
~/.local/bin/uvx "hf>=1.3.5" version
```

### Machine paths

`configs/gen.yaml` holds the development machine's paths. Copy it and change two lines:

```bash
cd $WS/nyc-drone-sim
sed -e "s|repo_dir: .*|repo_dir: $WS/src/cosmos-transfer2.5|" \
    -e "s|env_dir: .*|env_dir: $WS/nyc-drone-sim/env_gen|" configs/gen.yaml > configs/gen_pod.yaml
```

Pass `--config configs/gen_pod.yaml` to every `nyc_drone_sim.gen` command below.
`configs/gen_pod.yaml` is machine-specific; do not commit it.

### Verify, then run

```bash
cd $WS/nyc-drone-sim && source env_gen/bin/activate
export CUDA_HOME=$WS/nyc-drone-sim/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc

# every weight file is found with the network switched off
HF_HUB_OFFLINE=1 UV_OFFLINE=1 python -m nyc_drone_sim.gen.offline_check

# build the depth control and geometry edges, write and validate the specs for one run folder
python -m nyc_drone_sim.gen.controls --config configs/gen_pod.yaml --run-folder runs/run_006/hudson-park-run --altitude 150 --altitude 500
PYTHONPATH=$WS/src/cosmos-transfer2.5 python -m nyc_drone_sim.gen.plan --config configs/gen_pod.yaml --run-folder runs/run_006/hudson-park-run

# node check and self-test only: the model's own example, then one 93-frame chunk of our data
python -m nyc_drone_sim.gen.job --config configs/gen_pod.yaml --run-folder runs/run_006/hudson-park-run --self-test-only

# the full run list; safe to rerun, finished runs are skipped
nohup python -m nyc_drone_sim.gen.job --config configs/gen_pod.yaml --run-folder runs/run_006/hudson-park-run > gen_job.out 2>&1 &
```

`CUDA_HOME` is needed because one dependency looks for the CUDA runtime compiler library in a
system CUDA install; the line points it at the copy inside `env_gen`. If the pod image already
ships a CUDA toolkit this is harmless.

The job writes `reports/gen_job_local.md` after every run, with seconds per chunk and peak VRAM,
and puts each result in `runs/<run>/<scene>/<altitude>/gen/cosmos-transfer2.5-2b/<weather>_<edge>/`.
As of this writing the full model has never been run end to end: the self-test is the first
time all its parts load together. If it fails, report the error text rather than working around
it.

A pod bills by the hour whether or not anything runs. The ten runs for one scene are estimated
at 8 to 15 hours depending on the card; that figure is scaled from the model's published timing
and has not been measured.

## 6. Moving rendered clips between machines

Rendered output is not in git (`runs/` is ignored). A clip bundle is
`runs/<run>/<scene>/<altitude>/` and holds everything the video model reads: the colour,
segmentation and edge videos, the PNG frames, metric depth and `bundle.json`.

- Copy whole bundles, for example with `rsync -a` over SSH or `runpodctl send`. Do not copy a
  subset of a bundle: the control builder needs `metric_depth/` and `clip_0000/segmentation/`.
- The scene the bundle was rendered from lives in `runs/scenes/<scene>/`. The checks need it;
  the video model does not.
- After copying, confirm the hash recorded beside each control still matches:
  `sha256sum runs/<run>/<scene>/<altitude>/bundle.json`.

## 7. What is not in the repository

| Item | Where it comes from |
| --- | --- |
| The project briefs with the working rules | The owner, on request. Kept out of git on purpose. |
| `runs/` (clips, scenes), `data/cache/`, `data/assets/` | Rebuilt by the commands above, or copied |
| `env_cpu`, `env_sim`, `env_gen` | Built by sections 3 to 5 |
| Model weights and the Hugging Face token | Section 5; the token never goes in a file here |

## 8. Known open items

- The video model has not produced an output yet; quality on aerial views is unknown.
- `checks/stage_6.py` and the evidence sheets for generated clips are not written.
- The placement check finds one vehicle at 29.7 degrees to its street in the
  `canal-shore-street` scene (limit 10 degrees).
- `reports/stage_N.json` is overwritten by whichever location was checked last.
