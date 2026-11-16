# Setting up on a rented GPU machine (RunPod)

How to bring this project up from a clean clone on a RunPod pod and run the video model on one or
two scenes. It was developed on a Slurm cluster; nothing below has been run on RunPod yet, so
treat each step as "do, then verify", and stop at the first verification that fails.

Rules that apply on any machine are in the project brief kept outside the repository: one author
on every commit, no generated-by footers, never relax a check threshold, never invent an API
name. Ask the owner for that brief before changing code.

## 0. Scope, cost and approval: read before renting anything

A pod bills by the hour from the moment it starts, whether or not anything is running.

**Scope: two scenes at most.** Do not render or generate anything else without being asked.

| Order | Scene | Run folder | Frames per clip | Why |
| --- | --- | --- | --- | --- |
| 1 | Hudson River Park run | `runs/run_006/hudson-park-run` | 480 (30 s) | Passed its render check; controls and specs already built |
| 2 | Shore then street (Canal Street) | `runs/run_009/canal-shore-street` | 768 (48 s) | Only if scene 1 looks right and the owner says to continue |

**Do not rent a simulator pod.** Both scenes are already rendered on the development machine.
Copy their clip bundles over (section 5) and rent only the video-model pod. Section 7 covers the
simulator, and applies only if the owner asks for a new render.

**Prefer the cheapest GPU that fits.** The video model needs at least 70 GB of VRAM.

1. List the pods on offer with 80 GB or more (A100 80 GB, H100 80 GB, H200) and their current
   hourly prices from the RunPod page. Do not quote a price from memory.
2. For each, multiply the price by the hours in the table below. The cheapest card per hour is
   not always the cheapest per job, because it is slower.
3. Pick the lowest total. On a tie, pick the cheaper hourly rate, since an idle or failed hour
   then costs less. Use on-demand, not spot, unless the owner accepts a run being interrupted;
   the job is resumable, but the pod's volume must survive.

Estimated hours, model loading included. **None of these is measured**: they are scaled from the
model's published time for one control on an H100, and this job uses three controls. Replace
them with the self-test's measured seconds per chunk as soon as it exists.

| Stage | What runs | A100 80 GB | H100 80 GB | H200 |
| --- | --- | --- | --- | --- |
| A. Setup and self-test | Environment, 52 GB of weights, the model's own example, one 93-frame chunk | 2 to 3 h | 2 to 3 h | 2 to 3 h |
| B. Scene 1, short list | 150 ft only: clear, fog, rain, snow (4 runs, 6 chunks each) | 6 h | 4 h | 3 h |
| C. Scene 1, full list | All 10 runs in `configs/gen.yaml`, both altitudes | 14.5 h | 9.5 h | 8 h |
| D. Scene 2, short list | 150 ft only: clear, fog, rain, snow (4 runs, 9 chunks each) | 8.5 h | 5.5 h | 4.5 h |

**Ask before each paid stage, and wait for the answer.** Send the owner one message in this form
and do not start the pod, or the next stage, until they reply:

```text
Stage:      B. Scene 1, short list (4 runs)
GPU:        <card>, <VRAM> GB, <on-demand or spot>
Price:      $<x> per hour (read from the RunPod page at <time>)
Estimate:   <h> hours, about $<x * h>; measured so far: <seconds per chunk, or "nothing yet">
Alternative: <the next cheapest card>, <h> hours, about $<total>
Stop rule:  I stop the pod when the stage ends or at <1.5 x estimate> hours, whichever is first
Proceed?
```

Further rules for spending:

- Stage A first, alone. If the self-test fails, stop the pod and report the error text. Do not
  try a bigger card or a different model on your own.
- After stage A, recompute every estimate from the measured seconds per chunk before asking for
  stage B.
- Default to the short list (stages B and D). Run the full list (stage C) only if asked.
- Stop the pod as soon as a stage ends. Do not leave it idle while waiting for an answer: the
  volume keeps the environment, the weights and the outputs.
- If a stage reaches 1.5 times its estimate, stop it and report. Finished runs are kept, and a
  later start skips them.

## 1. The pod

- GPU: 80 GB of VRAM or more, chosen as in section 0. 48 GB cards (L40S, A6000) do not fit.
- Image: Linux x86-64 with glibc 2.35 or newer (Ubuntu 22.04 or newer) and NVIDIA driver
  570.124.06 or newer. Check with `ldd --version` and `nvidia-smi` before installing anything;
  if either is too old, stop the pod and pick another image.
- Disk: a network volume of at least 120 GB mounted at `/workspace`. Everything goes there, not
  on the container disk, which is lost when the pod stops. Budget 12 GB for the environment,
  52 GB for weights, 3 to 5 GB per copied clip bundle and 1 GB per generated run.

## 2. System packages and the repository

`scripts/pod_setup.sh` runs sections 2 to 4 and the CPU part of section 6 unattended, and can be
run again after any failure. On a later pod with the same volume, `scripts/pod_setup.sh system`
restores only what lives on the container disk. The commands below are what it runs.

The repository is uploaded to `/workspace/nyc-drone-sim` with its `.git` folder, not cloned: it is
private and a fresh pod has no key for it.

You are root inside a container, so `apt` works:

```bash
apt update && apt -y install git git-lfs curl ffmpeg libgl1 libglib2.0-0 libx11-6 libxext6 rsync tree
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

export WS=/workspace
cd $WS/nyc-drone-sim
git config user.name          # must print satyam19k before any commit; do not change user.email
```

`libgl1` matters: the video model's OpenCV build links against `libGL.so.1`.

Keep every cache on the volume. Put these in `~/.bashrc`:

```bash
export WS=/workspace
export UV_CACHE_DIR=$WS/cache/uv
export UV_PYTHON_INSTALL_DIR=$WS/cache/uv-python
export UV_TOOL_DIR=$WS/cache/uv-tools
export UV_LINK_MODE=copy
export HF_HOME=$WS/.huggingface
```

## 3. Video-model environment

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
The token arrives as the pod's `HF_TOKEN` environment variable, which the download tool reads;
do not run an interactive login. Never print, log or commit the token.

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

To save paid hours, the environment and weights can be prepared on the cheapest pod that has
the same volume attached, then the 80 GB pod started only for the self-test and the runs. Ask
the owner before renting a second pod for this.

### Machine paths

`configs/gen.yaml` holds the development machine's paths and the full ten-run list. Make a pod
copy with the right paths and the short list (150 ft: clear, fog, rain, snow):

```bash
cd $WS/nyc-drone-sim
env_gen/bin/python - <<'EOF'
import os, yaml
ws = os.environ["WS"]
cfg = yaml.safe_load(open("configs/gen.yaml"))
cfg["model"]["repo_dir"] = f"{ws}/src/cosmos-transfer2.5"
cfg["model"]["env_dir"] = f"{ws}/nyc-drone-sim/env_gen"
cfg["scenes"]["canal-shore-street"] = (
    "Aerial view of the Hudson River waterfront and Canal Street in New York City from a drone, "
    "real photograph, river, piers, a waterfront park with people walking, trees, a wide street "
    "with traffic, detailed building facades and rooftops, natural colours.")
cfg["runs"] = [{"altitude": 150, "weather": w, "edge": "rendered" if w == "clear" else "geometry"}
               for w in ("clear", "fog", "rain", "snow")]
yaml.safe_dump(cfg, open("configs/gen_pod.yaml", "w"), sort_keys=False, width=100)
EOF
```

Pass `--config configs/gen_pod.yaml` to every `nyc_drone_sim.gen` command below. The file is
machine-specific and ignored by git. For the full list (stage C), keep the `runs` from
`configs/gen.yaml` instead of replacing them.

## 4. Offline check (free of GPU time)

Run this before the 80 GB pod is needed for anything else. It proves every weight file is found
with the network switched off, using the model's own lookup:

```bash
cd $WS/nyc-drone-sim && source env_gen/bin/activate
export CUDA_HOME=$WS/nyc-drone-sim/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc
HF_HUB_OFFLINE=1 UV_OFFLINE=1 python -m nyc_drone_sim.gen.offline_check
```

Pass is the line `all found offline`. `CUDA_HOME` is needed because one dependency looks for the
CUDA runtime compiler library in a system CUDA install; the line points it at the copy inside
`env_gen`. If the pod image already ships a CUDA toolkit this is harmless.

## 5. Copy the clip bundles

Rendered output is not in git (`runs/` is ignored). A clip bundle is
`runs/<run>/<scene>/<altitude>/` and holds everything the video model reads: the colour,
segmentation and edge videos, the PNG frames, metric depth and `bundle.json`.

The owner keeps a copy of the repository folder with these bundles and the two brief files.
Ask the owner to upload them to the pod from that copy, once the pod is running and its SSH
address is known (RunPod shows it under Connect, as `ssh root@<ip> -p <port>`):

```bash
# run on the owner's machine, from the root of their copy
POD="root@<ip>"; PORT=<port>
rsync -avhP -e "ssh -p $PORT" AGENTS.md AGENTS_stage6.md $POD:/workspace/nyc-drone-sim/
rsync -avhPR -e "ssh -p $PORT" ./runs/run_006/hudson-park-run/150 $POD:/workspace/nyc-drone-sim/
```

Upload the repository first (section 2), so the bundles land inside it. What exists and when
it is needed:

```text
runs/run_006/hudson-park-run/150      2.3 GB         scene 1; the short list needs only this one
runs/run_006/hudson-park-run/500      2.6 GB         scene 1; only for the full list
runs/run_009/canal-shore-street/150   4.1 GB         scene 2; only after the owner says to continue
runs/run_009/canal-shore-street/500   4.2 GB         scene 2; only for the full list
runs/scenes/hudson-park-run           42 MB          only for running the checks
runs/scenes/canal-shore-street        39 MB          only for running the checks
```

Upload only what the approved stage needs; add the others with the same `rsync -avhPR` form.

- Copy whole bundles. The control builder needs `metric_depth/` and `clip_0000/segmentation/`.
- Keep the same relative paths under `$WS/nyc-drone-sim/runs/`.
- A `gen/` folder inside a copied bundle was built on the development machine. Its control
  videos (`gen/inputs/`) are valid, but its spec files hold that machine's absolute paths:
  delete `gen/cosmos-transfer2.5-2b/specs/` and rebuild them (section 6).
- Confirm each bundle arrived intact: `ls clip_0000/rgb | wc -l` equals the clip's frame count
  (480 or 768), and `sha256sum bundle.json` matches the value the owner reports.
- Scene 2 must have passed its render check on the development machine before it is used. Ask
  the owner to confirm; as of this writing both clips were rendered but the render check had
  not been run on them, and the placement check had one failure (section 8).

## 6. Run the video model

All of this is CPU work except the last two commands. Do it before asking for stage B.

```bash
cd $WS/nyc-drone-sim && source env_gen/bin/activate
export CUDA_HOME=$WS/nyc-drone-sim/env_gen/lib/python3.10/site-packages/nvidia/cuda_nvrtc
export HF_HUB_OFFLINE=1 UV_OFFLINE=1 TRANSFORMERS_OFFLINE=1
S=runs/run_006/hudson-park-run

# depth control and geometry edges from the bundle's own metric depth and segmentation
python -m nyc_drone_sim.gen.controls --config configs/gen_pod.yaml --run-folder $S --altitude 150

# write one spec per run, validate them without a GPU, and print the plan with chunk counts
PYTHONPATH=$WS/src/cosmos-transfer2.5 python -m nyc_drone_sim.gen.plan --config configs/gen_pod.yaml --run-folder $S
```

The plan must list every run as `ok`. Every control is passed as a file and the blur control is
never used; a control without a file would be computed from the synthetic colour video.

Stage A, after approval. The self-test checks the card (70 GB, driver, glibc), runs the model's
own example, then one 93-frame chunk of scene 1:

```bash
python -m nyc_drone_sim.gen.job --config configs/gen_pod.yaml --run-folder $S --self-test-only
cat reports/gen_job_local.md
```

Read the seconds per chunk and peak VRAM from the report, recompute the stage B estimate
(runs x chunks x seconds per chunk, plus about 7 minutes of loading per run), and send the
approval message from section 0.

Stage B, after approval. Use `tmux` or `nohup` so a dropped connection does not kill it:

```bash
nohup python -m nyc_drone_sim.gen.job --config configs/gen_pod.yaml --run-folder $S --skip-self-test > gen_job.out 2>&1 &
tail -f reports/gen_job_local.md
```

- Each run writes to a temporary folder and is renamed only on success, so a killed run never
  looks finished. Starting the command again skips finished runs.
- A failed run is logged with the end of its output and the next run starts.
- Results land in `runs/<run>/<scene>/<altitude>/gen/cosmos-transfer2.5-2b/<weather>_<edge>/`
  with `out.mp4`, the spec, prompt, seed, model commit, bundle hash and timing.
- The model's guardrail is left on. If it blocks a prompt or an output, report it; do not
  switch it off without asking.

When the stage ends: copy the `gen/` folders and `reports/gen_job_local.md` back to the owner,
then **stop the pod**. Show the owner a clear-weather frame beside the simulator frame before
asking for anything further. Stage D is the same commands with
`S=runs/run_009/canal-shore-street`, and needs its own approval.

As of this writing the full model has never been run end to end anywhere: the self-test is the
first time all its parts load together, and quality on aerial views is unknown.

## 7. Only if a new render is requested: the simulator

Skip this unless the owner asks for a scene that has not been rendered. It needs a different,
cheaper pod: Isaac Sim requires an RTX card with ray-tracing cores, and the scenes peak near
3 GB of VRAM, so a 24 GB card is enough. Prefer the cheapest of RTX 4090, RTX A5000, RTX A6000,
L40S. A100, H100 and H200 cannot run it. Ask before renting it, in the same form as section 0;
a 30-second clip at both altitudes took about 20 minutes on an L40S, plus up to an hour for the
first launch.

CPU environment (fetching city data, building scenes, planning paths, every check):

```bash
cd $WS/nyc-drone-sim
uv venv env_cpu --python 3.12
uv pip install --python env_cpu/bin/python \
    shapely pyproj usd-core numpy scipy pillow matplotlib pyyaml requests mapbox_earcut \
    static-ffmpeg pytest
env_cpu/bin/python -m pytest -q tests          # verify: all tests pass
scripts/fetch_assets.sh                        # 0.6 GB of vehicle, tree and pedestrian models
env_cpu/bin/python -m nyc_drone_sim.assets     # verify: data/assets/usd/manifest.json exists
```

Do not add the `peds24/bodies` folder: its licence forbids building datasets with it.

Simulator environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
uv venv env_sim --python 3.12
uv pip install --python env_sim/bin/python "isaacsim[all,extscache]==6.2.0" \
    --extra-index-url https://pypi.nvidia.com
```

The exact command used on the development machine was not recorded; the line above is the
documented pip route, and the result there was isaacsim 6.2.0.0 with the `extscache` packages.
The simulator's Python has no shapely: code under `nyc_drone_sim/sim/` and
`nyc_drone_sim/common.py` may only import what `env_sim` has.

Build one scene, probe it, render it, check it. The first launch builds a shader cache and can
take tens of minutes; do not interrupt it.

```bash
L=hudson-park-run; V="--variant long"; R=run_pod
for m in geo layers populate scene probes shotlist; do env_cpu/bin/python -m nyc_drone_sim.$m --location $L; done
env_cpu/bin/python -m nyc_drone_sim.paths --location $L $V
env_cpu/bin/python checks/stage_4.py --location $L $V
source env_sim/bin/activate
python -m nyc_drone_sim.sim.probe --config configs/default.yaml --location $L --run $R
for A in 150 500; do python -m nyc_drone_sim.sim.render --config configs/default.yaml --location $L --altitude $A --run $R $V; done
deactivate
env_cpu/bin/python checks/stage_3.py --location $L        # depth must read 45.72, 152.4 and 50 m
env_cpu/bin/python -m nyc_drone_sim.postrender --location $L --run $R $V
env_cpu/bin/python checks/stage_5.py --location $L --run $R $V
```

There is no Slurm on a pod; `slurm/batch.sbatch` is only a loop over a text file of commands.
A render script that fails prints its traceback and exits non-zero. Confirm every step by its
output files.

## 8. What is not in the repository, and open items

| Item | Where it comes from |
| --- | --- |
| The project briefs with the working rules | The owner, on request. Kept out of git on purpose. |
| `runs/` (clips, scenes), `data/cache/`, `data/assets/` | Copied (section 5) or rebuilt (section 7) |
| `env_cpu`, `env_sim`, `env_gen` | Built by sections 3 and 7 |
| Model weights and the Hugging Face token | Section 3; the token never goes in a file here |

Open items:

- The video model has not produced an output yet.
- Every time estimate in section 0 is unmeasured.
- `checks/stage_6.py` and the evidence sheets for generated clips are not written.
- The placement check finds one vehicle at 29.7 degrees to its street in the
  `canal-shore-street` scene (limit 10 degrees).
