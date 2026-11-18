# Quick test: check your install end to end

Seven steps check every part of the install with short clips:

- four renders in the simulator;
- four restyles by the video model, one 93-frame window each.

## Before you start

1. **Install.** Follow the README's install steps first, all three environments, including the
   weight download for the video model. Then run everything below from the repository root, in
   one shell.
2. **Times.** The times below are rough. They are scaled from measurements on an L40S (render) and
   an H200 (model); they have not been measured on an RTX PRO 6000.
3. **First start of the simulator.** The first start on a new machine also builds a shader cache.
   That can take tens of minutes; do not interrupt it.

```bash
source env_cpu/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES                        # the simulator, after reading its licence
export MODEL_REPO=$PWD/external/cosmos-transfer2.5     # the video model's checkout
export MODEL_PYTHON=$PWD/env_gen/bin/python            # the video model's Python
export HF_HUB_OFFLINE=1                                # use the downloaded weights, do not go online
# without the advance download, leave HF_HUB_OFFLINE unset and export HF_TOKEN=<your read token>
```

## Why steps 1, 2, 4, 5 and 7 use module commands

`nyc-drone route` and `nyc-drone reproduce` cannot take a frame range, and they render every
altitude in the route file. Canal Street has two altitudes and 768 frames.

The shortest commands for one 93-frame stretch at one altitude therefore do this:

1. build the Canal Street scene with `nyc-drone scene`;
2. render with the simulator module and `--frames`;
3. run the video model with the generate module.

Steps 3 and 6 use `nyc-drone` itself.

## The commands

### 0. Canal Street scene

**What it does:** data, street layers, vehicles and people, the USD scene, and the flight paths
for the Canal Street route. CPU only.

**Time:** about 10 to 15 minutes.

**Writes:**
- `runs/quick/canal/scene.usd`
- `runs/quick/canal/flight.json`
- `runs/quick/canal/config.yaml`
- `runs/quick/canal/work/scenes/canal-shore-street/`

```bash
nyc-drone scene canal-shore-street --out runs/quick/canal
```

### 1. Canal Street, enhanced look, frames 114 to 206, 150 ft

**What it does:** renders the shoreline stretch.

**Time:** about 3 minutes.

**Writes:** `runs/quick/canal/work/render/canal-shore-street/150/`, which holds:
- `clip_0000/rgb.mp4`, `depth.mp4`, `segmentation.mp4`, `edges.mp4` and their PNG frames;
- `metric_depth/`, `semantic/`, `camera/`;
- `bundle.json`.

```bash
env_sim/bin/python -m nyc_drone_sim.sim.render --config runs/quick/canal/config.yaml \
    --location canal-shore-street --variant long --altitude 150 --frames 114:93 --look enhanced --run render
```

### 2. The same frames in the renderer's heavy fog

**What it does:** renders frames 114 to 206 again in heavy fog, at fog brightness 10, the 150 ft
default.

**Time:** about 3 minutes.

**Writes:** `runs/quick/canal/work/fog/canal-shore-street/150/`, with the same files as step 1.

The CLI's `--sim-weather fog` is the renderer's `--sim-weather fog_heavy` used here.

```bash
env_sim/bin/python -m nyc_drone_sim.sim.render --config runs/quick/canal/config.yaml \
    --location canal-shore-street --variant long --altitude 150 --frames 114:93 --sim-weather fog_heavy --run fog
```

### 3. "1 East 161st Street, Bronx", 6 seconds, enhanced look, 150 ft

**What it does:** the whole pipeline for an address: scene, render and checks.

**Time:** about 20 to 60 minutes for the scene, 3 minutes to render, and a few minutes for the
checks.

**Writes, under `runs/quick/bronx/`:**
- `150/rgb.mp4`, `150/depth.mp4`, `150/segmentation.mp4`, `150/edges.mp4`
- `150/camera_poses.json`
- `scene.usd`, `flight.json`, `report.json`, `side_by_side.mp4`, `config.yaml`
- `validation/`
- `work/render/1-east-161st-street-bronx/150/`

The command exits 1 if a check fails.

```bash
nyc-drone run --address "1 East 161st Street, Bronx" --duration-s 6 --look enhanced --altitude-ft 150 \
    --out runs/quick/bronx
```

### 4. Model, setting M2, on the render from step 1

**What it does:** restyles the render with the model. M2 uses depth, edges and segmentation, and
takes its colours from the enhanced render.

**Time:** about 15 to 25 minutes, most of it loading the model.

**Writes:** `runs/quick/canal/work/render/canal-shore-street/150/gen/cosmos-transfer2.5-2b/clear_M2_window_0000/`,
which holds `out.mp4`, `spec.json`, `prompt.txt`, `clip.json`, `timing.json` and `run.log`.

```bash
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/render/canal-shore-street \
    --altitude 150 --method window --start 0 --weather clear --setting M2
```

### 5. Model, fog, guided by the fog render from step 2

**What it does:** the fog setting. The geometry comes from step 1 (depth 0.5, geometry edges 0.3,
segmentation 0.5). The colours come from the fog render (0.3, blur high). The prompt uses the
fog paragraph.

**Checks it makes:** the fog render must have the same frame count and camera poses as step 1,
and its `bundle.json` must record fog.

**Time:** about 15 to 25 minutes.

**Writes:** `runs/quick/canal/work/render/canal-shore-street/150/gen/cosmos-transfer2.5-2b/fog_fog_guide_window_0000/`,
with the same files as step 4.

```bash
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/render/canal-shore-street \
    --altitude 150 --method window --start 0 --weather fog --setting fog_guide \
    --guide-video runs/quick/canal/work/fog/canal-shore-street/150/clip_0000/rgb.mp4
```

### 6. Model, setting M2, on the Bronx render from step 3

**What it does:** restyles the Bronx render, the way a user does after a render.

**Time:** about 15 to 25 minutes.

**Writes, in `runs/quick/bronx/150/weather/`:**
- `clear_M2_window_0000.mp4`
- `clear_M2_window_0000.spec.json`, `.prompt.txt` and `.clip.json` beside it

```bash
nyc-drone generate --address "1 East 161st Street, Bronx" --duration-s 6 --altitude-ft 150 \
    --model-setting M2 --out runs/quick/bronx
```

### 7. Model, snow, setting t4, on the render from step 1

**What it does:** restyles the render as snow. t4 uses depth 1.0 and geometry edges 1.0, with no
segmentation.

**Time:** about 15 to 25 minutes.

**Writes:** `runs/quick/canal/work/render/canal-shore-street/150/gen/cosmos-transfer2.5-2b/snow_t4_window_0000/`,
with the same files as step 4.

```bash
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/render/canal-shore-street \
    --altitude 150 --method window --start 0 --weather snow --setting t4
```

## Re-running

- **Renders (steps 1 and 2):** a second run of the same command replaces its render folder.
- **Model steps (4, 5 and 7):** a second run writes a new folder ending in `_r2`, `_r3` and so on.
- **Bronx (steps 3 and 6):** a second run reuses the scene when its options are unchanged.
- **Logs:** every model step also adds a section to `runs/logs/gen_summary.md`.

## What to look at

Compare each `out.mp4` with the render it was made from.

| Step | Output | Compare with |
|---|---|---|
| 4 (M2) | `.../gen/cosmos-transfer2.5-2b/clear_M2_window_0000/out.mp4` | step 1's `runs/quick/canal/work/render/canal-shore-street/150/clip_0000/rgb.mp4` |
| 5 (fog) | `.../gen/cosmos-transfer2.5-2b/fog_fog_guide_window_0000/out.mp4` | step 2's `runs/quick/canal/work/fog/canal-shore-street/150/clip_0000/rgb.mp4` |
| 7 (snow) | `.../gen/cosmos-transfer2.5-2b/snow_t4_window_0000/out.mp4` | step 1's render |
| 6 (Bronx) | `runs/quick/bronx/150/weather/clear_M2_window_0000.mp4` | `runs/quick/bronx/150/rgb.mp4` |

The first three outputs are under `runs/quick/canal/work/render/canal-shore-street/150/`.

Also look at the `"checks"` block of `runs/quick/bronx/report.json`: every check should read
`passed`.
