# Reproduce the results

Every command that made the v1.0 videos and the snow window, in the order it ran, with its output, wall time and
peak GPU memory.

- Code: this repository at commit `674cc7e`.
- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB), driver 595.112.07. The renders
  and the model ran on it one after the other, never at the same time.
- Simulator: Isaac Sim 6.2.0. Model: Cosmos-Transfer2.5-2B at commit `2ff49d0`.
- Install: as in the [README](../README.md#install). Run everything from the repository root, in
  one shell, after the setup block of [quick_test.md](quick_test.md):

```bash
source env_cpu/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export MODEL_REPO=$PWD/external/cosmos-transfer2.5
export MODEL_PYTHON=$PWD/env_gen/bin/python
export HF_HUB_OFFLINE=1
```

Wall times include starting the simulator or loading the model. Peak GPU memory for the renders is
from `nvidia-smi` every 30 s; for the model runs it is the run's own `timing.json`.

## The frames of the Canal Street route

`configs/routes/canal-shore-street.yaml` is 768 frames (48 s at 16 fps), flown at 150 ft and
500 ft over the same ground track: about 20 s along the Hudson River Park shoreline, one turn of
125 degrees on the ground track, then about 16 s along Canal Street. In the saved camera poses
(`canal-shore-street.expected_poses.json`) the camera heading turns by about 120 degrees between
frames 305 and 522.

- The two 30 s clips are route frames 288 to 767: just before the turn, through it, and down
  Canal Street. The model runs them as two chains of 248 frames, 288 to 535 and 520 to 767,
  which share frames 520 to 535; `nyc-drone join` cross-fades them over those 16 frames into
  248 + 248 - 16 = 480 frames.
- The fog clip is route frames 114 to 353, 240 frames along the shoreline.
- The fog render covers all 768 frames, not only 114 to 353: `--guide-video` must have the same
  frame count as the render it guides, and the model step cuts it to the clip's frames itself.

The Bronx and Grand Army Plaza clips are 15 s address runs (240 frames), restyled whole.

## 0. Canal Street scene

```bash
nyc-drone scene canal-shore-street --out runs/quick/canal
```

City data, street layers, vehicles and people, the USD scene and the flight paths of both
altitudes, in `runs/quick/canal/`. CPU only. 7.8 min. This is step 0 of
[quick_test.md](quick_test.md); the Canal Street renders below use its `config.yaml`.

## 1. Renders

| Step | Command | Output | Wall time | Peak GPU |
| --- | --- | --- | --- | --- |
| Canal Street 150 ft | `env_sim/bin/python -m nyc_drone_sim.sim.render --config runs/quick/canal/config.yaml --location canal-shore-street --variant long --altitude 150 --frames 0:768 --look enhanced --run final_render` | `runs/quick/canal/work/final_render/canal-shore-street/150/`, 768 frames | 15.6 min (768 frames in 828.7 s) | 4.1 GB |
| Canal Street 500 ft | `env_sim/bin/python -m nyc_drone_sim.sim.render --config runs/quick/canal/config.yaml --location canal-shore-street --variant long --altitude 500 --frames 0:768 --look enhanced --run final_render` | `.../final_render/canal-shore-street/500/`, 768 frames | 29.5 min (768 frames in 1,658.9 s) | 4.1 GB |
| Render checks | `env_cpu/bin/python checks/render.py --config runs/quick/canal/config.yaml --location canal-shore-street --run final_render --variant long` | `docs/validation/reports/render_canal-shore-street_final_render_long.json` and two sheets in `docs/validation/images/` | 5.2 min | |
| Canal Street fog, 150 ft | `env_sim/bin/python -m nyc_drone_sim.sim.render --config runs/quick/canal/config.yaml --location canal-shore-street --variant long --altitude 150 --frames 0:768 --sim-weather fog_heavy --run final_fog` | `runs/quick/canal/work/final_fog/canal-shore-street/150/`, 768 frames, `"weather": "fog_heavy"` in `bundle.json` | 16.8 min (768 frames in 903.9 s) | 4.1 GB |
| Bronx 150 ft | `nyc-drone run --address "1 East 161st Street, Bronx" --altitude-ft 150 --duration-s 15 --look enhanced` | `runs/1-east-161st-street-bronx/`: `150/rgb.mp4` and the other videos, 240 frames; checks scene, path and render passed | 14.7 min | 4.1 GB |
| Grand Army Plaza 500 ft | `nyc-drone run --address "Grand Army Plaza, Brooklyn" --altitude-ft 500 --duration-s 15 --look enhanced` | `runs/grand-army-plaza-brooklyn/`: `500/rgb.mp4` and the other videos, 240 frames; checks scene, path and render passed | 19.4 min | 4.2 GB |

The `nyc-drone run` times include the scene, the render and the checks; the city data of both
places was already cached. The Canal Street render checks passed at both altitudes: 768 frames,
camera pose error 0.00 cm, centre depth 64.66 m (expected 64.7) and 215.53 m (expected 215.5),
camera height from ground pixels 45.720 m, and the repeat render of frame 0 with depth and
classes equal on 100% of pixels (colour PSNR 40.52 dB and 37.92 dB). One warning: on the
waterfront, building pixels cover less than 5% of many frames (render-6), which is expected there.

## 2. Model runs, in the order they ran

Each chain is three 93-frame chunks, each starting from the last frame of the one before.

**a. Canal Street 150 ft, clear, 30 s**

```bash
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/final_render/canal-shore-street --altitude 150 --method chain --weather clear --setting M2 --frames 288:248 --guide-video runs/quick/canal/work/final_render/canal-shore-street/150/clip_0000/rgb.mp4
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/final_render/canal-shore-street --altitude 150 --method chain --weather clear --setting M2 --frames 520:248 --guide-video runs/quick/canal/work/final_render/canal-shore-street/150/clip_0000/rgb.mp4
nyc-drone join runs/quick/canal/work/final_render/canal-shore-street/150/gen/cosmos-transfer2.5-2b/clear_M2_chain_0288_0248/out.mp4 runs/quick/canal/work/final_render/canal-shore-street/150/gen/cosmos-transfer2.5-2b/clear_M2_chain_0520_0248/out.mp4 --overlap 16 --out canal-shore-street_150ft_clear_M2_30s.mp4
```

The guide video of a and b is the enhanced render's own colour video, which M2 would use anyway.

**c. Canal Street 150 ft, fog, 15 s**

```bash
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/final_render/canal-shore-street --altitude 150 --method chain --weather fog --setting fog_guide --frames 114:240 --guide-video runs/quick/canal/work/final_fog/canal-shore-street/150/clip_0000/rgb.mp4
```

Output: `.../150/gen/cosmos-transfer2.5-2b/fog_fog_guide_chain_0114_0240/out.mp4`.

**d. Bronx 150 ft, clear, 15 s**

```bash
nyc-drone generate --address "1 East 161st Street, Bronx" --duration-s 15 --altitude-ft 150 --model-setting M2 --method chain --weather clear --guide-video runs/1-east-161st-street-bronx/150/rgb.mp4
```

Output: `runs/1-east-161st-street-bronx/150/weather/clear_M2_chain_0000_0240.mp4`.

**e. Grand Army Plaza 500 ft, clear, 15 s**

```bash
nyc-drone generate --address "Grand Army Plaza, Brooklyn" --duration-s 15 --altitude-ft 500 --model-setting M2 --method chain --weather clear --guide-video runs/grand-army-plaza-brooklyn/500/rgb.mp4
```

Output: `runs/grand-army-plaza-brooklyn/500/weather/clear_M2_chain_0000_0240.mp4`.

**b. Canal Street 500 ft, clear, 30 s**

```bash
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/final_render/canal-shore-street --altitude 500 --method chain --weather clear --setting M2 --frames 288:248 --guide-video runs/quick/canal/work/final_render/canal-shore-street/500/clip_0000/rgb.mp4
env_gen/bin/python -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/final_render/canal-shore-street --altitude 500 --method chain --weather clear --setting M2 --frames 520:248 --guide-video runs/quick/canal/work/final_render/canal-shore-street/500/clip_0000/rgb.mp4
nyc-drone join runs/quick/canal/work/final_render/canal-shore-street/500/gen/cosmos-transfer2.5-2b/clear_M2_chain_0288_0248/out.mp4 runs/quick/canal/work/final_render/canal-shore-street/500/gen/cosmos-transfer2.5-2b/clear_M2_chain_0520_0248/out.mp4 --overlap 16 --out canal-shore-street_500ft_clear_M2_30s.mp4
```

| Run | Frames | Wall time | Peak GPU (`timing.json`) | Seam check |
| --- | --- | --- | --- | --- |
| a, chain 288:248 | 248 | 43.8 min | 70,619 MiB | no flagged join |
| a, chain 520:248 | 248 | 41.8 min | 70,621 MiB | no flagged join |
| a, join | 480 | 4 s | | |
| c, chain 114:240 | 240 | 42.3 min | 71,178 MiB | join into frame 93 flagged (see below) |
| d, chain 0:240 | 240 | 41.6 min | 70,619 MiB | no flagged join |
| e, chain 0:240 | 240 | 41.1 min | 70,619 MiB | no flagged join |
| b, chain 288:248 | 248 | 42.4 min | 70,619 MiB | no flagged join |
| b, chain 520:248 | 248 | 41.4 min | 70,638 MiB | no flagged join |
| b, join | 480 | 4 s | | |

Each chain folder holds `out.mp4`, `spec.json`, `prompt.txt`, `clip.json`, `run.log`,
`timing.json`, `seams.json` and `sheet.png` (eight frames of the render above the same frames of
the result). `seams.json` flags a join between chunks whose frame-to-frame change is more than
four standard deviations above the clip's mean. In clip c the second chunk starts about 4.5%
brighter (change 2.18 against a limit of 1.93); fog makes the change between frames small and
even, so its limit is tight. The fog thickens in one step there, at 5.8 s.

## Snow window

The repository's built-in snow prompt does not give snow (quick test step 7: green trees, dry
paving): its scene sentences describe a dry, green scene and outweigh the snow paragraph. The
snow window was made with the same spec and only the prompt replaced. It is one 93-frame window
(5.8 s) of Canal Street at 150 ft, route frames 114 to 206, setting t4 (depth 1.0, geometry
edges 1.0, no segmentation), seed 2026, 35 steps. S1 uses guidance 3, like every other clip; S2
uses guidance 6.

1. The render: quick test step 1 (it needs the scene of step 0).

```bash
env_sim/bin/python -m nyc_drone_sim.sim.render --config runs/quick/canal/config.yaml \
    --location canal-shore-street --variant long --altitude 150 --frames 114:185 --look enhanced --run render
```

2. The window's spec. The snow window used the `spec.json` that quick test step 7 wrote
   (`runs/quick/canal/work/render/canal-shore-street/150/gen/cosmos-transfer2.5-2b/snow_t4_window_0000/spec.json`),
   copied to `runs/snow_test/S1/` and `runs/snow_test/S2/`. Without running step 7, the same
   command with `--specs-only` writes the controls, `spec.json`, `prompt.txt` and `clip.json`
   into a new, empty folder, with paths relative to the spec (this way was not used for the
   snow window):

```bash
${MODEL_PYTHON:-env_gen/bin/python} -m nyc_drone_sim.gen.makeclip --run-folder runs/quick/canal/work/render/canal-shore-street \
    --altitude 150 --method window --start 0 --weather snow --setting t4 --specs-only runs/snow_test/S1
cp -R runs/snow_test/S1 runs/snow_test/S2
```

3. Replace only the prompt with [snow_prompt.txt](snow_prompt.txt); for S2 also set guidance 6:

```bash
python3 -c "import json, sys; p = sys.argv[1]; s = json.load(open(p)); s['prompt'] = open('docs/snow_prompt.txt').read().strip(); json.dump(s, open(p, 'w'), indent=1)" runs/snow_test/S1/spec.json
python3 -c "import json, sys; p = sys.argv[1]; s = json.load(open(p)); s['prompt'] = open('docs/snow_prompt.txt').read().strip(); s['guidance'] = 6; json.dump(s, open(p, 'w'), indent=1)" runs/snow_test/S2/spec.json
```

4. Run the model on each spec, one at a time, as the model step itself calls it (in the model
   checkout):

```bash
R=$PWD
(cd "$MODEL_REPO" && HF_HUB_OFFLINE=1 "$MODEL_PYTHON" examples/inference.py -i "$R/runs/snow_test/S1/spec.json" -o "$R/runs/snow_test/S1/output")
(cd "$MODEL_REPO" && HF_HUB_OFFLINE=1 "$MODEL_PYTHON" examples/inference.py -i "$R/runs/snow_test/S2/spec.json" -o "$R/runs/snow_test/S2/output")
```

| Run | Guidance | Output | Wall time | Peak GPU |
| --- | --- | --- | --- | --- |
| S1 | 3 | `runs/snow_test/S1/output/out.mp4`, 93 frames | 13.95 min (837 s) | 62,915 MiB |
| S2 | 6 | `runs/snow_test/S2/output/out.mp4`, 93 frames | 13.97 min (838 s) | 62,915 MiB |

Peak GPU memory for the snow runs is from `nvidia-smi` every 5 s. The next step is to put
weather-specific scene and class sentences into `configs/gen_prompts_windows.yaml` and
`configs/gen_prompts_classes.yaml`, so that `--weather snow` builds this kind of prompt itself.

## Totals

- Renders: 1 h 36 min for the five renders, plus 5.2 min of render checks.
- Model: 4 h 54 min for the seven chains of the five videos. Each chain loads the model again
  (5 to 7.5 min). Each snow window: 14 min.
- Joins: 8 s.

The videos, the simulator clip of the same frames, the side-by-sides, every model run's inputs
(`specs.zip`), `MANIFEST.md` and `SHA256SUMS` are in the
[v1.0 release](https://github.com/satyam19k/nyc-drone-sim/releases/tag/v1.0).
