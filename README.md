# nyc-drone-sim

Drone video of any New York City location, rendered in NVIDIA Isaac Sim from city open data,
with optional photo-realism from a video model.

Give it an address or a route. It geocodes the address, downloads the buildings, streets, trees
and street furniture the city publishes for that area, builds a 3D scene in metres, plans a
collision-free flight at each altitude, renders the flight in Isaac Sim, and checks the result
against the data it started from. With `--realism model` it then passes a 93-frame window of each
clip through a video model (Cosmos-Transfer2.5) for a realistic look. Everything runs on the
machine you start it on.

    nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30
    nyc-drone route configs/routes/hudson-park-run.yaml
    nyc-drone reproduce canal-shore-street

## Quickstart

After the [installation](#install) below:

```bash
source env_cpu/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES                  # after reading the Isaac Sim licence
nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30 --dry-run
nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30
nyc-drone reproduce canal-shore-street --poses-only        # CPU only: replans a saved route
pytest tests                                     # unit tests, CPU only, about 15 s
```

`--dry-run` prints the resolved coordinates, the scene size, the frame count, the hardware each
step needs and a time estimate, and runs nothing. Before any work, `run`, `route` and
`reproduce` stop at once with what to install if Isaac Sim, the GPU or (with `--realism model`)
the video model's environment is missing, and download the vehicle, tree and pedestrian models
if they are not there yet.

## What a run produces

Everything is under `--out` (default `runs/<location or route name>`):

    scene.usd                     the 3D scene
    flight.json                   the planned camera path per altitude
    <altitude>/rgb.mp4            colour video
    <altitude>/depth.mp4, segmentation.mp4, edges.mp4
    <altitude>/camera_poses.json  camera pose of every frame
    <altitude>/weather/clear_window_<start>.mp4   the video model's version (--realism model)
    side_by_side.mp4              all altitudes next to each other
    report.json                   data sources, object counts, check results, timings
    validation/                   check reports and pictures
    config.yaml                   every setting the run used
    work/                         intermediate files and logs of each step

## Commands

| Command | What it does |
| --- | --- |
| `nyc-drone run --address ...` | scene, render, video model (with `--realism model`) and checks for an address or `--lat`/`--lon` |
| `nyc-drone route FILE` | the same for a route file |
| `nyc-drone reproduce NAME` | `route` on `configs/routes/NAME.yaml`, then compares the planned camera poses with `NAME.expected_poses.json` and reports the largest difference; exits 1 above 5 cm or 0.2 degrees |
| `nyc-drone scene`, `render`, `generate`, `check` | one step, with the same location or route arguments; a scene already built with the same options is reused |

## Options

| Group | Options |
| --- | --- |
| Location | `--address` (street address, intersection or place), or `--lat` and `--lon`; `--radius-m` (default 800) |
| Flight | `--altitude-ft` (one or more), `--duration-s`, `--speed-mps`, `--path street` (default) or `auto` |
| Camera | `--pitch-deg` (default -45), `--focal-mm` or `--fov-deg`, `--resolution WIDTHxHEIGHT`, `--fps` |
| Scene | `--seed`, `--objects on` or `off` (vehicles, pedestrians, trees, street furniture), `--look standard` or `enhanced` |
| Video model | `--realism none` (default) or `model`; `--start` (default 0); `--method window` (default); `--model-setting` |
| Weather | `--weather clear` (default). Other weather is not available yet: the command stops with that message |
| Route | `--poses-only`: fetch the buildings and plan the flight (and compare, for `reproduce`), nothing else; CPU only |
| Run | `--out`, `--config file.yaml`, `--dry-run`, `--offline`, `--sim-python` |

- `--path street` follows street centre lines and works for any length that fits in the scene.
  `--path auto` picks a short flyover, orbit or hover-and-pan around the location.
- `--config` takes a YAML file whose values replace those in `configs/default.yaml`.
- `--look enhanced` adds a sky-coloured ambient light with a soft sun, dark rippled water,
  ground textures sized in metres, and more varied walls, glass and roofs. It changes only light
  and materials: `checks/appearance.py` confirms that depth and semantic classes match a
  standard render of the same frames.
- With a route, the flight options come from the route file; `reproduce` also keeps the camera
  the poses were saved with.

## Route files

A route is a YAML file in `configs/routes/`:

```yaml
origin: [40.719955, -74.012988]      # [latitude, longitude], or an address in quotes
altitudes_ft: [150, 500]
speed_mps: {150: 8, 500: 12}         # or one value for every altitude
seed: 2026
look: standard                       # or enhanced
sync_ground_track: false             # true: both altitudes share one ground track of the image centre
frames: 480                          # 30 s at 16 fps
waypoints_m: [[-45, -290], [-20, -150], [0, 0], [25, 150], [45, 280]]   # metres east, north of the origin
# or waypoints_lonlat: [[-74.0135, 40.7173], ...]
```

Optional: `motion_frames` (length of the recorded traffic, default `frames`), `buffer_m` (scene
extent around the route, default 768), `max_yaw_rate_deg_s`, and, without `sync_ground_track`,
`focus` (where along the route the clip is centred, 0 to 1, per altitude) and `yaw_offset_deg`.
`canal-shore-street.yaml` (Hudson River Park, then Canal Street, 48 s) and `hudson-park-run.yaml`
(30 s along the esplanade) are the routes of the existing renders; their
`*.expected_poses.json` hold every frame's camera pose from those renders. The modules also take
a route name as `--location` (`python -m nyc_drone_sim.paths --location hudson-park-run --variant
long`).

## The video model (experimental)

`--realism model`, or `nyc-drone generate` after a render, builds the control videos from the
render (depth, edges, segmentation), cuts one 93-frame window from `--start`, and runs the model
on it with a prompt for the scene and clear weather. 93 frames is the model's own clip length.
Two methods restyle a whole clip and are experimental: `--method windows` (overlapping windows
joined on the CPU) and `--method chain` (one model run, each 93-frame chunk continuing the last).
The settings are in `configs/gen_prompts_windows.yaml`; `configs/gen.yaml` names the one used for
clear weather. A place without its own prompt gets the generic city description under
`scenes: default`. `nyc_drone_sim.gen.makeclip` also runs alone on any render.

## Hardware per step

| Step | Needs | Measured time |
| --- | --- | --- |
| `scene`: data, scene, flight paths | CPU; internet on the first run of a location | under 1 min for a 400 m scene; 13 to 60 min for an 800 m scene |
| `render` | NVIDIA RTX GPU with ray-tracing cores (tested on an L40S, 48 GB; a one-minute render used 3.5 GB of it) | 1.16 to 1.31 s per frame plus about 1 min per altitude: about 20 min per altitude for one minute of video |
| `generate` | NVIDIA GPU with 80 GB of memory (the model needs about 65 GB), Linux x86-64, glibc 2.35 or newer | not measured through this command |
| `check` | CPU | about 1 min for a 400 m scene; the scene check took 22 min on an 800 m scene with 2,851 buildings |

Disk: one minute of render at one altitude takes 6.5 GB (videos plus every frame as PNG and
metric depth); a 10 s run at two altitudes on a 400 m scene took 1.9 GB.

## Install

Linux x86-64 with an NVIDIA driver. Install [uv](https://docs.astral.sh/uv/) and git, then clone:

```bash
git clone git@github.com:satyam19k/nyc-drone-sim.git && cd nyc-drone-sim
```

### 1. CPU environment: data, scenes, flight paths, checks

```bash
uv venv env_cpu --python 3.12
uv pip install --python env_cpu/bin/python -e ".[test]"
env_cpu/bin/python -m pytest -q tests               # all tests pass
```

### 2. Simulator environment: Isaac Sim 6.2 for the render step

Tested with Isaac Sim 6.2.0 on an NVIDIA L40S with driver 610.43.02. The minimum driver for
Isaac Sim 6.2 was not checked; see NVIDIA's Isaac Sim requirements.

```bash
uv venv env_sim --python 3.12
uv pip install --python env_sim/bin/python "isaacsim[all,extscache]==6.2.0" --extra-index-url https://pypi.nvidia.com
export OMNI_KIT_ACCEPT_EULA=YES                     # after reading the Isaac Sim licence
env_sim/bin/python -c "from isaacsim import SimulationApp; app = SimulationApp({'headless': True}); print('Isaac Sim works'); app.close()"
```

The test line starts the simulator the way the render step does; it was not run in exactly this
form. The first start on a machine builds a shader cache and can take tens of minutes; do not
interrupt it. The command finds the simulator at `env_sim/bin/python`, or wherever
`NYC_DRONE_SIM_PYTHON` or `--sim-python` points.

### 3. Video-model environment (only for `--realism model`)

The model, [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5), is pinned
to commit `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`. Its own requirements: an NVIDIA GPU of the
Ampere generation or newer, driver 570.124.06 or newer (CUDA 12.8), glibc 2.35 or newer,
[git lfs](https://git-lfs.com/), and the system library `libGL.so.1` (package `libgl1` on Ubuntu)
for its OpenCV build.

```bash
git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git external/cosmos-transfer2.5
git -C external/cosmos-transfer2.5 checkout 2ff49d0717af02057ae79bc75c00fbff9da1b4e7
git -C external/cosmos-transfer2.5 lfs pull
(cd external/cosmos-transfer2.5 && UV_PROJECT_ENVIRONMENT=$PWD/../../env_gen uv sync --extra=cu128 --python 3.10)
```

`--python 3.10` is required: the model's pinned packages do not exist for newer Pythons. The
command finds the checkout at `external/cosmos-transfer2.5` and its Python at
`env_gen/bin/python`; `MODEL_REPO` and `MODEL_PYTHON`, or `model.repo_dir` and `model.env_dir` in
`configs/gen.yaml`, point elsewhere.

**Hugging Face access.** The weights are on Hugging Face. Create an account and a
[read token](https://huggingface.co/settings/tokens), and accept the licence on each of these
three gated pages while logged in:

- [nvidia/Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
- [nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)

Put the token in the environment as `HF_TOKEN` (never in a file of this repository). The model
downloads its weights on first use; to download them in advance (42 GB), with the revisions the
model's pinned commit asks for:

```bash
export HF_TOKEN=...                                 # your read token
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
HF_HUB_OFFLINE=1 UV_OFFLINE=1 env_gen/bin/python -m nyc_drone_sim.gen.offline_check    # every file found offline
```

All four control checkpoints are needed even when fewer controls are used: the model loads every
branch. "Access denied ... requires approval" means that page's licence has not been accepted.

### What gets downloaded

Sizes were measured on our machine; times were not measured.

| Item | Automatic or manual | When | Size | Time | Account needed |
| --- | --- | --- | --- | --- | --- |
| CPU Python packages | manual (step 1) | install | 0.9 GB installed | not measured | none |
| Isaac Sim 6.2 packages, with its extension cache | manual (step 2) | install | 23 GB installed (6.6 GB of it the extension cache download) | not measured | none; its licence must be accepted |
| Isaac Sim shader cache | automatic | first render on a machine | not measured | tens of minutes (not measured) | none |
| Vehicle, tree and pedestrian models (Valdrada v0.7.1) | automatic in `run`, `route`, `reproduce`; or `scripts/fetch_assets.sh` | first scene with objects | 587 MB downloaded, 21 MB converted | not measured | none |
| City data for a location | automatic | first scene of a location | about 10 MB per 800 m location (62 MB for our 7 locations) | not measured | none |
| Video-model checkout | manual (step 3) | install | 584 MB | not measured | none |
| Video-model environment | manual (step 3) | install | 12 GB installed | not measured | none |
| Video-model weights | automatic on first use, or manual in advance (step 3) | first `--realism model` | 42 GB | not measured | Hugging Face account, read token, three licences |

## Checks

`nyc-drone check` runs the scene, flight-path and render checks on a run. Each check reads values
back from the files a step wrote, never from the step's own bookkeeping:

| Check | File | What it confirms |
| --- | --- | --- |
| data | `checks/data.py` | geocoding, complete download, heights in metres, a byte-identical rerun from cache |
| scene | `checks/scene.py` | units and axis, one closed mesh per building, volumes, footprint overlap, labels |
| probe | `checks/probe.py` | depth read in the simulator equals the altitude: catches unit errors |
| path | `checks/path.py` | constant altitude and speed, clearance from every building surface, field of view |
| render | `checks/render.py` | frame counts and rates, camera pose, depth against altitude, a repeat render of frame 0 |
| populate | `checks/populate.py` | placed vehicles, pedestrians and trees: positions, scale, motion, labels |
| appearance | `checks/appearance.py` | a look render keeps the depth and classes of a standard render |

Check reports and pictures of the reference locations are in `docs/validation/`. `pytest tests`
runs the unit tests on the CPU without Isaac Sim, the model or the network, including a replan of
both saved routes against their expected poses.

## Example output

Frame sheets (colour, depth, segmentation and edges at four moments) are in
`docs/validation/images/`, for example:

- `render_175-5th-avenue-manhattan_new_address_001_150_street.jpg`: Flatiron, 150 ft
- `render_209-joralemon-street-brooklyn_brooklyn_60s_001_500_street.jpg`: Brooklyn Heights, 500 ft, 60 s
- `render_canal-shore-street_150_long.png`: Hudson River Park, then Canal Street
- `gen_clear_grid_93f_canal-shore-street_150.jpg`: the video model on 93-frame windows of the
  Canal Street route, clear weather, several settings

## Running the steps by hand

Each step is a module with `--location` and `--config`:

```bash
python -m nyc_drone_sim.geo       --location "120 Broadway"   # buildings
python -m nyc_drone_sim.layers    --location "120 Broadway"   # street layers
python -m nyc_drone_sim.populate  --location "120 Broadway"   # vehicles, people, trees, furniture
python -m nyc_drone_sim.scene     --location "120 Broadway"   # USD scene
python -m nyc_drone_sim.paths     --location "120 Broadway" --variant street
env_sim/bin/python -m nyc_drone_sim.sim.render --location "120 Broadway" --altitude 150 --run r1 --variant street
python -m nyc_drone_sim.postrender --location "120 Broadway" --run r1 --variant street
```

`configs/default.yaml` also lists named places under `places`. `python -m
nyc_drone_sim.shorestreet` and `python -m nyc_drone_sim.scout` score candidate routes from the
street layers before anything is rendered, and `scripts/cut_window.py` cuts a window out of a
long clip with its depth, segmentation and edge maps.

## Known limits

- New York City only: every data source is a city dataset.
- Weather other than clear is not available yet.
- Ground is flat at z = 0; terrain is not modelled.
- Buildings are footprints extruded to the recorded roof height with a generated window
  pattern; there is no real facade detail.
- A street clip must fit inside the scene. In an 800 m scene a 60 s clip at 150 ft may need a
  lower speed, and the route may not keep the address in view.
- Recorded traffic and pedestrian motion lasts as long as the clip the scene was built for.
- Vehicles and pedestrians are flat-coloured models. The enhanced look does not vary their colour
  or the trees', and its sky is a uniform colour, not an image.
- The repeat render of frame 0 (render-7) must agree to 35 dB. Views over tree canopy and some
  new addresses have scored 29 to 33 dB; the cause is not established.
- The `generate` step has not yet been run end to end through `nyc-drone`.

## Data sources and licences

City data comes from NYC Open Data (`data.cityofnewyork.us`), used under the city's open-data
terms of use: Building Footprints, the NYC Planimetric Database (Roadbed, Sidewalk, Median,
Public Plazas, Parking Lot, Open Space, Hydrography, Hydrography Structures, Boardwalk,
Transportation Structures), Centerline, Forestry Tree Points, Hydrants, Bus Stop Shelters,
Bicycle Parking and LinkNYC Kiosk Locations. Geocoding is by NYC Planning Labs GeoSearch.
Every response is cached under `data/cache/` with its URL, time and SHA-256.

3D models come from the Valdrada dataset by M. K. Turkcan
(`huggingface.co/datasets/mehmetkeremturkcan/valdrada`, tag `v0.7.1`), each part under its own
licence:

- Vehicles (`fleet24`): CARLA-derived models, credit "CARLA Simulator (carla.org)", CC BY 4.0;
  NYC buses, service and delivery vehicles by M. K. Turkcan, CC BY 4.0.
- Street trees (`bxtrees`): M. K. Turkcan, CC BY 4.0; leaf textures from ambientCG and bark
  textures from Poly Haven and ambientCG, CC0 1.0.
- Pedestrians (`peds24/rb27`): converted from the Microsoft Rocketbox Avatar Library,
  MIT licence, Copyright (c) 2020 Microsoft.

Rendered imagery that shows vehicles carries the credit "CARLA Simulator (carla.org), CC BY 4.0".
The models were converted to USD and, for vehicles and pedestrians, recoloured with flat colours.
The Valdrada pedestrian set under `peds24/bodies` is not used: its licence notice forbids
building datasets or training or testing models with it. Models and converted files live under
`data/assets/` and are not part of the repository.

Facade, roof, ground and water textures are generated by the code (`scene.py`,
`sim/appearance.py`); none are downloaded. The video model, Cosmos-Transfer2.5, is NVIDIA's,
under its own licence, and is installed separately.
