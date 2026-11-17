# nyc-drone-sim

Drone video of any New York City location, rendered in NVIDIA Isaac Sim from city open data,
with optional weather and photo-realism from a video model.

Give it an address. It geocodes the address, downloads the buildings, streets, trees and street
furniture the city publishes for that area, builds a 3D scene in metres, plans a collision-free
flight at each altitude, renders the flight in Isaac Sim, and checks the result against the data
it started from. With `--realism model` it then passes each clip through a video model
(Cosmos-Transfer2.5) for a realistic look in clear weather, fog, rain or snow.

    nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30

## Quickstart

```bash
git clone git@github.com:satyam19k/nyc-drone-sim.git && cd nyc-drone-sim
python -m venv env_cpu && source env_cpu/bin/activate
pip install -e ".[test]"                         # data, scene, flight paths, checks (CPU)
scripts/fetch_assets.sh                          # vehicle, tree and pedestrian models, once
export NYC_DRONE_SIM_PYTHON=/path/to/isaac-sim/python   # Isaac Sim 6.2.0, for the render step
export OMNI_KIT_ACCEPT_EULA=YES                  # after reading the Isaac Sim licence
nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30 --dry-run
nyc-drone run --address "175 5th Avenue, Manhattan" --altitude-ft 150 500 --duration-s 30
pytest tests                                     # unit tests, CPU only
```

`--dry-run` prints the resolved coordinates, the scene size, the frame count, the hardware each
step needs and a time estimate, and runs nothing.

## What a run produces

Everything is under `--out` (default `runs/<location>`):

    scene.usd                     the 3D scene
    flight.json                   the planned camera path per altitude
    <altitude>/rgb.mp4            colour video
    <altitude>/depth.mp4, segmentation.mp4, edges.mp4
    <altitude>/camera_poses.json  camera pose of every frame
    <altitude>/weather/<w>.mp4    the video model's version, one per weather (--realism model)
    side_by_side.mp4              all altitudes next to each other
    report.json                   data sources, object counts, check results, timings
    validation/                   check reports and pictures
    config.yaml                   every setting the run used
    work/                         intermediate files and logs of each step

## Commands and options

`nyc-drone <step> [options]`, where the step is `run` (everything, in order) or one of `scene`,
`render`, `generate`, `check`. A step run alone takes the same options as `run`; a scene that
was already built with the same options is reused.

| Group | Options |
| --- | --- |
| Location | `--address` (street address, intersection or place), or `--lat` and `--lon`; `--radius-m` (default 800) |
| Flight | `--altitude-ft` (one or more), `--duration-s`, `--speed-mps`, `--path street` (default) or `auto` |
| Camera | `--pitch-deg` (default -45), `--focal-mm` or `--fov-deg`, `--resolution WIDTHxHEIGHT`, `--fps` |
| Scene | `--seed`, `--objects on` or `off` (vehicles, pedestrians, trees, street furniture) |
| Renderer look | `--look standard` (default) or `enhanced`; `--sim-weather clear` (default), `fog_light`, `fog_heavy` or `wet` |
| Video model | `--realism none` (default) or `model`; `--weather clear fog rain snow` (one or more); `--model-setting` |
| Run | `--out`, `--config file.yaml`, `--dry-run`, `--offline`, `--sim-python`, `--slurm` |

- `--path street` follows street centre lines and works for any length that fits in the scene.
  `--path auto` picks a short flyover, orbit or hover-and-pan around the location.
- `--config` takes a YAML file whose values replace those in `configs/default.yaml`.
- `--slurm` writes a batch script for the run and submits it; `--slurm-account`,
  `--slurm-partition` and `--slurm-time` are passed through.
- `--look enhanced` adds a sky-coloured ambient light with a soft sun, dark rippled water,
  ground textures sized in metres, and more varied walls, glass and roofs. `--sim-weather` draws
  the renderer's own distance fog under an overcast sky, or a wet look with darker, smoother road
  and paving. Both change only light and materials: `checks/appearance.py` confirms that depth
  and semantic classes match a standard render of the same frames.

## The video model

`--realism model` (or `nyc-drone generate` after a render) builds the control videos from the
render (depth, edges, segmentation), cuts the clip into overlapping 93-frame windows, runs the
model on each window with a prompt for the scene and the weather, and joins the windows. The
setting used for each weather is named in `configs/gen.yaml` (`generate.default_setting`) and
defined in `configs/gen_prompts_windows.yaml`; `--model-setting` picks another one. A place
without its own prompt gets the generic city description under `scenes: default`.

The model is not part of this repository. Install
[Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) and its weights by its
own instructions, then tell this project where it is:

| Setting | Environment variable | Config (`configs/gen.yaml`) | Default |
| --- | --- | --- | --- |
| Model checkout | `MODEL_REPO` | `model.repo_dir` | `external/cosmos-transfer2.5` |
| Python of the model's environment | `MODEL_PYTHON` | `model.env_dir` (`<env_dir>/bin/python`) | `env_gen` |

Relative paths are relative to this repository; the environment variables win.
`python -m nyc_drone_sim.gen.offline_check` confirms, with the network off, that every weight
file the model needs is already cached. `nyc_drone_sim.gen.makeclip` also runs alone on any
render bundle.

## Hardware per step

| Step | Needs | Time |
| --- | --- | --- |
| `scene`: data, scene, flight paths | CPU; internet on the first run of a location | 15 to 60 min for an 800 m scene; under a minute for 400 m |
| `render` | RTX GPU with ray-tracing cores, Isaac Sim 6.2.0 (tested on an L40S, 48 GB) | about 1.2 s per frame plus about a minute per altitude: about 20 min per altitude for one minute of video |
| `generate` | GPU with 80 GB of memory, the model's environment | depends on the GPU |
| `check` | CPU | about a minute for a 400 m scene; the scene check alone took 22 min on an 800 m scene with 2,851 buildings, and the render check a few minutes per altitude for a one-minute clip |

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
| appearance | `checks/appearance.py` | a look or weather render keeps the depth and classes of a standard render |

Check reports and pictures of the reference locations are in `docs/validation/`. `pytest tests`
runs the unit tests on the CPU in about ten seconds, without Isaac Sim, the model or the network.

## Example output

Frame sheets (colour, depth, segmentation and edges at four moments) are in
`docs/validation/images/`, for example:

- `render_175-5th-avenue-manhattan_new_address_001_150_street.png`: Flatiron, 150 ft
- `render_209-joralemon-street-brooklyn_brooklyn_60s_001_500_street.png`: Brooklyn Heights, 500 ft, 60 s
- `render_canal-shore-street_150_long.png`: Hudson River Park, then Canal Street
- `gen_weather_window_hudson-park-run_150.png`: an early video-model test on one window, the
  render above clear, fog, rain and snow

## Named places and long clips

`configs/default.yaml` lists named places under `places`. A place has its own origin and is
either a circle or a corridor around a camera route given as waypoints in metres (`route_m`).
Pass its name as `--location` to the modules: `python -m nyc_drone_sim.paths --location <place>
--variant long` flies the route, and `--variant long` selects it in the renderer and the checks.
`python -m nyc_drone_sim.shorestreet` and `python -m nyc_drone_sim.scout` score candidate routes
from the street layers before anything is rendered, and `scripts/cut_window.py` cuts a window
out of a long clip with its depth, segmentation and edge maps. The `nyc-drone` command does not
fly named routes yet.

## Running the steps by hand

Each step is a module with `--location` and `--config`:

```bash
python -m nyc_drone_sim.geo       --location "120 Broadway"   # buildings
python -m nyc_drone_sim.layers    --location "120 Broadway"   # street layers
python -m nyc_drone_sim.populate  --location "120 Broadway"   # vehicles, people, trees, furniture
python -m nyc_drone_sim.scene     --location "120 Broadway"   # USD scene
python -m nyc_drone_sim.paths     --location "120 Broadway" --variant street
python -m nyc_drone_sim.sim.render --location "120 Broadway" --altitude 150 --run r1 --variant street   # Isaac Sim Python
python -m nyc_drone_sim.postrender --location "120 Broadway" --run r1 --variant street
```

## Known limits

- New York City only: every data source is a city dataset.
- Ground is flat at z = 0; terrain is not modelled.
- Buildings are footprints extruded to the recorded roof height with a generated window
  pattern; there is no real facade detail.
- A street clip must fit inside the scene. In an 800 m scene a 60 s clip at 150 ft may need a
  lower speed, and the route may not keep the address in view.
- Recorded traffic and pedestrian motion lasts as long as the clip the scene was built for.
- Vehicles and pedestrians are flat-coloured models. The enhanced look does not vary their colour
  or the trees', and its sky is a uniform colour, not an image.
- The renderer's fog has been seen only on short test frames; its brightness was set by eye.
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
