# nyc-drone-sim

Drone video of any NYC location, rendered in Isaac Sim from city open data.

## Populated scene

Besides buildings, the scene can carry street surfaces, road paint, trees, vehicles,
pedestrians and street furniture (`populate.enabled` in `configs/default.yaml`).

```bash
scripts/fetch_assets.sh                                    # download models, build the USD library
python -m nyc_drone_sim.geo      --location "120 Broadway" # buildings
python -m nyc_drone_sim.layers   --location "120 Broadway" # street layers from city records
python -m nyc_drone_sim.populate --location "120 Broadway" # seeded placement
python -m nyc_drone_sim.scene    --location "120 Broadway" # USD scene
```

Models and converted files live under `data/assets/` and are not part of the repository.

## Attribution

City data: NYC Open Data (`data.cityofnewyork.us`): Building Footprints, NYC Planimetric
Database (Roadbed, Sidewalk, Median, Public Plazas, Parking Lot, Open Space, Hydrography),
Centerline, Forestry Tree Points, Hydrants, Bus Stop Shelters, Bicycle Parking and LinkNYC
Kiosk Locations, used under the city's open-data terms of use. Geocoding by NYC Planning Labs
GeoSearch.

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
building datasets or training or testing models with it.

## Named places and long clips

`configs/default.yaml` lists named places under `places`. A place has its own origin and is
either a circle or a corridor around a camera route; pass its name as `--location`. For a
corridor place, `python -m nyc_drone_sim.paths --location <place> --variant long` flies the
route for one 30-second clip per altitude, and `--variant long` selects it in the renderer and
the checks. `python -m nyc_drone_sim.scout` scores candidate routes from the layers before
anything is rendered, and `scripts/cut_window.py` cuts any 93-frame window out of a long clip
with its depth, segmentation and edge maps.
