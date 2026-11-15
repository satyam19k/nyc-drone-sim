# Populating the scene: investigation

Phase A of the populate brief. Everything below was confirmed by fetching the source on
14 Nov 2026 unless a line says otherwise.

## 1. What the pipeline produced before this work

Read from the code and from `scene.usdc`, not from the brief.

| Step | Code | What it contained |
| --- | --- | --- |
| Fetch | `geo.fetch_buildings()` | One dataset, building footprints (`5zhs-2jue`) |
| Normalise | `geo.normalise()` | 756 polygons for 120 Broadway, 741 of them feature code 2100 |
| Scene | `scene.build_scene()` | 756 extruded meshes, one ground quad, a sun, a sky light |
| Labels | `scene._label()` | Two classes: `building`, `ground` |

Nothing was hidden by visibility, scale or camera distance. Roads, vehicles, people and trees
were never fetched or built. The "before" frames are frame 46 of `runs/run_003` at each altitude.

## 2. Missing layers and what each needs

| Layer | Geographic data | 3D models | Placement rules | Motion | Materials |
| --- | --- | --- | --- | --- | --- |
| Ground surfaces | yes | no (flat meshes) | no | no | yes |
| Trees | yes (points) | yes | size from trunk diameter | no | yes |
| Parked vehicles | street records only | yes | yes | no | yes |
| Moving vehicles | street records only | yes | yes | yes | yes |
| Road markings | street records only | no (flat meshes) | yes | no | yes |
| Street furniture | yes for four kinds | yes | yes for lamps and signals | no | yes |
| Pedestrians | no | yes | yes | optional | yes |

## 3. City data

All on `data.cityofnewyork.us`, fetched with the Stage 1 cache and paging. "In scene" is the
row count intersecting the 120 Broadway query box. Borough coverage was tested with a
2 km box in each borough; the counts are Manhattan, Bronx, Brooklyn, Queens, Staten Island.

| Layer | Dataset | Gives | Rows citywide | In scene | Five-borough test | Used |
| --- | --- | --- | --- | --- | --- | --- |
| Roadbed | `i36f-5ih7` | polygons | 104,961 | 590 | 711 / 643 / 672 / 795 / 227 | yes |
| Sidewalk | `52n9-sdep` | polygons | 50,865 | 266 | 315 / 336 / 289 / 399 / 191 | yes |
| Median | `ees7-4ufv` | polygons | 19,346 | 121 | 144 / 98 / 61 / 114 / 44 | yes |
| Public plazas | `ue2e-9jm2` | polygons | 1,762 | 75 | 82 / 15 / 12 / 12 / 3 | yes |
| Parking lots | `7cgt-uhhz` | polygons | 20,429 | 21 | 33 / 187 / 110 / 93 / 38 | yes |
| Open space, parks | `y6ja-fw4f` | polygons | 13,205 | 47 | 78 / 95 / 26 / 64 / 11 | yes |
| Hydrography | `pjs3-c3z5` | polygons | 2,206 | 2 | 2 / 28 / 0 / 0 / 39 | yes |
| Open space, other | `b7j8-z8a7` | polygons | 14,369 | 23 | not tested | no: different field names, overlaps the parks layer |
| Street centre lines | `inkn-q76z` | lines with lane counts, parking lanes, direction, posted speed | 122,312 | 979 | 1,222 / 808 / 544 / 850 / 318 | yes |
| Trees, current | `hn5i-inap` | points with trunk diameter, species, condition | 1,125,266 | 3,438 | 4,249 / 7,057 / 6,039 / 7,345 / 3,037 | yes |
| Trees, 2015 census | `uvpi-gqnh` | points | 683,788 | not queried | not tested | no: nine years old, superseded by the current layer |
| Hydrants | `5bgh-vtsn` | points | 109,725 | 701 | 813 / 597 / 772 / 713 / 268 | yes |
| Bus stop shelters | `t4f2-8md7` | points | 3,312 | 15 | 21 / 28 / 32 / 25 / 1 | yes |
| Bicycle parking | `592z-n7dk` | points | 38,255 | 438 | 468 / 209 / 428 / 96 / 0 | yes |
| LinkNYC kiosks | `s4kf-3yrf` | points (latitude and longitude columns) | 2,255 | 9 live | by borough field: 1,231 / 216 / 357 / 401 / 50 | yes |
| Parks properties | `enfh-gkve` | polygons | 2,061 | 14 | not tested | no: the planimetric parks layer already covers it |
| Pedestrian plazas (DOT) | `k5k6-6jex` | polygons | 93 | 5 | not tested | no: covered by the planimetric plaza layer |

Corrections to the brief's table:

- The older centre-line id `exjm-f27b` returns "Not found". `inkn-q76z` is the live one.
- The five "other planimetric" ids resolve to: `ees7-4ufv` Median, `7cgt-uhhz` Parking Lot,
  `ue2e-9jm2` Public Plazas, `y6ja-fw4f` Open Space (Parks), `b7j8-z8a7` Open Space (Other).
- The bicycle parking layer has 38,255 rows, not about 47,267.
- The 2015 tree census has 683,788 rows, not 666,134.
- The centre lines carry `number_travel_lanes`, `number_park_lanes`, `trafdir` and
  `posted_speed`. Lane count is missing on 175 of the 979 segments in the scene.
- The catalogue lists a licence ("Public Domain") for the parking-lot layer only; the other
  layers show no licence field and fall under the portal's general terms of use.

No city dataset gives positions of vehicles, pedestrians, street lamps or traffic signals.

## 4. Models

### Valdrada (`mehmetkeremturkcan/valdrada` on Hugging Face)

Public, not gated. Tags run from `v0.1.0` to `v0.7.1`; this work pins `v0.7.1`. The whole
`Content/models/` folder is 0.80 GB (812 files), not 6.8 GB; 0.61 GB was downloaded.

| Set | Format found | Licence in `LICENSE.md` | Decision |
| --- | --- | --- | --- |
| `fleet24/` vehicles, 33 files | glTF binary with compressed buffers and quantised vertices; textures as KTX2 | CC BY 4.0 | **Used.** Geometry converts exactly; textures are not converted |
| `bxtrees/` street trees, 20 trees | a packed binary of its own with a JSON header, not glTF; leaf images WebP, bark JPEG | CC BY 4.0, textures CC0 | **Used**, with textures |
| `peds24/rb27/` Rocketbox people, 49 bodies | glTF binary as above; textures in KTX2 arrays | MIT | **Used**, geometry only |
| `peds24/bodies/` CARLA walkers, 25 bodies | glTF binary | CC BY 4.0 **plus Epic MetaHuman terms** | **Rejected and not downloaded.** The notice forbids using them to build a dataset or to train or test a model, which is what our clips feed |
| `trees25/`, `trees36/` | packed binaries with KTX2 leaf atlases | project and MIT | Not used: `bxtrees` has decodable leaf images |
| `carla/` props, 536 files | glTF binary | CC BY 4.0 | Not used: furniture is 4 to 40 pixels wide in our frames, so simple generated shapes do the job |
| Compiled tiles, USD export | custom binary; browser export of one recorded take | ODbL | Not used, as the brief expected |

Two facts differ from the brief: the tree sets are not glTF, and the vehicle and people files
need a decoder for `EXT_meshopt_compression`. No decoder was available on the cluster (no
compiler, no Node). A short reader was written in `nyc_drone_sim/gltf.py`; it handles the
vertex codec version 0 and index codec versions 0 and 1 that these files use.

### Isaac Sim's own assets

Not tested. The install reads its asset location from the setting
`/persistent/isaac/asset_root/default`; whether a compute node can reach that server was not
checked, and `isaacsim.replicator.agent` was not tried. Skipped because the Rocketbox meshes
convert on the CPU, and a pedestrian is about 7 x 22 pixels at 150 ft and 2 x 7 at 500 ft,
where limb animation would not show.

### Other USD-native sources

None was looked at. The Valdrada sets covered vehicles, trees and people after conversion.

## 5. Chosen approach per layer

| # | Layer | Positions from | Shape from | Motion class | Rejected alternatives |
| --- | --- | --- | --- | --- | --- |
| 1 | Roadbed, sidewalk, median, plaza, parking lot, park, water | planimetric polygons, through the buildings' projection | flat meshes, cut so no two overlap, all at z = 0 | static | generating roadbed from centre lines (less exact than the surveyed polygons) |
| 2 | Trees | Forestry Tree Points, living trees only | Valdrada `bxtrees`, form chosen by genus, scaled by trunk diameter | static | 2015 census; generated blob crowns |
| 3 | Parked vehicles | rule: kerb of segments with parking lanes, plus parking-lot polygons | Valdrada `fleet24` | static | none |
| 4 | Moving vehicles | rule: lanes derived from lane count and measured road width, direction from `trafdir` | Valdrada `fleet24` | recorded animation | an agent or physics traffic system (not needed for 6 to 40 second clips) |
| 5 | Road markings | rule: crosswalk bars where a segment meets an intersection, lane and centre lines from lanes | thin flat meshes 2 cm above the road | static | cutting paint into the road mesh |
| 6 | Street furniture | records for hydrants, shelters, kiosks, racks; rules for lamps and signals | generated shapes | static | CARLA props |
| 7 | Pedestrians | rule: sidewalk and plaza polygons, denser near intersections | Rocketbox bodies, arms lowered, flat colours | static, or recorded animation (straight walk) | CARLA walkers (licence); limb animation |

No layer is independently simulated.
