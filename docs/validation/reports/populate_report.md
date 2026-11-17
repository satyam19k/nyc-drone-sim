# Populating the scene: report

Location 120 Broadway, scene radius 800 m, seed 2026. Render `runs/run_005` (jobs 23203971 and
23203973, 15 Nov 2026). `python checks/populate.py --run run_005` exits 0 and writes
`docs/validation/reports/populate.json`: P2 to P12 pass, P1 passes with a warning that needs a ruling (Section 6).

## 1. Approach

Ground layers come from the city's planimetric polygons, passed through the same projection as
the buildings and cut so that no two overlap. Trees stand at the city's tree points and use
Valdrada's street-tree models, scaled from trunk diameter. Vehicles use Valdrada's `fleet24`
models and are placed by rule: parked along kerbs where the centre-line record has parking lanes
and in parking lots, moving along lanes derived from lane counts and measured road width.
Pedestrians use the Rocketbox bodies from Valdrada and are placed on sidewalks and plazas.
Hydrants, bus shelters, kiosks and bike racks come from city records; lamps, signals and road
paint from rules. Every placed model is an instanceable reference to one converted USD file.

Rejected, with reasons (detail in `docs/validation/reports/populate_investigation.md`):

| Option | Reason |
| --- | --- |
| Valdrada's USD export or compiled tiles | browser export of one take in its own frame; undocumented binary; ODbL |
| Valdrada's CARLA pedestrian set | licence notice forbids dataset building and model training or testing |
| Isaac Sim characters and agent system | not tested; limb animation is at most a couple of pixels at our altitudes |
| Generated blob trees | not needed once the tree files were decoded |
| CARLA street props | furniture is 4 to 40 pixels wide; generated shapes are enough |
| Point instancer | copies lose their separate ids in `instance_segmentation` |

## 2. Evidence for the prototype questions

Numbers are in `docs/validation/reports/prototype.json`; the picture is `docs/validation/images/populate_prototype_120-broadway.png`.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Does a Valdrada model convert with the right scale, up axis and materials? | Scale and axis yes: taxi renders 5.35 m long and 1.56 m high against 5.36 and 1.56 m in the file; converted wheelbase 2.916 m against 2.913 m for the real Ford Crown Victoria. Materials only partly: vehicle and pedestrian textures are KTX2 and were not converted, so parts carry flat colours. Tree textures convert. |
| 2 | Do labels survive conversion? | The model files carry none. Labels are applied to each placed prim after import, and every class then appears in the rendered segmentation. |
| 3 | Distinct instance id per copy? | Plain references: yes in both annotators. Instanceable references: yes in both. Point instancer: yes in `instance_id_segmentation`, no in `instance_segmentation`. Instanceable references kept. |
| 4 | How much fits, at what frame time? | Prototype (150 m, 619 models): 0.45 s per frame, 3.3 GB in use. Full scene (800 m, 13,538 models, 73 million triangles drawn, 0.8 million stored): 1.45 s per frame at 150 ft and 1.56 s at 500 ft, 3.4 GB in use at the end, peak 3.1 GB of 46 GB in the probe job. Buildings only was 0.31 s per frame. |
| 5 | Do thin markings render cleanly? | Partly answered. Crosswalks are clear at both altitudes. The count of marking pixels changes by a median of 1.5% (150 ft) and 2.1% (500 ft) from frame to frame, with single-frame steps up to 21% that were not traced to a cause. Lane lines are under one pixel at 500 ft. The clips were not inspected frame by frame for shimmer. |
| 6 | Do shadows appear? | Yes, but weakly for low objects: 1.8% of the ground within 6 m of a taxi and 7.5% within 12 m of a tree falls under 75% of open-ground brightness. The sky light fills shadows; the lights were not retuned. |
| 7 | Clean depth and segmentation for a moving object, no lag? | Yes. A car moved between shots sits at the same place in RGB, depth and segmentation within 0.09 m; a one-frame lag would show as 5 m. |

Leaf cut-outs work: 78% of pixels inside a tree crown are vegetation and 22% show the ground
through the leaves, in depth and segmentation alike.

## 3. Sources and assets

City data, all from `data.cityofnewyork.us`, fetched 14 Nov 2026 and cached:

| Layer | Dataset id |
| --- | --- |
| Roadbed, sidewalk, median, public plazas, parking lots, parks, hydrography | `i36f-5ih7`, `52n9-sdep`, `ees7-4ufv`, `ue2e-9jm2`, `7cgt-uhhz`, `y6ja-fw4f`, `pjs3-c3z5` |
| Street centre lines | `inkn-q76z` |
| Forestry Tree Points | `hn5i-inap` |
| Hydrants, bus stop shelters, bicycle parking, LinkNYC kiosks | `5bgh-vtsn`, `t4f2-8md7`, `592z-n7dk`, `s4kf-3yrf` |

Models, from Valdrada (`mehmetkeremturkcan/valdrada`, tag `v0.7.1`), 0.61 GB downloaded:

| Set | Used | Licence |
| --- | --- | --- |
| `fleet24` | 18 vehicle models in 33 colour variants | CC BY 4.0, credit "CARLA Simulator (carla.org)" for the CARLA-derived ones |
| `bxtrees` | 20 street trees with leaf and bark textures | CC BY 4.0; textures CC0 1.0 |
| `peds24/rb27` | 12 adult bodies | MIT, Microsoft Rocketbox Avatar Library |

The attribution text is in the README under "Attribution". Model files, textures and converted
USD files sit under `data/assets/`, which is gitignored; `scripts/fetch_assets.sh` downloads and
converts them. The script was tested on one folder (matching file hash), not end to end.

## 4. Changes made

New files:

| File | Purpose |
| --- | --- |
| `nyc_drone_sim/layers.py` | fetch and project the street layers |
| `nyc_drone_sim/populate.py` | seeded placement, recorded motion, road paint |
| `nyc_drone_sim/gltf.py` | reader for the compressed glTF model files |
| `nyc_drone_sim/assets.py` | convert models to the USD library |
| `nyc_drone_sim/dress.py` | add surfaces, paint and placed models to the scene |
| `nyc_drone_sim/shotlist.py`, `nyc_drone_sim/sim/shots.py` | prepare and render the verification shots |
| `checks/populate.py`, `checks/prototype.py` | P1 to P12, and the prototype measurements |
| `scripts/fetch_assets.sh`, `scripts/populate_evidence.py` | download models; build the comparisons |
| `tests/test_populate.py` | unit tests (7 pass) |

Changed files: `scene.py` (ground cut around the street surfaces, calls `dress.py`),
`sim/render.py` (applies recorded poses per frame to objects near the camera), `probes.py`
(ground probe avoids placed objects), `README.md`, `.gitignore`.

New config sections: `assets`, `layers`, `populate`; new keys `scene.flat_classes`,
`scene.surface_colors`, `scene.surface_roughness`, `scene.marking_colors`,
`probe.ground_object_clear_m`, `render.motion_radius_per_altitude`, `render.motion_radius_base_m`.

Existing checks altered:

| Check | Change | Why |
| --- | --- | --- |
| scene-8 | counts every mesh drawn, with placed copies; still warns above 5 million | it counted buildings and ground only; now warns at 73 million |
| probe-4, render-3 | a `ground` expectation accepts any class in `scene.flat_classes` | the pixel under the camera is now road, sidewalk or plaza |
| path-2 | also measures the path against every placed model's bounding box | trees and lamps are new obstacles; nearest is 38 m at 150 ft |
| render-6 | adds the mean share of each other class | as the brief asked |

No tolerance was changed.

## 5. Motion

| Layer | Motion class |
| --- | --- |
| Ground surfaces, road markings | static |
| Trees | static |
| Parked vehicles (2,698) | static |
| Moving vehicles (513) | recorded animation: constant speed along a lane, easing to a stop at the end of the route |
| Street furniture (2,663) | static |
| Pedestrians standing (4,068) | static |
| Pedestrians walking (1,305) | recorded animation: straight line along the sidewalk, no limb motion |

Nothing is independently simulated. Motion is authored for 640 frames (40 s).

## 6. Limitations

- **P1 needs a ruling.** Roadbed overlaps building footprints by 0.75% of building area and
  sidewalk by 0.97%, each under 1%; together 1.71%. The check passes on the per-layer reading and
  warns. The measure is also a weak test of alignment here: shifting the roadbed 3 m only raises
  its overlap to 0.99%.
- **The default camera paths show little of the new content.** Buildings fill 77% of the 150 ft
  clip and 92% of the 500 ft clip. Vehicles are 0.07% of the 150 ft clip.
- **No textures on vehicles or people.** Flat colours per part; some box trucks render dark.
- **Shadows of low objects are faint**, and the lighting was not tuned.
- **Everything is flat.** No kerb height; elevated roads and bridge ramps lie on the ground.
- **Parking is generous.** Kerbs fill wherever the record lists a parking lane, at 70% occupancy.
- **Traffic is simple.** No signals or turns; cross traffic is thinned until nothing collides.
- **Walkers slide.** No limb animation, and walkers can pass through each other and through trees.
- **Trees:** 60 of 2,351 records were not placed; species outside the 11 mapped genera get a
  form by record id.
- **Furniture:** lamps and signals are rule-placed, not surveyed; 280 items were dropped for not
  landing on a walkable surface.
- **Segmentation of trees is speckled**, because leaf cut-outs let the surface behind show through.
- **Still open from before:** the 150 ft street-variant clip fails render-5; the street variant was
  not re-rendered with the populated scene.

## 7. Comparisons

Under `docs/validation/images/` (the `.mp4` files are gitignored like all videos):

| File | Content |
| --- | --- |
| `populate_before_after_120-broadway_150.png`, `_500.png` | same pose, frame 46, before and after |
| `populate_sheet_120-broadway_150.png`, `_500.png` | RGB, depth and segmentation of one populated frame |
| `populate_before_after_120-broadway_150.mp4`, `_500.mp4` | side-by-side clips |
| `populate_close_120-broadway.png` | 150 ft view of a street with vehicles, trees and pedestrians, with an enlarged crop |
| `populate_prototype_120-broadway.png` | the prototype stage |

**Verdict at 150 ft.** Populated, and credible where the street is in view: road, sidewalk,
plaza, a row of trees, lamps and pedestrians read correctly, and the close view shows a queue of
recognisable cars and taxis. The weakest element is the pedestrians, flat-coloured figures that
stand or slide. In the default clip the larger weakness is the framing: a blank building wall
fills most of the picture.

**Verdict at 500 ft.** Populated but only in the gaps: streets show crosswalks, taxis and cars,
and they read as traffic. Untextured building boxes still fill over 90% of the frame and are now
the weakest element by a wide margin; trees do not appear in this clip at all.
