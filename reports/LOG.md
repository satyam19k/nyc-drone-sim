# Verification log

## Stage 1 — 2026-11-14

Notes:
- G1.4 height identity is checked on rows with a measured height. Imputed rows have no raw feet value (0 or NULL in the source), so they are checked for "no raw height and >= floor_m" instead.
- G1.2 compares the paged rows against the count(*) response; both were fetched in the same run and are replayed from data/cache/.
- Feature code 1002 (storage tank) has no rows in the live dataset; code 1003 had no rows inside this box.

```
$ env_cpu/bin/python checks/stage_1.py
[PASS] G1.1: bin=1001026 lon=-74.010542 lat=40.708233; footprint(s) with that bin: 1, nearest 0.00 m from point
[PASS] G1.2: rows fetched 822, server count(*) 822
[PASS] G1.3: missing-field rows {'height_roof': 0, 'ground_elevation': 0, 'feature_code': 0, 'bin': 0, 'doitt_id': 0, 'construction_year': 0, 'last_status_type': 0, 'base_bbl': 0, 'mappluto_bbl': 0, 'geom_source': 0}; null geometries 0; valid after repair 100.0000% (0 repaired); all 753 output geometries valid: True
[PASS] G1.4: max |height_m - 0.3048*height_ft_raw| = 0.00e+00 over 748 measured rows; 5 imputed rows have no raw height and are >= floor: True; median 26.39 m, max 429.27 m
[PASS] G1.5: farthest vertex 935.6 m, limit 950 m
[PASS] G1.6: imputed share 0.66% (5 of 753)
[PASS] G1.7: offline rerun from cache byte-identical: True (545018 bytes)
[PASS] G1.8: evidence/stage_1_120-broadway.png (433696 bytes)
stage 1: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_1.json
exit=0
```

## Stage 2 — 2026-11-14

Notes:
- G2.2: the installed usd-core 26.11 has no `UsdUtils.ComplianceChecker`. The check runs every validator from its successor, `UsdValidation` (30 validators), and fails on any Error.
- G2.5 and G2.6 read meshes from scene.usdc and use the Stage 1 footprints only as the reference to compare against.
- `UsdSemantics` is present, so labels are authored in Stage 2.

```
$ env_cpu/bin/python checks/stage_2.py
[PASS] G2.1: metersPerUnit=1.0, upAxis=Z
[PASS] G2.2: UsdValidation (successor of UsdUtils.ComplianceChecker, which usd-core 0.26.11 no longer ships): 30 validators, 0 errors, 0 warnings
[PASS] G2.3: USD building meshes 753, Stage 1 kept 753
[PASS] G2.4: over 753 meshes: max |zmin - 0| = 0.00e+00 m, max |zmax - height_m| = 1.45e-05 m
[PASS] G2.5: 50 random meshes (seed 2026): worst relative volume error 3.92e-06 (B_559242), negative volumes 0; over all 753 meshes worst 7.27e-06
[PASS] G2.6: top-down IoU at 1 m/px = 1.00000 (USD 743354 px, footprints 743354 px)
[PASS] G2.7: class labels resolved per mesh: {'building': 753, 'ground': 1}; meshes without exactly one label: 0 []
[PASS] G2.8: 38562 triangles, scene.usdc 1.40 MB
[PASS] G2.9: evidence/stage_2_120-broadway.png (500822 bytes)
stage 2: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_2.json
exit=0
```

## Stage 4 — 2026-11-14

Notes:
- 150 ft at 120 Broadway: none of the 168 flyover/orbit candidates clears the 15 m lateral margin (108 of 181 buildings within 300 m are taller than the camera), so the planner used the `hover_pan` fallback. G4.3 speed and G4.6 target-in-view apply to moving paths; for hover_pan the check confirms the camera is stationary and reports the target share anyway.
- 500 ft: flyover, heading 0 deg (east), 45 of 168 candidates survive.
- G4.2 measures 3D point-to-triangle distance against every building triangle in scene.usdc.

```
$ env_cpu/bin/python -m pytest -q tests
3 passed in 4.58s
$ env_cpu/bin/python checks/stage_4.py
[PASS] G4.1: 150 ft: max |z - 45.72| = 0.0e+00 m; matrix translation = position; rotation orthonormal; pitch -45.000 deg; 500 ft: max |z - 152.40| = 0.0e+00 m; matrix translation = position; rotation orthonormal; pitch -45.000 deg
[PASS] G4.2: from 38560 USD triangles; 150 ft: nearest building surface 15.78 m; 500 ft: nearest building surface 21.06 m; required >= 10 m
[PASS] G4.3: 150 ft: stationary (hover_pan), max yaw rate 15.65 deg/s; 500 ft: speed 12.0000-12.0000 m/s (config 12), max yaw rate 0.00 deg/s
[PASS] G4.4: 150 ft: 93 frames at 16.000 fps, 5.8125 s; 500 ft: 93 frames at 16.000 fps, 5.8125 s
[PASS] G4.5: 150 ft: farthest corner ground hit 144.2 m from origin; 500 ft: farthest corner ground hit 379.7 m from origin; limit 800 m
[PASS] G4.6: 150 ft (hover_pan): not applicable; target in image in 100.0% of frames; 500 ft (flyover): target in image in 100.0% of frames
[PASS] G4.7: 150 ft: 73.740 x 45.747 deg; 500 ft: 73.740 x 45.747 deg
[PASS] G4.8: evidence/stage_4_120-broadway.png (580118 bytes)
stage 4: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_4.json
exit=0
```

## Stage 3 — 2026-11-14

Notes:
- Ran on NVIDIA L40S (node gl026), Isaac Sim 6.2.0.0, omni.replicator.core 1.13.38. Depth annotator `distance_to_image_plane`, documented in the installed package as float32 metres.
- First run (job 23121725): checks FAILED. The first captured frame (ground_150ft) came back with zero-size RGB, depth and segmentation arrays; the other two probes read 152.4000 m and 50.0000 m. Fix: one discarded warm-up frame before the probes, plus an error if any annotator returns no data. No threshold changed.
- Second run (job 23122654): checks passed.

```
$ sbatch --account=torch_pr_54_general --export=ALL,LOCATION="120 Broadway" slurm/probe.sbatch   # job 23122654
probe ground_150ft rendered
probe ground_500ft rendered
probe roof rendered
probe done: load 2.1 s, peak VRAM 2858 / 46068 MiB on NVIDIA L40S
$ env_cpu/bin/python checks/stage_3.py
[PASS] G3.1: ground_150ft: centre depth 45.7200 m, expected 45.72 m +/- 0.25 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.2: ground_500ft: centre depth 152.4000 m, expected 152.40 m +/- 0.5 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.3: roof: centre depth 50.0000 m, expected 50.00 m +/- 0.5 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.4: ground_150ft: centre 'ground' (expected 'ground'), unlabelled 0.00%, shares {'building': 0.0413, 'ground': 0.9587}; ground_500ft: centre 'ground' (expected 'ground'), unlabelled 0.00%, shares {'building': 0.5739, 'ground': 0.4261}; roof: centre 'building' (expected 'building'), unlabelled 0.00%, shares {'building': 0.8768, 'ground': 0.1232}
[PASS] G3.5: ground_150ft: mean 146.9, clipped 0.00%; ground_500ft: mean 149.1, clipped 0.00%; roof: mean 156.2, clipped 0.00%
[PASS] G3.6: peak 2858 MiB of 46068 MiB (6.2%) on NVIDIA L40S; scene load 2.1 s
[PASS] G3.7: evidence/stage_3_120-broadway.png (376741 bytes)
stage 3: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_3.json
exit=0
```

## Stage 5 — 2026-11-14

Notes:
- Rendered on NVIDIA L40S, job 23126414, run id run_001, both altitudes on the first submission.
- The brief points at standalone_examples/replicator/cosmos_writer_warehouse.py; the pip install of Isaac Sim 6.2.0 does not ship it. The capture loop follows the installed CosmosWriter source (omni.replicator.core 1.13.38, writers_default/cosmos.py) and its test in tests/test_writers.py.
- The writer encoded all MP4s itself at 16 fps (frame rate taken from the stage timeline); ffprobe confirmed and nothing was re-encoded. ffmpeg/ffprobe come from the static-ffmpeg package in env_cpu because the cluster has none.
- G5.3 WARN: at 500 ft no frame has ground at the image centre (the centre is on buildings throughout), so the altitude identity is only exercised by the 150 ft clip (24 frames, 0.000% error).
- G5.6 WARN: at 500 ft buildings cover more than 95% of the frame in 22 of 93 frames. Treated as flag-for-review, not a failure, following the wording of G5.6.
- G5.7 compares the clip frame 0 with a second render of the same pose taken during warm-up.

```
$ sbatch --account=torch_pr_54_general --export=ALL,LOCATION="120 Broadway",RUN_ID=run_001 slurm/render.sbatch   # job 23126414
render done: 93 frames in 31.1 s -> /scratch/sk12075/nyc-drone-sim/runs/run_001/120-broadway/150
render done: 93 frames in 30.6 s -> /scratch/sk12075/nyc-drone-sim/runs/run_001/120-broadway/500
$ env_cpu/bin/python -m nyc_drone_sim.postrender --location "120 Broadway" --run run_001
$ env_cpu/bin/python checks/stage_5.py --run run_001
[PASS] G5.1: 150 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none); 500 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none)
[PASS] G5.2: 150 ft: max position error 0.0000 cm, max orientation error 0.00000 deg over 93 frames; 500 ft: max position error 0.0000 cm, max orientation error 0.00001 deg over 93 frames
[WARN] G5.3: 150 ft: 24 frames with ground at centre, centre depth 64.66 m (expected 64.7), worst altitude error 0.000%; 500 ft: no frame has ground at the image centre, so this clip gives nothing to test
[PASS] G5.4: 150 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0; 500 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0
[PASS] G5.5: 150 ft: frame-to-frame mean abs difference min 2.038, mean 3.889, max 5.437 (4-sigma limit 7.608); 500 ft: frame-to-frame mean abs difference min 2.066, mean 2.658, max 3.620 (4-sigma limit 3.921)
[WARN] G5.6: 150 ft: building share per frame min 72.0%, mean 77.3%, max 84.2%; 0 frames outside 5-95%; 500 ft: building share per frame min 88.9%, mean 91.8%, max 97.9%; 22 frames outside 5-95%; flagged for review
[PASS] G5.7: 150 ft: PSNR of second frame-0 render against the clip's frame 0 = 42.34 dB; 500 ft: PSNR of second frame-0 render against the clip's frame 0 = 40.78 dB
[PASS] G5.8: evidence/stage_5_120-broadway_150.png (264129 bytes); evidence/stage_5_120-broadway_500.png (380280 bytes)
RGB video 150 ft: /scratch/sk12075/nyc-drone-sim/runs/run_001/120-broadway/150/clip_0000/rgb.mp4
RGB video 500 ft: /scratch/sk12075/nyc-drone-sim/runs/run_001/120-broadway/500/clip_0000/rgb.mp4
stage 5: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_5.json
exit=0
```

## Stage 5 — depth picture fix, 2026-11-14

Notes:
- Defect found by eye in run_001: the writer colourised depth over its fixed 0.1-100 m log range, so the depth PNGs and depth.mp4 were almost black (grey range 60 at 150 ft, 10 at 500 ft). Metric depth was unaffected.
- Fix: render.py uses a subclass of the writer whose depth range comes from config (5-500 m). New check G5.9 requires the depth picture to span at least 64 grey levels; that threshold is our own, not from the brief.
- Re-rendered as run_002 (job 23132248). G5.3 and G5.6 warnings are unchanged from run_001.

```
$ sbatch --account=torch_pr_54_general --export=ALL,LOCATION="120 Broadway",RUN_ID=run_002 slurm/render.sbatch   # job 23132248
$ env_cpu/bin/python -m nyc_drone_sim.postrender --location "120 Broadway" --run run_002
$ env_cpu/bin/python checks/stage_5.py --run run_002
[PASS] G5.1: 150 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none); 500 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none)
[PASS] G5.2: 150 ft: max position error 0.0000 cm, max orientation error 0.00000 deg over 93 frames; 500 ft: max position error 0.0000 cm, max orientation error 0.00001 deg over 93 frames
[WARN] G5.3: 150 ft: 24 frames with ground at centre, centre depth 64.66 m (expected 64.7), worst altitude error 0.000%; 500 ft: no frame has ground at the image centre, so this clip gives nothing to test
[PASS] G5.4: 150 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0; 500 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0
[PASS] G5.5: 150 ft: frame-to-frame mean abs difference min 2.039, mean 3.891, max 5.444 (4-sigma limit 7.610); 500 ft: frame-to-frame mean abs difference min 2.066, mean 2.658, max 3.620 (4-sigma limit 3.921)
[WARN] G5.6: 150 ft: building share per frame min 72.0%, mean 77.3%, max 84.2%; 0 frames outside 5-95%; 500 ft: building share per frame min 88.9%, mean 91.8%, max 97.9%; 22 frames outside 5-95%; flagged for review
[PASS] G5.7: 150 ft: PSNR of second frame-0 render against the clip's frame 0 = 42.28 dB; 500 ft: PSNR of second frame-0 render against the clip's frame 0 = 40.78 dB
[PASS] G5.8: evidence/stage_5_120-broadway_150.png (276486 bytes); evidence/stage_5_120-broadway_500.png (422268 bytes)
[PASS] G5.9: 150 ft: colourised depth grey levels p1 87 to p99 182 (range 95, need >= 64); 500 ft: colourised depth grey levels p1 35 to p99 104 (range 69, need >= 64)
RGB video 150 ft: /scratch/sk12075/nyc-drone-sim/runs/run_002/120-broadway/150/clip_0000/rgb.mp4
RGB video 500 ft: /scratch/sk12075/nyc-drone-sim/runs/run_002/120-broadway/500/clip_0000/rgb.mp4
stage 5: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_5.json
exit=0
```

## Full rerun after fetch fix — 2026-11-14

Notes:
- Defect found by cross-check: `within_box` only returns footprints lying fully inside the box. An `intersects` query over the same box returned 856 rows against 822, and 3 of the 34 extra footprints touch the 800 m scene circle (BINs 1085792, 1087170, 1078984). G1.2 could not see this because it compares against the count for the same filter.
- Fix: Stage 1 fetches with `intersects(the_geom, POLYGON(box))`. New check G1.9 confirms the filter, that the box holds the scene circle, and reports the within_box count for comparison. 756 buildings kept (was 753).
- Everything downstream rebuilt: scene.usdc (756 meshes), paths (unchanged choices), probes (job 23134170), render run_003 (job 23134183).

```
$ env_cpu/bin/python checks/stage_1.py
[PASS] G1.1: bin=1001026 lon=-74.010542 lat=40.708233; footprint(s) with that bin: 1, nearest 0.00 m from point
[PASS] G1.2: rows fetched 856, server count(*) 856
[PASS] G1.3: missing-field rows {'height_roof': 0, 'ground_elevation': 0, 'feature_code': 0, 'bin': 0, 'doitt_id': 0, 'construction_year': 0, 'last_status_type': 0, 'base_bbl': 0, 'mappluto_bbl': 0, 'geom_source': 0}; null geometries 0; valid after repair 100.0000% (0 repaired); all 756 output geome
[PASS] G1.4: max |height_m - 0.3048*height_ft_raw| = 0.00e+00 over 751 measured rows; 5 imputed rows have no raw height and are >= floor: True; median 26.45 m, max 429.27 m
[PASS] G1.5: farthest vertex 935.6 m, limit 950 m
[PASS] G1.6: imputed share 0.66% (5 of 756)
[PASS] G1.7: offline rerun from cache byte-identical: True (551918 bytes)
[PASS] G1.8: evidence/stage_1_120-broadway.png (438177 bytes)
[PASS] G1.9: filter is intersects over a box reaching at least 850 m from the target (radius 800 m); 856 rows against 822 from within_box, so 34 straddling footprints were recovered
stage 1: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_1.json
$ env_cpu/bin/python checks/stage_2.py
[PASS] G2.1: metersPerUnit=1.0, upAxis=Z
[PASS] G2.2: UsdValidation (successor of UsdUtils.ComplianceChecker, which usd-core 0.26.11 no longer ships): 30 validators, 0 errors, 0 warnings
[PASS] G2.3: USD building meshes 756, Stage 1 kept 756
[PASS] G2.4: over 756 meshes: max |zmin - 0| = 0.00e+00 m, max |zmax - height_m| = 1.45e-05 m
[PASS] G2.5: 50 random meshes (seed 2026): worst relative volume error 3.40e-06 (B_393702), negative volumes 0; over all 756 meshes worst 7.27e-06
[PASS] G2.6: top-down IoU at 1 m/px = 1.00000 (USD 764261 px, footprints 764261 px)
[PASS] G2.7: class labels resolved per mesh: {'building': 756, 'ground': 1}; meshes without exactly one label: 0 []
[PASS] G2.8: 39578 triangles, scene.usdc 1.43 MB
[PASS] G2.9: evidence/stage_2_120-broadway.png (509931 bytes)
stage 2: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_2.json
$ env_cpu/bin/python checks/stage_4.py
[PASS] G4.1: 150 ft: max |z - 45.72| = 0.0e+00 m; matrix translation = position; rotation orthonormal; pitch -45.000 deg; 500 ft: max |z - 152.40| = 0.0e+00 m; matrix translation = position; rotation orthonormal; pitch -45.000 deg
[PASS] G4.2: from 39576 USD triangles; 150 ft: nearest building surface 15.78 m; 500 ft: nearest building surface 21.06 m; required >= 10 m
[PASS] G4.3: 150 ft: stationary (hover_pan), max yaw rate 15.65 deg/s; 500 ft: speed 12.0000-12.0000 m/s (config 12), max yaw rate 0.00 deg/s
[PASS] G4.4: 150 ft: 93 frames at 16.000 fps, 5.8125 s; 500 ft: 93 frames at 16.000 fps, 5.8125 s
[PASS] G4.5: 150 ft: farthest corner ground hit 144.2 m from origin; 500 ft: farthest corner ground hit 379.7 m from origin; limit 800 m
[PASS] G4.6: 150 ft (hover_pan): not applicable; target in image in 100.0% of frames; 500 ft (flyover): target in image in 100.0% of frames
[PASS] G4.7: 150 ft: 73.740 x 45.747 deg; 500 ft: 73.740 x 45.747 deg
[PASS] G4.8: evidence/stage_4_120-broadway.png (580118 bytes)
stage 4: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_4.json
$ env_cpu/bin/python checks/stage_3.py
$ env_cpu/bin/python checks/stage_5.py --run run_003
[PASS] G3.1: ground_150ft: centre depth 45.7200 m, expected 45.72 m +/- 0.25 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.2: ground_500ft: centre depth 152.4000 m, expected 152.40 m +/- 0.5 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.3: roof: centre depth 50.0000 m, expected 50.00 m +/- 0.5 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.4: ground_150ft: centre 'ground' (expected 'ground'), unlabelled 0.00%, shares {'building': 0.0413, 'ground': 0.9587}; ground_500ft: centre 'ground' (expected 'ground'), unlabelled 0.00%, shares {'building': 0.5739, 'ground': 0.4261}; roof: centre 'building' (expected 'building'), unlabell
[PASS] G3.5: ground_150ft: mean 148.2, clipped 0.00%; ground_500ft: mean 151.0, clipped 0.00%; roof: mean 154.7, clipped 0.00%
[PASS] G3.6: peak 2826 MiB of 46068 MiB (6.1%) on NVIDIA L40S; scene load 0.2 s
[PASS] G3.7: evidence/stage_3_120-broadway.png (380580 bytes)
stage 3: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_3.json
[PASS] G5.1: 150 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none); 500 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none)
[PASS] G5.2: 150 ft: max position error 0.0000 cm, max orientation error 0.00000 deg over 93 frames; 500 ft: max position error 0.0000 cm, max orientation error 0.00001 deg over 93 frames
[WARN] G5.3: 150 ft: 24 frames with ground at centre, centre depth 64.66 m (expected 64.7), worst altitude error 0.000%; 500 ft: no frame has ground at the image centre, so this clip gives nothing to test
[PASS] G5.4: 150 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0; 500 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0
[PASS] G5.5: 150 ft: frame-to-frame mean abs difference min 2.043, mean 3.890, max 5.443 (4-sigma limit 7.610); 500 ft: frame-to-frame mean abs difference min 2.061, mean 2.658, max 3.617 (4-sigma limit 3.917)
[WARN] G5.6: 150 ft: building share per frame min 72.0%, mean 77.3%, max 84.2%; 0 frames outside 5-95%; 500 ft: building share per frame min 88.9%, mean 91.8%, max 97.9%; 22 frames outside 5-95%; flagged for review
[PASS] G5.7: 150 ft: PSNR of second frame-0 render against the clip's frame 0 = 42.29 dB; 500 ft: PSNR of second frame-0 render against the clip's frame 0 = 40.71 dB
[PASS] G5.8: evidence/stage_5_120-broadway_150.png (275669 bytes); evidence/stage_5_120-broadway_500.png (424682 bytes)
[PASS] G5.9: 150 ft: colourised depth grey levels p1 87 to p99 182 (range 95, need >= 64); 500 ft: colourised depth grey levels p1 35 to p99 104 (range 69, need >= 64)
stage 5: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_5.json
[exited with code 0]
```

## Street variant (long clips) — 2026-11-14

Notes:
- New path variant `street`: 640 frames (40 s) per altitude, route along street centre lines, same camera, speeds and clearances. Files `path_<alt>_street.json`, checks run with `--variant street`, reports `stage_4_street.json` and `stage_5_street.json`. Default 93-frame paths are byte-identical to before.
- Render job 23138941 -> runs/run_004.
- G5.5 FAILED at 150 ft (first failure). Six consecutive frame pairs, 132-137, read 4.53-4.67 against a 4-sigma limit of 4.42. Neighbouring pairs read 4.37 before and 4.27 after, so it is a gradual rise in image change over that stretch, not a single-frame jump. Threshold left unchanged; waiting for a decision.

```
$ env_cpu/bin/python checks/stage_4.py --variant street
[PASS] G4.1: 150 ft: max |z - 45.72| = 0.0e+00 m; matrix translation = position; rotation orthonormal; pitch -45.000 deg; 500 ft: max |z - 152.40| = 0.0e+00 m; matrix translation = position; rotation orthonormal; pitch -45.000 deg
[PASS] G4.2: from 39576 USD triangles; 150 ft: nearest building surface 17.46 m; 500 ft: nearest building surface 17.98 m; required >= 10 m
[PASS] G4.3: 150 ft: speed 7.9995-8.0000 m/s (config 8), max yaw rate 7.39 deg/s; 500 ft: speed 11.9995-12.0000 m/s (config 12), max yaw rate 6.82 deg/s
[PASS] G4.4: 150 ft: 640 frames at 16.000 fps, 40.0000 s; 500 ft: 640 frames at 16.000 fps, 40.0000 s
[PASS] G4.5: 150 ft: farthest corner ground hit 396.8 m from origin; 500 ft: farthest corner ground hit 622.2 m from origin; limit 800 m
[PASS] G4.6: 150 ft (street): not applicable; target in image in 0.0% of frames; 500 ft (street): not applicable; target in image in 37.7% of frames
[PASS] G4.7: 150 ft: 73.740 x 45.747 deg; 500 ft: 73.740 x 45.747 deg
[PASS] G4.8: evidence/stage_4_120-broadway_street.png (721282 bytes)
stage 4: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/stage_4_street.json
$ env_cpu/bin/python checks/stage_5.py --run run_004 --variant street
[PASS] G5.1: 150 ft: 5 folders x 640 PNGs at 1280x720; 5 MP4s at 640 frames, 16 fps (re-encoded: none); 500 ft: 5 folders x 640 PNGs at 1280x720; 5 MP4s at 640 frames, 16 fps (re-encoded: none)
[PASS] G5.2: 150 ft: max position error 0.0000 cm, max orientation error 0.00000 deg over 640 frames; 500 ft: max position error 0.0000 cm, max orientation error 0.00000 deg over 640 frames
[PASS] G5.3: 150 ft: 640 frames with ground at centre, centre depth 64.66 m (expected 64.7), worst altitude error 0.000%; 500 ft: 343 frames with ground at centre, centre depth 215.53 m (expected 215.5), worst altitude error 0.000%
[PASS] G5.4: 150 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0; 500 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0
[FAIL] G5.5: 150 ft: frame-to-frame mean abs difference min 0.461, mean 1.724, max 4.669 (4-sigma limit 4.420); 500 ft: frame-to-frame mean abs difference min 0.994, mean 2.159, max 4.553 (4-sigma limit 4.697)
[PASS] G5.6: 150 ft: building share per frame min 19.4%, mean 47.0%, max 66.5%; 0 frames outside 5-95%; 500 ft: building share per frame min 47.9%, mean 80.6%, max 94.8%; 0 frames outside 5-95%
[PASS] G5.7: 150 ft: PSNR of second frame-0 render against the clip's frame 0 = 46.83 dB; 500 ft: PSNR of second frame-0 render against the clip's frame 0 = 46.83 dB
[PASS] G5.8: evidence/stage_5_120-broadway_150_street.png (272497 bytes); evidence/stage_5_120-broadway_500_street.png (413444 bytes)
[PASS] G5.9: 150 ft: colourised depth grey levels p1 80 to p99 148 (range 68, need >= 64); 500 ft: colourised depth grey levels p1 24 to p99 126 (range 102, need >= 64)
stage 5: FAILED -> /scratch/sk12075/nyc-drone-sim/reports/stage_5_street.json
[exited with code 0]
```

## Populate the scene — 2026-11-14 to 2026-11-15

Decisions made without asking (routine, reversible):

- Pinned the Valdrada model set to tag `v0.7.1`; downloaded 0.61 GB into `data/assets/` (gitignored).
- Did not download `peds24/bodies`: its notice forbids dataset building and model training or testing.
- The vehicle and people files use compressed buffers. A Node-based converter could not be installed
  here, so `nyc_drone_sim/gltf.py` reads the format directly. Checked by a taxi wheelbase of 2.916 m
  against 2.913 m for the real car.
- Tree files are a packed format of their own. First reading was wrong (unsigned positions, leaf cards
  as two quads); the prototype render showed it. Fixed to signed positions and eight-sided cards.
- Vehicle and pedestrian textures are not converted (KTX2). Parts get flat colours by material role.
- Street furniture is generated from simple shapes, not taken from the CARLA props.
- Tree source is Forestry Tree Points (`hn5i-inap`), living trees only, not the 2015 census.
- Surfaces are cut so they never overlap and all sit at z = 0. Road paint sits 2 cm above the road.
- Instanceable references chosen for placed models after the prototype (distinct ids in both
  instance annotators; a point instancer loses them in `instance_segmentation`).
- Tree records inside the roadbed line by up to 2 m are moved 0.4 m onto the kerb; deeper ones and
  ones inside buildings are not placed.
- A parked vehicle must be nearer its own centre line than any other (found by P3: one car at a
  street merge was aligned with the wrong street).
- Two-way streets: lane direction follows the side of the centre line; a centre lane is used only
  on one-lane streets (found by P3 on the prototype).

Existing checks changed, and why:

- G2.8 counted building and ground triangles only. It now counts every mesh drawn, with placed
  copies, and still warns above 5 million. The populated scene draws 73 million (0.8 million stored).
- G3.4 and G5.3 expected the class `ground` under the camera. They now accept any class in
  `scene.flat_classes` (ground, road, sidewalk, median, plaza, parking, park, water, road_marking).
  Buildings, vehicles, trees, people and furniture are still rejected. Tolerances unchanged.
- The Stage 3 ground probe point also keeps 3 m clear of placed objects and outside tree crowns.
- G4.2 also measures the camera path against the bounding box of every placed model.
- G5.6 keeps the building share and adds the mean share of every other class.

Prototype (job 23200641, 150 m radius, through the Stage 5 path): 93 frames in 42 s per altitude,
3.3 GB VRAM in use at the end.

Full scene (jobs 23203971 probe, 23203973 shots and render -> runs/run_005):

```
$ env_cpu/bin/python checks/prototype.py
[PASS] Q1 scale: taxi rendered 5.35 m long and 1.56 m high (from depth and segmentation), model file 5.36 and 1.56 m; rendered width 1.74 m against 1.93 m in the file, which includes the mirrors; converted wheelbase 2.916 m against 2.913 m for the real Ford Cr
[PASS] Q2 labels: labels were applied to each placed prim after import (the model files carry none); class found at each object: Furniture/instanceable: ['furniture', 'ground']; Inst/instanceable: ['vehicle']; People/instanceable: ['ground', 'person']; Ref/ref
[PASS] Q3 instances: reference / instance_id: ids per copy [3, 3, 3], distinct between copies True; reference / instance: ids per copy [1, 1, 1], distinct between copies True; instanceable / instance_id: ids per copy [3, 3, 3], distinct between copies True; in
[PASS] Q6 shadows: sun elevation 55 deg: 1.8% of the ground within 6 m of the taxi and 7.5% within 12 m of the tree is under 75% of open-ground brightness; darkest ground beside the taxi 93%, under the tree 63% of open ground
[PASS] Q7 motion: car set to x = -9.0 m: centre in segmentation -8.99, depth -8.99, rgb -9.09; car set to x = -4.0 m: centre in segmentation -3.99, depth -3.99, rgb -4.04; car set to x = +1.0 m: centre in segmentation +0.98, depth +0.98, rgb +0.97; largest dis
[PASS] leaf cut-out: inside 80% of the crown radius 78.2% of pixels are vegetation and 21.8% show through to the ground, in depth and segmentation alike, so the leaf cut-out works (solid leaf cards would give 100%); vegetation heights 0.0-12.9 m, model 13.0 m
[PASS] small objects: pixels at 150 ft straight down: ped_a: 111 px, top 1.72 m; ped_b: 87 px, top 1.72 m; lamp: 378 px, top 8.50 m; hydrant: 35 px, top 0.75 m
[PASS] sheet: evidence/populate_prototype_120-broadway.png (230506 bytes)
prototype: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/prototype.json
[WARN] P1: overlap with building footprints as a share of building area: roadbed 0.75%, sidewalk 0.97% (each under 1%); the two together 1.71%; for scale, roadbed shifted 3 m east would overlap 0.99%
[PASS] P2: 3211 vehicles (2698 static, 513 moving, every one of 640 frames tested): footprints outside roadbed and parking lots 0, overlapping pairs 0
[PASS] P3: 16056 on-street poses tested away from intersections: worst angle to the street 1.75 deg (limit 10), misaligned 0; 13612 moving poses, against the direction of travel 0
[PASS] P4: 5373 pedestrians (1305 walking, tested at 161 frames each): off sidewalk, plaza and crosswalk 0, inside a building 0, touching a vehicle 0
[PASS] P5: 2291 trees placed from 2351 records (165 recorded just inside the roadbed line were moved onto the kerb, 60 deeper in the roadbed or inside a building were not placed); placed trees outside roadbed and buildings 100.0%; heights 3.0-22.4 m, median 7.
[PASS] P6: 13538 placed models read back from the USD (placement list has 13538): largest |lowest z| 0.00 cm, over 5 cm: 0 []
[PASS] P7: USD: 50383 meshes (model copies included) each resolve to one class {'building': 756, 'furniture': 4476, 'ground': 1, 'median': 1, 'park': 1, 'parking': 1, 'person': 19702, 'plaza': 1, 'road': 1, 'road_marking': 3, 'sidewalk': 1, 'vegetation': 4582,
[PASS] P8: two copies of veh_mini_1 (vehicle_00002, vehicle_00005) in one frame; instance_id: ids [59832] vs [59831], distinct True; instance: ids [7615] vs [12096], distinct True
[PASS] P9: veh_stepvan_0 (vehicle_00007): roof 2.970 m above ground in its model file, camera at 45.72 m, expected depth 42.750 m, rendered 42.750 m (error 0.0 cm, limit 30); centre pixel class 'vehicle'
[PASS] P10: rebuilt with seed 2026: placement list identical True (13538 objects), file byte-identical True, recorded motion identical True, road paint identical True
[PASS] P11: probe job: peak VRAM 3089 of 46068 MiB (6.7%), sampled every 0.5 s while the full scene loaded and rendered on NVIDIA L40S; scene load 1.0 s; 150 ft clip: 3347 MiB in use when the render ended (7.3%), 1.45 s per frame over 93 frames, load and warm-
[PASS] P12: stage 1: exit 0, 9 checks, failed none, warnings none; stage 2: exit 0, 9 checks, failed none, warnings ['[WARN] G2.8']; stage 3: exit 0, 7 checks, failed none, warnings none; stage 4: exit 0, 8 checks, failed none, warnings none; stage 5: exit 0, 
[PASS] G3.1: ground_150ft: centre depth 45.7200 m, expected 45.72 m +/- 0.25 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.2: ground_500ft: centre depth 152.4000 m, expected 152.40 m +/- 0.5 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.3: roof: centre depth 50.0000 m, expected 50.00 m +/- 0.5 (annotator distance_to_image_plane, stage metersPerUnit 1.0)
[PASS] G3.4: ground_150ft: centre 'road' (expected 'ground', any ground-level class), unlabelled 0.00%, shares {'building': 0.0189, 'road_marking': 0.0219, 'ground': 0.0027, 'plaza': 0.4959, 'road': 0.2889, 'sidewalk': 0.1313, 'vehicle': 0.0326, 'person': 0.00
[PASS] G3.5: ground_150ft: mean 126.5, clipped 0.00%; ground_500ft: mean 132.8, clipped 0.00%; roof: mean 152.9, clipped 0.00%
[PASS] G3.6: peak 3089 MiB of 46068 MiB (6.7%) on NVIDIA L40S; scene load 1.0 s
[PASS] G3.7: evidence/stage_3_120-broadway.png (545233 bytes)
[PASS] G5.1: 150 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none); 500 ft: 5 folders x 93 PNGs at 1280x720; 5 MP4s at 93 frames, 16 fps (re-encoded: none)
[PASS] G5.2: 150 ft: max position error 0.0000 cm, max orientation error 0.00000 deg over 93 frames; 500 ft: max position error 0.0000 cm, max orientation error 0.00001 deg over 93 frames
[WARN] G5.3: 150 ft: 21 frames with ground at centre, centre depth 64.66 m (expected 64.7), worst altitude error 0.000%; 500 ft: no frame has ground at the image centre, so this clip gives nothing to test
[PASS] G5.4: 150 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0; 500 ft: empty metric-depth pixels 0.0000%, sky/background pixels 0
[PASS] G5.5: 150 ft: frame-to-frame mean abs difference min 4.493, mean 6.093, max 7.807 (4-sigma limit 9.941); 500 ft: frame-to-frame mean abs difference min 2.370, mean 3.281, max 4.446 (4-sigma limit 4.958)
[WARN] G5.6: 150 ft: building share per frame min 72.0%, mean 77.3%, max 84.2%; 0 frames outside 5-95%; 150 ft mean share of other classes: furniture 0.24%, ground 1.46%, person 0.16%, plaza 4.28%, road 7.51%, road_marking 0.41%, sidewalk 5.70%, vegetation 2.8
[PASS] G5.7: 150 ft: PSNR of second frame-0 render against the clip's frame 0 = 40.63 dB; 500 ft: PSNR of second frame-0 render against the clip's frame 0 = 40.58 dB
$ env_cpu/bin/python checks/populate.py --run run_005
[WARN] P1: overlap with building footprints as a share of building area: roadbed 0.75%, sidewalk 0.97% (each under 1%); the two together 1.71%; for scale, roadbed shifted 3 m east would overlap 0.99%
[PASS] P2: 3211 vehicles (2698 static, 513 moving, every one of 640 frames tested): footprints outside roadbed and parking lots 0, overlapping pairs 0
[PASS] P3: 16056 on-street poses tested away from intersections: worst angle to the street 1.75 deg (limit 10), misaligned 0; 13612 moving poses, against the direction of travel 0
[PASS] P4: 5373 pedestrians (1305 walking, tested at 161 frames each): off sidewalk, plaza and crosswalk 0, inside a building 0, touching a vehicle 0
[PASS] P5: 2291 trees placed from 2351 records (165 recorded just inside the roadbed line were moved onto the kerb, 60 deeper in the roadbed or inside a building were not placed); placed trees outside roadbed and buildings 100.0%; heights 3.0-22.4 m, median 7.
[PASS] P6: 13538 placed models read back from the USD (placement list has 13538): largest |lowest z| 0.00 cm, over 5 cm: 0 []
[PASS] P7: USD: 50383 meshes (model copies included) each resolve to one class {'building': 756, 'furniture': 4476, 'ground': 1, 'median': 1, 'park': 1, 'parking': 1, 'person': 19702, 'plaza': 1, 'road': 1, 'road_marking': 3, 'sidewalk': 1, 'vegetation': 4582,
[PASS] P8: two copies of veh_mini_1 (vehicle_00002, vehicle_00005) in one frame; instance_id: ids [59832] vs [59831], distinct True; instance: ids [7615] vs [12096], distinct True
[PASS] P9: veh_stepvan_0 (vehicle_00007): roof 2.970 m above ground in its model file, camera at 45.72 m, expected depth 42.750 m, rendered 42.750 m (error 0.0 cm, limit 30); centre pixel class 'vehicle'
[PASS] P10: rebuilt with seed 2026: placement list identical True (13538 objects), file byte-identical True, recorded motion identical True, road paint identical True
[PASS] P11: probe job: peak VRAM 3089 of 46068 MiB (6.7%), sampled every 0.5 s while the full scene loaded and rendered on NVIDIA L40S; scene load 1.0 s; 150 ft clip: 3347 MiB in use when the render ended (7.3%), 1.45 s per frame over 93 frames, load and warm-
[PASS] P12: stage 1: exit 0, 9 checks, failed none, warnings none; stage 2: exit 0, 9 checks, failed none, warnings ['[WARN] G2.8']; stage 3: exit 0, 7 checks, failed none, warnings none; stage 4: exit 0, 8 checks, failed none, warnings none; stage 5: exit 0, 
populate: PASSED -> /scratch/sk12075/nyc-drone-sim/reports/populate.json
```

## Render check: G5.3 centre block and scene-aware G5.9, 2026-11-15

- G5.3 failed on the 120 Broadway street clips (run_008): 3 of 627 frames at 150 ft (worst 9.14%) and 1 of 309 at 500 ft (1.98%). In all four frames a vehicle, pole or tree covered part of the 2 x 2 centre block whose depth is averaged, while the class was read from one pixel. With all four pixels on ground the worst error was 0.044% and 0.013%. Camera height recovered from every ground pixel of 12 frames per clip is 45.72 m and 152.40 m in both run folders.
- Fix: a frame counts only when every pixel of the centre block is a ground class. The 1% limit is unchanged.
- G5.9 failed on every waterfront clip (55 and 59 grey levels for Hudson, limit 64). Bare flat ground at -45 deg spans 50 of 255 levels on the 5-500 m log scale at either altitude, so a fixed 64 cannot be met by a flat scene.
- Fix, approved by the owner: the expected range is computed per frame from metric depth through the configured scale (formula matches stored pictures within 0.5 grey level) and the picture must reach `render.depth_grey_range_share` (0.9) of it. `render.depth_min_grey_range` is removed. The stock 0.1-100 m scale, simulated on one frame of each of four clips, is still rejected (30, 0, 45, 8 against 55, 51, 67, 73 needed).
- `env_cpu/bin/python checks/stage_5.py --location hudson-park-run --run run_006 --variant long`

```
[PASS] G5.3: 150 ft: 478 frames with ground over the whole centre block, centre depth 64.66 m (expected 64.7), worst altitude error 0.000%; 500 ft: 472 frames ..., worst altitude error 0.000%
[PASS] G5.9: 150 ft: ... (range 55); the scene's metric depth gives 55 on the 5-500 m scale, need >= 90% of it; 500 ft: ... (range 59); ... gives 59 ...
stage 5: PASSED -> reports/stage_5_long.json
```

- `env_cpu/bin/python checks/stage_5.py --run run_008 --variant street`

```
[PASS] G5.3: 150 ft: 623 frames with ground over the whole centre block, centre depth 64.66 m (expected 64.7), worst altitude error 0.044%; 500 ft: 306 frames ..., worst altitude error 0.013%
[PASS] G5.9: 150 ft: ... (range 68); the scene's metric depth gives 68 ...; 500 ft: ... (range 102); ... gives 102 ...
stage 5: PASSED -> reports/stage_5_street.json
```

- Two additions asked for by the owner before committing. G5.3 now needs at least `render.centre_depth_min_frames` (10) counted frames per clip and otherwise reports "not evaluated" instead of passing. New check G5.10 recovers the camera height from every ground pixel (metric depth times the ray's downward slope, every 8th frame) and requires the per-frame median within 1% of the altitude.

```
hudson-park-run, run_006, long
[PASS] G5.3: 150 ft: 478 frames ..., worst altitude error 0.000%; 500 ft: 472 frames ..., worst altitude error 0.000%
[PASS] G5.10: 150 ft: camera height from 49.5 M ground pixels in 60 frames: per-frame median 45.720 to 45.720 m (altitude 45.72 m, worst error 0.000%, limit 1%); 100.00% of the pixels are within 1%; 500 ft: 34.6 M ground pixels in 60 frames: 152.400 to 152.400 m
stage 5: PASSED -> reports/stage_5_long.json
120 Broadway, run_008, street
[PASS] G5.3: 150 ft: 623 frames ..., worst altitude error 0.044%; 500 ft: 306 frames ..., worst altitude error 0.013%
[PASS] G5.10: 150 ft: camera height from 36.9 M ground pixels in 80 frames: 45.720 to 45.720 m; 500 ft: 12.3 M ground pixels in 80 frames: 152.400 to 152.400 m
stage 5: PASSED -> reports/stage_5_street.json
```
