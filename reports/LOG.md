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
