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
