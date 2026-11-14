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
