# Results, v1.0

Five videos and a snow window made with commit `674cc7e`, all 1280 x 720 at 16 fps, restyled by
Cosmos-Transfer2.5-2B from Isaac Sim renders. Every file below is an asset of the
[v1.0 release](https://github.com/satyam19k/nyc-drone-sim/releases/tag/v1.0). The commands that
made them are in [reproduce_results.md](reproduce_results.md).

## The five videos

**[canal-shore-street_150ft_clear_M2_30s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_clear_M2_30s.mp4)**:
Hudson River Park, the turn, then Canal Street at 150 ft, 30 s. Water, paving, trees and traffic
look real and stay in place through the turn; a tall white facade softens in the last seconds.

**[canal-shore-street_500ft_clear_M2_30s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_500ft_clear_M2_30s.mp4)**:
the same flight at 500 ft. The street plan, the river and the parks read well; large buildings
lose their window grid and become plain blocks.

**[canal-shore-street_150ft_fog_fogguide_15s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_fog_fogguide_15s.mp4)**:
the shoreline at 150 ft in fog, 15 s, with the renderer's fog render as the colour guide. Pale,
low-contrast haze with the shoreline and crossings readable; the fog thickens in one step at 5.8 s.

**[bronx-1-east-161st-street_150ft_clear_M2_15s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/bronx-1-east-161st-street_150ft_clear_M2_15s.mp4)**:
1 East 161st Street, Bronx, at 150 ft, 15 s, from the park over the crossings to a brick block.
Foliage and road surfaces look natural; cars, crossings and the facade follow the render.

**[grand-army-plaza-brooklyn_500ft_clear_M2_15s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/grand-army-plaza-brooklyn_500ft_clear_M2_15s.mp4)**:
Grand Army Plaza, Brooklyn, at 500 ft, 15 s. The plaza's oval, the roads and the park canopy match
the render; the tall white tower and the curved glass building lose their windows.

## The snow window

**[canal-shore-street_150ft_snow_S1_5.8s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_snow_S1_5.8s.mp4)**:
the shoreline at 150 ft in snow, one 93-frame window (5.8 s, route frames 114 to 206), setting t4
with the fixed snow prompt ([snow_prompt.txt](snow_prompt.txt)) at guidance 3. Snow covers the
paving, roofs and trees, footprint trails cross the plaza and the river is dark; the roads come
out white rather than wet and dark. This is a verified prompt fix, not yet the repository's
built-in snow prompt.

**[canal-shore-street_150ft_snow_S2_5.8s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_snow_S2_5.8s.mp4)**:
the same window at guidance 6: the same scene with more contrast, deeper footprints and a
darker river.

## Simulator clips and side-by-sides

For each video the release has the simulator render of exactly the same frames (`<name>_sim.mp4`:
the enhanced render, or for the fog clip the renderer's fog render) and a side-by-side
(`<name>_sim_vs_gen.mp4`, simulator left, generated right):

| Video | Simulator | Side-by-side |
| --- | --- | --- |
| Canal Street 150 ft | [sim](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_clear_M2_30s_sim.mp4) | [side-by-side](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_clear_M2_30s_sim_vs_gen.mp4) |
| Snow window (S1) | [sim](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_snow_5.8s_sim.mp4) | [side-by-side](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_snow_S1_5.8s_sim_vs_gen.mp4) |
| Canal Street 500 ft | [sim](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_500ft_clear_M2_30s_sim.mp4) | [side-by-side](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_500ft_clear_M2_30s_sim_vs_gen.mp4) |
| Canal Street fog | [sim](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_fog_fogguide_15s_sim.mp4) | [side-by-side](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150ft_fog_fogguide_15s_sim_vs_gen.mp4) |
| Bronx 150 ft | [sim](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/bronx-1-east-161st-street_150ft_clear_M2_15s_sim.mp4) | [side-by-side](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/bronx-1-east-161st-street_150ft_clear_M2_15s_sim_vs_gen.mp4) |
| Grand Army Plaza 500 ft | [sim](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/grand-army-plaza-brooklyn_500ft_clear_M2_15s_sim.mp4) | [side-by-side](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/grand-army-plaza-brooklyn_500ft_clear_M2_15s_sim_vs_gen.mp4) |

[canal-shore-street_150_vs_500ft_clear_M2_30s.mp4](https://github.com/satyam19k/nyc-drone-sim/releases/download/v1.0/canal-shore-street_150_vs_500ft_clear_M2_30s.mp4)
puts the two generated Canal Street clips side by side: 150 ft left, 500 ft right.

`specs.zip` holds, for every model run, its `spec.json`, `prompt.txt`, `clip.json`, `run.log`,
`timing.json`, `seams.json` and `sheet.png`, and the exact command. `MANIFEST.md` lists each video
with its frames, setting, source render, wall time, peak GPU memory and SHA-256; `SHA256SUMS`
covers the videos, `MANIFEST.md` and every file inside `specs.zip`.

## Earlier experiments

Before the final run, the settings were chosen on 93-frame windows of the Canal Street route:
control weights ([01](ablations/01_control_weights.md)), prompts ([02](ablations/02_prompts.md)),
weather in the model ([03](ablations/03_weather_model.md)), long clips
([04](ablations/04_long_clips.md)), the simulator's own look and weather
([05](ablations/05_look_and_renderer_weather.md)), validation ([06](ablations/06_validation.md)),
new addresses ([07](ablations/07_new_addresses.md)), and the M0, M1, M2 comparison that chose the
final setting ([08](ablations/08_final_run.md)). The [index](ablations/index.md) has one line per
experiment.
