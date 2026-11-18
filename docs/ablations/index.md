# Video-model ablations

Cosmos-Transfer2.5-2B at commit `2ff49d0717af`; seed 2026, 35 steps, 1280 x 720 at 16 fps,
93-frame chunks. The grids and long-clip tests (pages 01 to 04) ran on one H200; the M0, M1 and
M2 comparison and the final run (page 08) on one RTX PRO 6000 (96 GB). The numbers on these pages were measured from the generated clips and
are kept in `measurements.json`; the JPEG sheets in `img/` are downscaled from the full-size sheets.

| Experiment | Question | Finding | Page |
| --- | --- | --- | --- |
| Clear grid K1 to K11 | Which control weights look real and keep the geometry? | Depth 0.5 / edge 0.3 / segmentation 0.5 (K2) looks real and passes; full weights are dull; depth alone breaks the layout | [01](01_control_weights.md) |
| Snow grid t1 to t7 | Which settings let snow appear? | Only without segmentation (t4: 53% white ground against 14% with it) | [01](01_control_weights.md) |
| Seeds | How much does the seed change a window? | 27 to 29 grey levels between seeds, as much as a change of weather | [01](01_control_weights.md) |
| Short and long prompts | Does naming each surface help? | Yes for clear weather; length alone does not bring weather | [02](02_prompts.md) |
| River sentence | What happens when the prompt names something not in frame? | The model paints it: a turquoise pool on a lawn with 0 water pixels | [02](02_prompts.md) |
| One prompt against a prompt per window | Does a single prompt for a 48 s flight work? | No: it invents a lawn, a car park and water. Per-window prompts fix it | [02](02_prompts.md) |
| First four weathers | Does the weather prompt change the weather? | Barely: geometry kept, fog, rain and snow hardly visible | [03](03_weather_model.md) |
| Weather windows and F01 to F06 | Snow, fog and rain on Canal Street windows | All one grey overcast look; no snow cover, haze or rain | [03](03_weather_model.md) |
| Full snow clips | Snow over a full clip | Grey and flat, no snow cover | [03](03_weather_model.md) |
| Chained K1 | One model run over 48 s, full weights | Washes out: brightness 82 rising to 138 | [04](04_long_clips.md) |
| Chained K2 | The same with K2 | No washout, no joins, but invents content from the single prompt | [04](04_long_clips.md) |
| Chained, 2 conditioning frames | Does more carry-over help? | Collapses into a blur | [04](04_long_clips.md) |
| Independent windows | Ten windows with their own prompts, joined | Right layout; the look changes at every join (up to 47.6 grey levels) | [04](04_long_clips.md) |
| Two-chunk pieces | Fewer, longer independent pieces | Four joins in place of nine; still visible | [04](04_long_clips.md) |
| Simulator checks | Are geometry and altitude right? | Yes, to 0.044% and 0.0000 m | [06](06_validation.md) |
| Scores on generated video | Can a number choose the setting? | No: structure gives a blank clip infinity; ripple, white ground and brightness drift each reward a failure | [06](06_validation.md) |
| Fog with a fog guide | Can the model make fog? | Yes, with the renderer's fog render as the colour guide (0.3) and the fog prompt | [03](03_weather_model.md#since-then-the-final-run) |
| Snow, cause found | Why does snow not appear? | The scene sentences of the prompt (scene text for a window, class sentences for a chain) describe green trees and dry paving, and override the snow paragraph | [03](03_weather_model.md#since-then-the-final-run) |
| Snow, prompt fix S1 and S2 | Does a snow version of every scene sentence give snow? | Yes, on one 93-frame window with t4: snow on paving, roofs and trees, footprints, a dark river; guidance 6 adds contrast. Not yet in the code | [03](03_weather_model.md#since-then-the-final-run) |
| Chained M2 | Are M2 chains continuous, and do they fit in memory? | Yes: no flagged join in six clear chains; three-chunk chains peak at 70.6 to 71.2 GB | [04](04_long_clips.md#result-in-the-final-run) |
| M0, M1, M2 | Which input and colour control for the final clips? | M2 (enhanced render, colour control 0.3): keeps the render's lawns and colours; M0 and M1 bleach the park and repaint cars | [08](08_final_run.md) |
| Final run | Five clips at three places, two altitudes, clear and fog | All made; facades lose detail at 500 ft; the fog clip thickens in one step at 5.8 s | [08](08_final_run.md) |
