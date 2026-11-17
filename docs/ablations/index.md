# Video-model ablations

Cosmos-Transfer2.5-2B at commit `2ff49d0717af` on one H200; seed 2026, 35 steps, 1280 x 720 at
16 fps, 93-frame chunks. The numbers on these pages were measured from the generated clips and
are kept in `measurements.json`; the JPEG sheets in `img/` are downscaled from the full-size sheets.

| Experiment | Question | Finding | Page |
| --- | --- | --- | --- |
| Clear grid K1 to K11 | Which control weights look real and keep the geometry? | Depth 0.5 / edge 0.3 / segmentation 0.5 (K2) looks real and passes; full weights are dull; depth alone breaks the layout | [01](01_control_weights.md) |
| Snow grid t1 to t7 | Which settings let snow appear? | Only without segmentation (t4: 53% white ground against 14% with it) | [01](01_control_weights.md) |
| Seeds | How much does the seed change a window? | 27 to 29 grey levels between seeds, as much as a change of weather | [01](01_control_weights.md) |
| Short and long prompts | Does naming each surface help? | Yes for clear weather; length alone does not bring weather | [02](02_prompts.md) |
| River sentence | What happens when the prompt names something not in frame? | The model paints it: a turquoise pool on a lawn with 0 water pixels | [02](02_prompts.md) |
| One prompt against a prompt per window | Does a single prompt for a 48 s flight work? | No: it invents a lawn, a car park and water. Per-window prompts fix it | [02](02_prompts.md) |
| **P1 to P3 (RunPod)** | Does the pool on the street window come from the water words, or can stronger weights overrule the full-clip prompt? | **Gap: sheet and findings to come** | [02](02_prompts.md#p1-to-p3-the-turquoise-pool-tested-on-runpod) |
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
