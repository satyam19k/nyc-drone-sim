# 04 Long clips

**Question.** The model makes 93 frames (5.8 s at 16 fps) at a time. How do we make a 30 to 48
second clip that keeps its look from start to end, with no visible joins?

## Settings

Same model, seed (2026), steps (35) and resolution as in 01. Test clip: Canal Street 150 ft,
768 frames (48 s): 20 s along the shoreline, a 125° turn, then up Canal Street. Folders are under
`runs/run_009/canal-shore-street/150/gen/cosmos-transfer2.5-2b/`.

| Method | Folder | How it works |
| --- | --- | --- |
| Chained, K1 | `clear_K1_full/` | One run of the model over the whole clip. Each 93-frame chunk starts from the last frame of the previous one. One prompt. Full weights |
| Chained, K2 | `clear_K2_full/` | The same with K2 weights (0.5 / 0.3 / 0.5) |
| Chained, 2 conditioning frames | `clear_K2_chain_cond2/` | K2, each chunk starts from the last two frames. The model allows only 0, 1 or 2 |
| Independent windows | `clear_K2_windows/` | Ten 93-frame windows, each generated alone with its own prompt, 16 frames overlap, joined afterwards by cross-fade or hard cut |
| Independent two-chunk pieces | `clear_K2_pieces185/` | Five 185-frame pieces, chained inside a piece, independent between pieces, 16 frames overlap |

## What we saw

Brightness of each 93-frame stretch (0 to 255):

| Version | Brightness per stretch |
| --- | --- |
| Simulator | 176, 174, 177, 180, 163, 116, 103, 103, 112 |
| Chained, K1 | 82, 89, 102, 108, 144, 144, 142, 137, 138 |
| Chained, K2 | 91, 86, 87, 92, 117, 98, 91, 84, 84 |
| Chained, 2 conditioning frames | 140, 140, 138, 134, 128, 127, 131, 132, 132 |
| Independent windows | 106, 106, 108, 97, 104, 95, 91, 89, 80 |
| Two-chunk pieces | 106, 101, 106, 102, 93, 93, 84, 91, 99 |

The simulator itself darkens by about 40% along this flight, as the camera leaves the bright
esplanade for the street.

Structure score over eight frames of each clip (threshold 1.5) and the joins:

| Version | Structure (lowest frame) | Joins |
| --- | --- | --- |
| Chained, K1 | 5.03 (2.66) | 8 chunk joins, none flagged |
| Chained, K2 | 2.63 (2.17) | 8 chunk joins, none flagged (change at a join 5.1 to 11.6, limit 16.0) |
| Chained, 2 frames | infinite (3.33) | 6 of 8 joins flagged |
| Independent windows | 2.87 (1.89) | 9 joins; colour difference between neighbours 5.8, 6.9, 9.7, 13.5, **47.6**, **28.5**, 11.7, 20.9, 1.9 |
| Two-chunk pieces | 2.83 (2.28) | 4 joins; colour difference 8.9, 17.4, 9.8, 0.0 |

By eye:

- **Chained, K1:** washes out steadily to pink and beige by the street leg. The same happens at
  500 ft (brightness 66 rising to 120).
- **Chained, K2:** no washout and no visible joins; looks like real footage throughout. But it
  invents a lawn on the paved esplanade, a car park at 7 s and turquoise water over a building
  at the end, because one prompt covers the whole flight (see 02).
- **Chained, 2 conditioning frames:** collapses into a featureless blue-grey blur from the first
  chunk. Its structure score is infinite because the blank frames have no edges at all (see 06).
- **Independent windows:** the right layout everywhere, but the look changes at every join. The
  same paving is brown brick in one window and blue-grey slabs in the next; a building changes
  its facade. The cross-fade turns this into a one-second dissolve about every 4.8 s. The joins
  across the turn (47.6 and 28.5) are the worst.
- **Two-chunk pieces:** like the windows, with four joins in place of nine.

Other window clips, colour difference between neighbours: Hudson 150 ft 3.6 to 13.1; Canal
Street 500 ft 2.7 to 16.9; 120 Broadway 150 ft 7.2 to 32.9.

Why windows disagree: nothing ties one window's appearance to the next. Three seeds of the same
window and setting differ by 27 to 29 grey levels (03), so two windows generated apart differ by
about that much in their overlap.

Cost on one H200: a chained 768-frame clip took about 59 minutes and peaked at 108 GB of GPU
memory, so it does not fit an 80 GB card. Ten independent windows took 65 minutes at about
66 GB, which does fit.

![The four versions against the simulator](img/04_canal_four_versions.jpg)

![Chained K1 washing out](img/04_chained_K1_drift.jpg)

## What failed along the way

- **The first window joiner** colour-matched each window to the one before it. The corrections
  added up along the clip and the end came out far too dark (−76% first to last). It was
  replaced by a plain cross-fade plus a version matched only within each overlap, and the clips
  were joined again.
- **The first two-chunk run** stopped with an error after four minutes and was run again.
- **The drift rule** (more than 10% brightness change, first chunk to last, means "use windows")
  passed the chained K2 clip at −8.4%, so the next clips were chained, although that clip invents
  content. Brightness was the wrong measure.
- **I called the window clip the best result** after looking at eight still frames, before
  watching it. In motion the joins are its main defect.

## Conclusion

- **Chaining is the only method that is continuous**, because each chunk starts from the last
  frame of the previous one. With K1 it drifts; with K2 it holds its brightness but follows a
  single prompt and invents content.
- **Independent windows get the content right but cannot be joined invisibly.** Cross-fading,
  hard cuts and colour matching all leave the change of look visible.
- **More conditioning frames is not available.** The model accepts 0, 1 or 2; 2 failed.
- **Next to try, untested:** chained K2 with one prompt per chunk, which keeps chaining's
  continuity and fixes its prompt problem; and windows that all share one style reference image,
  which may make them agree.

The findings above are kept as they were measured. What the final run showed:

## Result in the final run

With setting M2 (the enhanced render's colours as a blurred guide, weight 0.3) and a prompt built
from the classes in view of each chain, chaining is continuous: the colour guide holds the look,
and the class-based prompt stops the invented content. On one RTX PRO 6000 (96 GB):

- A 240- or 248-frame chain is three 93-frame chunks. Peak GPU memory was 70.6 to 71.2 GB for
  every three-chunk chain (two-chunk, 185 frames: 71.6 GB). Each took 41.1 to 43.8 min.
- The 30 s clips are two 248-frame chains (route frames 288 to 535 and 520 to 767) cross-faded by
  `nyc-drone join` over their 16 shared frames: 248 + 248 - 16 = 480 frames.
- `seams.json` checks each join between chunks against four standard deviations above the clip's
  mean frame-to-frame change:

| Clip | Mean change | Limit | Join at 93 | Join at 185 | Flagged |
| --- | --- | --- | --- | --- | --- |
| Canal Street 150 ft, frames 288-535 | 6.34 | 13.29 | 6.01 | 9.19 | none |
| Canal Street 150 ft, frames 520-767 | 9.33 | 13.36 | 10.13 | 7.28 | none |
| Canal Street 500 ft, frames 288-535 | 6.25 | 13.07 | 6.74 | 8.30 | none |
| Canal Street 500 ft, frames 520-767 | 4.65 | 5.56 | 4.73 | 4.97 | none |
| Bronx 150 ft | 11.21 | 14.18 | 10.87 | 11.09 | none |
| Grand Army Plaza 500 ft | 6.09 | 8.60 | 6.25 | 5.36 | none |
| Canal Street 150 ft, fog | 1.15 | 1.93 | **2.18** | 1.04 | **93** |

Only the fog clip has a flagged join: its second chunk starts about 4.5% brighter. Fog makes the
change between frames small and even, so its limit is tight. See 08 for the clips themselves.
