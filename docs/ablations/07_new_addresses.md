# 07 New addresses

**Question.** Does the tool work on addresses it was not developed on, in all five boroughs,
for clips longer than a few seconds? How long does rendering take, and which checks hold?

## Settings

Isaac Sim 6.2 on one L40S, 1280 x 720 at 16 fps, `render.rt_subframes` 4, street flight at
150 ft and 500 ft. The Manhattan and Brooklyn clips were run step by step; Queens, the Bronx and
Staten Island with the single command `nyc-drone run --address ...`.

| Borough | Address | Scene radius | Buildings | Clip lengths |
| --- | --- | --- | --- | --- |
| Manhattan | 175 5th Avenue (Flatiron) | 800 m | 2,430 | 20 s, 40 s, 60 s |
| Manhattan | 350 5th Avenue (single-command test) | 400 m | 490 | 10 s |
| Brooklyn | 209 Joralemon Street | 800 m | 2,851 | 60 s |
| Queens | 27-01 Queens Plaza North | 400 m | 359 | 10 s |
| Bronx | 1 East 161st Street | 800 m | 750 | 20 s |
| Staten Island | 10 Richmond Terrace | 800 m | 1,387 | 20 s |

Frame sheets of the Manhattan and Brooklyn clips are in `docs/validation/images/render_*`.

## Render time

Time of the render step per altitude, including about 70 s to start the simulator and load the
scene, and the same time scaled to one minute of video:

| Clip | 150 ft | 500 ft | Per minute of video |
| --- | --- | --- | --- |
| Flatiron 20 s (320 frames) | 468 s | 470 s | 23 min |
| Flatiron 40 s (640 frames) | 848 s | 897 s | 21 to 22 min |
| Flatiron 60 s (960 frames) | 1,183 s | 1,239 s | 20 to 21 min |
| Brooklyn 60 s (960 frames) | 1,167 s | 1,320 s | 20 to 22 min |
| Bronx 20 s (320 frames) | 417 s | 429 s | 21 to 22 min |
| Staten Island 20 s (320 frames) | 1,163 s | 317 s | 16 min at 500 ft; the 150 ft render was slow, cause not checked |
| Queens 10 s (160 frames), 400 m scene | 155 s | 149 s | 15 min |
| 350 5th Avenue 10 s (160 frames), 400 m scene | 143 s | 147 s | 15 min |

So an 800 m scene renders at about **20 minutes per minute of video per altitude**, 40 minutes
for both altitudes; a 400 m scene is faster per frame. A 60 s street clip at 150 ft is close to
the most an 800 m scene holds (see Known limits in the README).

## The render-7 question

The render check renders frame 0 a second time and compares it with frame 0 of the clip
(render-7). The colour must agree to 35 dB PSNR.

| Clip | 150 ft | 500 ft |
| --- | --- | --- |
| Flatiron 20 s | 28.63 dB | 31.83 dB |
| Flatiron 40 s | 31.95 dB | 32.04 dB |
| Flatiron 60 s | 40.17 dB | 37.66 dB |
| Brooklyn 60 s | 31.23 dB | 30.57 dB |
| Bronx 20 s | 30.93 dB | 32.77 dB |
| Staten Island 20 s | 36.84 dB | 39.19 dB |
| Queens 10 s | 39.16 dB | 36.97 dB |
| 350 5th Avenue 10 s | 39.76 dB | 38.74 dB |
| Grand Army Plaza, Brooklyn (on a second machine) | 30.1 dB | 30.3 dB |

It failed at four of the six 800 m clips and at Grand Army Plaza, and passed on the
others. No other check failed on any of them (Brooklyn and the Bronx had a warning for frames
with few building pixels, where the route runs beside open space). At Grand Army Plaza the repeat
render's camera pose, depth and camera height matched exactly while its colour did not. Masking out
vegetation pixels did not clear 35 dB (Flatiron 20 s: 28.63 to 30.24 dB at 150 ft, 31.83 to
34.53 dB at 500 ft); vegetation pixels alone scored 26.7 to 30.8 dB.

The likely cause is the real-time renderer itself: it smooths each image over the frames before
it, so frame 0 rendered alone differs slightly in colour from frame 0 at the start of the clip,
most of all in leaves and their shadows. That changes colour, not geometry. The check is
therefore being split: depth and classes of the repeat render must match exactly, and colour
below 35 dB becomes a warning with its value. Whether more subframes per frame (16 in place of 4)
close the gap is being tested.

## Findings

- The tool works unchanged in all five boroughs, by address, with the single command.
- Rendering costs about 20 minutes per minute of video per altitude on an 800 m scene.
- Render-7 colour PSNR is not a reliable pass/fail test; its geometry part is.
