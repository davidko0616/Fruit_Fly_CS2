# Dust II visible-target calibration v1

This calibration converts a visible enemy bounding box at 2560 x 1440 into
local forward/right target coordinates without using opponent radar markers at
runtime. It was fitted from a controlled Dust II warmup capture containing one
stationary enemy and a moving player.

The visible radar supplied calibration truth only. A bright red diamond was
treated as a confirmed current enemy position. A red question mark was treated
as a last-known cue and excluded from the live-target fit. The extractor found
274 confirmed frames, 20 last-known frames, and 6 frames with no usable marker
across the complete 300-frame recording. Temporal support rejected two red
background shapes visible through the translucent radar.

Twenty evenly spaced frames were manually labeled. Nineteen contained one
confirmed radar diamond and formed calibration pairs; frame 0 contained a
question mark and was excluded. The fitted range model is:

```text
radar-relative range = 7779.45 / visible box height + 0.0731
```

Across the 19 calibration pairs it achieved 7.56 radar pixels RMSE, 6.50 pixels
MAE, 14.47 pixels maximum error, and R² 0.932. The observed calibration domain
is 71.3 to 762.0 pixels of box height and 13.0 to 111.1 radar pixels of range;
runtime estimates are clipped to those bounds.

Horizontal bearing uses the center of the visible box and a 53.13-degree
horizontal half-FOV. Against 15 radar pairs at least 25 pixels away, its direct
angular agreement was 3.01 degrees RMSE, 2.74 degrees MAE, and 4.89 degrees
maximum error. Very-close pairs were omitted from this check because overlapping
radar icons make their angular difference unstable.

The machine-readable artifact is
[`dust2_visible_target_calibration_v1.json`](dust2_visible_target_calibration_v1.json).
The offline replay accepts it through `--target-calibration` and attaches a
`visible_target` only to screen detections classified as enemies. Question-mark
coordinates are never supplied as a live target; the existing round-scoped
encoder memory handles loss of visual contact.
