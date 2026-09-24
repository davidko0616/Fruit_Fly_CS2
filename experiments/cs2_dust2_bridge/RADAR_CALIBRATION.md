# Dust II fixed-radar calibration v1

The first real Dust II localization walk was recorded on September 24, 2026 in
Practice with Bots. CS2 ran at 2560 x 1440 with `exec flywire_radar`; the capture
stored the screen rectangle `(20, 10, 720, 520)` at a requested five frames per
second while the player traversed both sites, spawns, mid, tunnels, and connecting
routes.

The local raw capture contains 3,000 lossless frames over 599.805 seconds. All
timestamps increase strictly. Mean frame interval was 200.002 ms (p99 218.5 ms,
maximum 219.0 ms), and mean screen-grab time was 34.826 ms.

Pose extraction was limited to the visible radar-map rectangle
`(180, 100, 500, 400)` in global screen coordinates. The detector recovered
2,987 poses (99.57%). Thirteen frames lacked a valid compact player marker;
every frame is accounted for. White-heading candidates take priority over the
red fallback so the gold site labels cannot replace the live player marker.
The temporal audit rejected no remaining pose, found no consecutive jump above
40 pixels, and measured a maximum consecutive jump of 29.009 pixels.

Measured route bounds before the fixed eight-pixel margin were:

| Axis | Minimum | Maximum |
| --- | ---: | ---: |
| Radar x | 232.4498 | 441.4901 |
| Radar y | 166.7740 | 343.5990 |

The versioned controller bounds are stored in
[`dust2_calibration_v1.json`](dust2_calibration_v1.json). They use visible-radar
pixels as map coordinates and a local-distance scale of 225.0403 pixels. Raw
screenshots and per-frame poses stay under ignored `artifacts/cs2_bridge/` because
they are large local evidence.

This calibration depends on the recorded resolution, HUD scale, crop origin,
and `flywire_radar.cfg`. Regenerate it if any of those settings change. This run
validates localization only; visible-target detection, clearance estimation, GSI
synchronization, offline policy replay, and map-specific training remain separate
acceptance stages.
