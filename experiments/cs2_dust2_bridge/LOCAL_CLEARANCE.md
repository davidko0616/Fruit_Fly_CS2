# Dust II local-clearance estimator

The local-clearance stage replaces placeholder navigation values with four
normalized ray casts in forward, backward, left, and right order. It remains an
offline, read-only perception component and never emits game input.

The walkability source is the version-matched `de_dust2.nav` shipped with the
locally installed game. Source 2 Viewer 20.0 decoded that NAV file to glTF. The
builder rasterizes its 3,155 triangles using the official overview origin and
4.4-world-unit pixel scale, then registers the result to the fixed-radar player
track from `dust2_local_clearance_calibration_01`.

Registration uses alternating frames for fitting and holds the other frames out.
This checks the transform on observations that did not affect optimization while
retaining coverage across T Start, Long/A, mid, tunnels, and B. Small radar and
rasterization errors may place an icon just outside the binary mask; poses may be
snapped by at most 30 world units. Larger errors are rejected.

The fitted mask contains 396,104 walkable pixels. It placed 410 of 441 held-out
poses (93.0%) directly inside NAV space. All 441 held-out poses and all 882 clean
capture poses were accepted after bounded edge correction; the largest correction
was 19.9 world units and the 95th percentile was 4.3. A 16-location visual audit
covered open lanes, walls, boxes, corners, doors, stairs, tunnels, and both sites.

Rebuild the local artifacts with:

```powershell
python tools/build_dust2_clearance.py `
  --nav-glb artifacts/cs2_bridge/de_dust2_nav.glb `
  --poses artifacts/cs2_bridge/dust2_local_clearance_calibration_01_radar_poses.jsonl `
  --output-calibration artifacts/cs2_bridge/dust2_clearance_calibration_v1.json `
  --output-mask artifacts/cs2_bridge/dust2_walkability_mask_v1.png
```

The resulting calibration can be supplied to
`tools/build_cs2_perception_replay.py --clearance-calibration ...`. Each emitted
record then contains `local_clearances`. Values saturate at 600 world units. The
20-frame held-out perception replay emitted all 20 records with target and
clearance calibration enabled and no drops.

The mask is a two-dimensional union of navigation areas. Dust II has limited
vertical overlap, but a future floor-aware refinement should use the NAV area's
height when a reliable player-height signal becomes available.
