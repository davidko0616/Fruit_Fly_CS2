# Dust II local-clearance capture 01

The first local-clearance route contains 900 lossless 2560 x 1440 frames at a
nominal 0.2-second interval. It covers open ground, walls, boxes, doors,
corridors, tunnels, both bombsites, and multiple player headings. The raw
capture remains immutable under
`artifacts/cs2_bridge/dust2_local_clearance_calibration_01`.

Frames 707 through 724 contain a Codex desktop overlay caused by a temporary
focus switch. Those 18 frames are excluded rather than deleted. Frames 706 and
725 were visually checked and contain clean gameplay, leaving 882 usable
frames.

The fixed-radar extractor recovered a valid own-player pose from all 882 usable
frames:

- raw pose recovery: 882/882 (100%);
- temporally supported poses: 882/882 (100%);
- detection failures: 0;
- temporal rejections: 0.

The audit command was:

```powershell
.\.venv\Scripts\python.exe -u tools/extract_dust2_radar.py `
  --capture artifacts/cs2_bridge/dust2_local_clearance_calibration_01 `
  --output artifacts/cs2_bridge/dust2_local_clearance_calibration_01_radar_poses.jsonl `
  --search-bounds 180,100,500,400 --exclude-range 707-724
```

Aggregating the translucent in-game radar preserves too much changing world
background to serve as a trustworthy collision mask. The capture is therefore
accepted for pose coverage and visual clearance validation, while the static
walkability mask must come from the installed Dust II overview asset or an
equivalent clean, version-matched source. The next implementation step is to
align that mask to the calibrated radar coordinates, ray-cast forward,
backward, left, and right clearances, and compare the results with manually
checked frames from this route.
