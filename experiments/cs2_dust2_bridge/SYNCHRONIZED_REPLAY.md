# First synchronized Dust II perception replay

## Scope

This milestone combines previously recorded screen frames, fixed-radar pose,
and sanitized GSI state by monotonic timestamp. It is read-only: the output
contains no proposed or executed keyboard or mouse input. Raw detector boxes
remain screen-space evidence; the primary enemy target is derived only through
the separately validated visible-target calibration.

The current replay uses the 20 independently labeled validation frames, the team-aware
SSDLite320 epoch-five checkpoint, the real Dust II radar calibration, and the
continuously recorded GSI stream. GSI matching is causal: each image receives
the latest state at or before its capture time. A 15-second maximum age covers
the event-driven GSI heartbeat without using future state.

Version four also applies the screen-only visible-target calibration and the
NAV-derived local-clearance calibration. This produces the target and navigation
fields needed by the policy without reading opponent state from GSI.

## Result

- 20 selected frames and 20 emitted rows;
- all 20 rows identified as active `de_dust2` play;
- zero stale-GSI, radar-detection, or out-of-calibration drops;
- strictly increasing screen timestamps and nonpositive GSI time deltas;
- maximum GSI age 10.211 seconds;
- radar confidence from 0.85 to 1.00;
- 11 visible-player detections across the replay;
- six frames with at least one enemy candidate;
- six frames with a primary screen-visible target;
- four finite local-clearance values on every emitted frame;
- about 31 ms detector inference per frame on CPU in the version-four run;
- no authentication data retained;
- every row explicitly marked `read_only: true`.

The generated JSONL and checkpoint remain local under `artifacts/cs2_bridge/`.
Reproduce the artifact with:

```powershell
.\.venv\Scripts\python.exe tools/build_cs2_perception_replay.py `
  --capture artifacts/cs2_bridge/dust2_visible_players_validation_01 `
  --labels artifacts/cs2_bridge/dust2_visible_players_validation_01_labels.json `
  --checkpoint artifacts/cs2_bridge/dust2_team_detector_pilot_v3_threshold_015/best.pt `
  --gsi artifacts/cs2_bridge/dust2_gsi_walk_01.jsonl `
  --calibration experiments/cs2_dust2_bridge/dust2_calibration_v1.json `
  --target-calibration experiments/cs2_dust2_bridge/dust2_visible_target_calibration_v1.json `
  --clearance-calibration artifacts/cs2_bridge/dust2_clearance_calibration_v1.json `
  --output artifacts/cs2_bridge/dust2_validation_perception_replay_v4.jsonl `
  --score-threshold 0.15 --nms-threshold 0.30 `
  --radar-search-bounds 180,100,500,400 --max-gsi-delta-ms 15000
```

## Remaining integration

The perception record now contains own pose, four clearances, screen-derived
target state when visible, and causal round/player state. The next stage is a
deterministic adapter into `BridgeFrame`, including fire cooldown and action-mask
rules, followed by an offline policy replay. It must remain read-only until that
combined replay has been audited.
