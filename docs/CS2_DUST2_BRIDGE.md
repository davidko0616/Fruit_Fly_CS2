# Dust II observation bridge

The first Counter-Strike stage was deliberately read-only. It records player and
map status from Valve Game State Integration (GSI), combines it with visible-radar
localization and later screen perception, converts synchronized frames to the
existing 14-value controller input, and replays them through a fixed policy without
emitting keyboard or mouse input. A separate guarded executor now exists for the
next local-practice validation stage; the original shadow command remains
read-only.

This stage uses the internal map name `de_dust2`. It is designed for Practice
with Bots or another controlled local session. The toy-combat weights are used
only to validate the data path; they are not assumed to control Dust II well
before map-specific data collection and training.

## Implemented boundary

- `cs2_bridge/gamestate_integration_flywire.cfg` requests provider, map, round,
  own-player identity/state/weapons, and own-player position. It does not request
  `allplayers`, grenades, or opponent positions.
- `tools/serve_cs2_gsi.py` listens only on loopback, validates an optional token,
  assigns monotonic sequence and receive times, flushes every record, and refuses
  to overwrite an existing capture. Authentication is never written. Unrequested
  top-level fields such as `allplayers` are discarded rather than retained.
- `Dust2ObservationEncoder` accepts only `de_dust2`, rejects stale or reordered
  frames, resets memory at round boundaries, and converts player pose, four local
  clearances, firing state, and a visible target into the 14-value policy input.
- A target detection is represented in the player's local forward/right axes and
  may enter the bridge only while visible in the captured frame. The encoder
  converts that observation to a world-space last-seen point. During occlusion it
  recomputes remembered relative geometry from the current player pose.
- `tools/run_cs2_bridge_replay.py` loads recorded bridge frames and the confirmed
  policy, applies action masks, and writes observations, hashes, logits,
  probabilities, complete connectome activity, memory status, and proposed
  actions. Every row is marked
  `offline_replay_only`; no executor is present in this stage.
- Planner-enabled policies require `--waypoint-calibration`. While a target is
  hidden, the bridge replaces its direct remembered vector with a collision-free
  waypoint up to six NAV cells ahead. Live visible-target vectors are unchanged.
  Target corrections to walkable space are recorded and limited by
  `--waypoint-target-max-snap-world` (90 by default); player pose retains the
  stricter limit stored in the clearance calibration. A remembered target that
  cannot be mapped within this bound is cleared, recorded as a planner
  rejection, and never supplied to the policy.
- `tools/calibrate_dust2_bridge.py` derives provisional bounds from a walking
  fixed-radar capture. Calibration is explicit and versionable rather than
  embedding guessed Dust II coordinates in code.
- `tools/capture_cs2_screen.py` records lossless timestamped full frames or radar
  crops and flushes the manifest after every frame. It can fall back to DXcam's
  Windows Desktop Duplication path for full-screen Direct3D capture.
  `tools/extract_dust2_radar.py`
  detects the compact yellow player marker and adjacent white or red heading
  marker without reading game memory. A visible radar-map search rectangle and
  temporal-support check reject scenery and menu lookalikes.

## Install the GSI configuration

Steam reports CS2 at `C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike
Global Offensive` on the current PC. Generate a local token and install without
overwriting an existing configuration:

```powershell
$token = [guid]::NewGuid().ToString('N')
.\.venv\Scripts\python.exe tools/install_cs2_gsi.py --cfg-directory "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg" --token $token
.\.venv\Scripts\python.exe tools/serve_cs2_gsi.py --output artifacts/cs2_bridge/dust2_gsi_walk_01.jsonl --token $token
```

Launch a controlled Dust II practice session after the receiver starts. Walk the
full intended training region only after installing the fixed radar config:

```powershell
.\.venv\Scripts\python.exe tools/install_cs2_radar.py --cfg-directory "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo\cfg"
```

In the CS2 developer console run `exec flywire_radar`. This disables radar
rotation and always-centered behavior, sets a stable scale, and does not expose
anything absent from the normal HUD. Capture the upper-left radar region while
walking both spawns, mid, both bombsites, and the connecting routes:

```powershell
.\.venv\Scripts\python.exe tools/capture_cs2_screen.py --output artifacts/cs2_bridge/dust2_radar_walk_01 --frames 3000 --interval 0.2 --region 20,10,720,520
.\.venv\Scripts\python.exe tools/extract_dust2_radar.py --capture artifacts/cs2_bridge/dust2_radar_walk_01 --output artifacts/cs2_bridge/dust2_radar_poses_01.jsonl --origin 20,10 --search-bounds 180,100,500,400
.\.venv\Scripts\python.exe tools/calibrate_dust2_bridge.py --input artifacts/cs2_bridge/dust2_radar_poses_01.jsonl --output artifacts/cs2_bridge/dust2_calibration_v1.json
```

The walking capture must cover both map axes. The calibration tool adds an
eight-pixel margin by default and refuses to overwrite an earlier version.
After capture, inspect its raw bounds and repeat with broader coverage if future
positions fall outside them.

The first real walk at 2560 x 1440 recorded 3,000 frames over 599.805 seconds
and recovered 2,987 valid poses (99.57%), with zero timestamp reversals and no
consecutive localization jump above 40 pixels. Its measured bounds and settings
are preserved in the [calibration report](../experiments/cs2_dust2_bridge/RADAR_CALIBRATION.md)
and [versioned calibration](../experiments/cs2_dust2_bridge/dust2_calibration_v1.json).
Regenerate it after changing resolution, HUD scale, crop, or fixed-radar settings.

## Bridge-frame contract

Each policy frame contains:

- increasing sequence and monotonic timestamp, round ID, optional source tick,
  and `de_dust2` map identity;
- own-player x/y position and yaw;
- forward, backward, left, and right clearance in `[0, 1]`;
- a visible target's local forward/right displacement and detector confidence,
  or null when no target is visible;
- normalized fire cooldown and an eight-action availability mask.

The [live GSI probe](../experiments/cs2_dust2_bridge/GSI_POSE_PROBE.md) found that
the current CS2 build omits own-player pose while actively playing. GSI therefore
supplies map, round, health, and weapon state; the visible fixed radar supplies
pose. The team-aware CPU visible-player pilot now supplies the first read-only
detector baseline; its validation result is documented in the
[detector report](../experiments/cs2_dust2_bridge/DETECTOR_PILOT.md). Visible-box
bearing/range and NAV-derived local clearance are now calibrated for the
read-only bridge. Opponent coordinates from observer feeds, server
plugins, demos, or `allplayers` may be retained separately as evaluation labels
only; they must never populate the policy observation.

The current [synchronized perception replay](../experiments/cs2_dust2_bridge/SYNCHRONIZED_REPLAY.md)
now combines causal GSI snapshots, same-frame radar pose, and team-aware visible
detections for all 20 selected validation frames without drops. All rows include
four local clearances, and visible enemy detections include calibrated local
target geometry. The subsequent
[offline policy replay](../experiments/cs2_dust2_bridge/OFFLINE_POLICY_REPLAY.md)
converts all 20 records to `BridgeFrame` and proposes masked FlyWire actions
without executing input.

## Verified synthetic bridge replay

The committed fixture represents a visible target, a subsequent occluded frame
after player movement, and a new round. Its checks confirm:

- live geometry is converted to the controller input;
- the remembered world point transforms correctly as the player moves;
- target geometry becomes zero after a round reset;
- reordered and out-of-calibration frames are rejected;
- masked actions cannot be selected;
- GSI secrets and privileged opponent fields are not retained;
- output files cannot silently replace earlier captures or decisions.

Run the fixture against the confirmed policy:

```powershell
.\.venv\Scripts\python.exe tools/run_cs2_bridge_replay.py --frames tests/fixtures/dust2_bridge_frames.jsonl --calibration tests/fixtures/dust2_calibration_test.json --policy-run artifacts/toy_combat/partial_observability_flywire_seed_77_confirmation_v1 --policy-version 100 --output artifacts/cs2_bridge/fixture_replay.jsonl --mode greedy
```

The fixture and calibration are synthetic interface tests, not Dust II training
evidence. The real 20-frame acceptance replay is documented separately above.

The frozen planner-enabled policy can be reproduced with:

```powershell
.\.venv\Scripts\python.exe tools/run_cs2_bridge_replay.py --frames artifacts/cs2_bridge/dust2_validation_bridge_frames_v1.jsonl --calibration experiments/cs2_dust2_bridge/dust2_calibration_v1.json --waypoint-calibration artifacts/cs2_bridge/dust2_clearance_calibration_v1.json --policy-run artifacts/toy_combat/dust2_nav_flywire_seed_84_exploratory_v9_waypoint_full_map --policy-version 80 --output artifacts/cs2_bridge/dust2_validation_flywire_waypoint_v80_stochastic.jsonl --mode stochastic --seed 9200000
```

## Continuous live shadow

`tools/run_cs2_shadow.py` runs the complete bridge continuously while a human
plays. It captures each screen frame in memory, matches only an earlier GSI row,
localizes the player on the fixed radar, removes the radar map's per-frame pan
from the orange A/B site anchors, estimates NAV clearance, detects visible
enemies, updates round-scoped target memory and the waypoint planner, and records
the frozen policy's proposed action. It contains no keyboard or mouse output and
marks every evidence row and the summary with `input_emitted: false`.

The default output is compact: `shadow.jsonl`, sanitized `gsi.jsonl`, and
`summary.json`. Full-resolution images are not retained. `--audit-every N` can
save a sparse JPEG sample for visual audit. Run a short preflight with:

```powershell
.\.venv\Scripts\python.exe -u tools\run_cs2_shadow.py `
  --output artifacts\cs2_bridge\dust2_live_shadow_grid_v2_preflight_20260929_01 `
  --checkpoint artifacts\cs2_bridge\dust2_team_detector_pilot_v3_threshold_015\best.pt `
  --calibration experiments\cs2_dust2_bridge\dust2_policy_grid_calibration_v2.json `
  --target-calibration experiments\cs2_dust2_bridge\dust2_visible_target_calibration_v1.json `
  --clearance-calibration experiments\cs2_dust2_bridge\dust2_clearance_grid_calibration_v2.json `
  --waypoint-calibration experiments\cs2_dust2_bridge\dust2_clearance_grid_calibration_v2.json `
  --radar-map-calibration experiments\cs2_dust2_bridge\dust2_radar_map_calibration_v2.json `
  --policy-run artifacts\toy_combat\dust2_nav_flywire_seed_84_exploratory_v9_waypoint_full_map `
  --policy-version 80 --mode greedy --duration-seconds 30 --hz 4 `
  --backend dxcam --region 0,0,2560,1440 `
  --radar-search-bounds 180,100,500,400 --delay 5 --sound-cues `
  --audit-every 20
```

Keep `exec flywire_radar` active, play on `de_dust2`, and keep friendly bots
removed so their radar markers cannot be mistaken for the player marker. The
summary reports accepted and dropped frames, causal GSI volume, action counts,
mask violations, target-memory and waypoint use, capture latency, processing
latency, and effective frame rate.

The first 15-second preflight passed the CPU integration and timing checks: it
accepted 57 of 60 frames, sustained 4 Hz, kept processing p95 at 113.8 ms, used
causal GSI, and produced no masked-action violations. A later five-minute run
exposed a coordinate error: fixed radar orientation does not stop the map from
panning, so raw player-marker screen coordinates are not global positions. Its
729 accepted decisions remain useful for timing and safety evidence, but their
action distribution is not valid behavioral evidence.

The v2 localizer tracks the A/B site labels to remove that pan and maps both the
player and visible targets into the policy's 128 by 128 grid. Reanalysis of the
independent 600-frame dense route accepted 588 frames (98.0%): two poses were
ambiguous, ten lacked a usable map anchor, and none failed NAV clearance. A
sparse independent audit of the live run placed all 20 usable frames within the
90-world-unit correction bound. See the
[preflight result](../experiments/cs2_dust2_bridge/LIVE_SHADOW_PREFLIGHT_RESULT.json)
and [five-minute diagnostic](../experiments/cs2_dust2_bridge/LIVE_SHADOW_5MIN_RESULT.json).
The September 30 v2 preflight passed: 113 of 120 frames were accepted (94.2%),
the loop sustained 4 Hz, processing p95 was 117.5 ms, and all decisions obeyed
their action masks. Four startup frames lacked an A/B anchor and three poses
exceeded the configured NAV correction bound. All six audit images showed active
Dust II gameplay, every GSI match was causal, and no input was emitted. See the
[v2 preflight result](../experiments/cs2_dust2_bridge/LIVE_SHADOW_GRID_V2_PREFLIGHT_RESULT.json).
The October 4 five-minute v2 run then covered both sites, both tunnels, mid,
Long, Short, Catwalk, and the spawn routes. It accepted 918 of 949 eligible
gameplay frames (96.7%); 99 death frames and 125 pose-less buy-menu frames were
excluded before policy execution. Processing p95 was 99.1 ms, effective rate was
3.83 Hz, and 247 accepted frames contained a live target. All 30 audit images
were reviewed: 24 showed active gameplay, five showed the buy menu, one showed a
death, and none showed a black screen, settings, console, desktop, or Codex.

The game crashed at the end of the run. The final capture stalled for 13.1
seconds and the run retained 1,173 rather than 1,200 frames, but its last frame
was still valid gameplay and the preceding 293 seconds remain usable. See the
[five-minute v2 result](../experiments/cs2_dust2_bridge/LIVE_SHADOW_GRID_V2_5MIN_RESULT.json).
## Guarded local-practice controller

`tools/run_cs2_controller.py` wraps the same perception and policy loop with a
disabled-by-default native input boundary. It requires both `--enable-input` and
the exact `--confirm-local-practice LOCAL_PRACTICE_ONLY` phrase. Before every
action it verifies F12 is not pressed, the frame and causal GSI row are fresh,
the player is alive and active on `de_dust2`, the proposed action is allowed by
its mask, and the foreground window title contains `Counter-Strike 2`.

Movement is one bounded W/A/S/D hold per accepted frame (60 ms by default), and
turns are bounded relative mouse movements (32 pixels by default). All movement
keys are released on exit. F12 latches an emergency stop, releases the movement
keys, and ends the loop. Fire is blocked unless the separate `--enable-fire`
flag is supplied; the first validation run therefore remains no-fire. Every row
records whether an input was emitted and why an action was executed or blocked,
and the summary records the safety configuration and execution counts.

The guarded executor first passed its fake-backend tests. Its separately
approved 10-second no-fire CS2 test then processed all 40 frames, emitted three bounded
left turns, and blocked 37 decisions on stale-state guards. It emitted no fire,
retained no screenshots, and had no action-mask violations. This passed the
native-input and fail-closed safety gate, but the installed 10-second GSI
heartbeat was too slow for the one-second execution freshness limit. The
versioned GSI config now uses a 0.25-second heartbeat; the installed config must
be updated before a separately approved repeat. See the
[preflight result](../experiments/cs2_dust2_bridge/GUARDED_CONTROLLER_NOFIRE_PREFLIGHT_RESULT.json).

The second approved 10-second run verified that correction: 64 GSI rows kept
all post-startup decisions within the one-second freshness limit. One right
strafe was emitted; 38 fire proposals were blocked and no shot was sent. Because
blocking fire only at the executor prevented the greedy policy from choosing
its next-best action, no-fire mode now also masks fire before policy selection.
The executor keeps its independent fire guard. See the
[heartbeat result](../experiments/cs2_dust2_bridge/GUARDED_CONTROLLER_NOFIRE_HEARTBEAT_RESULT.json).

The third approved 10-second run passed the complete no-fire movement gate. All
40 frames were accepted and all 40 decisions emitted bounded input: ten forward
steps, nine right strafes, and 21 left turns. GSI age stayed between 94 and 609
ms, frame age stayed between 85.5 and 234 ms, fire remained masked before policy
selection, and there were no safety blocks or mask violations. No screenshots
were retained, so this proves the guarded integration path rather than Dust II
gameplay quality. See the
[movement result](../experiments/cs2_dust2_bridge/GUARDED_CONTROLLER_NOFIRE_MOVEMENT_RESULT.json).

The player confirmed that the character moved and turned without behaving
wildly, but that four instantaneous turn steps per second looked unnatural. The
next actuator trial keeps approximately the same turn speed while using 8 Hz
policy updates and 16-pixel turn steps. A separate live F12 drill is not required
before that short trial; the stop path remains available, is covered by the fake
backend tests, and keys are released on normal exit or capture failure.

The approved 8 Hz actuator trial sustained all 80 requested capture intervals
and emitted 45 bounded actions with 16-pixel turn steps. It then rejected frames
45–79 because both site anchors had left the usable radar view and the
two-second recent-anchor window expired. No dropped frame emitted input. This is
a safety pass and localization-continuity failure; anchorless short-horizon map
pan tracking is required before a longer controller run. Player feedback is
still needed to judge whether the 8 Hz turns looked smoother. See the
[8 Hz result](../experiments/cs2_dust2_bridge/GUARDED_CONTROLLER_NOFIRE_8HZ_RESULT.json).

The player confirmed that the 8 Hz motion looked much smoother. The bridge now
tracks short radar-map translations from the visible map texture for at most six
seconds after the last direct site anchor, rejects shifts above eight pixels per
frame or weak correlations, and resets the horizon whenever an anchor returns.
On the 600-frame regression route this increased accepted localization from 588
to 598 frames, eliminated all ten radar-map failures, and used tracked pan on 54
frames. Tracked frames required at most 18.4 world units of NAV correction. The
two remaining drops were ambiguous player-marker frames. See the
[tracking reanalysis](../experiments/cs2_dust2_bridge/RADAR_MAP_TRACKING_REANALYSIS.json).

The first live tracking validation began at the anchorless position left by the
previous run. All 80 raw player-marker poses were valid, but the localizer had no
absolute site anchor from which to initialize texture tracking, so all frames
failed closed and no input was emitted. The current tracker bridges temporary
anchor loss; it does not yet provide global anchorless startup. Short trials must
therefore begin after a local round restart where a site anchor is visible. See
the [bootstrap result](../experiments/cs2_dust2_bridge/RADAR_TRACKING_LIVE_BOOTSTRAP_RESULT.json).

After a one-frame diagnostic confirmed both site labels and the player marker,
the final 8 Hz retry accepted and executed all 79 captured frames with zero
localization or safety failures. Direct anchors remained visible throughout, so
the texture tracker was not exercised live. With no visible or remembered
target, the frozen greedy policy chose 78 left turns and one forward step. The
next behavior gate is therefore map-specific no-target exploration rather than
a longer spinning run. See the
[continuity result](../experiments/cs2_dust2_bridge/GUARDED_CONTROLLER_NOFIRE_8HZ_CONTINUITY_RESULT.json).

Short no-fire trials should use an empty local Practice with Bots session and
`--audit-every 0` so they retain no screenshots. Use the same arguments as the
passing v2 preflight, replace `run_cs2_shadow.py` with
`run_cs2_controller.py`, choose a new output directory, and append:

```powershell
--enable-input --confirm-local-practice LOCAL_PRACTICE_ONLY
```

Visible-player captures and labels follow the separate
[perception protocol](../experiments/cs2_dust2_bridge/PERCEPTION_PROTOCOL.md).
The loopback labeler supports tight full/partial player boxes, team class,
verified negative frames, and a fixed split for the whole capture session.
Audited sessions can be exported with `tools/export_cs2_detector_dataset.py`;
the export preserves capture-level train/validation/test boundaries and retains
team and visibility metadata alongside YOLO labels.
