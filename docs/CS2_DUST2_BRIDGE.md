# Dust II observation bridge

The first Counter-Strike stage is deliberately read-only. It records player and
map status from Valve Game State Integration (GSI), combines it with visible-radar
localization and later screen perception, converts synchronized frames to the
existing 14-value controller input, and replays them through a fixed policy without
emitting keyboard or mouse input.

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
- `tools/calibrate_dust2_bridge.py` derives provisional bounds from a walking
  fixed-radar capture. Calibration is explicit and versionable rather than
  embedding guessed Dust II coordinates in code.
- `tools/capture_cs2_screen.py` records lossless timestamped full frames or radar
  crops and flushes the manifest after every frame. `tools/extract_dust2_radar.py`
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
pose. The next increment must supply visible-target detection, local-clearance
estimation, and timestamp synchronization. Opponent coordinates from observer feeds, server
plugins, demos, or `allplayers` may be retained separately as evaluation labels
only; they must never populate the policy observation.

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
evidence. The first real acceptance run begins after GSI and screen frames have
been captured from the same controlled session.

## Next acceptance run

Before enabling any action executor:

1. Add a visible-player detector and local-clearance estimator, with held-out
   labeled frames measuring detection precision, recall, range error, and false
   positives through walls.
2. Synchronize perception and GSI, then replay the combined frames through the
   policy. Require finite observations, strictly increasing timestamps, explicit
   round resets, and a report of dropped or late frames.
3. Collect map-specific demonstrations or controlled rewards and train a Dust II
   policy. Keep capture sessions separated between training, validation, and the
   final held-out route evaluation.

Only after this read-only run passes should proposed actions be mapped to local
practice controls behind an explicit enable flag and immediate stop control.
