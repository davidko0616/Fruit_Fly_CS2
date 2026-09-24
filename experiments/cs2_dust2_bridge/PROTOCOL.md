# Dust II read-only bridge protocol

This protocol fixes the first real-game integration boundary before collecting
CS2 data. The purpose is to validate observation timing and semantics on Dust II,
not to measure gameplay performance or claim that toy-combat weights transfer.

## Environment and data sources

- Map: `de_dust2`, Practice with Bots or another controlled local session.
- Valve GSI supplies map, round, and own-player state over
  loopback. The shipped configuration does not request `allplayers` or opponent
  positions. The recorder removes authentication and discards unrequested
  top-level fields.
- Timestamped screen capture will supply only information visible to the player.
  A fixed, non-rotating visible radar supplies own-player map position and facing.
  The target detector may emit a local forward/right displacement only when a
  player is visibly detected in that frame.
- Any privileged labels used to score perception remain outside the policy-frame
  file and cannot be passed to the observation encoder.

## Observation and memory

- Preserve the existing 14-value interface: normalized player x/y, facing vector,
  four target-geometry values, live-visibility flag, firing state, and four local
  clearances.
- Calibrate player-position bounds from a versioned fixed-radar walking capture.
  Reject frames outside those bounds rather than silently clipping them.
- Convert each visible local target estimate to a world-space last-seen point.
  Recompute its relative geometry from current player pose during later
  occlusion. Reset target memory on every round ID change.
- Reject non-finite values and non-increasing sequence or monotonic timestamps.

## Recording and validation

- Retain raw screen-frame identity, GSI receive time, source time/tick when
  available, synchronization skew, actual policy observation, memory/live flags,
  action mask, logits, probabilities, and proposed action.
- Refuse to overwrite capture, calibration, or decision files.
- The first real run is read-only. It must not send keyboard, mouse, console, or
  network control commands to CS2.
- Report GSI position availability, covered coordinate bounds, frame counts,
  join rate, dropped/late frames, timestamp reversals, observation validity,
  visible detections, remembered frames, and round resets.
- Pass requires no privileged opponent fields in policy records, no authentication
  secrets on disk, zero invalid observations, zero timestamp reversals, explicit
  accounting for every captured frame, and successful offline replay through the
  fixed policy.

After this protocol passes, freeze a separate detector/data-collection protocol
before map-specific training. Training and final gameplay evaluation must use
session-level splits so adjacent frames from one route cannot leak across train,
validation, and test sets.
