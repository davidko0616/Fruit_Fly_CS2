# Dust II bridge status

The read-only bridge foundation, first real fixed-radar calibration, and guarded
local-practice execution boundary are implemented. The bridge records
own-player/map GSI, encodes visibility-gated
Dust II frames with last-seen target memory, and replays them through the
confirmed 100-neuron FlyWire policy without game input.

The full 68-test suite passes (with one expected CUDA skip), and the committed
three-frame fixture completes an
end-to-end replay through policy version 100. This validates interfaces and
memory transformations only; it is not Dust II training or performance evidence.

The [real calibration walk](RADAR_CALIBRATION.md) recovered 2,987 of 3,000 poses
(99.57%) over ten minutes and produced the versioned
[`de_dust2` bounds](dust2_calibration_v1.json). The visible-player pilot now has
95 training and 20 validation frames. Its team-aware CPU checkpoint reaches
63.6% overall precision and recall, 71.4% enemy precision, and 83.3% enemy recall
at about 36 ms/frame, with no detections on 11 negative frames. It is sufficient
to test the read-only integration path but not to claim final perception
accuracy or enable firing. The
[visible-target calibration](VISIBLE_TARGET_CALIBRATION.md) now converts a
screen-visible enemy box to local bearing and range with 7.56 radar-pixel range
RMSE and 3.01-degree bearing RMSE on its controlled calibration pairs. Red
question-mark cues are excluded from live targets. The
[local-clearance estimator](LOCAL_CLEARANCE.md) now derives a static Dust II
walkability mask from the version-matched game NAV mesh and emits normalized
forward, backward, left, and right ray casts. The first
[local-clearance route](LOCAL_CLEARANCE_CAPTURE.md) supplies 882 clean validation
frames with 100% own-pose recovery after excluding one 18-frame desktop-overlay
interval. The
[live GSI probe](GSI_POSE_PROBE.md) showed
that active-player GSI omits pose, so localization uses the visible fixed radar.
The [first synchronized replay](SYNCHRONIZED_REPLAY.md) emitted all 20 selected
validation frames with causal GSI state, in-calibration radar poses, detector
outputs, visible-target estimates, local clearances, and no drops or input
execution. The [first real policy replay](OFFLINE_POLICY_REPLAY.md) then converted
all 20 records to `BridgeFrame`, applied wall and visible-fire masks plus a
five-second target-memory horizon, and produced 20 valid offline FlyWire
decisions. This verifies the integration path; it does not establish Dust II
policy performance.
The [map-specific training protocol](DUST2_TRAINING_PROTOCOL.md) now fixes
hash-disjoint training, validation, and held-out route sets on the NAV mask. A
privileged oracle solved 50 of 50 routes in every split, and a recorded FlyWire
CPU smoke update passed exact replay. The first exploratory Dust II run is
documented there. The local 8–24-cell curriculum reached 93.75% stochastic
validation hit rate at version 30. Its unrestricted full-map transfer failed to
improve the 37.5% transferred baseline, so that run was rejected and the held-out
route bucket remains unused by learned policies. A subsequent 24–48-cell
curriculum tied its 62.5% transferred stochastic validation hit rate but did not
improve it, so it was also rejected. A smaller 16–32-cell expansion succeeded:
version 20 improved stochastic validation hit and acquisition from 62.5% to
68.75% and was selected for the next curriculum stage. It retained 68.75% when
transferred to 24–48-cell validation routes, although further training did not
improve that rate. On unrestricted routes it fell to a 25.0% hit baseline, and
all trained checkpoints were equal or worse. The next stage therefore adds a
NAV waypoint planner rather than another curriculum-only PPO run. The planner
keeps the 14-value interface and exposes a collision-free local waypoint only
while the remembered target is hidden. A reference follower solved all 16 fixed
unrestricted validation routes through this interface. Planner-enabled FlyWire
training then improved stochastic unrestricted validation hit and acquisition
from 43.75% at version 0 to 87.5% at the selected version 80. The checkpoint is
frozen for one final 64-route held-out confirmation. On that confirmation, the
stochastic actor hit 51 of 64 routes (79.7%) and acquired line of sight on 56
(87.5%); the greedy actor hit 14 (21.9%), and the oracle solved all 64.
The frozen stochastic version 80 now also completes the real 20-frame read-only
bridge replay with the NAV planner: six frames use live target geometry, four
hidden frames use planned remembered-target waypoints, every proposed action is
allowed by its mask, and no game input is emitted. This establishes that the
trained hybrid controller crosses the real bridge interface; it does not yet
measure live Dust II control performance.
An independent 600-frame route was captured after the policy was frozen, but it
is invalid for acceptance. Frames 71–91 showed the scoreboard or settings, and
the settings change persisted by changing radar zoom from frame 92 onward.
Direct review also found single-frame localization switching among friendly bot
markers from the start.
Although 358 rows passed the old numeric bounds and completed offline inference,
their radar coordinates are not consistently in the frozen calibration. The run
is retained only as a diagnostic. Strict candidate reanalysis rejected all 579
retained images as ambiguous and accepted zero. No input executor may use its
results.
See the
[fixed read-only protocol](PROTOCOL.md) and the
[setup and data contract](../../docs/CS2_DUST2_BRIDGE.md). The capture backend,
label schema, session-level split, and held-out detector acceptance metrics are
fixed in the [visible-perception protocol](PERCEPTION_PROTOCOL.md).

The October 4 five-minute v2 shadow run passed 918 of 949 eligible gameplay
frames (96.7%) across the main Dust II routes, with zero action-mask or input
violations. A terminal CS2 crash caused one 13.1-second capture stall after the
preceding 293 seconds of valid evidence. The next-stage controller is now
implemented separately from the shadow runner. It requires explicit
local-practice enablement, keeps fire disabled by default, stops and releases
movement on F12, and rejects stale, inactive, masked, or unfocused actions. The
first separately approved 10-second no-fire trial passed all 40 perception
frames. Three fresh-state left-turn decisions emitted bounded mouse input, and
the
executor rejected the other 37 decisions because one frame and 36 causal GSI
rows were outside their execution-age limits. No fire or masked input was
emitted and no screenshots were retained. The guard therefore failed closed as
designed. The installed GSI config still used its 10-second read-only heartbeat,
so the versioned heartbeat is reduced to 0.25 seconds before repeating the short
trial; behavioral quality is not yet assessed. See
[`GUARDED_CONTROLLER_NOFIRE_PREFLIGHT_RESULT.json`](GUARDED_CONTROLLER_NOFIRE_PREFLIGHT_RESULT.json).

The approved repeat verified the 0.25-second heartbeat with 64 GSI rows and no
stale-GSI rejection. It emitted one bounded right strafe and safely blocked 38
fire proposals. The no-fire path now masks fire before policy selection so the
greedy policy selects its next-best movement or turn, while the executor retains
its independent fire block. No behavioral-quality claim is made from either
short run. See
[`GUARDED_CONTROLLER_NOFIRE_HEARTBEAT_RESULT.json`](GUARDED_CONTROLLER_NOFIRE_HEARTBEAT_RESULT.json).

The third approved 10-second trial passed after preselection fire masking. It
accepted all 40 frames and emitted all 40 bounded decisions: ten forward, nine
strafe-right, and 21 turn-left actions. Every frame and GSI row met the strict
freshness limits, with no firing, safety blocks, saved screenshots, or mask
violations. This validates real no-fire input integration; observed navigation
quality still needs player feedback. See
[`GUARDED_CONTROLLER_NOFIRE_MOVEMENT_RESULT.json`](GUARDED_CONTROLLER_NOFIRE_MOVEMENT_RESULT.json).

The player observed controlled movement and slight spinning, but the 4 Hz turn
steps did not look natural. The next short trial will use 8 Hz updates and
16-pixel turn steps, which preserves approximately the same turn speed while
doubling its temporal resolution. A dedicated F12 drill is skipped; F12 remains
available and the executor releases movement keys on exit or capture failure.

The approved 8 Hz tuning run sustained 8 Hz and emitted 45 safe actions, but it
accepted only the first 45 of 80 frames. Both A/B site anchors then left the
usable radar view; after the two-second recent-anchor window expired, the bridge
rejected all remaining frames and emitted no further input. The next engineering
gate is anchorless short-horizon radar-map pan tracking. Player feedback remains
the evidence for whether the 16-pixel 8 Hz turn steps looked smoother. See
[`GUARDED_CONTROLLER_NOFIRE_8HZ_RESULT.json`](GUARDED_CONTROLLER_NOFIRE_8HZ_RESULT.json).

The player confirmed that 8 Hz looked much smoother. Short-horizon phase
correlation now tracks radar-map pan from map texture when site labels disappear,
with strict correlation, shift, and six-second age limits. It raised regression
route acceptance from 588/600 to 598/600, eliminated all radar-map failures, and
kept all 54 tracked frames within 18.4 world units of walkable NAV space. The two
remaining rejections were ambiguous player markers. See
[`RADAR_MAP_TRACKING_REANALYSIS.json`](RADAR_MAP_TRACKING_REANALYSIS.json).

The first live tracker validation started from the anchorless position left by
the preceding run. The player marker was valid on all 80 frames, but no site
anchor was available to initialize an absolute map origin. The bridge correctly
emitted no input. Until global reference matching is added, live trials must
start after a local round restart with a visible site anchor. See
[`RADAR_TRACKING_LIVE_BOOTSTRAP_RESULT.json`](RADAR_TRACKING_LIVE_BOOTSTRAP_RESULT.json).

A one-frame diagnostic then confirmed a valid anchored radar view. The final
8 Hz retry accepted and executed all 79 frames with no localization or safety
failures. Anchors stayed visible, so live texture tracking remains unexercised.
The no-target greedy policy produced 78 left turns and one forward step, showing
that the next controller problem is map-specific exploration rather than input
safety or timing. See
[`GUARDED_CONTROLLER_NOFIRE_8HZ_CONTINUITY_RESULT.json`](GUARDED_CONTROLLER_NOFIRE_8HZ_CONTINUITY_RESULT.json).
