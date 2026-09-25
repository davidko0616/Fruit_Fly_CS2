# Dust II bridge status

The read-only bridge foundation and first real fixed-radar calibration are
implemented. The bridge records own-player/map GSI, encodes visibility-gated
Dust II frames with last-seen target memory, and replays them through the
confirmed 100-neuron FlyWire policy without game input.

The full 52-test suite passes (with one expected CUDA skip), and the committed
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
See the
[fixed read-only protocol](PROTOCOL.md) and the
[setup and data contract](../../docs/CS2_DUST2_BRIDGE.md). The capture backend,
label schema, session-level split, and held-out detector acceptance metrics are
fixed in the [visible-perception protocol](PERCEPTION_PROTOCOL.md).
