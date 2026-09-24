# Dust II bridge status

The read-only bridge foundation and first real fixed-radar calibration are
implemented. The bridge records own-player/map GSI, encodes visibility-gated
Dust II frames with last-seen target memory, and replays them through the
confirmed 100-neuron FlyWire policy without game input.

Eleven bridge tests pass, and the committed three-frame fixture completes an
end-to-end replay through policy version 100. This validates interfaces and
memory transformations only; it is not Dust II training or performance evidence.

The [real calibration walk](RADAR_CALIBRATION.md) recovered 2,987 of 3,000 poses
(99.57%) over ten minutes and produced the versioned
[`de_dust2` bounds](dust2_calibration_v1.json). The visible-player pilot now has
95 training and 20 validation frames. Its team-aware CPU checkpoint reaches
63.6% overall precision and recall, 71.4% enemy precision, and 83.3% enemy recall
at about 36 ms/frame, with no detections on 11 negative frames. It is sufficient
to test the read-only integration path but not to claim final perception
accuracy or enable firing. The next required artifact is
synchronized detector, radar, and GSI replay, followed by a local-clearance
perception set. The [live GSI probe](GSI_POSE_PROBE.md) showed
that active-player GSI omits pose, so localization uses the visible fixed radar.
See the
[fixed read-only protocol](PROTOCOL.md) and the
[setup and data contract](../../docs/CS2_DUST2_BRIDGE.md). The capture backend,
label schema, session-level split, and held-out detector acceptance metrics are
fixed in the [visible-perception protocol](PERCEPTION_PROTOCOL.md).
