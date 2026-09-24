# Dust II bridge status

The read-only bridge foundation and first real fixed-radar calibration are
implemented. The bridge records own-player/map GSI, encodes visibility-gated
Dust II frames with last-seen target memory, and replays them through the
confirmed 100-neuron FlyWire policy without game input.

Nine bridge tests pass, and the committed three-frame fixture completes an
end-to-end replay through policy version 100. This validates interfaces and
memory transformations only; it is not Dust II training or performance evidence.

The [real calibration walk](RADAR_CALIBRATION.md) recovered 2,987 of 3,000 poses
(99.57%) over ten minutes and produced the versioned
[`de_dust2` bounds](dust2_calibration_v1.json). The next required artifact is a
labeled visible-target and local-clearance perception set, followed by
synchronized GSI/screen capture. The [live GSI probe](GSI_POSE_PROBE.md) showed
that active-player GSI omits pose, so localization uses the visible fixed radar.
See the
[fixed read-only protocol](PROTOCOL.md) and the
[setup and data contract](../../docs/CS2_DUST2_BRIDGE.md).
