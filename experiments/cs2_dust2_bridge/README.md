# Dust II bridge status

The read-only bridge foundation is implemented. It records own-player/map GSI,
encodes visibility-gated Dust II frames with last-seen target memory, and replays
them through the confirmed 100-neuron FlyWire policy without game input.

Seven bridge tests pass, and the committed three-frame fixture completes an
end-to-end replay through policy version 100. This validates interfaces and
memory transformations only; it is not Dust II training or performance evidence.

The next required artifact is a real `de_dust2` fixed-radar calibration walk,
followed by timestamped screen capture and visible-target perception. The
[live GSI probe](GSI_POSE_PROBE.md) showed that active-player GSI omits pose, so
localization now uses the visible fixed radar. See the
[fixed read-only protocol](PROTOCOL.md) and the
[setup and data contract](../../docs/CS2_DUST2_BRIDGE.md).
