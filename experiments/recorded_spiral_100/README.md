# First complete activity recording

The September 21, 2026 CPU run captured every sample processed during 200
epochs of the existing 100-neuron spiral experiment. Test accuracy remained
99.324% (147/148), matching the unrecorded CPU run.

`summary.json` contains compact metrics, counts, source fingerprints and the
separate-process audit result. The full approximately 712 MiB recording remains
local at `artifacts/synthetic/recorded_spiral_100_cpu_v2/` and is ignored by Git.
It contains 2,604 forwards, 312,904 sample-forwards, 402 compressed activation
chunks, and 2,201 exact parameter versions.

See [instructions and format](../../docs/RECORDING.md) to reproduce a recording
and launch the offline viewer. This experiment establishes traceability and
replay for the classifier; it does not demonstrate a biological advantage or
game-playing ability. It uses computational IO roles and a non-anatomical view.
