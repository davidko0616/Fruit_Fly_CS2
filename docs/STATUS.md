# Project handoff — updated September 22, 2026

## Current PC: CPU baseline comparison complete

The current machine is the Ryzen 5 7600 desktop, using Python 3.12 and
PyTorch 2.7.1+cpu in `.venv`. All 15 recorded runs (FlyWire, matched random
destinations and matched MLP; seeds 42-46; 200 epochs) completed and passed
independent audits. Mean test accuracy is 99.46% for FlyWire and 99.32% for both
baselines. MLP reaches 95% validation accuracy sooner in all five paired seeds.
This does not establish a biological wiring advantage. See the
[comparison report, protocol and per-seed results](../experiments/baseline_comparison_100/README.md).
Full traces total 8.78 GiB locally; 4,693,560 sample-forwards were retained.
The original laptop recording and the first activity analysis are also local.

## Previous laptop: NVIDIA CUDA validated

The project moved to a Windows laptop with an NVIDIA GeForce MX570 A (4 GB).
That laptop's isolated `.venv` had PyTorch 2.7.1+cu126. The archived 100-neuron model
completed 200 training epochs on CUDA with 99.324% test accuracy in 63.73 s;
the same laptop's four-thread CPU run achieved 99.324% in 39.65 s. CUDA sparse
forward/backward and invariant checks passed. Peak PyTorch allocated GPU memory
was 66.24 MiB. Separate-process CUDA checkpoint reload reproduced the result.
All nine existing tests passed. See [reproduction and results](../tools/CUDA.md).

This validates the small model on NVIDIA; larger graphs remain untested. The
CPU is faster for this particular small checked workload. Prior AMD/ZLUDA
limitations below describe the previous computer. The original trainer remains
CPU-only; `tools/validate_cuda.py` runs the archived experiment on either device.

## Completed

- First descriptive activity analysis of recorded seed 42 on the Ryzen PC:
  both splits reach 95% accuracy at epoch 27; neurons 36/37 are silent only at
  the final processing step, and class-associated responses appear in output
  neurons. No causal or biological-advantage claim follows. See
  [report and plots](../experiments/recorded_spiral_100/activity_analysis/README.md).
  A CPU-only `.venv` with PyTorch 2.7.1+cpu is installed here; the copied full
  recording passed audit, and 16 CPU-compatible tests passed (CUDA test skipped).

- Full classifier activity recorder and local offline network viewer. The
  recorded 200-epoch CPU run saved all 312,904 sample-forwards and 2,201 parameter
  versions, retaining 99.324% test accuracy. CPU/CUDA equivalence, interruption,
  disk-full and file-lock tests pass (17 tests total). See
  [usage, data format and results](RECORDING.md).
- FlyWire v783 data download and inspection (large files remain local).
- Graph extraction, sparse network, and initial baseline implementations.
- Disjoint computational IO fallbacks, deterministic degree tie-breaking, and
  normalized synapse-count initialization.
- Nine passing graph, model, and synthetic data tests.
- Configurable CPU spiral trainer with separate train/validation/test sets,
  validation-based checkpoint selection, invariant checks, saved artifacts,
  package versions, and data/source fingerprints.
- First synthetic classification run: seed 42, 99.3% held-out test accuracy.
  See [archived results](../experiments/cpu_spiral_100/README.md).

## ZLUDA result

An isolated environment was tested with the RX 7800 XT. Excluding the integrated
GPU fixed library initialization failures. Dense GPU math passed. PyTorch
2.9.1+cu128 failed at a kernel lacking PTX in the wheel; 2.7.1+cu118 got further
but failed in the existing sparse layer at `cusparseSetStream`, which the supplied
ZLUDA build does not implement. GPU backward/training were not reached.
See [reproduction and diagnostics](../tools/ZLUDA.md).

## Next work

The [activity recording and visualization plan](planning/activity_recording_and_visualization.md)
now has a working classifier recorder and offline viewer. Use
`training.train_recorded` for new inspectable runs. Environment-action recording,
live display and anatomical layouts remain planned. Every future environment
decision must retain synchronized observations, activity, actions, rewards and
outcomes. Older experiments cannot retroactively supply full activity traces.

1. Develop a minimal toy combat environment with synchronized observations,
   actions, rewards, outcomes and model activity; retain the matched baselines.
2. For stronger scientific conclusions, predeclare harder synthetic tasks,
   independent dataset splits, or less dense circuits. The first comparison
   is close to the accuracy ceiling and uses a single fixed split.
3. Extend replay to feedforward MLP schema-2 recordings if visual comparison
   is needed. They are fully recorded and numerically audited, but the current
   recurrent viewer intentionally accepts schema 1 only.
4. Benchmark larger models only when needed. Current experiments run on CPU;
   NVIDIA laptop results remain historical evidence for a CUDA option.

Do not claim biological advantages from the first classification run. The current
100-neuron circuit is dense, and its IO roles are computational fallbacks.

The user chose to complete the small CPU milestone before setting up WSL2.
