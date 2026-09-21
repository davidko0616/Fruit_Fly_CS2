# Project handoff — updated September 21, 2026

## Current laptop: NVIDIA CUDA validated

The project moved to a Windows laptop with an NVIDIA GeForce MX570 A (4 GB).
An isolated `.venv` now has PyTorch 2.7.1+cu126. The archived 100-neuron model
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

1. Run seeds 43–46 and aggregate variability. These runs were proposed but did
   not complete before the session was interrupted; only seed 42 is archived.
2. Verify reloading the original CPU checkpoint in a separate evaluation process
   (the new NVIDIA run's checkpoint has now passed this check).
3. Fix the random sparse baseline's duplicate-edge sampling and unmatched sign
   distribution; match the MLP parameter budget before controlled comparisons.
4. Compare architectures under equivalent data, seeds, initialization policies,
   training budgets, and evaluation conditions.
5. Continue to the toy combat environment after synthetic validation. Revisit
   larger CUDA models on the current NVIDIA laptop when GPU scaling is needed.

Do not claim biological advantages from the first classification run. The current
100-neuron circuit is dense, and its IO roles are computational fallbacks.

The user chose to complete the small CPU milestone before setting up WSL2.
