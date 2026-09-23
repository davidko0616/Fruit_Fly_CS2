# Project handoff — updated September 23, 2026

## Current PC: integrated navigation milestone complete

The current machine is the Ryzen 5 7600 desktop, using Python 3.12 and
PyTorch 2.7.1+cpu in `.venv`. The first fixed navigation confirmation now passes:
the FlyWire controller reaches a 97.3% held-out hit rate after starting every
episode behind occlusion. All 51,200 recorded decisions replayed successfully.
The subsequent integrated curriculum removes phase masking and reaches 99.6%
held-out hit rate with movement and firing available together. Its independent
51,200-decision replay also passes.

All 15 recorded classifier runs (FlyWire, matched random
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

- Integrated toy-combat navigation: curriculum transfer followed by 100 PPO
  updates produced 99.6% held-out stochastic hit rate with movement, aiming, and
  firing simultaneously available. Mean return was +2.857 and greedy hit rate
  was 74.6%. The source-policy lineage, topology/signs, and all 51,200 decisions
  passed their checks. See the
  [protocol, report, and curve](../experiments/toy_combat_integrated_navigation/README.md).

- First toy-combat navigation milestone: the recorded FlyWire PPO policy learned
  to move from initial occlusion, acquire line of sight, aim, and fire. Held-out
  stochastic hit rate improved from 48.4% to 97.3%, with final 100% line-of-sight
  acquisition and firing alignment. All 51,200 CPU decisions passed replay and
  topology/sign invariants passed. See the
  [fixed protocol, report, and learning curve](../experiments/toy_combat_navigation/README.md).

- Five-seed toy-combat aiming comparison: all 15 matched CPU runs and all
  768,000 rollout decisions passed replay. Final stochastic hit rate was
  90.7 ± 4.1% for FlyWire, 96.1 ± 3.3% for randomized wiring, and 100.0 ± 0.0%
  for the parameter-matched MLP. FlyWire trailed both controls on primary
  endpoints in all five paired seeds. See the
  [full comparison](../experiments/toy_combat_baseline_comparison/README.md).

- First toy-combat aiming milestone: the recorded FlyWire PPO policy improved
  from 35.2% to 95.3% stochastic hit rate on 256 held-out episodes; mean return
  improved from -0.89 to +1.49. All 51,200 decisions and internal activations
  passed independent policy/environment replay. See the
  [protocol, report and learning curve](../experiments/toy_combat_aiming/README.md).

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
now has a working classifier recorder, offline viewer, and audited environment
action recorder. Use `training.train_recorded` for new inspectable classifiers
and `training.train_toy_combat` for recorded combat policies. Live environment
display and anatomical layouts remain planned. Every future environment decision
must retain synchronized observations, activity, actions, rewards and outcomes.
Older experiments cannot retroactively supply full activity traces.

1. Add controlled target motion while retaining integrated actions, synchronized
   activity recording, and stochastic evaluation.
2. Hide target coordinates during occlusion to introduce memory-dependent partial
   observability and longer credit assignment.
3. Confirm each harder curriculum across multiple seeds before connecting the
   controller to a live Counter-Strike observation/action bridge. Repeat matched
   actor comparisons when they answer a specific scientific question rather than
   as the main optimization target.
4. For stronger scientific conclusions, predeclare harder synthetic tasks,
   independent dataset splits, or less dense circuits. The first comparison
   is close to the accuracy ceiling and uses a single fixed split.
5. Extend replay to feedforward MLP schema-2 recordings if visual comparison
   is needed. They are fully recorded and numerically audited, but the current
   recurrent viewer intentionally accepts schema 1 only.
6. Benchmark larger models only when needed. Current experiments run on CPU;
   NVIDIA laptop results remain historical evidence for a CUDA option.

Do not claim biological advantages from the first classification run. The current
100-neuron circuit is dense, and its IO roles are computational fallbacks.

The user chose to complete the small CPU milestone before setting up WSL2.
