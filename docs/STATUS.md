# Project handoff — September 20, 2026

## Completed

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

1. Run seeds 43–46 and aggregate variability. These runs were proposed but did
   not complete before the session was interrupted; only seed 42 is archived.
2. Verify reloading the saved checkpoint in a separate evaluation process.
3. Fix the random sparse baseline's duplicate-edge sampling and unmatched sign
   distribution; match the MLP parameter budget before controlled comparisons.
4. Compare architectures under equivalent data, seeds, initialization policies,
   training budgets, and evaluation conditions.
5. Continue to the toy combat environment after synthetic validation. Revisit
   WSL2/native ROCm when GPU training is needed.

Do not claim biological advantages from the first classification run. The current
100-neuron circuit is dense, and its IO roles are computational fallbacks.

The user chose to complete the small CPU milestone before setting up WSL2.
