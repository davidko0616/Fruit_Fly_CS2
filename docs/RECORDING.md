# Activity recording and offline replay

## Baseline comparison recordings

`training.train_recorded` also accepts `--architecture random` or
`--architecture mlp` (default: `flywire`). Random graphs use the same recurrent
schema and viewer. Each run saves its actual randomized adjacency, with source
signs and IO roles preserved. See the
[fixed comparison protocol](../experiments/baseline_comparison_100/PROTOCOL.md).

MLPs use schema 2: `hidden_preactivations` and `hidden_activations` contain all
hidden-layer values concatenated along the feature axis, with layer offsets and
widths in `manifest.json`. Inputs, logits, predictions, probabilities, losses,
sample IDs, every parameter version, update diagnostics and checkpoints are
retained with the same durability policy. `architecture.json` identifies the
feedforward architecture. Copied `graph.npz` and `adjacency.npz` are reference
budget artifacts for MLP runs, not its computation graph. MLP weights have no
sign constraints; `topology_and_signs_preserved` is null rather than true.

`tools/verify_recording.py` audits both schemas. The existing offline recurrent
viewer accepts schema 1 only; use numerical analysis for MLP traces. No hidden
layers are mislabeled as biological neurons or recurrent time steps.

Run the full sequential CPU comparison with a fresh output directory:

```powershell
.\.venv\Scripts\python.exe tools/run_comparison.py --output artifacts/synthetic/baseline_comparison_100 --results experiments/baseline_comparison_100/results
.\.venv\Scripts\python.exe tools/check_comparison_matching.py
.\.venv\Scripts\python.exe tools/summarize_comparison.py
```

The default schedule is seeds 42-46, 200 epochs each, rotating architecture order
between seeds. Full traces remain ignored by Git. Compact per-run results retain
configuration, source hashes, histories and audit results. The matching audit
also checks actual graph budgets, initial recurrent IO weights, paired minibatch
orders, and one final test evaluation per run.

Use the recorded runner for new inspectable experiments. It trains the existing
100-neuron FlyWire model with the archived graph and spiral dataset; no large raw
connectome download is needed. CPU is the default; CUDA is also supported.
The original `training.train_synthetic` and hardware validation scripts remain
available as unrecorded historical workflows.

## Run and inspect

From the repository root with the dependencies installed:

```powershell
.\.venv\Scripts\python.exe -m training.train_recorded --output artifacts/synthetic/my_recorded_run --epochs 200
.\.venv\Scripts\python.exe tools/verify_recording.py --run artifacts/synthetic/my_recorded_run
.\.venv\Scripts\python.exe -m visualization.serve --run artifacts/synthetic/my_recorded_run
```

Open http://127.0.0.1:8765 in a browser. The viewer uses local files and loopback
HTTP only, has no CDN dependencies, and does not run inference or retrain the
model. Keep the server running while viewing; Ctrl+C stops it. Choose a different
`--port` if necessary. On other platforms replace the Windows Python executable
with your environment's `python`.

For a quick smoke test use `--epochs 2`. For GPU recording add `--device cuda`.
Every training run requires a new output directory. `--seed` changes model and
batch-order seeds; the archived dataset and its split remain fixed at seed 42.

Select **Training** or **Validation**, a forward-pass type, a point on the
training timeline, and a sample. Use **Play**, the arrows, and the processing-step
buttons to explore activity. Select a neuron to see its stable FlyWire ID, role,
connections, and activity across steps. Color scales are fixed within a selected
split/pass type, not rescaled for each frame. Test data requires explicitly
selecting **Held-out test**; use it for final evaluation, not tuning.

## What is recorded

Every actual classifier forward is recorded, including initial evaluation,
optimizer batches, epoch train/validation evaluation, and final selected-model
evaluation. There is no trace sampling. Each sample stores:

- Original dataset ID, input coordinates, label, split, epoch, event ID, and
  exact parameter version. Inputs are unnormalized and state resets each forward.
- Sensory projection, injected neuron state, pre-ReLU values and post-ReLU
  activations for every neuron at all three internal steps.
- Logits, probabilities, prediction, and per-sample cross-entropy loss.

Each optimizer update stores exact parameters, gradient/weight/update norms,
learning rate, batch loss, backward timing, finite-gradient results, and
topology/sign checks. Full model/Adam/RNG/batch-generator checkpoints are saved
initially and every 25 epochs, including the final epoch. The selected model is
saved separately. Initial/final graph/data metadata and source fingerprints
allow the records to be interpreted without the original training process.

These are classification events, not game actions. Environment observations,
executed actions, rewards, and episode boundaries will be added with toy combat.
Positions in the viewer are a computational layout, not anatomical coordinates;
activation colors are artificial model values, not biological measurements.

## On-disk format (schema 1)

| File | Contents |
|---|---|
| `manifest.json` | Schema, run ID, state, capture policy, configuration and counts |
| `graph.json` | String neuron IDs, roles, directed source/target indices and edge signs |
| `chunks/NNNNNN.npz` | Compressed arrays and embedded event metadata/row offsets |
| `weights/NNNNNNN.npz` | Exact parameter arrays `p0...pN`, update diagnostics |
| `checkpoints/*.pt` | Full continuation state at scheduled boundaries |
| `model.pt` | Selected model state and its parameter version |
| `dataset.npz`, `graph.npz`, `adjacency.npz` | Self-contained input data, splits and model structure |
| `metrics.json`, `history.json`, `provenance.json` | Results, learning history and source hashes |

Chunk `states` has shape `[samples, steps+1, neurons]`; `preactivations` has
shape `[samples, steps, neurons]`. Other arrays are `inputs`, `labels`,
`sample_ids`, `sensory`, `logits`, `probabilities`, `predictions`, and `losses`.
Each event records its chunk-local `offset` and `count`. A sample can appear
many times across epochs and phases; join on run ID + event ID + sample position,
and use `sample_ids` to trace the same original input. Parameter ordering and
shapes are specified in the manifest. FlyWire IDs are strings in browser JSON
to preserve integers larger than JavaScript's exact numeric range.

Trace and parameter archives use numerical arrays without pickle. PyTorch
checkpoint files should only be loaded from trusted runs. The viewer reads the
numerical archives, not checkpoint pickle data.

## Durability and cost

The recorder detaches/copies activations and buffers at most 1,024 samples before
flushing a compressed chunk. It retains no autograd graphs in the recording
buffer. Capturing CUDA activity synchronizes transfers, so it adds overhead.
`forward_with_capture_seconds` includes forward execution and capture;
manifest `recording_seconds` measures post-forward processing, file writes and
checkpoint writes, not all hook/transfer overhead. Compare total run times to
measure end-to-end overhead; these counters are not a full profiler.

Completed chunks are atomically renamed into place; readers rebuild the index
from these chunks even if the manifest is stale. Temporary files are ignored.
Handled exceptions mark runs interrupted and flush recoverable data. Abrupt
termination can lose the buffered tail (up to 1,024 samples); a non-finalized
manifest remains a visible warning. The recorder retries transient Windows
replacement locks for about three seconds, then fails. Disk-full and persistent
write errors stop training rather than silently dropping data. An interrupted
checkpoint is attempted where possible. Automatic mid-epoch resume is not yet
implemented; scheduled checkpoints retain model/optimizer/RNG state.

The current complete run uses about **712 MiB**. Storage grows with samples,
epochs, neurons and internal steps; the runner prints a raw activation-size
estimate before recording. Large artifacts are ignored by Git. Do not reduce
trace coverage silently when scaling.

## Verified result — September 21, 2026

`artifacts/synthetic/recorded_spiral_100_cpu_v2` completed 200 epochs with:

- 2,604 recorded forwards, 312,904 sample-forwards, 402 activation chunks.
- 2,201 exact parameter versions, including initialization.
- 100% train/validation accuracy and 99.324% test accuracy (147/148).
- 85.49 seconds of training including recording; about 712 MiB on disk.
- The same final test loss and accuracy as the earlier unrecorded CPU run.

These are single runs, not controlled throughput benchmarks. CPU and CUDA
equivalence tests check outputs, gradients and RNG state. Tests also cover
interrupted writes, transient/persistent locks, disk-full failure and parameter
reload. Two-epoch CPU/CUDA recordings passed separate-process audits. The audit
checks every committed sample's data/split assignment and reproduces selected
forward passes from exact weights. A failed first full run remains locally
available as an interrupted recording; its 106,640 committed sample-forwards
were successfully audited after a transient Windows file lock.
