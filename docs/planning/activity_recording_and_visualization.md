# Active plan addition: model activity recording and visualization

Added September 21, 2026 at the user's request. This extends the existing
phases 0–10+ roadmap. The classifier recorder and offline viewer are implemented;
see [usage and verification](../RECORDING.md). The remaining roadmap is tracked below.

## Goal

Show how the connectome model responds to observations and actions, and save
enough detail to replay behavior, identify patterns, diagnose failures, and
evaluate improvements. Recording is a core experiment requirement, not just
a feature of the viewer. It must work on CPU and CUDA without an open viewer.

## Development order

- [x] Before further controlled experiments: implement a versioned recording
  format and instrument the current 100-neuron synthetic model. Its events are
  input samples and classification decisions, not game actions.
- [x] Add an offline network viewer using those recordings: neuron activation
  colors, input/internal/output roles, directed connections, a timeline,
  pause/step/replay, and selection of individual neurons or annotated groups.
- [ ] During phases 6–7: require recording in the toy combat environment and
  RL loop from the first run. Synchronize observations, policy decisions,
  executed actions, neuron activity, rewards, and resulting states.
- [ ] Add a live viewer fed by the same recorder, with an independently
  throttled display rate. Display throttling must not drop recorded decisions.
- [ ] Obtain matching FlyWire anatomical coordinates or morphology, including
  version, provenance, licensing, and stable neuron-ID mappings. Add an
  anatomical view with the selected circuit inside a brain outline; clearly
  identify missing coordinates and inactive/unmodeled regions. Anatomical
  assets are optional for the first network viewer.
- [ ] During phases 8–10+: instrument the CS2 bridge before training against
  it. Record tick IDs, observation/action times, action acknowledgments,
  latency, and missing/out-of-order messages alongside game events.

## Required record for every environment decision

Join all records using run ID, episode ID, environment/worker ID, decision
index, and policy/checkpoint version. Include monotonic timestamps and game
tick IDs where available; wall-clock timestamps alone are insufficient.

1. **What the model saw:** raw observation, transformed model input,
   normalization state/version, visibility and action masks, and any recurrent
   state carried into the decision. Record privileged environment state
   separately from the policy's actual observation.
2. **Internal response:** input projections, per-neuron activations at every
   message-passing step, final state, and readout. Preserve neuron IDs and
   activation definitions. Record whether state resets between decisions;
   internal message-passing steps are not environment timesteps.
3. **Decision:** raw outputs/logits, action probabilities or continuous-policy
   distribution parameters, sampled/chosen action, log probability, value
   estimate and entropy when applicable, and exploration settings.
4. **Execution:** action actually sent and applied, timing, duration/repeated
   ticks, rejected/clamped actions, and acknowledgment where available.
5. **Consequence:** next observation/state, total reward and each reward
   component, hit/miss/damage/kill/movement events as supported, termination,
   truncation, reset reason, and final observation before automatic reset.

For the current classifier, substitute sample and batch IDs, dataset split,
input coordinates, labels, logits/probabilities, predictions, and losses for
environment-specific fields. Distinguish training, validation, and test passes,
and link each forward pass to the correct optimizer/model version. For RL,
distinguish rollout decisions from later optimization passes over those data.

## Learning diagnostics and reproducibility

- [ ] Save per-update loss components, learning rate, gradient norms and
  finite-value checks, weight/update norms, activation distributions, inactive
  neuron fractions, topology/sign invariant results, and algorithm-specific
  diagnostics such as PPO KL divergence and clipping fraction.
- [ ] Save initial/final and scheduled intermediate checkpoints containing
  model, optimizer, scheduler, normalization, RNG, and training-progress state.
  For each fixed-policy rollout, retain the exact policy version used. For the
  small synthetic experiment, retain parameter updates or per-update states
  so recorded reactions can be traced to exact weights. Do not duplicate
  unchanged topology or full weights inside every action row.
- [ ] Save immutable graph/IO/sign metadata, dataset and split references,
  environment/reward definitions, configuration, all seeds, source revision
  and dirty-source fingerprints, package versions, device, and data hashes.
  Version any transforms needed to decode observations or action encodings.
- [ ] Measure training throughput, forward/backward and environment latency,
  recorder overhead, CPU/RAM and GPU-memory use, bytes written, and queue depth.
  Save raw measurements as well as aggregates where practical.

## Storage and completeness

For the current 100-neuron experiments, default to full per-neuron traces at
every internal step for every forward pass, and retain every environment
decision/action/outcome when environments are introduced. No silent sampling.
Keep raw numerical values; viewer colors and aggregate charts are derived data.

Use compressed, chunked numerical arrays for activation tensors and structured
tables for events, with a versioned manifest linking them to graph and checkpoint
files. Separate small committed summaries from large local artifacts. Use
bounded buffered writes, detached tensors, periodic flushes, and recoverable
chunks so recording neither retains autograd graphs nor exhausts memory.
Never overwrite an existing run.

Before larger runs, estimate storage from neuron count, internal steps, dtype,
decision rate, and duration; report measured recording overhead. If full traces
become impractical, explicitly document and agree on the reduced capture policy
before the run. Continue to retain every action/observation/reward transition;
identify exactly which neuron or optimizer details were reduced. On disk-full
or recorder failure, flush what is recoverable and stop/checkpoint the run rather
than silently continuing unrecorded. Mark interrupted runs and incomplete
chunks in the manifest. Backpressure must not silently drop events.

## Analysis and visualization

- Compare activity aligned to actions and outcomes: before/during/after firing,
  turning, hits, misses, damage, and episode success/failure.
- Inspect neuron/group activation heatmaps, correlations, saturation/inactivity,
  recurring activity patterns, action entropy, and reward/latency relationships.
- Compare successful and failed trajectories, learning stages, seeds, and
  baseline architectures under equivalent recording and training conditions.
- Use a consistent, labeled color scale with signed values where applicable;
  distinguish model activations from measured biological activity. Anatomical
  location or correlation with an action does not establish biological function
  or causality. Validate proposed explanations with controlled ablations.
- Keep held-out test traces separate from optimization work. Use training and
  validation traces to choose improvements; do not repeatedly tune against the
  existing test set. Save analysis definitions and links to their source runs.

## Acceptance checks

- [ ] A short run has a complete decision-to-observation-to-action-to-outcome
  mapping, including parallel environments, action repeats, and episode resets.
- [x] Saved traces match captured model tensors and preserve neuron ordering;
  enabling recording does not change model outputs, gradients, or RNG behavior
  under a controlled deterministic test. Report timing overhead separately.
- [ ] A separate process loads recordings/checkpoints and replays activations,
  decisions, and outcomes without retraining or requiring a live environment.
  Replaying recorded data is distinct from guaranteeing deterministic CS2 replay.
- [x] Interrupted writes recover complete chunks and explicitly expose gaps.
- [ ] The viewer works for the existing CPU model and the CUDA test, without
  requiring a dedicated GPU merely to record data.
- [ ] Each experiment produces a human-readable summary and machine-readable
  raw records sufficient for action-aligned analysis and reproducible comparisons.
