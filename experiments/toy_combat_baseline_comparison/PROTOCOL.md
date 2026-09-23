# Toy-combat aiming baseline protocol

This protocol was fixed before the full comparison runs. One-update schema
smoke tests and the earlier single-seed FlyWire curriculum run are excluded.
All three architectures will be rerun through the finalized comparison code.

## Runs and controls

- Architectures: FlyWire, randomized destinations, and a parameter-matched MLP.
- Seeds: 42, 43, 44, 45, and 46. Run order rotates by seed to reduce systematic
  time-order effects. No completed run will be excluded.
- Task: the fixed stationary-target aiming curriculum in
  `experiments/toy_combat_aiming/PROTOCOL.md`. Movement actions remain masked.
- PPO: 100 updates, 8 workers, horizon 64, action repeat 2, four optimization
  epochs, minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, entropy
  coefficient .01, value coefficient .5, and gradient clipping .5.
- RNG streams for seed S: actor initialization S, sampled actions S+1,
  minibatch order S+2, and value-network initialization S+3. Training episode
  seeds use the same `S * 100000 + episode_id` mapping; policies can advance
  through that common per-worker stream at different rates.
- The separate 10-64-1 value network and its initialization are identical in
  each same-seed architecture triplet.

## Architecture matching

- All actors have exactly 7,913 trainable parameters and the same 10 inputs and
  eight action logits.
- FlyWire uses 100 neurons, 7,615 fixed directed edges, source-neuron signs,
  normalized synapse-count initialization, and three recurrent processing steps.
- Random uses the same neuron count, edge count, IO neuron sets, source signs,
  recurrent steps, and parameter count. Destinations are randomized independently
  for each source without replacement while preserving source degree, self-loops,
  and each source's synapse-count multiset. Incoming degree and motifs may differ.
  The dense graph is expected to retain many original edge positions, which
  limits the strength of this topology perturbation.
- FlyWire and random receive identical same-seed input/output projection
  initialization. The random graph generator has an isolated NumPy RNG.
- MLP is 10 -> 78 -> 81 -> 8 with ReLU hidden layers and exactly 7,913
  parameters. It uses PyTorch Linear initialization. This conventional baseline
  matches parameter count, not recurrence, signs, operation count, or inductive
  bias.

## Recording, audit, and evaluation

- Retain every rollout decision, observation, action mask, sampled/executed
  action, probability, reward component, outcome, value estimate, exact actor
  version, and all actor hidden activity. Save every policy version, PPO update
  diagnostics, source hashes, configuration, and model invariants.
- Independently replay every recorded policy forward and environment transition.
  All 15 runs must finish with a complete manifest and pass the audit.
- Evaluate versions 0, 10, ..., 100 with stochastic and greedy action selection
  on the same 256 held-out seeds 9,000,000-9,000,255. Evaluation sampling uses
  the same pre-generated uniform values for every architecture, seed, and version.
- Primary endpoints: final stochastic held-out hit rate and mean return.
  Secondary endpoints: stochastic hit-rate learning-curve area, final greedy hit
  rate, episode length, recorded CPU time, and training-episode diagnostics.
- Report every seed, mean and sample standard deviation, and paired same-seed
  FlyWire-minus-baseline differences. Five training seeds on one task support a
  descriptive comparison, not a general biological-wiring claim.

Full traces remain under ignored `artifacts/toy_combat/`; compact per-run data,
the protocol, aggregate tables, and plots are committed under `experiments/`.
