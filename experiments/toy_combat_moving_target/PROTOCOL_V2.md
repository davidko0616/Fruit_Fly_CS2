# Moving-target replacement confirmation protocol

This replacement protocol was fixed after the original seed-61 confirmation
failed and after zero-entropy diagnostics on seeds 42 and 61. The failed result
remains governed by the original protocol and is not reclassified.

## Fixed change

The sole training change from the original protocol is an entropy coefficient of
zero for all 100 updates. The environment, transferred source actor, PPO budget,
rewards, recording, evaluation seeds, and endpoints remain unchanged. This tests
the prospective finding that the moving-target controller benefits from a sharper
action distribution.

## Seeds and training

- Initialize from version 100 of run
  `integrated_navigation_flywire_seed_53_confirmation_v1`, run ID
  `fe67e1db-d400-469a-9028-d3b29125076e`, weight SHA-256
  `ca6a550673745fd33e9ff5952dbb8dd02f6bbd13af13817c661f81f8667b47ca`.
- Confirmation seed 69; sampled-action seed 70; minibatch seed 71; freshly
  initialized value-network seed 72.
- Eight workers, horizon 64, action repeat one, 100 PPO updates, four optimization
  epochs, minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, value coefficient
  .5, gradient clipping .5, and entropy coefficient zero.
- The target moves one cell after every nonterminal decision using the deterministic
  obstacle-aware rule in the original protocol. All eight agent actions coexist.

## Evaluation and success

- Evaluate versions 0, 10, ..., 100 in stochastic and greedy modes on held-out
  seeds 9,000,000–9,000,255 and report all original endpoints.
- Replay all 51,200 decisions and verify policy activity, expanded environment
  state, rewards, outcomes, source-policy lineage, topology, and signs.
- Confirmation succeeds if final stochastic hit rate is at least 90%, improves
  by at least 10 percentage points over transferred version 0, has positive mean
  return, experiences target motion in all held-out episodes with at least 20
  mean target moves, preserves topology/signs, and passes complete replay.
