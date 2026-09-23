# Integrated FlyWire navigation confirmation protocol

This protocol was fixed after the two seed-42 explorations and before the
independent confirmation run. The objective is to confirm that a FlyWire-derived
policy can retain navigation and aiming skills while movement, turning, waiting,
and firing coexist in one action space.

## Starting policy and environment

- Initialize the actor from version 100 of recorded run
  `navigation_flywire_seed_47_confirmation_v1`, run ID
  `9130167b-43f0-4cf5-b77d-bbdb603973dc`, weight SHA-256
  `41aff46ebdcbda7147ca7668f3da6d7ce1e2712868ca5a25c68fd0939635604d`.
- Record the source run ID, version, and hash in the new manifest. Saved policy
  version 0 must exactly equal the source actor.
- Use the confirmed deterministic 9x9 navigation environment: five obstacles,
  stationary target, initial occlusion, 14 observations, and maximum 120 ticks.
- Keep all eight actions available in both occluded and visible states. Only the
  existing two-tick fire cooldown may temporarily mask fire. No phase-based
  movement or firing restriction is allowed.
- Keep the confirmed reward unchanged: time -0.01, aim progress ×0.2, distance
  progress ×0.1, line-of-sight progress ×0.2, collision -0.05, miss -0.1, and
  hit +3.

## Policy and training

- FlyWire actor with 100 neurons, 7,615 fixed directed edges and fixed source
  signs, 14 inputs, eight logits, three recurrent message-passing steps, and
  7,953 trainable parameters.
- Confirmation seed 53; sampled-action seed 54; minibatch seed 55; freshly
  initialized value-network seed 56.
- Eight workers, horizon 64, action repeat one, 100 PPO updates, four optimization
  epochs, minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, value coefficient
  .5, and gradient clipping .5.
- Entropy coefficient decreases linearly from .01 to zero. The stochastic policy
  is the primary controller; greedy behavior is a diagnostic.

## Evaluation, recording, and success

- Evaluate versions 0, 10, ..., 100 in stochastic and greedy modes on fixed
  held-out seeds 9,000,000–9,000,255. Report hit rate, return, episode length,
  line-of-sight acquisition, firing alignment, firing rate, and oracle results.
- Record and independently replay all 51,200 decisions, including full internal
  activity, masks, actions, rewards, environment state, and policy versions.
- Confirmation succeeds if final stochastic hit rate is at least 95%, improves
  by at least 10 percentage points over transferred version 0, has positive mean
  return, preserves topology/signs, uses no navigation phase masking, and passes
  complete replay.

After confirmation, the next curriculum will introduce target motion before
removing direct target coordinates behind occlusion to create a memory-dependent
partial-observability task.
