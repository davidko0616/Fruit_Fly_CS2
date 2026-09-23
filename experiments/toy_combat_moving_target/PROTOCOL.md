# Moving-target FlyWire confirmation protocol

This protocol was fixed after the two seed-42 explorations and before the
independent confirmation. The objective is to confirm tracking and firing against
a continuously moving target while retaining integrated navigation and combat
actions.

## Starting policy and environment

- Initialize the actor from version 100 of confirmed run
  `integrated_navigation_flywire_seed_53_confirmation_v1`, run ID
  `fe67e1db-d400-469a-9028-d3b29125076e`, weight SHA-256
  `ca6a550673745fd33e9ff5952dbb8dd02f6bbd13af13817c661f81f8667b47ca`.
- Save the source run ID, policy version, and hash in the confirmation manifest;
  saved version 0 must reproduce the source actor exactly.
- Use the deterministic 9x9 environment with five obstacles, initial occlusion,
  a 120-tick limit, and the same 14 observations. Current relative target
  coordinates remain visible; partial observability is reserved for the next
  stage.
- Move the target one cardinal cell after every nonterminal agent decision. It
  continues straight when possible, then tries right, left, and reverse. It may
  not enter an obstacle, leave the grid, or collide with the agent.
- Keep all eight agent actions available. Only the two-tick firing cooldown may
  temporarily mask fire. Keep the integrated reward unchanged.

## Policy and training

- FlyWire actor with 100 neurons, 7,615 fixed directed edges and source signs,
  14 inputs, eight action logits, three recurrent message-passing steps, and
  7,953 trainable parameters.
- Confirmation seed 61; sampled-action seed 62; minibatch seed 63; freshly
  initialized value-network seed 64.
- Eight workers, horizon 64, action repeat one, 100 PPO updates, four optimization
  epochs, minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, value coefficient
  .5, and gradient clipping .5.
- Entropy coefficient decreases linearly from .01 to zero. Stochastic evaluation
  is primary and greedy evaluation is diagnostic.

## Evaluation, recording, and success

- Evaluate versions 0, 10, ..., 100 in stochastic and greedy modes on held-out
  seeds 9,000,000–9,000,255. Report hit rate, return, episode length, line-of-sight
  acquisition, firing alignment, firing, target-motion exposure, action counts,
  and the privileged reactive benchmark.
- Record and independently replay all 51,200 decisions. The expanded privileged
  state includes target position and motion heading.
- Confirmation succeeds if final stochastic hit rate is at least 88%, improves
  by at least eight percentage points over transferred version 0, has positive
  mean return, experiences target motion in all held-out episodes with at least
  20 mean target moves, preserves topology/signs, and passes complete replay.

After confirmation, remove direct target coordinates while line of sight is
blocked. That next stage will require memory or belief-state inference rather
than purely reactive tracking.
