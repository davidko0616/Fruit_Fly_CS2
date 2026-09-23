# First FlyWire navigation confirmation protocol

This protocol was fixed after three seed-42 exploratory runs and before the
confirmation run. The objective is an engineering milestone: demonstrate that
the FlyWire policy can move around occlusion, acquire line of sight, aim, and hit
a stationary target. It is not an architecture comparison.

## Environment and interface

- Deterministic 9x9 grid with one stationary target and five obstacles.
- Every episode begins without line of sight. Rejection sampling retains only
  layouts with a reachable firing position. A privileged breadth-first oracle,
  used only for solvability checks, solves at least 1,000 fixed test layouts.
- Stable eight actions: wait, forward, backward, strafe left/right, turn
  left/right, and fire. Action repeat is one for cell-level movement control.
- Fourteen observations: normalized agent position, heading vector, target
  coordinates in the local frame, normalized target distance, aim alignment,
  line-of-sight flag, fire cooldown, and forward/back/left/right clearance.
- First navigation scaffold: while occluded, fire is masked; after acquiring
  line of sight, movement is masked and the policy must aim and shoot. This
  separates navigation from aiming without changing the action interface.
- Maximum 120 decisions. Reward components per tick: time -0.01; change in aim
  alignment ×0.2; distance reduction ×0.1; line-of-sight change ×0.2; collision
  -0.05; miss -0.1; hit +3. Potential-difference shaping prevents accumulating
  reward by oscillating between positions or visibility states.

## Policy and PPO

- FlyWire actor: 100 neurons, 7,615 fixed directed edges, fixed source signs,
  14 inputs, eight action logits, three recurrent message-passing steps, and
  7,953 trainable parameters.
- Separate 14-64-1 value network. Actor seed 47, sampled-action seed 48,
  minibatch seed 49, and value seed 50.
- Eight workers, horizon 64, 100 PPO updates, four optimization epochs,
  minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, value coefficient .5,
  and gradient clipping .5.
- Entropy coefficient decreases linearly from .01 at update 1 to zero at update
  100. The stochastic policy is the primary controller and evaluation endpoint;
  greedy behavior is retained as a diagnostic rather than a success criterion.

## Recording, evaluation, and success

- Record every rollout decision, full connectome activity, policy version,
  observation, mask, sampled action, probabilities, value, all reward components,
  environment state, outcome, and PPO update diagnostic. Save source hashes,
  every actor version, optimizer checkpoints, and model invariants.
- Independently replay all 51,200 recorded policy forwards and environment
  transitions. Any unexplained mismatch fails the run.
- Evaluate versions 0, 10, ..., 100 with stochastic and greedy selection on
  held-out seeds 9,000,000-9,000,255. Report hit rate, return, episode length,
  line-of-sight acquisition, firing alignment, and firing rate. Report the
  privileged oracle separately.
- Confirmation succeeds if the final stochastic policy reaches at least 85%
  hit rate, improves at least 30 percentage points over version 0, has positive
  mean return, preserves topology/signs, and passes complete replay.

After confirmation, the next curriculum can relax phase masking so movement and
shooting coexist, then introduce moving targets and partial observability.
