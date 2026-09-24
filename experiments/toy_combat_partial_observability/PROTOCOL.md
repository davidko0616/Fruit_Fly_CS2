# Partial-observability FlyWire confirmation protocol

This protocol was fixed after the seed-42 environment and policy explorations and
before the independent confirmation. Its objective is to confirm that a
FlyWire-derived controller can navigate, track, and fire when it cannot observe
the live target through obstacles and must act from a remembered target state.

## Starting policy and environment

- Initialize the actor from version 100 of confirmed run
  `moving_target_flywire_seed_69_confirmation_v2`, run ID
  `b06ecac8-58b0-44cd-996f-6f202cd4d7b3`, weight SHA-256
  `b4590817ab26630d3c5f90f43550553fbd685f9370631c7a5e7b9ff117b0895d`.
- Save the source run ID, version, and hash in the confirmation manifest; saved
  version 0 must reproduce the source actor exactly.
- Use a deterministic 9x9 environment with 16 obstacles, initial occlusion, a
  120-tick limit, and target motion after every nonterminal decision. Keep all
  eight integrated actions available except fire during its two-tick cooldown.
- Keep the 14-value observation interface. Exact live target geometry is visible
  only with line of sight. It is zero before first sighting and becomes the last
  visible target geometry during later occlusion. Record whether this geometry is
  live, whether memory exists, and its age.
- Mask aim-progress and distance-progress shaping unless line of sight exists on
  both sides of a transition. This prevents hidden live coordinates from leaking
  through rewards. Keep time, line-of-sight, collision, and shot rewards.

## Policy and training

- FlyWire actor with 100 neurons, 7,615 fixed directed edges and source signs,
  14 inputs, eight action logits, three message-passing steps, and 7,953 trainable
  parameters.
- Use explicit last-seen target state and reset it every episode. Do not carry
  persistent neuron activity between decisions in this confirmation.
- Confirmation seed 77; sampled-action seed 78; minibatch seed 79; freshly
  initialized value-network seed 80.
- Eight workers, horizon 64, action repeat one, 100 PPO updates, four optimization
  epochs, minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, value coefficient
  .5, gradient clipping .5, and entropy coefficient zero.

## Evaluation, intervention, recording, and success

- Evaluate versions 0, 10, ..., 100 in stochastic and greedy modes on held-out
  seeds 9,000,000–9,000,255. Report hit rate, return, episode length, line-of-sight
  acquisition, firing alignment, target motion, occluded decisions, memory use,
  memory age, and action counts.
- At the final version, repeat stochastic evaluation with identical weights and
  seeds while zeroing remembered target geometry during occlusion. This
  no-memory intervention is the causal comparison.
- Report the privileged scripted benchmark separately. It can read environment
  state and is a solvability reference, not a learned policy.
- Record and independently replay all 51,200 training decisions. Verify policy
  activity, observations, privileged hidden state, target transitions, rewards,
  outcomes, source-policy lineage, topology, and signs.
- Confirmation succeeds if the final stochastic policy reaches at least 80% hit
  rate and positive mean return; acquires line of sight in at least 85% of
  episodes; spends at least 60% of decisions occluded; receives remembered target
  geometry in at least 30% of occluded decisions; experiences target movement in
  at least 95% of episodes with at least 20 mean target moves; exceeds the
  no-memory intervention by at least 60 percentage points while that intervention
  remains at or below 20%; preserves topology and signs; and passes complete
  replay.

The confirmation does not require improvement over transferred version 0. The
source already contains navigation, aiming, and moving-target skills; this stage
tests whether those skills remain effective under occlusion and whether success
depends on the remembered target representation.
