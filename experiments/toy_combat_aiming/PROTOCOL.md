# First toy-combat aiming protocol

This is an engineering learnability milestone, not an architecture comparison.
It was fixed after two exploratory runs exposed a reward-design failure: the
policy learned to avoid firing and rotate until timeout. Those local runs remain
available as `flywire_ppo_seed_42_v1` and `v2`; they are not reported as final.

- Environment: deterministic 9x9 grid, stationary visible target, five obstacles
  placed outside the initial line of sight, 80-tick limit, action repeat 2.
- Initial curriculum: wait, turn left, turn right, and fire. Movement actions
  exist in the stable eight-action interface but are masked until navigation.
- Turn: 11.25 degrees per tick. A shot hits within 11.25 degrees with clear line
  of sight. Fire cooldown: two ticks.
- Reward: hit +2, miss -0.15, time -0.01/tick, aim-progress 0.2 times the change
  in cosine alignment. Collision -0.05 applies once movement is enabled.
- Model: the 100-neuron FlyWire circuit, 10 observation features, 8 action
  logits, three connectome steps. Value function: separate 10-64-1 MLP.
- PPO: seed 42, 8 workers, horizon 64, 100 updates, four optimization epochs,
  minibatch 256, Adam 3e-4, gamma .99, GAE .95, clip .2, entropy coefficient
  .01, value coefficient .5, gradient clipping .5.
- Record every rollout decision and full connectome activity, outcome, reward
  components and exact policy version. Optimization forwards are identified in
  update diagnostics rather than mixed with environment decisions.
- Audit every recorded transition by replaying the environment and policy.
- Evaluate versions 0,10,...,100 on the same 256 held-out episode seeds
  9,000,000–9,000,255. Report stochastic and greedy policies. These seeds are
  not used for optimization. The fixed scripted controller is a solvability
  ceiling, not a learned baseline.

Success for this milestone requires the final stochastic policy to improve both
hit rate and return over version 0. Greedy behavior is reported separately.
Results from one training seed do not support architectural conclusions.
