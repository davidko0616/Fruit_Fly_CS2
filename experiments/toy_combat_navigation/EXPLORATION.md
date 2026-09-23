# Navigation curriculum exploration

These local seed-42 runs informed the fixed confirmation protocol and are not
confirmation evidence.

## Version 1: unrestricted actions and weaker hit incentive

Movement was enabled, every episode began occluded, and the observation added
four directional-clearance features. Fire and movement remained available at
all times. After 100 updates, held-out stochastic hit rate rose from 19.5% to
33.2%; greedy stayed at 0%. The policy improved shaped return but usually moved
without committing to a shot.

## Version 2: phase scaffold and stronger terminal reward

Fire was masked while occluded; after line-of-sight acquisition, movement was
masked so the policy could focus on aiming. Hit reward increased from +2 to +3,
miss penalty decreased from -0.15 to -0.1, and aim-progress scale increased from
0.1 to 0.2. Final stochastic hit rate reached 98.4% from 44.5% initially, with
mean return +2.81. All 51,200 decisions passed replay. Greedy remained at 0%:
it acquired line of sight and firing alignment in 160/256 episodes but never
selected fire as the single highest-probability action.

## Version 3: entropy annealing

The same scaffold used a linear entropy schedule from .01 to zero. Final
stochastic hit rate reached 98.0% and mean return +2.72; recent training hit rate
was 100%. Greedy remained at 0%, showing that entropy regularization alone did
not cause the argmax behavior. The confirmation protocol therefore treats the
sampled PPO policy as the controller and keeps greedy evaluation diagnostic.

The privileged shortest-path oracle solves 1,000/1,000 checked layouts, with a
maximum of 26 decisions and no access by the learned policy.
