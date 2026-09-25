# Dust II map-specific training protocol

## Environment

The first map-specific environment uses the version-matched Dust II NAV mask at
an eight-pixel grid step. Each episode contains one stationary target and one
player. The target begins behind geometry, while its last seen location is
available through the same local forward/right memory fields used by the bridge.
The player must navigate the real Dust II walkable topology, reacquire direct
line of sight, aim, and fire.

Exact start-target route pairs are assigned by a stable hash:

- buckets 0–7: training;
- bucket 8: validation;
- bucket 9: final held-out evaluation.

This assignment makes route pairs disjoint while keeping their map distribution
comparable. The final held-out bucket remains unused during exploration and
model selection.

## Environment acceptance

The privileged pathfinding oracle solved 50 of 50 routes in each split. Mean
episode lengths were 162.0 decisions for training, 161.5 for validation, and
172.2 for held-out routes; the longest was 340 decisions. Split route IDs had
no overlap. The FlyWire CPU smoke run recorded 16 decisions and replayed all
model activity, observations, actions, rewards, and environment transitions
within the existing numerical tolerances.

## Predeclared exploratory run

The first exploratory run used:

- FlyWire architecture and fixed connectome topology;
- seed 84;
- the confirmed integrated-navigation policy version 100 as initialization;
- 20 PPO updates, eight workers, and 64 decisions per worker per update;
- action repeat one, learning rate `3e-4`, four PPO epochs, and minibatches of
  256;
- entropy coefficient annealed from `0.01` to `0.0`;
- training routes only.

Policy version 0 and the final version are compared on the same 16 fixed
validation routes in stochastic and greedy modes. The scripted oracle is
reported alongside them. This run is exploratory and cannot consume the held-out
route bucket. A later confirmation protocol must freeze the chosen settings and
evaluate held-out routes once.

## Exploratory result and correction

The 20-update run recorded and replayed all 10,240 decisions exactly. On the 16
fixed validation routes, stochastic hit rate changed from 6.25% at version 0 to
0% at version 20, and line-of-sight acquisition changed from 6.25% to 0%.
Mean return improved from -15.94 to -12.24, so the primary navigation endpoint
did not improve.

Inspection found that distance-progress reward had been divided by map size,
unlike the established toy-navigation reward, and aim progress was disabled even
while remembered target geometry was present in the observation. This made the
available target direction much weaker as a learning signal. The failed run is
preserved. The corrected second exploration keeps seed, topology, route split,
optimizer, and warm start fixed, restores unscaled per-cell distance progress,
allows aim shaping whenever live or remembered target state is available, and
runs for 40 PPO updates. It used the same 16 validation routes and did not
access the held-out bucket.

The second exploration again reached 0% validation hit rate at version 40;
stochastic line-of-sight acquisition remained 6.25%, and greedy acquisition was
0%. Mean stochastic return improved from -13.39 to -11.18. This rules out the
reward-scale error as the only cause.

The environment still allowed blind fire and movement into blocked cells, while
the real bridge masks both. The third exploration corrects that interface and
uses a local-route curriculum: target displacement is 8–24 grid cells, fire is
available only with live aligned sight, and blocked directional movement is
masked from the policy. It keeps the same seed and warm start, runs 80 PPO
updates, and is evaluated on 16 hash-disjoint validation routes under the same
8–24-cell constraint. Full-map curriculum transfer will proceed only if this
local stage improves validation acquisition or hit rate. The held-out bucket
remains untouched.

The local curriculum recorded and replayed all 40,960 decisions exactly. Its
overall training hit rate was 72.8%. On the fixed validation routes, version 80
reached 75.0% stochastic and 68.8% greedy hit rate, compared with 87.5% and
43.8% for version 0. The predefined 10-update checkpoint scan selected version
30: it reached the best stochastic validation hit rate, 93.75%, and the best
mean return, +2.77.

The full-map transfer is initialized from local-curriculum version 30. It returns
to the original minimum displacement of 24 cells with no maximum, keeps the
corrected rewards and bridge-matched action masks, and runs 80 PPO updates with
the same seed and optimizer settings. Versions at fixed 10-update intervals are
selected on the 16 full-map validation routes. The held-out bucket remains
untouched until settings and checkpoint selection are frozen.

## Full-map transfer result

The full-map run recorded and replayed all 40,960 decisions exactly. Training
completed 87 episodes with a 16.1% hit rate. On the fixed validation routes, the
transferred version-0 policy reached 37.5% stochastic hit and line-of-sight
acquisition. Every trained checkpoint was worse: versions 10 and 20 were best at
31.25%, and version 80 reached 25.0%. The scripted oracle remained at 100%.

The full-map transfer is therefore rejected. The successful result is limited to
the local 8–24-cell curriculum; the held-out route bucket remains unused. The
next exploration should add an intermediate 24–48-cell curriculum or a planner
hierarchy before another unrestricted-map transfer. No live-control claim follows
from the local-route result.

## Intermediate 24–48-cell result

The overlapping curriculum was initialized from local-curriculum version 30 and
kept the same seed, optimizer, bridge-matched masks, and 80-update budget. It
recorded and replayed all 40,960 decisions exactly. Training completed 119
episodes with 53 hits, an overall hit rate of 44.5%.

On the same 16 hash-disjoint validation routes, the transferred version-0 policy
reached 62.5% stochastic hit and line-of-sight acquisition with mean return
+0.44. Versions 10, 20, and 80 also reached 62.5%; version 10 had the best mean
return at +0.56. No trained checkpoint improved the primary stochastic hit or
acquisition rate, and the remaining checkpoints were worse. The 24–48-cell
stage is therefore rejected rather than promoted to full-map transfer. The
held-out bucket remains unused by learned policies.

The next exploration should use a smaller overlapping 16–32-cell expansion.
This tests whether the 24-cell lower bound caused too abrupt a distribution
shift before introducing a map planner or changing the controller interface.

## Overlapping 16–32-cell result

The smaller expansion again started from local-curriculum version 30 and kept
the same seed, optimizer, masks, and 80-update budget. It recorded and replayed
all 40,960 decisions exactly. Training completed 170 episodes with 112 hits, an
overall hit rate of 65.9%.

On the fixed 16-route validation set, the transferred version-0 policy reached
62.5% stochastic hit and line-of-sight acquisition with mean return +0.40.
Version 20 improved both primary rates to 68.75% and mean return to +1.12, the
best stochastic result in the predefined 10-update scan. It is selected for the
next curriculum stage. The held-out bucket remains unused by learned policies.

The next exploration should retry 24–48-cell routes from this version-20
checkpoint. This tests gradual curriculum transfer without changing the
controller, rewards, or action interface.

## Gradual 24–48-cell result

The retry started from the selected 16–32-cell version 20 and kept all other
training settings fixed. It recorded and replayed all 40,960 decisions exactly.
Training completed 122 episodes with 57 hits, an overall hit rate of 46.7%.

The transferred version-0 policy reached 68.75% stochastic validation hit and
line-of-sight acquisition with mean return +1.11. Versions 20 and 70 tied both
primary rates; version 70 had the best mean return at +1.28. Because no trained
checkpoint improved the primary rate, this training stage is rejected under the
same rule used for the earlier 24–48-cell run. The result still shows that the
selected 16–32-cell policy transfers to 24–48-cell routes without losing its
validation hit rate.

The next exploration should measure that selected 16–32-cell checkpoint as the
version-0 baseline on unrestricted routes before deciding whether to add a map
planner. The held-out bucket remains unused by learned policies.

The environment is a navigation abstraction. It does not model recoil, weapon
selection, player acceleration, round economy, teammates, or moving opponents.
Success here establishes map-specific navigation learning, not complete CS2
play.
