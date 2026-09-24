# Partial-observability exploration

This exploration determined how to remove through-wall target information while
keeping the next confirmation difficult, solvable, and auditable. These results
were used to fix the confirmation protocol; they are not confirmation results.

## Environment difficulty

Simply hiding live coordinates in the existing five-obstacle environment was too
weak a test. The transferred moving-target policy still hit 94.5% of targets and
only 29.8% of its decisions occurred without line of sight. Increasing obstacle
count while leaving the 9x9 grid and 120-tick limit unchanged produced:

| Obstacles | Transferred hit rate | Occluded decisions |
|---:|---:|---:|
| 8 | 92.6% | 38.5% |
| 12 | 91.8% | 59.0% |
| 16 | 80.5% | 71.1% |
| 20 | 74.6% | 80.4% |

Sixteen obstacles were selected. A privileged scripted controller solved 982 of
1,000 layouts and acquired line of sight in 997, showing that the harder layouts
remain broadly solvable.

## Observation and reward choices

The live target geometry is present only while line of sight is clear. Before the
first sighting, its four observation values are zero. During later occlusion they
encode the last visible target position, and the environment reports the age of
that memory. The current target position remains only in the privileged recorded
state for replay and analysis.

Aim-progress and distance-progress rewards are also zero during occlusion. This
prevents the reward signal from leaking whether a blind action moved toward the
hidden live target. Line-of-sight progress, collision, time, and shot rewards are
unchanged.

## Policy experiments

Both experiments initialized the actor from version 100 of the confirmed
moving-target run and used seed 42, zero entropy, and 100 PPO updates.

The explicit last-seen policy reached 85.9% stochastic hit rate at version 30
and 82.4% at version 100 on 256 fixed held-out episodes. At version 30, 69.1% of
decisions were occluded and a remembered target was available for 44.9% of those
decisions. Applying the same environment seeds and same version-30 weights while
zeroing last-seen geometry during occlusion reduced hit rate to 0%. This is the
main causal check that the controller uses remembered target state.

A second experiment carried the 100-neuron activity vector between decisions.
The previous state was bounded with `tanh`, scaled by a temporal decay of .002,
and injected before the usual three message-passing steps. PPO used recorded
previous states with one-step truncated gradients. The temporal controller
reached 81.6% at version 100, so it did not outperform the simpler last-seen
representation. Its 51,200 recorded state transitions did pass independent
replay.

The confirmation therefore uses explicit last-seen state. Persistent circuit
state remains implemented and recorded for later sequence training, where
backpropagation can span multiple decisions rather than treating every recorded
state as fixed input.
