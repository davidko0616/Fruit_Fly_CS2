# Moving-target curriculum exploration

These seed-42 runs selected the motion schedule and confirmation thresholds. They
are exploratory evidence rather than the independent confirmation.

## Version 1: target moves every six ticks

The actor was transferred from version 100 of the confirmed integrated-navigation
policy. The target followed a deterministic obstacle-aware route and moved one
cell every six ticks. Target motion occurred in 95.3% of 256 held-out episodes,
averaging 4.2 moves, but transferred version 0 already scored 99.6%. Fine-tuning
reached 100%. The schedule validated motion, expanded-state recording, and replay,
but was too easy to demonstrate new tracking learning. All 51,200 decisions
passed replay.

## Version 2: target moves every tick

The same transferred actor faced a one-cell target move after every nonterminal
decision. The target attempts to continue straight and deterministically tries a
right turn, left turn, then reverse when blocked by the grid, an obstacle, or the
agent. All eight agent actions remain available.

On 256 held-out layouts, stochastic hit rate improved from **79.3%** at transferred
version 0 to **91.8%** after 100 PPO updates. Mean return rose from +1.903 to
+2.548 and mean episode length fell from 54.4 to 35.4 decisions. Greedy hit rate
improved from 84.8% to 93.4%. Every evaluated episode contained target motion,
averaging 53.6 target moves initially and 34.5 at the final policy. All 51,200
decisions passed replay and topology/sign invariants remained intact.

The privileged reactive benchmark solved all 256 held-out layouts and at least
99% of a broader 1,000-seed check. It is reported as a benchmark rather than a
proof of universal solvability because rare target cycles can outlast its simple
chasing rule.
