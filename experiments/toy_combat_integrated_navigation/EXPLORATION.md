# Integrated navigation exploration

These seed-42 runs informed the fixed confirmation protocol. They are exploratory
evidence and are not the independent confirmation.

## Version 1: train from scratch with all actions available

This run removed navigation phase masking while retaining the confirmed
environment, observations, rewards, and 100-update PPO schedule. Final held-out
stochastic hit rate reached **64.1%**, up from 19.5% at initialization, and mean
return reached +0.741. Line of sight was acquired in 99.2% of episodes and valid
firing alignment was reached in 94.9%, but the policy did not reliably coordinate
those states with firing. Greedy hit rate remained zero. All 51,200 decisions
passed replay.

## Version 2: transfer the confirmed navigation policy

The actor was initialized exactly from version 100 of the independent seed-47
masked-navigation confirmation, then fine-tuned for 100 updates with all eight
actions exposed. The source run ID, policy version, and weight SHA-256 are stored
in the recording manifest; version 0 reproduces the source actor exactly.

Under the integrated action mask, the transferred version-0 policy scored 80.9%
on 256 held-out layouts. Fine-tuning raised stochastic hit rate to **99.6%**, mean
return from +1.530 to +2.754, and reduced mean episode length from 64.3 to 31.4
decisions. Final line-of-sight acquisition, firing alignment, and firing rates
were each 100%. The final greedy diagnostic reached 69.1%. All 51,200 decisions
passed replay and FlyWire topology/sign invariants remained intact.

The large gap between versions 1 and 2 supports staged curriculum transfer: the
masked task teaches navigation and aiming primitives, while integrated
fine-tuning teaches when to move, stop, and fire with the complete action set.
