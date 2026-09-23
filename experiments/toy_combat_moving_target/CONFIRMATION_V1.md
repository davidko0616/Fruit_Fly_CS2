# First moving-target confirmation result: threshold miss

The seed-61 confirmation followed [the original fixed protocol](PROTOCOL.md) and
completed normally. Final held-out stochastic hit rate was **87.1%**, versus
79.3% at transferred version 0. This is a 7.8-point improvement with mean return
+2.075. Greedy hit rate was 90.6%.

The run failed the predeclared primary thresholds by narrow margins: final hit
rate was below 88%, and improvement was below eight percentage points. Earlier
or greedy checkpoints were not substituted for the fixed version-100 stochastic
endpoint. The result is therefore recorded as a failed confirmation despite
showing useful tracking behavior.

All 51,200 decisions passed independent replay, topology and source signs were
preserved, target motion occurred in every held-out episode, and training
completed without error. The failure concerns performance thresholds only.

Post-confirmation diagnostics are documented in [EXPLORATION.md](EXPLORATION.md).
They identify the entropy bonus as the main source of excess sampling diffusion.
The replacement protocol is separately frozen in [PROTOCOL_V2.md](PROTOCOL_V2.md).
