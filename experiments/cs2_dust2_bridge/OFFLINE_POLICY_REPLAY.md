# First real Dust II FlyWire policy replay

## Scope

This run converts the 20 synchronized validation records into the existing
14-value controller contract and evaluates the confirmed 100-neuron FlyWire
policy entirely offline. It produces proposed actions only. No keyboard, mouse,
or game input is emitted.

The adapter applies these fixed rules:

- forward, backward, left, and right are available only when the corresponding
  normalized NAV clearance is at least 0.03 (18 world units);
- wait and both turn actions remain available;
- fire is available only for a currently visible enemy with confidence at least
  0.15 and bearing within 11.25 degrees of the crosshair;
- passive captures use zero fire cooldown because the bridge executed no shots;
- screen-derived last-seen target memory expires after 5 seconds and resets on
  a round change.

## Result

All 20 perception records became valid `BridgeFrame` rows with no drops. Six
contained a live screen-derived target, 14 hid the target, and only two satisfied
the conservative fire gate. During policy encoding, four hidden frames used a
recent target memory; older target state expired as intended.

Greedy replay through policy version 100 from
`integrated_navigation_flywire_seed_53_confirmation_v1` produced 20 valid,
finite decisions. Every chosen action was allowed by its mask. The action counts
were one fire, two forward, two strafe-left, and 15 turn-left. The single proposed
fire occurred on a live, aligned target. These counts are an integration sanity
check, not evidence of Dust II competence: the policy was trained in the toy
navigation environment and the validation sequence is sparse.

Local artifacts:

- `dust2_validation_bridge_frames_v1.jsonl` and its summary;
- `dust2_validation_flywire_decisions_v1.jsonl`;
- hashes of the source perception file, bridge frames, policy run, policy
  version, and calibration are recorded with the outputs.

The next scientific stage is map-specific Dust II training and held-out route
evaluation. A live practice controller should not be enabled from this replay
alone.

## Planner-enabled frozen-policy replay

After map-specific training and the one-time held-out evaluation, the frozen
stochastic FlyWire version 80 was replayed through the same 20 real Dust II
records with the six-cell NAV waypoint planner enabled. All 20 frames produced
valid offline decisions. Six frames had a live target, four hidden frames used
recent target memory and a collision-free waypoint, and no selected action
violated its mask. The proposed actions were two fire, four forward, six
strafe-left, and eight turn-left. No input was sent to CS2.

The radar-to-screen target estimate in one frame landed 76.10 NAV-world units
outside the walkable mask. The planner therefore uses a recorded, configurable
90-unit maximum correction for remembered target estimates. Player localization
retains the stricter 30-unit calibration limit. This is an integration result,
not a gameplay-performance measurement; the sequence is sparse and was already
used during perception validation. The exact compact result is preserved in
[`WAYPOINT_BRIDGE_REPLAY_RESULT.json`](WAYPOINT_BRIDGE_REPLAY_RESULT.json).
