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

## Invalidated dense-route replay

After freezing the planner and policy, a new practice route captured 600
full-screen frames and synchronized GSI state. Frames 71 through 91 were deleted
because they showed the scoreboard or in-game settings, leaving 579 gameplay
frames with their original IDs and timestamps. The fixed perception pipeline
emitted 358 active-play records. It rejected 157 clearance-pose failures and 64
radar poses outside the existing calibration rather than clipping them.

The resulting 358-record replay is invalid for acceptance. Direct radar review
showed that the single-frame detector switched among several friendly bot
markers from the beginning. The settings interval also changed radar zoom
persistently beginning at frame 92, so the frozen radar-to-NAV calibration does
not apply to those later images. Some incorrect poses happened to land inside
NAV bounds. Deleting
the visible menu frames could not repair those semantic errors.

A corrected strict reanalysis enumerated all radar candidates and rejected all
579 retained images because each contained multiple white-heading friendly
markers. It accepted zero frames, confirming that the earlier 358-row replay was
based on an unsafe single-frame selection rather than a valid player track.

The run remains a diagnostic artifact: it exposed the need to fingerprint radar
geometry, enumerate player-marker candidates, and use temporal identity checks.
Its policy outputs must not be interpreted as a valid dense-route result. The
compact invalidation record is preserved in
[`DENSE_ROUTE_REPLAY_RESULT.json`](DENSE_ROUTE_REPLAY_RESULT.json).

## Corrected dense-route replay

The replacement September 29 capture restored `exec flywire_radar`, removed all
friendly bots, and used five enemy CT bots. A five-frame preflight first verified
one unambiguous local-player marker per image, direct placement inside the NAV
mask without snapping, and causal GSI alignment. The full capture then recorded
600 normal-gameplay frames across Long, A/Short, Mid, tunnels, and B without a
console, settings screen, or Codex overlay in the reviewed samples.

The strict localizer accepted 386 frames. It rejected 75 frames with transient
radar-marker ambiguity and 139 frames outside the 30-world-unit player-to-NAV
correction bound instead of guessing. All 386 accepted frames remained active
Dust II play and became valid `BridgeFrame` rows. They contained 155 live-target
frames and 231 hidden-target frames; the maximum causal GSI age was 10.289
seconds and CPU detector inference averaged 32.98 ms per source frame.

Frozen stochastic policy version 80 produced 386 offline decisions. Target
memory and the six-cell NAV waypoint planner were active on 163 hidden-target
frames. Eight remembered targets were rejected because they could not be mapped
safely, the largest accepted target correction was 67.02 world units, and no
selected action violated its mask. These results validate the corrected dense
read-only shadow path. They do not demonstrate live gameplay competence and do
not enable keyboard or mouse output. Exact counts and hashes are preserved in
[`DENSE_ROUTE_REPLAY_RESULT_V2.json`](DENSE_ROUTE_REPLAY_RESULT_V2.json).
