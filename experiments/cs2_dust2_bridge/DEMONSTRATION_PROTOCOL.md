# Dust II human demonstration protocol

The recorder is read-only. It does not move, turn, click, or fire for the player.
It samples W/A/S/D and the relevant buttons at 100 Hz while Counter-Strike 2 is
the foreground window, and pairs those controls with the existing 8 Hz bridge
observation. Camera-turn labels come from consecutive localized radar headings.
No screenshots or full-resolution video are retained.

Record independent ten-minute sessions. Ten minutes is long enough to contain
useful connected behavior and short enough to repeat after a crash. Keep whole
sessions together when splitting the data so adjacent moments never appear in
both training and validation.

1. **Navigation:** traverse Long, Short/Catwalk, Mid, both tunnels, both bomb
   sites, and both spawns. Include decisive forward movement, corners, stops,
   backing out, and recovery after taking a wrong route.
2. **Combat movement:** use active bots. Include peeking, counter-strafing,
   tracking, retreating, taking cover, and re-engaging. Normal deaths and round
   transitions are allowed; transition rows are excluded automatically.
3. **Mixed validation:** play naturally across the map. Hold this entire session
   out when judging imitation quality.

The stored training target is multi-head: signed forward/backward duty, signed
right/left strafe duty, yaw delta and yaw rate, plus fire, walk, crouch, jump,
and secondary-fire duty. This retains combinations such as forward + right +
turn that the current single-action controller cannot express.

Before the first ten-minute session, run a short recorder validation and inspect
its summary. The gate passes only when accepted bridge frames produce valid
demonstration rows, input coverage is sufficient, the game remains foreground,
turn labels vary when the camera turns, and no images were saved.

Use `tools/record_cs2_demonstration.py` with the same calibration, detector, and
policy arguments as the live shadow. Select `--session-kind navigation`,
`combat`, or `mixed`. The defaults are 600 seconds, 8 Hz perception, 100 Hz
input sampling, sound cues, and `--audit-every 0`.

Each output contains:

- `input.jsonl`: compact timestamp, foreground flag, and control bit mask.
- `shadow.jsonl`: synchronized perception and the exact 14-value observation.
- `demonstration.jsonl`: causal observation/control pairs and quality flags.
- `summary.json`: valid-row counts, invalid reasons, and file hashes.

Demonstrations establish how the player acts within the situations recorded.
They do not by themselves cover unseen routes or tactics, so more ten-minute
sessions can be appended later without changing the format.
