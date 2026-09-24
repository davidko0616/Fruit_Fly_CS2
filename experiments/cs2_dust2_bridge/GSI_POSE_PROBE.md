# Live CS2 GSI pose probe

The first live probe ran on September 24, 2026 against the locally installed CS2
build recorded by Steam as build ID `25492732`. The loopback receiver captured
120 GSI payloads, including 11 samples while the player was active on
`de_dust2`. GSI correctly reported the map, round, player activity, health,
money, and weapons. It supplied no `player.position` or `player.forward` in any
active sample, despite the configuration requesting `player_position`.

| Endpoint | Result |
|---|---:|
| Total retained GSI rows | 120 |
| Active `de_dust2` rows | 11 |
| Active rows with own pose | 0 |
| Map/round/activity available | Yes |
| Health and weapons available | Yes |
| Authentication retained | No |
| Opponent positions requested or retained | No |

The local raw capture remains at
`artifacts/cs2_bridge/dust2_gsi_walk_01.jsonl` and is ignored by Git. A diagnostic
screen frame confirmed that the visible radar contains a compact yellow local
player marker and adjacent white heading marker. The implemented detector located
them at `(325.6, 249.3)` in the diagnostic radar crop and measured a heading of
`-87.9°` with full detector confidence.

The bridge therefore uses GSI for map, round, health, and weapon state, and uses
the visible radar for own-player pose. A fixed, non-rotating, non-centered radar
is required so marker pixels remain map coordinates rather than staying at the
center while the map moves underneath them.
