"""Parse Valve Game State Integration payloads without opponent-state access."""
from dataclasses import dataclass
import math

from .schema import PlayerPose


def _vector(value, name):
    if isinstance(value, str):
        parts = value.split(',')
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        raise ValueError(f'{name} must be a comma-separated vector')
    if len(parts) < 2:
        raise ValueError(f'{name} must contain at least x and y')
    try:
        result = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} contains a nonnumeric value') from error
    if not all(math.isfinite(part) for part in result):
        raise ValueError(f'{name} contains a non-finite value')
    return result


@dataclass(frozen=True)
class GSISnapshot:
    map_name: str | None
    round_number: int | None
    player_activity: str | None
    health: int | None
    pose: PlayerPose | None
    provider_timestamp: int | None

    @property
    def round_id(self):
        if self.map_name is None or self.round_number is None:
            return None
        return f'{self.map_name}:{self.round_number}'


def parse_gsi_payload(payload):
    """Extract only own-player and map fields used by the bridge."""
    if not isinstance(payload, dict):
        raise ValueError('GSI payload must be an object')
    map_state = payload.get('map') or {}
    player = payload.get('player') or {}
    provider = payload.get('provider') or {}
    pose = None
    if player.get('position') is not None and player.get('forward') is not None:
        position = _vector(player['position'], 'player.position')
        forward = _vector(player['forward'], 'player.forward')
        if math.hypot(forward[0], forward[1]) == 0:
            raise ValueError('player.forward cannot be zero')
        pose = PlayerPose(position[0], position[1],
                          math.degrees(math.atan2(forward[1], forward[0])))
    state = player.get('state') or {}
    return GSISnapshot(
        map_name=map_state.get('name'),
        round_number=(int(map_state['round']) if map_state.get('round') is not None else None),
        player_activity=player.get('activity'),
        health=(int(state['health']) if state.get('health') is not None else None),
        pose=pose,
        provider_timestamp=(int(provider['timestamp'])
                            if provider.get('timestamp') is not None else None),
    )
