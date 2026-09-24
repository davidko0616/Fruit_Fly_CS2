"""Validated, visible-screen-only labels for CS2 perception captures."""
from dataclasses import asdict, dataclass
import math


TEAMS = frozenset({'enemy', 'friendly', 'unknown'})
VISIBILITIES = frozenset({'full', 'partial'})


@dataclass(frozen=True)
class PlayerBox:
    x1: float
    y1: float
    x2: float
    y2: float
    team: str = 'unknown'
    visibility: str = 'full'

    @classmethod
    def from_dict(cls, value, width, height):
        try:
            box = cls(*(float(value[name]) for name in ('x1', 'y1', 'x2', 'y2')),
                      team=str(value.get('team', 'unknown')),
                      visibility=str(value.get('visibility', 'full')))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid player box: {error}') from error
        coordinates = (box.x1, box.y1, box.x2, box.y2)
        if not all(math.isfinite(number) for number in coordinates):
            raise ValueError('Player-box coordinates must be finite')
        if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
            raise ValueError('Player box must have positive area inside the frame')
        if box.team not in TEAMS:
            raise ValueError(f'Player-box team must be one of {sorted(TEAMS)}')
        if box.visibility not in VISIBILITIES:
            raise ValueError(
                f'Player-box visibility must be one of {sorted(VISIBILITIES)}')
        return box

    def to_dict(self):
        return asdict(self)


def validate_frame_label(value, width, height):
    if not isinstance(value, dict):
        raise ValueError('Frame label must be an object')
    annotated = value.get('annotated')
    if annotated is not True:
        raise ValueError('A saved frame label must set annotated=true')
    players = value.get('players')
    if not isinstance(players, list):
        raise ValueError('players must be a list')
    return {
        'annotated': True,
        'players': [PlayerBox.from_dict(box, width, height).to_dict()
                    for box in players],
    }
