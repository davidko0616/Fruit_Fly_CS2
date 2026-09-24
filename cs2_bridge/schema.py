"""Versioned data contract for the read-only Dust II bridge."""
from dataclasses import asdict, dataclass
import math


ACTION_COUNT = 8
SCHEMA_VERSION = 1


def _finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


@dataclass(frozen=True)
class PlayerPose:
    x: float
    y: float
    yaw_degrees: float

    def __post_init__(self):
        object.__setattr__(self, 'x', _finite('x', self.x))
        object.__setattr__(self, 'y', _finite('y', self.y))
        object.__setattr__(self, 'yaw_degrees', _finite('yaw_degrees', self.yaw_degrees))


@dataclass(frozen=True)
class VisibleTarget:
    """A screen-visible target expressed in the player's local world-unit axes."""
    forward: float
    right: float
    confidence: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, 'forward', _finite('target forward', self.forward))
        object.__setattr__(self, 'right', _finite('target right', self.right))
        confidence = _finite('target confidence', self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError('Target confidence must be between zero and one')
        object.__setattr__(self, 'confidence', confidence)


@dataclass(frozen=True)
class Dust2Calibration:
    """Map bounds and observation scale measured in the same units as GSI position."""
    map_name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    local_distance_scale: float

    def __post_init__(self):
        if self.map_name != 'de_dust2':
            raise ValueError('The first bridge supports de_dust2 only')
        for name in ('min_x', 'max_x', 'min_y', 'max_y', 'local_distance_scale'):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError('Calibration bounds must have positive width and height')
        if self.local_distance_scale <= 0:
            raise ValueError('Local distance scale must be positive')

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BridgeFrame:
    """One synchronized policy input before encoding.

    Target values may only come from a visible screen detection. Local clearance
    values are normalized to [0, 1] in forward, backward, left, right order.
    """
    sequence: int
    monotonic_ns: int
    round_id: str
    map_name: str
    pose: PlayerPose
    clearances: tuple
    target: VisibleTarget | None = None
    fire_cooldown: float = 0.0
    action_mask: tuple = (True,) * ACTION_COUNT
    source_tick: int | None = None

    def __post_init__(self):
        if int(self.sequence) < 0 or int(self.monotonic_ns) < 0:
            raise ValueError('Sequence and monotonic time must be nonnegative')
        object.__setattr__(self, 'sequence', int(self.sequence))
        object.__setattr__(self, 'monotonic_ns', int(self.monotonic_ns))
        if not self.round_id:
            raise ValueError('round_id is required to reset target memory')
        if self.map_name != 'de_dust2':
            raise ValueError('Bridge frame must be from de_dust2')
        if len(self.clearances) != 4:
            raise ValueError('Exactly four local clearances are required')
        clearances = tuple(_finite('clearance', value) for value in self.clearances)
        if any(value < 0 or value > 1 for value in clearances):
            raise ValueError('Clearances must be normalized to [0, 1]')
        object.__setattr__(self, 'clearances', clearances)
        cooldown = _finite('fire cooldown', self.fire_cooldown)
        if not 0 <= cooldown <= 1:
            raise ValueError('Fire cooldown must be normalized to [0, 1]')
        object.__setattr__(self, 'fire_cooldown', cooldown)
        mask = tuple(bool(value) for value in self.action_mask)
        if len(mask) != ACTION_COUNT or not any(mask):
            raise ValueError('Action mask must contain eight values and allow an action')
        if cooldown > 0 and mask[7]:
            raise ValueError('Fire must be masked while cooldown is nonzero')
        object.__setattr__(self, 'action_mask', mask)
        if self.source_tick is not None:
            object.__setattr__(self, 'source_tick', int(self.source_tick))

    @classmethod
    def from_dict(cls, value):
        if value.get('schema_version', SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError('Unsupported bridge-frame schema version')
        fields = dict(value)
        fields.pop('schema_version', None)
        fields['pose'] = PlayerPose(**fields['pose'])
        if fields.get('target') is not None:
            fields['target'] = VisibleTarget(**fields['target'])
        for name in ('clearances', 'action_mask'):
            if name in fields:
                fields[name] = tuple(fields[name])
        return cls(**fields)

    def to_dict(self):
        value = asdict(self)
        value['schema_version'] = SCHEMA_VERSION
        return value
