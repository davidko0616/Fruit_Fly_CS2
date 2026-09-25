"""Convert synchronized Dust II bridge frames to the 14-value policy input."""
from dataclasses import dataclass
import math

import numpy as np

from .schema import BridgeFrame, Dust2Calibration


@dataclass(frozen=True)
class ObservationResult:
    observation: np.ndarray
    target_observation_is_live: bool
    target_memory_in_observation: bool
    has_last_seen_target: bool
    last_seen_age_ns: int | None


class Dust2ObservationEncoder:
    """Visibility-gated encoder matching the toy policy's observation semantics."""

    observation_size = 14

    def __init__(self, calibration: Dust2Calibration,
                 target_memory_timeout_ns=5_000_000_000):
        self.calibration = calibration
        if target_memory_timeout_ns is not None and target_memory_timeout_ns <= 0:
            raise ValueError('Target-memory timeout must be positive or None')
        self.target_memory_timeout_ns = target_memory_timeout_ns
        self.last_seen_world = None
        self.last_seen_ns = None
        self.round_id = None
        self.previous_sequence = None
        self.previous_monotonic_ns = None

    def reset(self, round_id=None):
        self.last_seen_world = None
        self.last_seen_ns = None
        self.round_id = round_id
        self.previous_sequence = None
        self.previous_monotonic_ns = None

    @staticmethod
    def _axes(yaw_degrees):
        angle = math.radians(yaw_degrees)
        forward = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)
        right = np.asarray((-forward[1], forward[0]), dtype=np.float32)
        return forward, right

    def _validate_order(self, frame):
        if self.previous_sequence is not None and frame.sequence <= self.previous_sequence:
            raise ValueError('Bridge-frame sequence must increase strictly')
        if (self.previous_monotonic_ns is not None and
                frame.monotonic_ns <= self.previous_monotonic_ns):
            raise ValueError('Bridge-frame monotonic time must increase strictly')
        self.previous_sequence = frame.sequence
        self.previous_monotonic_ns = frame.monotonic_ns

    def encode(self, frame: BridgeFrame):
        if frame.map_name != self.calibration.map_name:
            raise ValueError('Frame map does not match calibration')
        if self.round_id != frame.round_id:
            self.last_seen_world = None
            self.last_seen_ns = None
            self.round_id = frame.round_id
        self._validate_order(frame)

        width = self.calibration.max_x - self.calibration.min_x
        height = self.calibration.max_y - self.calibration.min_y
        x = (frame.pose.x - self.calibration.min_x) / width
        y = (frame.pose.y - self.calibration.min_y) / height
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError('Player position falls outside the Dust II calibration bounds')

        forward_axis, right_axis = self._axes(frame.pose.yaw_degrees)
        player_world = np.asarray((frame.pose.x, frame.pose.y), dtype=np.float32)
        live = frame.target is not None
        memory_used = False
        if live:
            local_forward, local_right = frame.target.forward, frame.target.right
            self.last_seen_world = (player_world + forward_axis * local_forward +
                                    right_axis * local_right)
            self.last_seen_ns = frame.monotonic_ns
        elif (self.last_seen_ns is not None and
              self.target_memory_timeout_ns is not None and
              frame.monotonic_ns - self.last_seen_ns >
              self.target_memory_timeout_ns):
            self.last_seen_world = None
            self.last_seen_ns = None
            local_forward = local_right = 0.0
        elif self.last_seen_world is not None:
            delta = self.last_seen_world - player_world
            local_forward = float(np.dot(delta, forward_axis))
            local_right = float(np.dot(delta, right_axis))
            memory_used = True
        else:
            local_forward = local_right = 0.0

        distance = math.hypot(local_forward, local_right)
        if not live and not memory_used:
            alignment = 0.0
        else:
            alignment = 1.0 if distance == 0 else local_forward / distance
        scale = self.calibration.local_distance_scale
        observation = np.asarray([
            x, y, forward_axis[0], forward_axis[1],
            local_forward / scale, local_right / scale,
            distance / (math.sqrt(2) * scale), alignment,
            float(live), frame.fire_cooldown, *frame.clearances,
        ], dtype=np.float32)
        if not np.isfinite(observation).all():
            raise ValueError('Encoded observation contains a non-finite value')
        age = (None if self.last_seen_ns is None else
               frame.monotonic_ns - self.last_seen_ns)
        return ObservationResult(
            observation=observation,
            target_observation_is_live=live,
            target_memory_in_observation=memory_used,
            has_last_seen_target=self.last_seen_world is not None,
            last_seen_age_ns=age,
        )
