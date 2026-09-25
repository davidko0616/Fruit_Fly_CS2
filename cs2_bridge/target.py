"""Screen-visible enemy boxes converted to local Dust II target coordinates."""
from dataclasses import dataclass
import math

from .schema import VisibleTarget


@dataclass(frozen=True)
class VisibleTargetCalibration:
    image_width: int
    image_height: int
    horizontal_half_fov_degrees: float
    inverse_height_scale: float
    range_offset: float
    min_box_height: float
    max_box_height: float
    min_range: float
    max_range: float

    @classmethod
    def from_dict(cls, value):
        if value.get('schema_version', 1) != 1:
            raise ValueError('Unsupported visible-target calibration schema')
        fields = value.get('model', value)
        return cls(**{name: fields[name] for name in cls.__dataclass_fields__})

    def __post_init__(self):
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError('Calibration image dimensions must be positive')
        if not 0 < self.horizontal_half_fov_degrees < 90:
            raise ValueError('Horizontal half FOV must be between 0 and 90 degrees')
        if self.inverse_height_scale <= 0:
            raise ValueError('Inverse-height scale must be positive')
        if not 0 < self.min_box_height <= self.max_box_height:
            raise ValueError('Calibrated box-height range is invalid')
        if not 0 <= self.min_range <= self.max_range:
            raise ValueError('Calibrated target range is invalid')

    def target_from_box(self, box, confidence=1.0):
        """Estimate a live target using a visible screen box only.

        The radar is used to create this calibration, never as an opponent input
        here. Values outside the observed height range are clipped to avoid
        extrapolating beyond the controlled capture.
        """
        x1, y1, x2, y2 = (float(value) for value in box)
        if not (0 <= x1 < x2 <= self.image_width and
                0 <= y1 < y2 <= self.image_height):
            raise ValueError('Visible target box must be inside the calibrated image')
        height = min(self.max_box_height, max(self.min_box_height, y2 - y1))
        distance = self.inverse_height_scale / height + self.range_offset
        distance = min(self.max_range, max(self.min_range, distance))
        center_x = (x1 + x2) / 2
        normalized_x = (center_x - self.image_width / 2) / (self.image_width / 2)
        bearing = math.atan(
            normalized_x * math.tan(math.radians(self.horizontal_half_fov_degrees)))
        return VisibleTarget(
            forward=distance * math.cos(bearing),
            right=distance * math.sin(bearing),
            confidence=confidence,
        )
