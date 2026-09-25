"""Directional local clearance from a registered Dust II navigation mask."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .schema import PlayerPose


@dataclass(frozen=True)
class ClearanceCalibration:
    map_name: str
    screen_scale_x: float
    screen_scale_y: float
    screen_offset_x: float
    screen_offset_y: float
    overview_units_per_pixel: float
    max_distance_world: float
    max_snap_world: float
    mask_file: str

    def __post_init__(self):
        if self.map_name != 'de_dust2':
            raise ValueError('Clearance calibration must be for de_dust2')
        positive = ('screen_scale_x', 'screen_scale_y',
                    'overview_units_per_pixel', 'max_distance_world',
                    'max_snap_world')
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            object.__setattr__(self, name, value)
        for name in ('screen_offset_x', 'screen_offset_y'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)
        if not self.mask_file:
            raise ValueError('mask_file is required')

    @classmethod
    def from_dict(cls, value):
        if value.get('schema_version', 1) != 1:
            raise ValueError('Unsupported clearance-calibration schema version')
        fields = dict(value)
        fields.pop('schema_version', None)
        return cls(**fields)

    @classmethod
    def from_json(cls, path):
        path = Path(path)
        calibration = cls.from_dict(json.loads(path.read_text(encoding='utf-8')))
        mask_path = Path(calibration.mask_file)
        if not mask_path.is_absolute():
            mask_path = path.parent / mask_path
        return calibration, mask_path


class Dust2ClearanceEstimator:
    """Ray-cast forward/back/left/right distance on a NAV-derived mask."""

    def __init__(self, calibration: ClearanceCalibration, mask):
        self.calibration = calibration
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask)
        self.mask = np.asarray(mask.convert('L') if isinstance(mask, Image.Image)
                               else mask) > 0
        if self.mask.ndim != 2 or min(self.mask.shape) < 32:
            raise ValueError('Walkability mask must be a two-dimensional image')
        _, self.nearest_inside = ndimage.distance_transform_edt(
            ~self.mask, return_indices=True)

    @classmethod
    def from_json(cls, path):
        calibration, mask_path = ClearanceCalibration.from_json(path)
        return cls(calibration, Image.open(mask_path))

    def screen_to_mask(self, x, y):
        calibration = self.calibration
        return ((float(x) - calibration.screen_offset_x) /
                calibration.screen_scale_x,
                (float(y) - calibration.screen_offset_y) /
                calibration.screen_scale_y)

    def _snap_inside(self, x, y):
        ix, iy = int(round(x)), int(round(y))
        height, width = self.mask.shape
        if not (0 <= ix < width and 0 <= iy < height):
            raise ValueError('Player pose falls outside the walkability mask')
        if self.mask[iy, ix]:
            return float(x), float(y), 0.0
        nearest_y = int(self.nearest_inside[0, iy, ix])
        nearest_x = int(self.nearest_inside[1, iy, ix])
        distance = math.hypot(nearest_x - x, nearest_y - y)
        if distance * self.calibration.overview_units_per_pixel > \
                self.calibration.max_snap_world:
            raise ValueError('Player pose is too far from walkable navigation space')
        return float(nearest_x), float(nearest_y), distance

    def _ray_distance(self, origin, screen_direction, step_pixels=0.5):
        calibration = self.calibration
        dx = screen_direction[0] / calibration.screen_scale_x
        dy = screen_direction[1] / calibration.screen_scale_y
        length = math.hypot(dx, dy)
        dx, dy = dx / length, dy / length
        max_pixels = (calibration.max_distance_world /
                      calibration.overview_units_per_pixel)
        height, width = self.mask.shape
        distance = 0.0
        while distance < max_pixels:
            distance = min(distance + step_pixels, max_pixels)
            x = int(round(origin[0] + dx * distance))
            y = int(round(origin[1] + dy * distance))
            if not (0 <= x < width and 0 <= y < height and self.mask[y, x]):
                return max(0.0, distance - step_pixels)
        return max_pixels

    def estimate_with_details(self, pose: PlayerPose):
        x, y = self.screen_to_mask(pose.x, pose.y)
        origin_x, origin_y, snap_pixels = self._snap_inside(x, y)
        angle = math.radians(pose.yaw_degrees)
        forward = (math.cos(angle), math.sin(angle))
        right = (-forward[1], forward[0])
        directions = (forward, (-forward[0], -forward[1]),
                      (-right[0], -right[1]), right)
        distances_pixels = tuple(
            self._ray_distance((origin_x, origin_y), direction)
            for direction in directions)
        max_pixels = (self.calibration.max_distance_world /
                      self.calibration.overview_units_per_pixel)
        clearances = tuple(float(min(1.0, distance / max_pixels))
                           for distance in distances_pixels)
        return {
            'clearances': clearances,
            'distances_world': tuple(
                float(distance * self.calibration.overview_units_per_pixel)
                for distance in distances_pixels),
            'mask_position': (origin_x, origin_y),
            'snap_world': float(
                snap_pixels * self.calibration.overview_units_per_pixel),
        }

    def estimate(self, pose: PlayerPose):
        return self.estimate_with_details(pose)['clearances']
