"""Pan-corrected Dust II radar localization in policy grid coordinates."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from .schema import PlayerPose, VisibleTarget


@dataclass(frozen=True)
class RadarMapCalibration:
    map_name: str
    screen_scale_x: float
    screen_scale_y: float
    site_b_overview_x: float
    site_b_overview_y: float
    site_separation_screen_x: float
    site_separation_screen_y: float
    overview_grid_step: float
    overview_grid_offset: float
    anchor_match_radius: float
    anchor_max_age_ms: float

    def __post_init__(self):
        if self.map_name != 'de_dust2':
            raise ValueError('Radar-map calibration must be for de_dust2')
        positive = (
            'screen_scale_x', 'screen_scale_y', 'overview_grid_step',
            'anchor_match_radius', 'anchor_max_age_ms')
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
            object.__setattr__(self, name, value)
        for name in (
                'site_b_overview_x', 'site_b_overview_y',
                'site_separation_screen_x', 'site_separation_screen_y',
                'overview_grid_offset'):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, value):
        if value.get('schema_version', 1) != 1:
            raise ValueError('Unsupported radar-map calibration schema')
        fields = value.get('calibration', value)
        return cls(**{name: fields[name] for name in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


def detect_site_anchor_candidates(
        image, origin=(0, 0), search_bounds=(170, 90, 510, 260)):
    """Detect orange A/B site-letter components in the fixed radar."""
    rgb = np.asarray(image.convert('RGB'))
    min_x, min_y, max_x, max_y = search_bounds
    x0 = max(0, int(math.floor(min_x - origin[0])))
    y0 = max(0, int(math.floor(min_y - origin[1])))
    x1 = min(rgb.shape[1], int(math.ceil(max_x - origin[0])) + 1)
    y1 = min(rgb.shape[0], int(math.ceil(max_y - origin[1])) + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Radar anchor bounds do not overlap the image')
    crop = rgb[y0:y1, x0:x1]
    red, green, blue = (crop[:, :, index].astype(int) for index in range(3))
    orange = ((red > 115) & (green > 75) & (green < 190) & (blue < 90) &
              (red > green * 1.1) & ((green - blue) > 30))
    orange = ndimage.binary_dilation(orange, iterations=1)
    labels, count = ndimage.label(orange)
    candidates = []
    for index in range(1, count + 1):
        yy, xx = np.where(labels == index)
        if not len(xx):
            continue
        width = int(xx.max() - xx.min() + 1)
        height = int(yy.max() - yy.min() + 1)
        if 10 <= width <= 35 and 10 <= height <= 35 and len(xx) >= 50:
            candidates.append((
                float(xx.mean() + origin[0] + x0),
                float(yy.mean() + origin[1] + y0)))
    return tuple(candidates)


class Dust2RadarMapLocalizer:
    """Remove HUD radar pan and return pose in the training grid system."""

    def __init__(self, calibration: RadarMapCalibration):
        self.calibration = calibration
        self.site_b_screen = None
        self.anchor_ns = None

    @classmethod
    def from_json(cls, path):
        return cls(RadarMapCalibration.from_json(path))

    def reset(self):
        self.site_b_screen = None
        self.anchor_ns = None

    def _select_site_b(self, candidates, timestamp_ns):
        calibration = self.calibration
        expected_dx = calibration.site_separation_screen_x
        expected_dy = calibration.site_separation_screen_y
        pairs = []
        for site_b in candidates:
            for site_a in candidates:
                dx = site_a[0] - site_b[0]
                dy = site_a[1] - site_b[1]
                error = math.hypot(dx - expected_dx, dy - expected_dy)
                if error <= calibration.anchor_match_radius:
                    pairs.append((error, site_b))
        if pairs:
            _, site_b = min(pairs, key=lambda item: item[0])
            source = 'site_pair'
        elif self.site_b_screen is not None and candidates:
            expected_a = (
                self.site_b_screen[0] + expected_dx,
                self.site_b_screen[1] + expected_dy)
            matches = []
            for candidate in candidates:
                b_error = math.dist(candidate, self.site_b_screen)
                a_error = math.dist(candidate, expected_a)
                if b_error <= calibration.anchor_match_radius:
                    matches.append((b_error, candidate))
                if a_error <= calibration.anchor_match_radius:
                    matches.append((a_error, (
                        candidate[0] - expected_dx,
                        candidate[1] - expected_dy)))
            if not matches:
                raise ValueError('No radar site anchor matches recent map pan')
            _, site_b = min(matches, key=lambda item: item[0])
            source = 'single_site'
        elif (self.site_b_screen is not None and self.anchor_ns is not None and
              timestamp_ns - self.anchor_ns <=
              calibration.anchor_max_age_ms * 1_000_000):
            return self.site_b_screen, 'recent_site'
        else:
            raise ValueError('No usable radar site anchors')
        self.site_b_screen = tuple(float(value) for value in site_b)
        self.anchor_ns = int(timestamp_ns)
        return self.site_b_screen, source

    def localize(self, raw_pose, image, timestamp_ns, origin=(0, 0)):
        candidates = detect_site_anchor_candidates(image, origin)
        site_b, source = self._select_site_b(candidates, int(timestamp_ns))
        calibration = self.calibration
        overview_x = (calibration.site_b_overview_x +
                      (raw_pose.x - site_b[0]) / calibration.screen_scale_x)
        overview_y = (calibration.site_b_overview_y +
                      (raw_pose.y - site_b[1]) / calibration.screen_scale_y)
        raw_angle = math.radians(raw_pose.yaw_degrees)
        yaw = math.degrees(math.atan2(
            math.sin(raw_angle) / calibration.screen_scale_y,
            math.cos(raw_angle) / calibration.screen_scale_x))
        step = calibration.overview_grid_step
        offset = calibration.overview_grid_offset
        pose = PlayerPose(
            (overview_x - offset) / step,
            (overview_y - offset) / step, yaw)
        return pose, {
            'source': source,
            'candidate_count': len(candidates),
            'site_b_screen': list(site_b),
            'overview_position': [overview_x, overview_y],
        }

    def convert_target(self, target, raw_yaw_degrees):
        """Convert a screen-radar-scaled local vector to training-grid units."""
        if target is None:
            return None
        calibration = self.calibration
        angle = math.radians(raw_yaw_degrees)
        forward = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
        right = np.asarray((-forward[1], forward[0]), dtype=float)
        screen_delta = (forward * target.forward + right * target.right)
        overview_delta = np.asarray((
            screen_delta[0] / calibration.screen_scale_x,
            screen_delta[1] / calibration.screen_scale_y))
        corrected_angle = math.atan2(
            math.sin(angle) / calibration.screen_scale_y,
            math.cos(angle) / calibration.screen_scale_x)
        corrected_forward = np.asarray(
            (math.cos(corrected_angle), math.sin(corrected_angle)))
        corrected_right = np.asarray(
            (-corrected_forward[1], corrected_forward[0]))
        step = calibration.overview_grid_step
        return VisibleTarget(
            forward=float(np.dot(overview_delta, corrected_forward) / step),
            right=float(np.dot(overview_delta, corrected_right) / step),
            confidence=target.confidence)
