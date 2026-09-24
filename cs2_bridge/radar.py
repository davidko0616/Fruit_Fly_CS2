"""Visible-HUD radar localization for Dust II."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class RadarPose:
    x: float
    y: float
    yaw_degrees: float
    confidence: float
    marker_pixels: int
    heading_pixels: int

    def to_dict(self):
        return {
            'x': self.x, 'y': self.y, 'yaw_degrees': self.yaw_degrees,
            'confidence': self.confidence, 'marker_pixels': self.marker_pixels,
            'heading_pixels': self.heading_pixels,
        }


def _components(mask):
    labels, count = ndimage.label(mask)
    for index in range(1, count + 1):
        points = np.argwhere(labels == index)
        if len(points):
            yield points


def detect_player_pose(image, origin=(0, 0)):
    """Find the local player's yellow marker and adjacent white heading arrow.

    Coordinates use image axes: x increases right and y increases down. A fixed,
    non-rotating radar is required for these coordinates to represent map pose.
    """
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    rgb = np.asarray(image.convert('RGB'))
    if rgb.ndim != 3 or rgb.shape[0] < 64 or rgb.shape[1] < 64:
        raise ValueError('Radar image is too small')
    yellow = ((rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 140) & (rgb[:, :, 2] < 120) &
              ((rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int)) > 100))
    candidates = []
    for points in _components(yellow):
        y0, x0 = points.min(0); y1, x1 = points.max(0)
        height, width = y1 - y0 + 1, x1 - x0 + 1
        if 40 <= len(points) <= 1000 and width <= 48 and height <= 48:
            candidates.append(points)
    if not candidates:
        raise ValueError('No compact yellow player marker found')
    marker = max(candidates, key=len)
    marker_y, marker_x = marker.mean(0)

    yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    nearby = (xx - marker_x) ** 2 + (yy - marker_y) ** 2 <= 36 ** 2
    white = (rgb.min(2) > 190) & nearby
    heading_candidates = []
    for points in _components(white):
        if 12 <= len(points) <= 600:
            center = points.mean(0)
            distance = math.hypot(center[1] - marker_x, center[0] - marker_y)
            if 4 <= distance <= 30:
                heading_candidates.append((distance, points))
    if not heading_candidates:
        raise ValueError('No white heading marker found beside player marker')
    _, heading = min(heading_candidates, key=lambda item: item[0])
    heading_y, heading_x = heading.mean(0)
    dx, dy = heading_x - marker_x, heading_y - marker_y
    yaw = math.degrees(math.atan2(dy, dx))
    marker_score = min(1.0, len(marker) / 150)
    heading_score = min(1.0, len(heading) / 40)
    return RadarPose(
        x=float(marker_x + origin[0]), y=float(marker_y + origin[1]),
        yaw_degrees=float(yaw), confidence=float((marker_score + heading_score) / 2),
        marker_pixels=int(len(marker)), heading_pixels=int(len(heading)),
    )
