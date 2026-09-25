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
    heading_color: str

    def to_dict(self):
        return {
            'x': self.x, 'y': self.y, 'yaw_degrees': self.yaw_degrees,
            'confidence': self.confidence, 'marker_pixels': self.marker_pixels,
            'heading_pixels': self.heading_pixels, 'heading_color': self.heading_color,
        }


@dataclass(frozen=True)
class RadarEnemyMarker:
    """An enemy cue exposed by the visible CS2 radar.

    ``confirmed`` is the bright filled diamond shown for a currently spotted
    enemy. ``last_known`` is the muted question mark left at the last observed
    position after visual contact is lost.
    """

    x: float
    y: float
    state: str
    confidence: float
    marker_pixels: int
    width: int
    height: int

    def to_dict(self):
        return {
            'x': self.x, 'y': self.y, 'state': self.state,
            'confidence': self.confidence, 'marker_pixels': self.marker_pixels,
            'width': self.width, 'height': self.height,
        }


def _components(mask):
    labels, count = ndimage.label(mask)
    for index in range(1, count + 1):
        points = np.argwhere(labels == index)
        if len(points):
            yield points


def _bounded_rgb(image, origin, search_bounds):
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    rgb = np.asarray(image.convert('RGB'))
    if rgb.ndim != 3 or rgb.shape[0] < 32 or rgb.shape[1] < 32:
        raise ValueError('Radar image is too small')
    if search_bounds is None:
        return rgb, origin
    min_x, min_y, max_x, max_y = search_bounds
    crop_x0 = max(0, int(math.floor(min_x - origin[0])))
    crop_y0 = max(0, int(math.floor(min_y - origin[1])))
    crop_x1 = min(rgb.shape[1], int(math.ceil(max_x - origin[0])) + 1)
    crop_y1 = min(rgb.shape[0], int(math.ceil(max_y - origin[1])) + 1)
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise ValueError('Radar search bounds do not overlap the image')
    return (rgb[crop_y0:crop_y1, crop_x0:crop_x1],
            (origin[0] + crop_x0, origin[1] + crop_y0))


def detect_enemy_markers(image, origin=(0, 0), search_bounds=None,
                         radar_ellipse=None):
    """Detect confirmed and last-known enemy cues on the visible radar.

    Bomb/objective rectangles and site lettering are rejected by the compact
    diamond/question-mark geometry. Coordinates use full-image axes even when
    ``search_bounds`` limits the radar region.
    """
    rgb, crop_origin = _bounded_rgb(image, origin, search_bounds)
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    saturated = ((red > 140) & ((red - green) > 40) &
                 ((red - blue) > 40) & (green < 175))
    labels, count = ndimage.label(saturated)
    components = []
    for index in range(1, count + 1):
        points = np.argwhere(labels == index)
        if len(points) < 4:
            continue
        y0, x0 = points.min(0); y1, x1 = points.max(0)
        width, height = int(x1 - x0 + 1), int(y1 - y0 + 1)
        colors = rgb[points[:, 0], points[:, 1]]
        bright = ((colors[:, 0] > 210) & (colors[:, 1] < 105) &
                  (colors[:, 2] < 105))
        rows = np.bincount(points[:, 0] - y0, minlength=height)
        components.append({
            'points': points, 'x0': int(x0), 'x1': int(x1),
            'y0': int(y0), 'y1': int(y1), 'width': width, 'height': height,
            'pixels': int(len(points)), 'fill': float(len(points) / (width * height)),
            'bright_ratio': float(bright.mean()),
            'edge_ratio': float((rows[0] + rows[-1]) / (2 * max(rows))),
        })

    def inside_radar(x, y):
        if radar_ellipse is None:
            return True
        center_x, center_y, radius_x, radius_y = radar_ellipse
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError('Radar ellipse radii must be positive')
        return (((x - center_x) / radius_x) ** 2 +
                ((y - center_y) / radius_y) ** 2) <= 1

    markers = []
    used = set()
    # Question marks are narrower than diamonds and have either a detached dot
    # or a narrow lower stem. Detect them first because their colour brightens
    # during the transition from live to last-known state.
    for component_index, component in enumerate(components):
        width, height = component['width'], component['height']
        if not (12 <= width <= 21 and 12 <= height <= 24):
            continue
        if not (0.20 <= component['fill'] <= 0.80):
            continue
        dot_index = None
        main_center_x = (component['x0'] + component['x1']) / 2
        for candidate_index, candidate in enumerate(components):
            if candidate_index == component_index or candidate_index in used:
                continue
            if not (2 <= candidate['width'] <= 8 and
                    2 <= candidate['height'] <= 8 and candidate['pixels'] >= 4):
                continue
            gap = candidate['y0'] - component['y1'] - 1
            dot_center_x = (candidate['x0'] + candidate['x1']) / 2
            if 0 <= gap <= 6 and abs(dot_center_x - main_center_x) <= 5:
                dot_index = candidate_index
                break
        joined_question = height >= 19 and width <= 20
        if dot_index is None and not joined_question:
            continue
        dot = components[dot_index] if dot_index is not None else None
        points = (np.concatenate((component['points'], dot['points']))
                  if dot is not None else component['points'])
        y, x = points.mean(0)
        global_x, global_y = x + crop_origin[0], y + crop_origin[1]
        if not inside_radar(global_x, global_y):
            continue
        if dot is not None:
            width = (max(component['x1'], dot['x1']) -
                     min(component['x0'], dot['x0']) + 1)
            height = (max(component['y1'], dot['y1']) -
                      min(component['y0'], dot['y0']) + 1)
            alignment = max(0.0, 1.0 - abs(dot_center_x - main_center_x) / 5)
        else:
            alignment = 0.75
        confidence = 0.75 + 0.2 * alignment
        markers.append(RadarEnemyMarker(
            x=float(global_x), y=float(global_y),
            state='last_known', confidence=float(confidence),
            marker_pixels=int(len(points)), width=int(width), height=int(height),
        ))
        used.add(component_index)
        if dot_index is not None:
            used.add(dot_index)

    for component_index, component in enumerate(components):
        if component_index in used:
            continue
        width, height = component['width'], component['height']
        if not (22 <= width <= 32 and 12 <= height <= 22):
            continue
        if not (0.35 <= component['fill'] <= 0.75):
            continue
        if component['bright_ratio'] < 0.5 or component['edge_ratio'] > 0.35:
            continue
        points = component['points']
        y, x = points.mean(0)
        global_x, global_y = x + crop_origin[0], y + crop_origin[1]
        if not inside_radar(global_x, global_y):
            continue
        shape_score = min(1.0, (1.0 - component['edge_ratio']) / 0.8)
        confidence = 0.5 * component['bright_ratio'] + 0.5 * shape_score
        markers.append(RadarEnemyMarker(
            x=float(global_x), y=float(global_y),
            state='confirmed', confidence=float(confidence),
            marker_pixels=component['pixels'], width=width, height=height,
        ))
        used.add(component_index)

    return sorted(markers, key=lambda marker: marker.confidence, reverse=True)


def detect_player_pose(image, origin=(0, 0), search_bounds=None):
    """Find the local player's yellow marker and adjacent heading arrow.

    Coordinates use image axes: x increases right and y increases down. A fixed,
    non-rotating radar is required for these coordinates to represent map pose.
    """
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    rgb = np.asarray(image.convert('RGB'))
    if rgb.ndim != 3 or rgb.shape[0] < 64 or rgb.shape[1] < 64:
        raise ValueError('Radar image is too small')
    if search_bounds is not None:
        min_x, min_y, max_x, max_y = search_bounds
        crop_x0 = max(0, int(math.floor(min_x - origin[0])))
        crop_y0 = max(0, int(math.floor(min_y - origin[1])))
        crop_x1 = min(rgb.shape[1], int(math.ceil(max_x - origin[0])) + 1)
        crop_y1 = min(rgb.shape[0], int(math.ceil(max_y - origin[1])) + 1)
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            raise ValueError('Radar search bounds do not overlap the image')
        rgb = rgb[crop_y0:crop_y1, crop_x0:crop_x1]
        origin = (origin[0] + crop_x0, origin[1] + crop_y0)
    yellow = ((rgb[:, :, 0] > 180) & (rgb[:, :, 1] > 140) & (rgb[:, :, 2] < 120) &
              ((rgb[:, :, 0].astype(int) - rgb[:, :, 2].astype(int)) > 100))
    candidates = []
    for points in _components(yellow):
        y0, x0 = points.min(0); y1, x1 = points.max(0)
        height, width = y1 - y0 + 1, x1 - x0 + 1
        if 40 <= len(points) <= 1000 and width <= 48 and height <= 48:
            marker_y, marker_x = points.mean(0)
            global_x, global_y = marker_x + origin[0], marker_y + origin[1]
            if search_bounds is not None:
                min_x, min_y, max_x, max_y = search_bounds
                if not (min_x <= global_x <= max_x and min_y <= global_y <= max_y):
                    continue
            candidates.append(points)
    if not candidates:
        raise ValueError('No compact yellow player marker found')
    yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    colors = (
        ('white', rgb.min(2) > 190),
        ('red', ((rgb[:, :, 0] > 170) &
                 (rgb[:, :, 0] > rgb[:, :, 1].astype(float) * 1.4) &
                 (rgb[:, :, 0] > rgb[:, :, 2].astype(float) * 1.4))),
    )
    matched = []
    for marker in candidates:
        marker_y, marker_x = marker.mean(0)
        nearby = (xx - marker_x) ** 2 + (yy - marker_y) ** 2 <= 36 ** 2
        for color, color_mask in colors:
            heading_candidates = []
            for points in _components(color_mask & nearby):
                if 12 <= len(points) <= 600:
                    center = points.mean(0)
                    distance = math.hypot(center[1] - marker_x, center[0] - marker_y)
                    if 4 <= distance <= 30:
                        heading_candidates.append((distance, points))
            if heading_candidates:
                distance, heading = min(heading_candidates, key=lambda item: item[0])
                matched.append((color == 'white', len(marker) + len(heading),
                                -distance, marker, heading, color))
                break
    if not matched:
        raise ValueError('No white or red heading marker found beside player marker')
    _, _, _, marker, heading, heading_color = max(
        matched, key=lambda item: item[:3])
    marker_y, marker_x = marker.mean(0)
    heading_y, heading_x = heading.mean(0)
    dx, dy = heading_x - marker_x, heading_y - marker_y
    yaw = math.degrees(math.atan2(dy, dx))
    marker_score = min(1.0, len(marker) / 150)
    heading_score = min(1.0, len(heading) / 40)
    return RadarPose(
        x=float(marker_x + origin[0]), y=float(marker_y + origin[1]),
        yaw_degrees=float(yaw), confidence=float((marker_score + heading_score) / 2),
        marker_pixels=int(len(marker)), heading_pixels=int(len(heading)),
        heading_color=heading_color,
    )
