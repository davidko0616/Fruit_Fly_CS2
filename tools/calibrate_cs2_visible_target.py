"""Fit screen-only enemy range and validate bearing against visible radar."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.radar import detect_player_pose


def _metrics(expected, predicted):
    expected = np.asarray(expected, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - expected
    residual = float(np.sum(errors ** 2))
    total = float(np.sum((expected - expected.mean()) ** 2))
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mae': float(np.mean(np.abs(errors))),
        'max_abs_error': float(np.max(np.abs(errors))),
        'r_squared': None if total == 0 else float(1 - residual / total),
    }


def calibrate(capture, labels_path, markers_path, output,
              radar_search_bounds=(180, 100, 500, 400),
              horizontal_half_fov_degrees=53.13,
              minimum_bearing_range=25.0):
    capture, labels_path = Path(capture), Path(labels_path)
    markers_path, output = Path(markers_path), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    labels = json.loads(labels_path.read_text(encoding='utf-8'))['labels']
    marker_rows = {
        int(row['frame_id']): row
        for row in (json.loads(line) for line in markers_path.read_text(
            encoding='utf-8').splitlines() if line.strip())
    }
    pairs, excluded = [], []
    image_width = image_height = None
    for frame_key, frame_label in labels.items():
        frame_id = int(frame_key)
        enemies = [player for player in frame_label['players']
                   if player.get('team') == 'enemy']
        markers = marker_rows.get(frame_id, {}).get('enemy_markers', [])
        confirmed = [marker for marker in markers
                     if marker['state'] == 'confirmed']
        if len(enemies) != 1 or len(confirmed) != 1:
            excluded.append({'frame_id': frame_id,
                             'reason': 'requires_one_enemy_and_confirmed_marker'})
            continue
        image_path = capture / f'{frame_id:07d}.png'
        with Image.open(image_path) as image:
            image_width, image_height = image.size
            pose = detect_player_pose(
                image, search_bounds=radar_search_bounds)
        enemy, marker = enemies[0], confirmed[0]
        dx, dy = marker['x'] - pose.x, marker['y'] - pose.y
        radar_range = math.hypot(dx, dy)
        radar_angle = math.degrees(math.atan2(dy, dx))
        radar_bearing = ((radar_angle - pose.yaw_degrees + 180) % 360) - 180
        box_height = float(enemy['y2'] - enemy['y1'])
        center_x = float((enemy['x1'] + enemy['x2']) / 2)
        screen_bearing = math.degrees(math.atan(
            ((center_x - image_width / 2) / (image_width / 2)) *
            math.tan(math.radians(horizontal_half_fov_degrees))))
        pairs.append({
            'frame_id': frame_id, 'radar_range': radar_range,
            'box_height': box_height, 'radar_bearing_degrees': radar_bearing,
            'screen_bearing_degrees': screen_bearing,
            'visibility': enemy.get('visibility', 'unknown'),
        })
    if len(pairs) < 10:
        raise ValueError('Need at least ten confirmed visible-target pairs')
    heights = np.asarray([pair['box_height'] for pair in pairs], dtype=float)
    ranges = np.asarray([pair['radar_range'] for pair in pairs], dtype=float)
    design = np.column_stack((1 / heights, np.ones(len(heights))))
    scale, offset = np.linalg.lstsq(design, ranges, rcond=None)[0]
    predicted_ranges = design @ np.asarray((scale, offset))
    bearing_pairs = [pair for pair in pairs
                     if pair['radar_range'] >= minimum_bearing_range]
    radar_bearings = [pair['radar_bearing_degrees'] for pair in bearing_pairs]
    screen_bearings = [pair['screen_bearing_degrees'] for pair in bearing_pairs]
    bearing_metrics = _metrics(radar_bearings, screen_bearings)
    bearing_metrics.pop('r_squared', None)
    bearing_metrics['mean_signed_error'] = float(np.mean(
        np.asarray(screen_bearings) - np.asarray(radar_bearings)))
    result = {
        'schema_version': 1,
        'map_name': 'de_dust2',
        'source': {
            'capture': str(capture), 'labels': str(labels_path),
            'enemy_markers': str(markers_path),
            'confirmed_pairs': len(pairs), 'excluded': excluded,
        },
        'model': {
            'image_width': image_width, 'image_height': image_height,
            'horizontal_half_fov_degrees': horizontal_half_fov_degrees,
            'inverse_height_scale': float(scale),
            'range_offset': float(offset),
            'min_box_height': float(heights.min()),
            'max_box_height': float(heights.max()),
            'min_range': float(ranges.min()), 'max_range': float(ranges.max()),
        },
        'validation': {
            'range': _metrics(ranges, predicted_ranges),
            'bearing': {
                **bearing_metrics,
                'pairs': len(bearing_pairs),
                'minimum_radar_range': minimum_bearing_range,
            },
        },
        'pairs': pairs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--enemy-markers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radar-search-bounds', default='180,100,500,400')
    parser.add_argument('--horizontal-half-fov-degrees', type=float, default=53.13)
    parser.add_argument('--minimum-bearing-range', type=float, default=25.0)
    args = parser.parse_args()
    bounds = tuple(float(part) for part in args.radar_search_bounds.split(','))
    if len(bounds) != 4:
        parser.error('radar-search-bounds must have four values')
    result = calibrate(
        args.capture, args.labels, args.enemy_markers, args.output, bounds,
        args.horizontal_half_fov_degrees, args.minimum_bearing_range)
    print(json.dumps({
        'status': 'complete',
        'confirmed_pairs': result['source']['confirmed_pairs'],
        'excluded_frames': len(result['source']['excluded']),
        'range_validation': result['validation']['range'],
        'bearing_validation': result['validation']['bearing'],
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
