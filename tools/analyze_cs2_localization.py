"""Audit radar-to-NAV localization on a recorded Dust II capture."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from cs2_bridge.clearance import Dust2ClearanceEstimator
from cs2_bridge.radar import (
    TemporalRadarPoseSelector, detect_player_pose_candidates)
from cs2_bridge.radar_map import Dust2RadarMapLocalizer
from cs2_bridge.schema import Dust2Calibration, PlayerPose


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines() if line.strip()]


def _ranges(rows):
    groups = []
    for row in rows:
        frame_id = int(row['frame_id'])
        if groups and groups[-1][-1] + 1 == frame_id:
            groups[-1].append(frame_id)
        else:
            groups.append([frame_id])
    return [[group[0], group[-1]] for group in groups]


def _required_snap_world(estimator, x, y):
    mask_x, mask_y = estimator.screen_to_mask(x, y)
    ix, iy = int(round(mask_x)), int(round(mask_y))
    height, width = estimator.mask.shape
    if not (0 <= ix < width and 0 <= iy < height):
        return None
    nearest_y = int(estimator.nearest_inside[0, iy, ix])
    nearest_x = int(estimator.nearest_inside[1, iy, ix])
    pixels = math.hypot(nearest_x - mask_x, nearest_y - mask_y)
    return float(pixels * estimator.calibration.overview_units_per_pixel)


def analyze(capture, calibration_path, clearance_path, output,
            radar_search_bounds=(180, 100, 500, 400),
            radar_map_calibration_path=None):
    capture, output = Path(capture), Path(output)
    summary_path = output.with_suffix(output.suffix + '.summary.json')
    if output.exists() or summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {summary_path}')
    frames = _read_jsonl(capture / 'frames.jsonl')
    calibration = Dust2Calibration.from_dict(json.loads(
        Path(calibration_path).read_text(encoding='utf-8')))
    estimator = Dust2ClearanceEstimator.from_json(clearance_path)
    radar_map_localizer = (
        None if radar_map_calibration_path is None else
        Dust2RadarMapLocalizer.from_json(radar_map_calibration_path))
    pose_selector = TemporalRadarPoseSelector()
    results = []
    for frame in frames:
        region = frame.get('region', [0, 0, frame['width'], frame['height']])
        origin = (int(region[0]), int(region[1]))
        result = {
            'frame_id': int(frame['frame_id']),
            'file': frame['file'],
            'monotonic_ns': int(frame['capture_midpoint_monotonic_ns']),
        }
        try:
            with Image.open(capture / frame['file']) as image:
                rgb = image.convert('RGB')
                candidates = detect_player_pose_candidates(
                    rgb, origin=origin,
                    search_bounds=radar_search_bounds)
        except ValueError as error:
            result.update(status='radar_pose_failure', reason=str(error))
            results.append(result)
            continue
        result['radar_candidate_count'] = len(candidates)
        result['white_heading_candidate_count'] = sum(
            pose.heading_color == 'white' for pose in candidates)
        try:
            raw_pose, pose_source = pose_selector.select(
                candidates, result['monotonic_ns'])
        except ValueError as error:
            result.update(
                status='ambiguous_radar_pose',
                reason=str(error),
                radar_candidates=[pose.to_dict() for pose in candidates])
            results.append(result)
            continue
        result['raw_radar_pose'] = raw_pose.to_dict()
        result['radar_pose_source'] = pose_source
        if radar_map_localizer is None:
            pose = PlayerPose(
                raw_pose.x, raw_pose.y, raw_pose.yaw_degrees)
        else:
            try:
                pose, map_details = radar_map_localizer.localize(
                    raw_pose, rgb, result['monotonic_ns'], origin)
            except ValueError as error:
                result.update(status='radar_map_failure', reason=str(error))
                results.append(result)
                continue
            result['radar_map'] = map_details
        result['radar_pose'] = {
            'x': pose.x, 'y': pose.y, 'yaw_degrees': pose.yaw_degrees}
        if not (calibration.min_x <= pose.x <= calibration.max_x and
                calibration.min_y <= pose.y <= calibration.max_y):
            result.update(status='radar_pose_out_of_calibration', reason=(
                'Radar pose falls outside recorded rectangular calibration'))
            results.append(result)
            continue
        try:
            details = estimator.estimate_with_details(PlayerPose(
                pose.x, pose.y, pose.yaw_degrees))
        except ValueError as error:
            result.update(
                status='clearance_pose_failure', reason=str(error),
                required_snap_world=_required_snap_world(
                    estimator, pose.x, pose.y))
            results.append(result)
            continue
        result.update(
            status='accepted', snap_world=details['snap_world'],
            mask_position=list(details['mask_position']))
        results.append(result)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as destination:
        for row in results:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
    counts = Counter(row['status'] for row in results)
    accepted_snaps = [row['snap_world'] for row in results
                      if row['status'] == 'accepted']
    summary = {
        'schema_version': 1,
        'frames': len(results),
        'counts': dict(sorted(counts.items())),
        'ranges': {
            status: _ranges([row for row in results if row['status'] == status])
            for status in sorted(counts)
        },
        'clearance_required_snap_world': {
            'maximum': max((row.get('required_snap_world') or 0
                            for row in results), default=0),
            'over_30': sum((row.get('required_snap_world') or 0) > 30
                           for row in results),
            'over_90': sum((row.get('required_snap_world') or 0) > 90
                           for row in results),
        },
        'accepted_snap_world': {
            'mean': (None if not accepted_snaps else
                     float(np.mean(accepted_snaps))),
            'p95': (None if not accepted_snaps else
                    float(np.quantile(accepted_snaps, .95))),
            'maximum': (None if not accepted_snaps else
                        float(max(accepted_snaps))),
            'over_30': sum(value > 30 for value in accepted_snaps),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--clearance-calibration', type=Path, required=True)
    parser.add_argument('--radar-map-calibration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radar-search-bounds', default='180,100,500,400')
    args = parser.parse_args()
    bounds = tuple(float(part) for part in args.radar_search_bounds.split(','))
    if len(bounds) != 4 or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
        parser.error('radar search bounds must be min_x,min_y,max_x,max_y')
    print(json.dumps(analyze(
        args.capture, args.calibration, args.clearance_calibration,
        args.output, bounds, args.radar_map_calibration), indent=2))


if __name__ == '__main__':
    main()
