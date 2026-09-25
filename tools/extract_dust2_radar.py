"""Extract player pose from timestamped fixed-radar screen captures."""
import argparse
import json
import math
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.radar import detect_player_pose


def _temporally_supported(rows, max_step_pixels, max_gap_frames):
    if max_step_pixels <= 0:
        return rows, []
    kept, rejected = [], []
    for index, row in enumerate(rows):
        pose = row['radar_pose']
        supported = False
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(rows):
                continue
            neighbor = rows[neighbor_index]
            if abs(neighbor['frame_id'] - row['frame_id']) > max_gap_frames:
                continue
            other = neighbor['radar_pose']
            if math.hypot(pose['x'] - other['x'], pose['y'] - other['y']) <= max_step_pixels:
                supported = True
                break
        (kept if supported else rejected).append(row)
    return kept, rejected


def extract(capture_directory, output, origin=(0, 0), search_bounds=None,
            max_step_pixels=20.0, max_gap_frames=2, excluded_frames=()):
    capture_directory = Path(capture_directory)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rows, failures = [], []
    excluded_frames = set(excluded_frames)
    skipped = 0
    for line_number, line in enumerate(
            (capture_directory / 'frames.jsonl').read_text().splitlines(), 1):
        frame = json.loads(line)
        if int(frame['frame_id']) in excluded_frames:
            skipped += 1
            continue
        try:
            pose = detect_player_pose(Image.open(capture_directory / frame['file']),
                                      origin, search_bounds)
        except ValueError as error:
            failures.append({'frame_id': frame['frame_id'], 'error': str(error)})
            continue
        rows.append({
            'schema_version': 1, 'frame_id': frame['frame_id'],
            'monotonic_ns': frame['capture_midpoint_monotonic_ns'],
            'radar_pose': pose.to_dict(),
        })
    raw_pose_count = len(rows)
    rows, temporal_rejections = _temporally_supported(
        rows, max_step_pixels=max_step_pixels, max_gap_frames=max_gap_frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
    frame_count = raw_pose_count + len(failures)
    return {
        'frames': frame_count, 'raw_poses': raw_pose_count, 'poses': len(rows),
        'detection_failures': len(failures),
        'temporal_rejections': len(temporal_rejections),
        'excluded_frames': skipped,
        'failure_examples': failures[:20],
        'raw_pose_rate': raw_pose_count / frame_count if frame_count else None,
        'pose_rate': len(rows) / frame_count if frame_count else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origin', default='0,0')
    parser.add_argument('--search-bounds', help='Global min_x,min_y,max_x,max_y')
    parser.add_argument('--max-step-pixels', type=float, default=20.0)
    parser.add_argument('--max-gap-frames', type=int, default=2)
    parser.add_argument('--exclude-range', action='append', default=[],
                        help='Inclusive frame range START-END; may be repeated')
    args = parser.parse_args()
    origin = tuple(int(part) for part in args.origin.split(','))
    if len(origin) != 2:
        raise ValueError('Origin must be x,y')
    search_bounds = None
    if args.search_bounds:
        search_bounds = tuple(float(part) for part in args.search_bounds.split(','))
        if len(search_bounds) != 4:
            raise ValueError('Search bounds must be min_x,min_y,max_x,max_y')
        if not (search_bounds[0] < search_bounds[2] and
                search_bounds[1] < search_bounds[3]):
            raise ValueError('Search bounds must have positive width and height')
    if args.max_step_pixels < 0 or args.max_gap_frames < 1:
        raise ValueError('Temporal filter values must be nonnegative and positive')
    excluded = set()
    for value in args.exclude_range:
        try:
            start, end = (int(part) for part in value.split('-', 1))
        except ValueError as error:
            raise ValueError('Excluded range must be START-END') from error
        if start < 0 or end < start:
            raise ValueError('Excluded range must be nonnegative and ordered')
        excluded.update(range(start, end + 1))
    print(json.dumps(extract(
        args.capture, args.output, origin, search_bounds,
        args.max_step_pixels, args.max_gap_frames, excluded), indent=2))


if __name__ == '__main__':
    main()
