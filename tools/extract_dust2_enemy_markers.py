"""Extract confirmed and last-known enemy cues from visible Dust II radar."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.radar import detect_enemy_markers


def _temporally_supported_markers(rows, max_step_pixels, max_gap_frames):
    filtered = []
    rejected = 0
    for row_index, row in enumerate(rows):
        kept = []
        for marker in row['enemy_markers']:
            supported = False
            for neighbor_index in (row_index - 1, row_index + 1):
                if not 0 <= neighbor_index < len(rows):
                    continue
                neighbor = rows[neighbor_index]
                if abs(neighbor['frame_id'] - row['frame_id']) > max_gap_frames:
                    continue
                for other in neighbor['enemy_markers']:
                    if math.hypot(marker['x'] - other['x'],
                                  marker['y'] - other['y']) <= max_step_pixels:
                        supported = True
                        break
                if supported:
                    break
            if supported:
                kept.append(marker)
            else:
                rejected += 1
        filtered.append({**row, 'enemy_markers': kept})
    return filtered, rejected


def extract(capture_directory, output, origin=(0, 0), search_bounds=None,
            radar_ellipse=None, max_step_pixels=20.0, max_gap_frames=2):
    capture_directory = Path(capture_directory)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rows = []
    raw_counts = Counter()
    manifest = (capture_directory / 'frames.jsonl').read_text().splitlines()
    for line in manifest:
        frame = json.loads(line)
        image = Image.open(capture_directory / frame['file'])
        markers = detect_enemy_markers(image, origin, search_bounds, radar_ellipse)
        for marker in markers:
            raw_counts[marker.state] += 1
        if not markers:
            raw_counts['absent'] += 1
        rows.append({
            'schema_version': 1,
            'frame_id': frame['frame_id'],
            'monotonic_ns': frame['capture_midpoint_monotonic_ns'],
            'enemy_markers': [marker.to_dict() for marker in markers],
        })
    rows, temporal_rejections = _temporally_supported_markers(
        rows, max_step_pixels, max_gap_frames)
    counts = Counter()
    for row in rows:
        if not row['enemy_markers']:
            counts['absent'] += 1
        for marker in row['enemy_markers']:
            counts[marker['state']] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
    return {'frames': len(rows), 'raw_states': dict(sorted(raw_counts.items())),
            'states': dict(sorted(counts.items())),
            'temporal_rejections': temporal_rejections,
            'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origin', default='0,0')
    parser.add_argument('--search-bounds', default='40,50,560,400',
                        help='Global min_x,min_y,max_x,max_y')
    parser.add_argument('--radar-ellipse', default='278,208,235,178',
                        help='Global center_x,center_y,radius_x,radius_y')
    parser.add_argument('--max-step-pixels', type=float, default=20.0)
    parser.add_argument('--max-gap-frames', type=int, default=2)
    args = parser.parse_args()
    origin = tuple(int(part) for part in args.origin.split(','))
    bounds = tuple(float(part) for part in args.search_bounds.split(','))
    ellipse = tuple(float(part) for part in args.radar_ellipse.split(','))
    if len(origin) != 2 or len(bounds) != 4 or len(ellipse) != 4:
        parser.error('origin needs 2, search-bounds 4, and radar-ellipse 4 values')
    if args.max_step_pixels < 0 or args.max_gap_frames < 1:
        parser.error('temporal filter values must be nonnegative and positive')
    print(json.dumps(extract(
        args.capture, args.output, origin, bounds, ellipse,
        args.max_step_pixels, args.max_gap_frames), indent=2))


if __name__ == '__main__':
    main()
