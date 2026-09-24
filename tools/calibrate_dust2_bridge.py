"""Build provisional Dust II map bounds from a fixed-radar walking capture."""
import argparse
import json
import math
from pathlib import Path


def calibrate(input_path, output_path, margin=8.0, minimum_positions=10):
    positions = []
    with Path(input_path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get('radar_pose') is not None:
                    pose = row['radar_pose']
                else:
                    snapshot = row['snapshot']
                    pose = snapshot.get('pose')
                if pose is not None:
                    position = (float(pose['x']), float(pose['y']))
                    if all(math.isfinite(value) for value in position):
                        positions.append(position)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f'Invalid GSI capture at line {line_number}: {error}') from error
    if len(positions) < minimum_positions:
        raise ValueError(f'Need at least {minimum_positions} valid Dust II radar positions')
    xs, ys = zip(*positions)
    raw_width, raw_height = max(xs) - min(xs), max(ys) - min(ys)
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError('Capture must cover movement on both map axes')
    calibration = {
        'map_name': 'de_dust2',
        'min_x': min(xs) - margin,
        'max_x': max(xs) + margin,
        'min_y': min(ys) - margin,
        'max_y': max(ys) + margin,
        'local_distance_scale': max(raw_width + 2 * margin, raw_height + 2 * margin),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output_path}')
    output_path.write_text(json.dumps(calibration, indent=2) + '\n')
    return {'positions': len(positions), 'raw_bounds': {
        'min_x': min(xs), 'max_x': max(xs), 'min_y': min(ys), 'max_y': max(ys)},
        'margin': margin, 'calibration': calibration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--margin', type=float, default=8.0)
    parser.add_argument('--minimum-positions', type=int, default=10)
    args = parser.parse_args()
    if args.margin < 0 or args.minimum_positions < 2:
        raise ValueError('Margin must be nonnegative and minimum positions at least two')
    print(json.dumps(calibrate(args.input, args.output, args.margin,
                               args.minimum_positions), indent=2))


if __name__ == '__main__':
    main()
