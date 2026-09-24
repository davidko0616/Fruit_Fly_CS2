"""Extract player pose from timestamped fixed-radar screen captures."""
import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.radar import detect_player_pose


def extract(capture_directory, output, origin=(0, 0)):
    capture_directory = Path(capture_directory)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rows, failures = [], []
    for line_number, line in enumerate(
            (capture_directory / 'frames.jsonl').read_text().splitlines(), 1):
        frame = json.loads(line)
        try:
            pose = detect_player_pose(Image.open(capture_directory / frame['file']), origin)
        except ValueError as error:
            failures.append({'frame_id': frame['frame_id'], 'error': str(error)})
            continue
        rows.append({
            'schema_version': 1, 'frame_id': frame['frame_id'],
            'monotonic_ns': frame['capture_midpoint_monotonic_ns'],
            'radar_pose': pose.to_dict(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
    return {'frames': len(rows) + len(failures), 'poses': len(rows),
            'failures': failures, 'pose_rate': len(rows) / (len(rows) + len(failures))
            if rows or failures else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origin', default='0,0')
    args = parser.parse_args()
    origin = tuple(int(part) for part in args.origin.split(','))
    if len(origin) != 2:
        raise ValueError('Origin must be x,y')
    print(json.dumps(extract(args.capture, args.output, origin), indent=2))


if __name__ == '__main__':
    main()
