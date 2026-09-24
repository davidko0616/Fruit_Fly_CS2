"""Audit visible-player labels against their immutable capture manifest."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.labels import validate_frame_label


def audit(capture, labels_path):
    capture = Path(capture)
    manifest = [json.loads(line) for line in
                (capture / 'frames.jsonl').read_text(encoding='utf-8').splitlines()
                if line.strip()]
    frames = {str(frame['frame_id']): frame for frame in manifest}
    if len(frames) != len(manifest):
        raise ValueError('Capture manifest has duplicate frame IDs')
    saved = json.loads(Path(labels_path).read_text(encoding='utf-8'))
    if saved.get('schema_version') != 1:
        raise ValueError('Unsupported label schema')
    if saved.get('split') not in ('train', 'validation', 'test'):
        raise ValueError('Label file has an invalid session split')
    labels = saved.get('labels')
    if not isinstance(labels, dict):
        raise ValueError('Label file must contain a labels object')
    teams, visibilities = Counter(), Counter()
    positive_frames = box_count = 0
    for frame_id, value in labels.items():
        frame = frames.get(frame_id)
        if frame is None:
            raise ValueError(f'Label references unknown frame {frame_id}')
        clean = validate_frame_label(value, frame['width'], frame['height'])
        if clean['players']:
            positive_frames += 1
        box_count += len(clean['players'])
        teams.update(box['team'] for box in clean['players'])
        visibilities.update(box['visibility'] for box in clean['players'])
    return {
        'status': 'pass', 'schema_version': 1, 'split': saved['split'],
        'capture_frames': len(manifest), 'labeled_frames': len(labels),
        'positive_frames': positive_frames,
        'negative_frames': len(labels) - positive_frames,
        'unlabeled_frames': len(manifest) - len(labels),
        'player_boxes': box_count, 'teams': dict(sorted(teams.items())),
        'visibilities': dict(sorted(visibilities.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.capture, args.labels), indent=2))


if __name__ == '__main__':
    main()
