"""Export audited visible-player sessions as a YOLO detection dataset."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import shutil

import yaml

from cs2_bridge.labels import validate_frame_label


TEAM_CLASSES = ('enemy', 'friendly', 'unknown')
SPLITS = ('train', 'validation', 'test')


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _safe_name(value):
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('._')
    if not cleaned:
        raise ValueError('Capture directory must have a usable name')
    return cleaned


def _load_session(capture, labels_path, class_mode):
    capture = Path(capture)
    labels_path = Path(labels_path)
    manifest_rows = _read_jsonl(capture / 'frames.jsonl')
    frames = {str(row['frame_id']): row for row in manifest_rows}
    if len(frames) != len(manifest_rows):
        raise ValueError(f'{capture} has duplicate frame IDs')

    saved = json.loads(labels_path.read_text(encoding='utf-8'))
    if saved.get('schema_version') != 1:
        raise ValueError(f'{labels_path} has an unsupported schema')
    split = saved.get('split')
    if split not in SPLITS:
        raise ValueError(f'{labels_path} has an invalid split')
    labels = saved.get('labels')
    if not isinstance(labels, dict):
        raise ValueError(f'{labels_path} must contain a labels object')

    samples = []
    for frame_id in sorted(labels, key=int):
        frame = frames.get(frame_id)
        if frame is None:
            raise ValueError(f'{labels_path} references unknown frame {frame_id}')
        width, height = int(frame['width']), int(frame['height'])
        clean = validate_frame_label(labels[frame_id], width, height)
        image = capture / frame['file']
        if not image.is_file():
            raise FileNotFoundError(image)
        boxes = []
        for box in clean['players']:
            if class_mode == 'team':
                class_id = TEAM_CLASSES.index(box['team'])
            else:
                class_id = 0
            boxes.append({**box, 'class_id': class_id})
        samples.append({
            'capture': capture,
            'labels_path': labels_path,
            'split': split,
            'frame_id': int(frame_id),
            'frame': frame,
            'image': image,
            'width': width,
            'height': height,
            'boxes': boxes,
        })
    return samples


def _yolo_line(box, width, height):
    center_x = (box['x1'] + box['x2']) / (2 * width)
    center_y = (box['y1'] + box['y2']) / (2 * height)
    box_width = (box['x2'] - box['x1']) / width
    box_height = (box['y2'] - box['y1']) / height
    return (f"{box['class_id']} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f}")


def export_dataset(sessions, output, class_mode='team', transfer='hardlink'):
    """Export labeled sessions without mixing capture sessions across splits."""
    if class_mode not in ('team', 'player'):
        raise ValueError('class_mode must be team or player')
    if transfer not in ('hardlink', 'copy'):
        raise ValueError('transfer must be hardlink or copy')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not sessions:
        raise ValueError('At least one capture/labels session is required')

    all_samples = []
    seen_captures = {}
    session_rows = []
    for capture, labels_path in sessions:
        capture = Path(capture)
        labels_path = Path(labels_path)
        samples = _load_session(capture, labels_path, class_mode)
        if not samples:
            raise ValueError(f'{labels_path} contains no labeled frames')
        split = samples[0]['split']
        resolved = capture.resolve()
        previous = seen_captures.get(resolved)
        if previous is not None:
            raise ValueError(f'Capture {capture} was supplied more than once ({previous}, {split})')
        seen_captures[resolved] = split
        session_rows.append({
            'capture': str(capture), 'labels': str(labels_path), 'split': split,
            'labeled_frames': len(samples),
        })
        all_samples.extend(samples)

    class_names = TEAM_CLASSES if class_mode == 'team' else ('player',)
    split_counts = defaultdict(lambda: Counter(frames=0, positive_frames=0,
                                                negative_frames=0, boxes=0))
    actual_transfers = Counter()
    output.mkdir(parents=True)
    try:
        metadata_handles = {}
        for split in sorted({sample['split'] for sample in all_samples}):
            (output / 'images' / split).mkdir(parents=True)
            (output / 'labels' / split).mkdir(parents=True)
            (output / 'metadata').mkdir(parents=True, exist_ok=True)
            metadata_handles[split] = (output / 'metadata' / f'{split}.jsonl').open(
                'w', encoding='utf-8')
        try:
            for sample in all_samples:
                split = sample['split']
                stem = f"{_safe_name(sample['capture'].name)}__{sample['frame_id']:07d}"
                image_name = stem + sample['image'].suffix.lower()
                destination = output / 'images' / split / image_name
                if destination.exists():
                    raise ValueError(f'Duplicate exported image name: {destination.name}')
                used_transfer = transfer
                if transfer == 'hardlink':
                    try:
                        os.link(sample['image'], destination)
                    except OSError:
                        shutil.copy2(sample['image'], destination)
                        used_transfer = 'copy_fallback'
                else:
                    shutil.copy2(sample['image'], destination)
                actual_transfers[used_transfer] += 1

                yolo = '\n'.join(_yolo_line(box, sample['width'], sample['height'])
                                 for box in sample['boxes'])
                if yolo:
                    yolo += '\n'
                (output / 'labels' / split / f'{stem}.txt').write_text(
                    yolo, encoding='utf-8')
                metadata_handles[split].write(json.dumps({
                    'schema_version': 1,
                    'image': f'images/{split}/{image_name}',
                    'label': f'labels/{split}/{stem}.txt',
                    'source_capture': str(sample['capture']),
                    'source_labels': str(sample['labels_path']),
                    'source_frame_id': sample['frame_id'],
                    'source_file': sample['frame']['file'],
                    'width': sample['width'], 'height': sample['height'],
                    'split': split, 'boxes': sample['boxes'],
                }, separators=(',', ':')) + '\n')

                counts = split_counts[split]
                counts['frames'] += 1
                counts['boxes'] += len(sample['boxes'])
                if sample['boxes']:
                    counts['positive_frames'] += 1
                else:
                    counts['negative_frames'] += 1
        finally:
            for handle in metadata_handles.values():
                handle.close()

        dataset_yaml = {
            'path': '.',
            'train': 'images/train',
            'val': 'images/validation',
            'names': {index: name for index, name in enumerate(class_names)},
        }
        if split_counts['test']['frames']:
            dataset_yaml['test'] = 'images/test'
        (output / 'dataset.yaml').write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False), encoding='utf-8')
        summary = {
            'schema_version': 1,
            'format': 'yolo_detection',
            'class_mode': class_mode,
            'class_names': list(class_names),
            'requested_transfer': transfer,
            'actual_transfers': dict(sorted(actual_transfers.items())),
            'splits': {split: dict(split_counts[split]) for split in SPLITS
                       if split_counts[split]['frames']},
            'sessions': session_rows,
        }
        (output / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        return summary
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
