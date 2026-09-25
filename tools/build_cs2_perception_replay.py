"""Build timestamp-synchronized, read-only CS2 perception records."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor

from cs2_bridge.detector import build_player_ssdlite
from cs2_bridge.radar import detect_player_pose
from cs2_bridge.schema import Dust2Calibration
from cs2_bridge.sync import TimestampMatcher
from cs2_bridge.target import VisibleTargetCalibration


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines() if line.strip()]


def _selected_frames(capture, labels=None, start_frame=0, end_frame=None,
                     stride=1):
    rows = _read_jsonl(Path(capture) / 'frames.jsonl')
    selected_ids = None
    if labels:
        saved = json.loads(Path(labels).read_text(encoding='utf-8'))
        selected_ids = {int(frame_id) for frame_id in saved['labels']}
    rows = [row for row in rows
            if int(row['frame_id']) >= start_frame and
            (end_frame is None or int(row['frame_id']) <= end_frame) and
            (int(row['frame_id']) - start_frame) % stride == 0 and
            (selected_ids is None or int(row['frame_id']) in selected_ids)]
    if not rows:
        raise ValueError('No capture frames match the selection')
    timestamps = [int(row['capture_midpoint_monotonic_ns']) for row in rows]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError('Selected capture timestamps must be strictly increasing')
    return rows


def build_replay(capture, checkpoint_path, gsi_path, calibration_path, output,
                 labels=None,
                 start_frame=0, end_frame=None, stride=1,
                 score_threshold=0.15, nms_threshold=0.30,
                 radar_search_bounds=(180, 100, 500, 400),
                 max_gsi_delta_ns=15_000_000_000, batch_size=2,
                 cpu_threads=6, target_calibration_path=None):
    capture, output = Path(capture), Path(output)
    summary_path = output.with_suffix(output.suffix + '.summary.json')
    if output.exists() or summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {summary_path}')
    if not 0 <= score_threshold <= 1 or not 0 <= nms_threshold <= 1:
        raise ValueError('Detector thresholds must be in [0, 1]')
    frames = _selected_frames(
        capture, labels, start_frame, end_frame, stride)
    gsi_rows = _read_jsonl(gsi_path)
    gsi_matcher = TimestampMatcher(gsi_rows, 'received_monotonic_ns')
    calibration = Dust2Calibration.from_dict(json.loads(
        Path(calibration_path).read_text(encoding='utf-8')))
    target_calibration = None
    if target_calibration_path is not None:
        target_calibration = VisibleTargetCalibration.from_dict(json.loads(
            Path(target_calibration_path).read_text(encoding='utf-8')))

    torch.set_num_threads(cpu_threads)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    class_names = tuple(checkpoint['classes'][1:])
    image_size = int(checkpoint.get('config', {}).get('image_size', 320))
    model = build_player_ssdlite(
        pretrained=False, image_size=image_size,
        foreground_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.nms_thresh = nms_threshold
    model.eval()

    output_rows, drops = [], Counter()
    detector_seconds = 0.0
    for offset in range(0, len(frames), batch_size):
        batch = frames[offset:offset + batch_size]
        loaded = []
        for frame in batch:
            with Image.open(capture / frame['file']) as source:
                loaded.append(source.convert('RGB'))
        tensors = [pil_to_tensor(image).float().div_(255) for image in loaded]
        started = time.perf_counter()
        with torch.inference_mode():
            predictions = model(tensors)
        detector_seconds += time.perf_counter() - started

        for frame, image, prediction in zip(batch, loaded, predictions):
            timestamp = int(frame['capture_midpoint_monotonic_ns'])
            gsi, gsi_delta = gsi_matcher.latest(timestamp, max_gsi_delta_ns)
            if gsi is None:
                drops['stale_or_missing_gsi'] += 1
                continue
            region = frame.get('region', [0, 0, frame['width'], frame['height']])
            origin = (int(region[0]), int(region[1]))
            try:
                radar_pose = detect_player_pose(
                    image, origin=origin, search_bounds=radar_search_bounds)
            except ValueError:
                drops['radar_pose_failure'] += 1
                continue
            if not (calibration.min_x <= radar_pose.x <= calibration.max_x and
                    calibration.min_y <= radar_pose.y <= calibration.max_y):
                drops['radar_pose_out_of_calibration'] += 1
                continue

            keep = prediction['scores'] >= score_threshold
            boxes = prediction['boxes'][keep].cpu().tolist()
            scores = prediction['scores'][keep].cpu().tolist()
            class_ids = prediction['labels'][keep].cpu().tolist()
            detections = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                class_index = int(class_id) - 1
                name = (class_names[class_index]
                        if 0 <= class_index < len(class_names) else 'invalid')
                screen_box = [
                    box[0] + origin[0], box[1] + origin[1],
                    box[2] + origin[0], box[3] + origin[1],
                ]
                detection = {
                    'class_id': int(class_id), 'class_name': name,
                    'score': float(score), 'screen_box': screen_box,
                }
                if target_calibration is not None and name == 'enemy':
                    target = target_calibration.target_from_box(
                        screen_box, confidence=float(score))
                    detection['visible_target'] = {
                        'forward': target.forward, 'right': target.right,
                        'confidence': target.confidence,
                    }
                detections.append(detection)
            primary_target = next(
                (detection['visible_target'] for detection in detections
                 if 'visible_target' in detection), None)
            snapshot = gsi.get('snapshot') or {}
            active = (snapshot.get('map_name') == calibration.map_name and
                      snapshot.get('round_id') is not None and
                      snapshot.get('player_activity') == 'playing')
            if not active:
                drops['inactive_gsi_frame'] += 1
            output_rows.append({
                'schema_version': 1,
                'sequence': len(output_rows),
                'monotonic_ns': timestamp,
                'source_frame_id': int(frame['frame_id']),
                'source_file': frame['file'],
                'read_only': True,
                'active_play': active,
                'gsi_delta_ns': int(gsi_delta),
                'gsi_sequence': int(gsi['sequence']),
                'gsi_snapshot': snapshot,
                'radar_pose': radar_pose.to_dict(),
                'detections': detections,
                'primary_visible_target': primary_target,
                'enemy_candidates': sum(
                    detection['class_name'] == 'enemy'
                    for detection in detections),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as destination:
        for row in output_rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
    summary = {
        'schema_version': 1,
        'status': 'complete',
        'read_only': True,
        'selected_frames': len(frames),
        'emitted_frames': len(output_rows),
        'active_play_frames': sum(row['active_play'] for row in output_rows),
        'frames_with_enemy_candidates': sum(
            row['enemy_candidates'] > 0 for row in output_rows),
        'detections': sum(len(row['detections']) for row in output_rows),
        'drops': dict(sorted(drops.items())),
        'max_abs_gsi_delta_ms': (max(
            (abs(row['gsi_delta_ns']) for row in output_rows), default=0) / 1e6),
        'detector_latency_ms_per_frame': (
            1000 * detector_seconds / len(frames)),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'score_threshold': score_threshold,
        'nms_threshold': nms_threshold,
        'class_names': list(class_names),
        'calibration': str(calibration_path),
        'target_calibration': (None if target_calibration_path is None else
                               str(target_calibration_path)),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--gsi', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--target-calibration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--score-threshold', type=float, default=0.15)
    parser.add_argument('--nms-threshold', type=float, default=0.30)
    parser.add_argument('--radar-search-bounds', default='180,100,500,400')
    parser.add_argument('--max-gsi-delta-ms', type=float, default=15000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int,
                        default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    bounds = tuple(float(part) for part in args.radar_search_bounds.split(','))
    if len(bounds) != 4 or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
        parser.error('radar search bounds must be min_x,min_y,max_x,max_y')
    if (args.start_frame < 0 or args.stride <= 0 or args.batch_size <= 0 or
            args.cpu_threads <= 0 or args.max_gsi_delta_ms < 0):
        parser.error('frame, batch, thread, and time values must be valid')
    summary = build_replay(
        args.capture, args.checkpoint, args.gsi, args.calibration, args.output,
        args.labels,
        args.start_frame, args.end_frame, args.stride,
        args.score_threshold, args.nms_threshold, bounds,
        int(args.max_gsi_delta_ms * 1e6), args.batch_size, args.cpu_threads,
        args.target_calibration)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
