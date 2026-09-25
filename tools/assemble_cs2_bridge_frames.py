"""Convert synchronized perception records into validated offline BridgeFrames."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.schema import BridgeFrame, PlayerPose, VisibleTarget


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding='utf-8').splitlines() if line.strip()]


def _action_mask(clearances, target, movement_threshold,
                 fire_confidence, fire_half_angle_degrees):
    movement = [value >= movement_threshold for value in clearances]
    fire_allowed = False
    if target is not None and target.confidence >= fire_confidence:
        distance = math.hypot(target.forward, target.right)
        alignment = 0.0 if distance == 0 else target.forward / distance
        fire_allowed = alignment >= math.cos(math.radians(
            fire_half_angle_degrees))
    return (True, *movement, True, True, fire_allowed)


def assemble(perception_path, output, movement_threshold=0.03,
             target_confidence=0.15, fire_confidence=0.15,
             fire_half_angle_degrees=11.25):
    perception_path, output = Path(perception_path), Path(output)
    summary_path = output.with_suffix(output.suffix + '.summary.json')
    if output.exists() or summary_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {summary_path}')
    if not 0 <= movement_threshold <= 1:
        raise ValueError('Movement threshold must be in [0, 1]')
    if not 0 <= target_confidence <= 1 or not 0 <= fire_confidence <= 1:
        raise ValueError('Confidence thresholds must be in [0, 1]')
    if not 0 <= fire_half_angle_degrees <= 180:
        raise ValueError('Fire half-angle must be in [0, 180]')

    rows, frames, drops = _read_jsonl(perception_path), [], Counter()
    previous_time = None
    for row in rows:
        snapshot = row.get('gsi_snapshot') or {}
        if not row.get('active_play'):
            drops['inactive_play'] += 1
            continue
        if snapshot.get('map_name') != 'de_dust2' or not snapshot.get('round_id'):
            drops['invalid_map_or_round'] += 1
            continue
        monotonic_ns = int(row['monotonic_ns'])
        if previous_time is not None and monotonic_ns <= previous_time:
            raise ValueError('Perception timestamps must increase strictly')
        previous_time = monotonic_ns
        clearances = row.get('local_clearances')
        if clearances is None or len(clearances) != 4:
            drops['missing_clearances'] += 1
            continue
        pose_value = row['radar_pose']
        pose = PlayerPose(pose_value['x'], pose_value['y'],
                          pose_value['yaw_degrees'])
        target = None
        target_value = row.get('primary_visible_target')
        if (target_value is not None and
                float(target_value['confidence']) >= target_confidence):
            target = VisibleTarget(**target_value)
        mask = _action_mask(clearances, target, movement_threshold,
                            fire_confidence, fire_half_angle_degrees)
        frame = BridgeFrame(
            sequence=len(frames), monotonic_ns=monotonic_ns,
            round_id=snapshot['round_id'], map_name='de_dust2', pose=pose,
            clearances=tuple(clearances), target=target,
            fire_cooldown=0.0, action_mask=mask)
        frames.append(frame)
    if not frames:
        raise ValueError('No valid active-play perception records')

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as destination:
        for frame in frames:
            destination.write(json.dumps(frame.to_dict(), separators=(',', ':')) + '\n')
    summary = {
        'schema_version': 1, 'status': 'complete',
        'execution': 'offline_replay_only',
        'input_records': len(rows), 'emitted_frames': len(frames),
        'live_target_frames': sum(frame.target is not None for frame in frames),
        'hidden_target_frames': sum(frame.target is None for frame in frames),
        'fire_allowed_frames': sum(frame.action_mask[7] for frame in frames),
        'movement_threshold': movement_threshold,
        'target_confidence': target_confidence,
        'fire_confidence': fire_confidence,
        'fire_half_angle_degrees': fire_half_angle_degrees,
        'passive_capture_fire_cooldown': 0.0,
        'drops': dict(sorted(drops.items())),
        'input_sha256': hashlib.sha256(perception_path.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perception', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--movement-threshold', type=float, default=0.03)
    parser.add_argument('--target-confidence', type=float, default=0.15)
    parser.add_argument('--fire-confidence', type=float, default=0.15)
    parser.add_argument('--fire-half-angle-degrees', type=float, default=11.25)
    args = parser.parse_args()
    print(json.dumps(assemble(
        args.perception, args.output, args.movement_threshold,
        args.target_confidence, args.fire_confidence,
        args.fire_half_angle_degrees), indent=2))


if __name__ == '__main__':
    main()
