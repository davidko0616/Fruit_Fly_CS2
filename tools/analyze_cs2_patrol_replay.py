"""Counterfactually replay accepted live-shadow poses with Dust II patrol goals."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.encoder import Dust2ObservationEncoder
from cs2_bridge.replay import PolicyRunner
from cs2_bridge.schema import BridgeFrame, Dust2Calibration, PlayerPose, VisibleTarget
from cs2_bridge.waypoint import Dust2PatrolPlanner, Dust2WaypointPlanner
from tools.evaluate_toy_combat import load_policy


def _load_records(path):
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def _frame(record, sequence):
    target = record.get('primary_visible_target')
    if target is not None:
        target = VisibleTarget(**target)
    pose = record['radar_pose']
    return BridgeFrame(
        sequence=sequence, monotonic_ns=record['monotonic_ns'],
        round_id=record['gsi_snapshot']['round_id'], map_name='de_dust2',
        pose=PlayerPose(pose['x'], pose['y'], pose['yaw_degrees']),
        clearances=tuple(record['local_clearances']), target=target,
        action_mask=tuple(record['decision']['action_mask']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shadow', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--waypoint-calibration', type=Path, required=True)
    parser.add_argument('--policy-run', type=Path, required=True)
    parser.add_argument('--policy-version', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-memory-ms', type=float, default=5000)
    parser.add_argument('--waypoint-target-max-snap-world', type=float, default=90)
    parser.add_argument('--patrol-goal-count', type=int, default=8)
    parser.add_argument('--patrol-arrival-cells', type=int, default=3)
    args = parser.parse_args()

    records = [record for record in _load_records(args.shadow)
               if record.get('status') == 'accepted']
    if not records:
        raise ValueError('Shadow contains no accepted records')
    calibration = Dust2Calibration.from_dict(json.loads(
        args.calibration.read_text(encoding='utf-8')))
    model, manifest = load_policy(args.policy_run, args.policy_version)
    environment = manifest['config'].get('environment', {})
    planner_kwargs = {
        'grid_step': int(environment.get('grid_step', 8)),
        'lookahead_cells': int(environment.get('waypoint_lookahead_cells', 6)),
        'target_max_snap_world': args.waypoint_target_max_snap_world,
    }
    target_planner = Dust2WaypointPlanner.from_json(
        args.waypoint_calibration, **planner_kwargs)
    patrol_planner = Dust2PatrolPlanner.from_json(
        args.waypoint_calibration, **planner_kwargs,
        goal_count=args.patrol_goal_count,
        arrival_cells=args.patrol_arrival_cells)
    runner = PolicyRunner(
        Dust2ObservationEncoder(
            calibration, int(args.target_memory_ms * 1_000_000),
            target_planner, patrol_planner),
        model, mode='greedy', execution='offline_counterfactual_replay')

    decisions = [runner.decide(_frame(record, sequence))
                 for sequence, record in enumerate(records)]
    baseline_actions = Counter(
        record['decision']['action_name'] for record in records)
    patrol_actions = Counter(decision['action_name'] for decision in decisions)
    goal_indices = sorted({decision['patrol_goal_index'] for decision in decisions
                           if decision['patrol_goal_index'] is not None})
    result = {
        'schema_version': 1,
        'status': 'complete',
        'execution': 'offline_counterfactual_replay',
        'source_shadow': str(args.shadow),
        'source_sha256': hashlib.sha256(args.shadow.read_bytes()).hexdigest(),
        'accepted_frames': len(records),
        'policy_run_id': manifest['run_id'],
        'policy_version': args.policy_version,
        'patrol_goal_count': args.patrol_goal_count,
        'patrol_arrival_cells': args.patrol_arrival_cells,
        'patrol_frames': sum(decision['patrol_active'] for decision in decisions),
        'target_memory_frames': sum(
            decision['target_memory_in_observation'] for decision in decisions),
        'waypoint_rejections': sum(
            decision['waypoint_planner_rejection'] is not None
            for decision in decisions),
        'patrol_goal_indices_seen': goal_indices,
        'baseline_action_counts': dict(sorted(baseline_actions.items())),
        'patrol_action_counts': dict(sorted(patrol_actions.items())),
        'behavioral_quality_claimed': False,
        'interpretation': (
            'Counterfactual policy decisions on recorded poses verify wiring only; '
            'a live guarded trial is required to measure closed-loop patrol behavior.'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
