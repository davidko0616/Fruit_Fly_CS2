"""Replay recorded Dust II bridge frames through a saved FlyWire policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.encoder import Dust2ObservationEncoder
from cs2_bridge.replay import read_frames, replay_frames, write_jsonl
from cs2_bridge.schema import Dust2Calibration
from cs2_bridge.waypoint import Dust2WaypointPlanner
from tools.evaluate_toy_combat import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--policy-run', type=Path, required=True)
    parser.add_argument('--policy-version', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('greedy', 'stochastic'), default='greedy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--target-memory-ms', type=float, default=5000)
    parser.add_argument('--waypoint-calibration', type=Path)
    parser.add_argument('--waypoint-target-max-snap-world', type=float, default=90)
    args = parser.parse_args()
    if args.target_memory_ms <= 0:
        parser.error('--target-memory-ms must be positive')
    if args.waypoint_target_max_snap_world <= 0:
        parser.error('--waypoint-target-max-snap-world must be positive')

    calibration = Dust2Calibration.from_dict(json.loads(args.calibration.read_text()))
    frames = read_frames(args.frames)
    model, manifest = load_policy(args.policy_run, args.policy_version)
    if manifest['config'].get('observation_size') != 14:
        raise ValueError('Bridge requires a policy with the 14-value navigation input')
    environment = manifest['config'].get('environment', {})
    planner_required = bool(environment.get('waypoint_planner_enabled'))
    if planner_required and args.waypoint_calibration is None:
        parser.error('--waypoint-calibration is required by this policy')
    planner = None
    if args.waypoint_calibration is not None:
        planner = Dust2WaypointPlanner.from_json(
            args.waypoint_calibration,
            grid_step=int(environment.get('grid_step', 8)),
            lookahead_cells=int(environment.get('waypoint_lookahead_cells', 6)),
            target_max_snap_world=args.waypoint_target_max_snap_world)
    decisions = replay_frames(
        frames, Dust2ObservationEncoder(
            calibration, int(args.target_memory_ms * 1_000_000), planner), model,
        args.mode, args.seed)
    calibration_sha256 = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    frame_file_sha256 = hashlib.sha256(args.frames.read_bytes()).hexdigest()
    waypoint_calibration_sha256 = (
        None if args.waypoint_calibration is None else
        hashlib.sha256(args.waypoint_calibration.read_bytes()).hexdigest())
    for decision in decisions:
        decision.update(policy_run_id=manifest['run_id'], policy_version=args.policy_version,
                        target_memory_ms=args.target_memory_ms,
                        waypoint_planner_enabled=planner is not None,
                        waypoint_target_max_snap_world=(
                            args.waypoint_target_max_snap_world
                            if planner is not None else None),
                        waypoint_calibration_sha256=waypoint_calibration_sha256,
                        calibration_sha256=calibration_sha256,
                        frame_file_sha256=frame_file_sha256)
    write_jsonl(args.output, decisions)
    print(json.dumps({
        'status': 'complete', 'execution': 'offline_replay_only',
        'frames': len(frames), 'decisions': len(decisions),
        'policy_run_id': manifest['run_id'], 'policy_version': args.policy_version,
        'target_memory_ms': args.target_memory_ms,
        'waypoint_planner_enabled': planner is not None,
        'waypoint_target_max_snap_world': (
            args.waypoint_target_max_snap_world if planner is not None else None),
        'waypoint_calibration_sha256': waypoint_calibration_sha256,
        'calibration_sha256': calibration_sha256, 'frame_file_sha256': frame_file_sha256,
        'output': str(args.output),
    }, indent=2))


if __name__ == '__main__':
    main()
