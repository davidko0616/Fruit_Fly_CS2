"""Run a continuous read-only CS2 perception and policy shadow on loopback."""
import argparse
from collections import Counter
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.clearance import Dust2ClearanceEstimator
from cs2_bridge.detector import build_player_ssdlite
from cs2_bridge.encoder import Dust2ObservationEncoder
from cs2_bridge.radar import (
    TemporalRadarPoseSelector, detect_player_pose_candidates)
from cs2_bridge.radar_map import Dust2RadarMapLocalizer
from cs2_bridge.replay import PolicyRunner
from cs2_bridge.schema import BridgeFrame, Dust2Calibration, PlayerPose
from cs2_bridge.target import VisibleTargetCalibration
from cs2_bridge.waypoint import Dust2PatrolPlanner, Dust2WaypointPlanner
from tools.assemble_cs2_bridge_frames import build_action_mask
from tools.capture_cs2_screen import (
    ScreenGrabber, notify_capture_complete, notify_capture_failed,
    parse_region, wait_for_capture)
from tools.evaluate_toy_combat import load_policy
from tools.serve_cs2_gsi import GSIRecorder, make_handler


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _latency_summary(values):
    if not values:
        return {'mean': None, 'p50': None, 'p95': None, 'maximum': None}
    values = np.asarray(values, dtype=float)
    return {
        'mean': float(values.mean()),
        'p50': float(np.quantile(values, .50)),
        'p95': float(np.quantile(values, .95)),
        'maximum': float(values.max()),
    }


def _load_detector(checkpoint_path, cpu_threads):
    torch.set_num_threads(cpu_threads)
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    class_names = tuple(checkpoint['classes'][1:])
    image_size = int(checkpoint.get('config', {}).get('image_size', 320))
    model = build_player_ssdlite(
        pretrained=False, image_size=image_size,
        foreground_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names, int(checkpoint['epoch'])


class ShadowProcessor:
    """Convert one screen image and causal GSI row into a proposed action."""

    def __init__(self, detector, class_names, calibration,
                 target_calibration, clearance_estimator, policy_runner,
                 radar_map_localizer=None,
                 score_threshold=.15, max_gsi_age_ns=15_000_000_000,
                 radar_search_bounds=(180, 100, 500, 400),
                 movement_threshold=.03, target_confidence=.15,
                 fire_confidence=.15, fire_half_angle_degrees=11.25,
                 execution='live_shadow_read_only', allow_fire_actions=True):
        self.detector = detector
        self.class_names = tuple(class_names)
        self.calibration = calibration
        self.target_calibration = target_calibration
        self.clearance_estimator = clearance_estimator
        self.policy_runner = policy_runner
        self.radar_map_localizer = radar_map_localizer
        self.radar_pose_selector = TemporalRadarPoseSelector()
        self.round_id = None
        self.score_threshold = float(score_threshold)
        self.max_gsi_age_ns = int(max_gsi_age_ns)
        self.radar_search_bounds = tuple(radar_search_bounds)
        self.movement_threshold = float(movement_threshold)
        self.target_confidence = float(target_confidence)
        self.fire_confidence = float(fire_confidence)
        self.fire_half_angle_degrees = float(fire_half_angle_degrees)
        self.execution = str(execution)
        self.allow_fire_actions = bool(allow_fire_actions)
        self.sequence = 0

    def process(self, image, timestamp_ns, gsi_row, origin=(0, 0)):
        started = time.perf_counter_ns()
        record = {
            'schema_version': 1,
            'frame_index': None,
            'monotonic_ns': int(timestamp_ns),
            'execution': self.execution,
            'fire_execution_enabled': self.allow_fire_actions,
            'input_emitted': False,
        }
        if gsi_row is None:
            record.update(status='dropped', drop_reason='missing_causal_gsi')
            return record
        gsi_delta_ns = int(gsi_row['received_monotonic_ns']) - int(timestamp_ns)
        record['gsi_delta_ns'] = gsi_delta_ns
        if gsi_delta_ns > 0:
            record.update(status='dropped', drop_reason='future_gsi')
            return record
        if -gsi_delta_ns > self.max_gsi_age_ns:
            record.update(status='dropped', drop_reason='stale_gsi')
            return record
        snapshot = gsi_row.get('snapshot') or {}
        record['gsi_sequence'] = int(gsi_row['sequence'])
        record['gsi_snapshot'] = snapshot
        if (snapshot.get('map_name') != self.calibration.map_name or
                snapshot.get('round_id') is None or
                snapshot.get('player_activity') != 'playing' or
                (snapshot.get('health') is not None and
                 int(snapshot['health']) <= 0)):
            record.update(status='dropped', drop_reason='inactive_gsi_frame')
            return record
        if snapshot['round_id'] != self.round_id:
            self.radar_pose_selector.reset()
            if self.radar_map_localizer is not None:
                self.radar_map_localizer.reset()
            self.round_id = snapshot['round_id']

        radar_started = time.perf_counter_ns()
        try:
            radar_candidates = detect_player_pose_candidates(
                image, origin=origin,
                search_bounds=self.radar_search_bounds)
            raw_radar_pose, pose_source = self.radar_pose_selector.select(
                radar_candidates, timestamp_ns)
        except ValueError as error:
            record.update(
                status='dropped', drop_reason='radar_pose_failure',
                detail=str(error))
            return record
        record['radar_ms'] = (time.perf_counter_ns() - radar_started) / 1e6
        record['radar_candidate_count'] = len(radar_candidates)
        record['white_heading_candidate_count'] = sum(
            pose.heading_color == 'white' for pose in radar_candidates)
        record['raw_radar_pose'] = raw_radar_pose.to_dict()
        record['radar_pose_source'] = pose_source
        if self.radar_map_localizer is None:
            radar_pose = PlayerPose(
                raw_radar_pose.x, raw_radar_pose.y,
                raw_radar_pose.yaw_degrees)
        else:
            try:
                radar_pose, map_details = self.radar_map_localizer.localize(
                    raw_radar_pose, image, timestamp_ns, origin)
            except ValueError as error:
                record.update(
                    status='dropped', drop_reason='radar_map_failure',
                    detail=str(error))
                return record
            record['radar_map'] = map_details
        record['radar_pose'] = {
            'x': radar_pose.x, 'y': radar_pose.y,
            'yaw_degrees': radar_pose.yaw_degrees,
        }
        if not (self.calibration.min_x <= radar_pose.x <= self.calibration.max_x and
                self.calibration.min_y <= radar_pose.y <= self.calibration.max_y):
            record.update(
                status='dropped',
                drop_reason='radar_pose_out_of_calibration')
            return record
        try:
            clearance_details = self.clearance_estimator.estimate_with_details(
                radar_pose)
            local_clearances = clearance_details['clearances']
        except ValueError as error:
            record.update(
                status='dropped', drop_reason='clearance_pose_failure',
                detail=str(error))
            return record

        detector_started = time.perf_counter_ns()
        tensor = pil_to_tensor(image).float().div_(255)
        with torch.inference_mode():
            prediction = self.detector([tensor])[0]
        detector_ms = (time.perf_counter_ns() - detector_started) / 1e6
        keep = prediction['scores'] >= self.score_threshold
        boxes = prediction['boxes'][keep].cpu().tolist()
        scores = prediction['scores'][keep].cpu().tolist()
        class_ids = prediction['labels'][keep].cpu().tolist()
        detections = []
        target_candidates = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            class_index = int(class_id) - 1
            name = (self.class_names[class_index]
                    if 0 <= class_index < len(self.class_names) else 'invalid')
            screen_box = [
                float(box[0] + origin[0]), float(box[1] + origin[1]),
                float(box[2] + origin[0]), float(box[3] + origin[1]),
            ]
            detection = {
                'class_id': int(class_id), 'class_name': name,
                'score': float(score), 'screen_box': screen_box,
            }
            if name == 'enemy':
                raw_target = self.target_calibration.target_from_box(
                    screen_box, confidence=float(score))
                target = (raw_target if self.radar_map_localizer is None else
                          self.radar_map_localizer.convert_target(
                              raw_target, raw_radar_pose.yaw_degrees))
                detection['visible_target_raw'] = {
                    'forward': raw_target.forward, 'right': raw_target.right,
                    'confidence': raw_target.confidence,
                }
                detection['visible_target'] = {
                    'forward': target.forward, 'right': target.right,
                    'confidence': target.confidence,
                }
                if float(score) >= self.target_confidence:
                    target_candidates.append(target)
            detections.append(detection)
        target = next(iter(target_candidates), None)
        mask = build_action_mask(
            local_clearances, target, self.movement_threshold,
            self.fire_confidence, self.fire_half_angle_degrees)
        if not self.allow_fire_actions:
            mask = tuple(value if index != 7 else False
                         for index, value in enumerate(mask))
        frame = BridgeFrame(
            sequence=self.sequence, monotonic_ns=timestamp_ns,
            round_id=snapshot['round_id'], map_name=self.calibration.map_name,
            pose=PlayerPose(radar_pose.x, radar_pose.y, radar_pose.yaw_degrees),
            clearances=tuple(local_clearances), target=target,
            fire_cooldown=0.0, action_mask=mask)
        policy_started = time.perf_counter_ns()
        decision = self.policy_runner.decide(frame)
        policy_ms = (time.perf_counter_ns() - policy_started) / 1e6
        self.sequence += 1
        record.update(
            status='accepted', sequence=frame.sequence,
            local_clearances=list(local_clearances),
            clearance_snap_world=clearance_details['snap_world'],
            clearance_mask_position=list(clearance_details['mask_position']),
            detections=detections,
            primary_visible_target=(None if target is None else {
                'forward': target.forward, 'right': target.right,
                'confidence': target.confidence,
            }),
            detector_ms=detector_ms, policy_ms=policy_ms,
            processing_ms=(time.perf_counter_ns() - started) / 1e6,
            decision=decision)
        return record


def run(args, action_executor=None):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    calibration = Dust2Calibration.from_dict(_load_json(args.calibration))
    target_calibration = VisibleTargetCalibration.from_dict(
        _load_json(args.target_calibration))
    clearance_estimator = Dust2ClearanceEstimator.from_json(
        args.clearance_calibration)
    radar_map_localizer = (None if args.radar_map_calibration is None else
                           Dust2RadarMapLocalizer.from_json(
                               args.radar_map_calibration))
    detector, class_names, checkpoint_epoch = _load_detector(
        args.checkpoint, args.cpu_threads)
    model, manifest = load_policy(args.policy_run, args.policy_version)
    if manifest['config'].get('observation_size') != 14:
        raise ValueError('Shadow requires a 14-value navigation policy')
    environment = manifest['config'].get('environment', {})
    planner = Dust2WaypointPlanner.from_json(
        args.waypoint_calibration,
        grid_step=int(environment.get('grid_step', 8)),
        lookahead_cells=int(environment.get('waypoint_lookahead_cells', 6)),
        target_max_snap_world=args.waypoint_target_max_snap_world)
    patrol_planner = (None if not args.enable_patrol else
                      Dust2PatrolPlanner.from_json(
                          args.waypoint_calibration,
                          grid_step=int(environment.get('grid_step', 8)),
                          lookahead_cells=int(environment.get(
                              'waypoint_lookahead_cells', 6)),
                          target_max_snap_world=(
                              args.waypoint_target_max_snap_world),
                          goal_count=args.patrol_goal_count,
                          arrival_cells=args.patrol_arrival_cells))
    execution = ('live_shadow_read_only' if action_executor is None else
                 'live_controller_guarded')
    policy_runner = PolicyRunner(
        Dust2ObservationEncoder(
            calibration, int(args.target_memory_ms * 1_000_000), planner,
            patrol_planner),
        model, mode=args.mode, seed=args.seed,
        execution=execution)
    processor = ShadowProcessor(
        detector, class_names, calibration, target_calibration,
        clearance_estimator, policy_runner, radar_map_localizer,
        score_threshold=args.score_threshold,
        max_gsi_age_ns=int(args.max_gsi_age_ms * 1_000_000),
        radar_search_bounds=args.radar_search_bounds, execution=execution,
        allow_fire_actions=(action_executor is None or
                            action_executor.config.fire_enabled))

    gsi_recorder = GSIRecorder(output / 'gsi.jsonl', args.gsi_token)
    server = ThreadingHTTPServer(
        (args.gsi_host, args.gsi_port), make_handler(gsi_recorder))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    grabber = ScreenGrabber(args.backend)
    counts = Counter()
    action_counts = Counter()
    capture_ms = []
    processing_ms = []
    clearance_snaps = []
    attempted_frames = 0
    accepted_frames = 0
    target_memory_frames = 0
    waypoint_frames = 0
    patrol_frames = 0
    masked_action_violations = 0
    emitted_inputs = 0
    execution_counts = Counter()
    emergency_stop = False
    started = None
    elapsed_seconds = None
    try:
        wait_for_capture(args.delay, args.sound_cues, args.spoken_prompt)
        started = time.monotonic()
        deadline = started
        with (output / 'shadow.jsonl').open(
                'x', encoding='utf-8', newline='\n') as destination:
            frame_index = 0
            while True:
                if (action_executor is not None and
                        action_executor.poll_emergency_stop()):
                    emergency_stop = True
                    break
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                if (args.duration_seconds is not None and
                        time.monotonic() - started >= args.duration_seconds):
                    break
                capture_started = time.perf_counter_ns()
                start_ns = time.monotonic_ns()
                image = grabber.grab(args.region)
                end_ns = time.monotonic_ns()
                current_capture_ms = (time.perf_counter_ns() - capture_started) / 1e6
                midpoint_ns = (start_ns + end_ns) // 2
                gsi = gsi_recorder.latest_before(midpoint_ns)
                origin = ((0, 0) if args.region is None else
                          (args.region[0], args.region[1]))
                record = processor.process(image, midpoint_ns, gsi, origin)
                record['frame_index'] = frame_index
                record['capture_ms'] = current_capture_ms
                record['capture_backend'] = grabber.backend
                record['screen_size'] = [image.width, image.height]
                if action_executor is not None:
                    input_execution = action_executor.execute(record)
                    record['input_execution'] = input_execution
                    record['input_emitted'] = input_execution['input_emitted']
                    emitted_inputs += int(input_execution['input_emitted'])
                    execution_counts[input_execution['reason']] += 1
                    if input_execution['emergency_stop_latched']:
                        emergency_stop = True
                if args.audit_every and frame_index % args.audit_every == 0:
                    audit_dir = output / 'audit_frames'
                    audit_dir.mkdir(exist_ok=True)
                    audit_name = f'{frame_index:07d}.jpg'
                    image.save(audit_dir / audit_name, quality=85)
                    record['audit_image'] = f'audit_frames/{audit_name}'
                destination.write(json.dumps(
                    record, separators=(',', ':')) + '\n')
                destination.flush()
                attempted_frames += 1
                counts[record.get('drop_reason', record['status'])] += 1
                capture_ms.append(current_capture_ms)
                if record['status'] == 'accepted':
                    accepted_frames += 1
                    processing_ms.append(record['processing_ms'])
                    clearance_snaps.append(record['clearance_snap_world'])
                    action_counts[record['decision']['action_name']] += 1
                    target_memory_frames += int(
                        record['decision']['target_memory_in_observation'])
                    waypoint_frames += int(
                        record['decision']['waypoint_planner_active'])
                    patrol_frames += int(record['decision']['patrol_active'])
                    masked_action_violations += int(not record['decision'][
                        'action_mask'][record['decision']['action']])
                frame_index += 1
                if emergency_stop:
                    break
                deadline += 1.0 / args.hz
                time.sleep(max(0.0, deadline - time.monotonic()))
        elapsed_seconds = time.monotonic() - started
    except Exception:
        notify_capture_failed(args.sound_cues)
        raise
    finally:
        if action_executor is not None:
            action_executor.close()
        grabber.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        gsi_recorder.close()

    summary = {
        'schema_version': 1,
        'status': ('emergency_stop' if emergency_stop else 'complete'),
        'execution': execution,
        'input_executor_enabled': (action_executor is not None and
                                   action_executor.enabled),
        'input_safety_config': (None if action_executor is None else
                                action_executor.config.to_dict()),
        'input_emitted': emitted_inputs > 0,
        'emitted_input_actions': emitted_inputs,
        'input_execution_counts': dict(sorted(execution_counts.items())),
        'duration_seconds': (None if elapsed_seconds is None else
                             float(elapsed_seconds)),
        'attempted_frames': attempted_frames,
        'accepted_frames': accepted_frames,
        'gsi_rows': gsi_recorder.sequence,
        'counts': dict(sorted(counts.items())),
        'action_counts': dict(sorted(action_counts.items())),
        'masked_action_violations': masked_action_violations,
        'target_memory_frames': target_memory_frames,
        'waypoint_frames': waypoint_frames,
        'patrol_frames': patrol_frames,
        'capture_latency_ms': _latency_summary(capture_ms),
        'accepted_processing_latency_ms': _latency_summary(processing_ms),
        'clearance_snap_world': _latency_summary(clearance_snaps),
        'clearance_snap_over_30_frames': sum(
            value > 30 for value in clearance_snaps),
        'requested_hz': args.hz,
        'effective_hz': (None if elapsed_seconds is None else
                         attempted_frames / max(elapsed_seconds, 1e-9)),
        'checkpoint_epoch': checkpoint_epoch,
        'policy_run_id': manifest['run_id'],
        'policy_version': args.policy_version,
        'radar_map_calibration': (None if args.radar_map_calibration is None
                                  else str(args.radar_map_calibration)),
        'mode': args.mode,
        'seed': args.seed,
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    notify_capture_complete(args.sound_cues)
    return summary


def build_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--target-calibration', type=Path, required=True)
    parser.add_argument('--clearance-calibration', type=Path, required=True)
    parser.add_argument('--waypoint-calibration', type=Path, required=True)
    parser.add_argument('--radar-map-calibration', type=Path)
    parser.add_argument('--policy-run', type=Path, required=True)
    parser.add_argument('--policy-version', type=int, default=80)
    parser.add_argument('--mode', choices=('greedy', 'stochastic'), default='greedy')
    parser.add_argument('--seed', type=int, default=9200300)
    parser.add_argument('--duration-seconds', type=float, default=60)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--hz', type=float, default=4)
    parser.add_argument('--region', type=parse_region,
                        default=(0, 0, 2560, 1440))
    parser.add_argument('--backend', choices=('auto', 'pillow', 'dxcam'),
                        default='dxcam')
    parser.add_argument('--radar-search-bounds', type=parse_region,
                        default=(180, 100, 500, 400))
    parser.add_argument('--score-threshold', type=float, default=.15)
    parser.add_argument('--target-memory-ms', type=float, default=5000)
    parser.add_argument('--waypoint-target-max-snap-world', type=float, default=90)
    parser.add_argument('--enable-patrol', action='store_true')
    parser.add_argument('--patrol-goal-count', type=int, default=8)
    parser.add_argument('--patrol-arrival-cells', type=int, default=3)
    parser.add_argument('--max-gsi-age-ms', type=float, default=15000)
    parser.add_argument('--cpu-threads', type=int,
                        default=min(6, os.cpu_count() or 1))
    parser.add_argument('--audit-every', type=int, default=0)
    parser.add_argument('--gsi-host', default='127.0.0.1')
    parser.add_argument('--gsi-port', type=int, default=3000)
    parser.add_argument('--gsi-token')
    parser.add_argument('--delay', type=float, default=5)
    parser.add_argument('--sound-cues', action='store_true')
    parser.add_argument('--spoken-prompt', default=(
        'Dust Two live shadow starting. Play normally until the completion sound.'))
    return parser


def validate_args(parser, args):
    if args.gsi_host not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('GSI host must be loopback')
    if (args.duration_seconds is not None and args.duration_seconds <= 0 or
            args.max_frames is not None and args.max_frames <= 0 or
            args.hz <= 0 or args.delay < 0 or args.audit_every < 0 or
            args.cpu_threads <= 0 or args.max_gsi_age_ms < 0):
        parser.error('Duration, frame, rate, delay, and age values are invalid')
    if not 0 <= args.score_threshold <= 1:
        parser.error('--score-threshold must be in [0, 1]')
    if args.patrol_goal_count < 2 or args.patrol_arrival_cells < 0:
        parser.error('Patrol goal count and arrival distance are invalid')


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
