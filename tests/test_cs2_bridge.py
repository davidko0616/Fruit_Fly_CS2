import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from cs2_bridge.encoder import Dust2ObservationEncoder
from cs2_bridge.clearance import ClearanceCalibration, Dust2ClearanceEstimator
from cs2_bridge.detector_dataset import export_dataset
from cs2_bridge.detector import build_player_ssdlite, evaluate_detection_records
from cs2_bridge.gsi import parse_gsi_payload
from cs2_bridge.labels import PlayerBox, validate_frame_label
from cs2_bridge.radar import detect_enemy_markers, detect_player_pose
from cs2_bridge.replay import read_frames, replay_frames, write_jsonl
from cs2_bridge.schema import BridgeFrame, Dust2Calibration, PlayerPose, VisibleTarget
from cs2_bridge.sync import TimestampMatcher
from cs2_bridge.target import VisibleTargetCalibration
from cs2_bridge.waypoint import Dust2WaypointPlanner
from tools.calibrate_dust2_bridge import calibrate
from tools.audit_cs2_labels import audit
from tools.assemble_cs2_bridge_frames import assemble
from tools.extract_dust2_radar import _temporally_supported
from tools.extract_dust2_enemy_markers import _temporally_supported_markers
from tools.label_cs2_frames import LabelSet
from http.server import ThreadingHTTPServer
from tools.serve_cs2_gsi import GSIRecorder, make_handler


FIXTURES = Path(__file__).parent / 'fixtures'


class FixedPolicy(torch.nn.Module):
    def forward(self, observations):
        logits = torch.arange(8, dtype=observations.dtype).repeat(len(observations), 1)
        return logits


class CS2BridgeTests(unittest.TestCase):
    def setUp(self):
        self.calibration = Dust2Calibration.from_dict(json.loads(
            (FIXTURES / 'dust2_calibration_test.json').read_text()))

    def test_visibility_last_seen_memory_and_round_reset(self):
        frames = read_frames(FIXTURES / 'dust2_bridge_frames.jsonl')
        encoder = Dust2ObservationEncoder(self.calibration)
        visible = encoder.encode(frames[0])
        self.assertTrue(visible.target_observation_is_live)
        self.assertFalse(visible.target_memory_in_observation)
        np.testing.assert_allclose(visible.observation[4:6], (0.4, 0.1), atol=1e-7)

        hidden = encoder.encode(frames[1])
        self.assertFalse(hidden.target_observation_is_live)
        self.assertTrue(hidden.target_memory_in_observation)
        self.assertEqual(hidden.last_seen_age_ns, 50_000_000)
        np.testing.assert_allclose(hidden.observation[4:6], (0.39, 0.1), atol=1e-7)

        reset = encoder.encode(frames[2])
        self.assertFalse(reset.has_last_seen_target)
        np.testing.assert_array_equal(reset.observation[4:8], np.zeros(4, dtype=np.float32))

    def test_target_memory_expires_after_configured_horizon(self):
        encoder = Dust2ObservationEncoder(
            self.calibration, target_memory_timeout_ns=100)
        visible = BridgeFrame(0, 1, 'de_dust2:1', 'de_dust2',
                              PlayerPose(1, 1, 0), (1, 1, 1, 1),
                              VisibleTarget(1, 0))
        recent = BridgeFrame(1, 101, 'de_dust2:1', 'de_dust2',
                             PlayerPose(1, 1, 0), (1, 1, 1, 1))
        expired = BridgeFrame(2, 102, 'de_dust2:1', 'de_dust2',
                              PlayerPose(1, 1, 0), (1, 1, 1, 1))
        encoder.encode(visible)
        self.assertTrue(encoder.encode(recent).target_memory_in_observation)
        result = encoder.encode(expired)
        self.assertFalse(result.target_memory_in_observation)
        self.assertFalse(result.has_last_seen_target)
        self.assertIsNone(result.last_seen_age_ns)

    def test_waypoint_planner_replaces_hidden_through_wall_vector(self):
        mask = Image.new('L', (160, 160), 255)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((76, 0, 83, 127), fill=0)
        draw.rectangle((76, 144, 83, 159), fill=0)
        clearance = ClearanceCalibration(
            map_name='de_dust2', screen_scale_x=1, screen_scale_y=1,
            screen_offset_x=0, screen_offset_y=0,
            overview_units_per_pixel=1, max_distance_world=24,
            max_snap_world=30, mask_file='unused.png')
        planner = Dust2WaypointPlanner(
            clearance, mask, grid_step=4, lookahead_cells=6)
        calibration = Dust2Calibration(
            'de_dust2', 0, 160, 0, 160, 160)
        encoder = Dust2ObservationEncoder(
            calibration, target_memory_timeout_ns=1_000, waypoint_planner=planner)
        live = encoder.encode(BridgeFrame(
            0, 1, 'de_dust2:1', 'de_dust2', PlayerPose(20, 80, 0),
            (1, 1, 1, 1), VisibleTarget(100, 0)))
        hidden = encoder.encode(BridgeFrame(
            1, 2, 'de_dust2:1', 'de_dust2', PlayerPose(72, 80, 0),
            (1, 1, 1, 1)))
        self.assertFalse(live.waypoint_planner_active)
        self.assertTrue(hidden.target_memory_in_observation)
        self.assertTrue(hidden.waypoint_planner_active)
        self.assertIsNotNone(hidden.waypoint_world)
        self.assertGreater(hidden.waypoint_path_remaining, 0)
        self.assertNotAlmostEqual(hidden.observation[5], 0.0)
        reset = encoder.encode(BridgeFrame(
            2, 3, 'de_dust2:2', 'de_dust2', PlayerPose(72, 80, 0),
            (1, 1, 1, 1)))
        self.assertFalse(reset.waypoint_planner_active)

    def test_timestamp_matcher_prefers_nearest_and_rejects_stale_rows(self):
        matcher = TimestampMatcher([
            {'time': 100, 'value': 'earlier'},
            {'time': 200, 'value': 'later'},
        ], 'time')
        row, delta = matcher.nearest(160, 50)
        self.assertEqual((row['value'], delta), ('later', 40))
        row, delta = matcher.nearest(150, 50)
        self.assertEqual((row['value'], delta), ('earlier', -50))
        row, delta = matcher.nearest(400, 50)
        self.assertIsNone(row)
        self.assertEqual(delta, -200)
        row, delta = matcher.latest(160, 100)
        self.assertEqual((row['value'], delta), ('earlier', -60))
        row, delta = matcher.latest(90, 100)
        self.assertIsNone(row)
        self.assertIsNone(delta)
        row, delta = matcher.latest(400, 50)
        self.assertIsNone(row)
        self.assertEqual(delta, -200)
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            TimestampMatcher([{'time': 100}, {'time': 100}], 'time')

    def test_encoder_rejects_out_of_order_and_out_of_bounds_frames(self):
        encoder = Dust2ObservationEncoder(self.calibration)
        frame = BridgeFrame(0, 1, 'de_dust2:1', 'de_dust2', PlayerPose(1, 1, 0),
                            (1, 1, 1, 1), VisibleTarget(1, 0))
        encoder.encode(frame)
        with self.assertRaisesRegex(ValueError, 'sequence'):
            encoder.encode(BridgeFrame(0, 2, 'de_dust2:2', 'de_dust2',
                                       PlayerPose(1, 1, 0), (1, 1, 1, 1)))
        encoder.reset()
        outside = BridgeFrame(1, 2, 'de_dust2:1', 'de_dust2', PlayerPose(-1, 1, 0),
                              (1, 1, 1, 1))
        with self.assertRaisesRegex(ValueError, 'outside'):
            encoder.encode(outside)
        encoder.reset()
        with self.assertRaisesRegex(ValueError, 'Fire must be masked'):
            BridgeFrame(3, 4, 'de_dust2:1', 'de_dust2', PlayerPose(1, 1, 0),
                        (1, 1, 1, 1), fire_cooldown=1)

    def test_gsi_parser_extracts_own_pose_without_opponent_fields(self):
        payload = {
            'provider': {'timestamp': 1234},
            'map': {'name': 'de_dust2', 'round': 3},
            'player': {'activity': 'playing', 'position': '10, 20, 30',
                       'forward': '0, 1, 0', 'state': {'health': 87}},
            'allplayers': {'enemy': {'position': '999,999,999'}},
        }
        snapshot = parse_gsi_payload(payload)
        self.assertEqual(snapshot.map_name, 'de_dust2')
        self.assertEqual(snapshot.round_id, 'de_dust2:3')
        self.assertEqual(snapshot.health, 87)
        self.assertAlmostEqual(snapshot.pose.yaw_degrees, 90)
        self.assertEqual((snapshot.pose.x, snapshot.pose.y), (10, 20))

    def test_offline_replay_respects_action_mask_and_refuses_overwrite(self):
        frames = read_frames(FIXTURES / 'dust2_bridge_frames.jsonl')
        decisions = replay_frames(frames, Dust2ObservationEncoder(self.calibration),
                                  FixedPolicy())
        self.assertEqual([row['action'] for row in decisions], [7, 6, 7])
        self.assertTrue(all(row['execution'] == 'offline_replay_only' for row in decisions))
        self.assertTrue(decisions[1]['target_memory_in_observation'])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'decisions.jsonl'
            write_jsonl(output, decisions)
            self.assertEqual(len(output.read_text().splitlines()), 3)
            with self.assertRaises(FileExistsError):
                write_jsonl(output, decisions)

    def test_perception_adapter_gates_walls_and_unaligned_fire(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            perception = root / 'perception.jsonl'
            rows = []
            for index, target in enumerate((
                    {'forward': 100, 'right': 0, 'confidence': .8},
                    {'forward': 0, 'right': 100, 'confidence': .8}, None)):
                rows.append({
                    'monotonic_ns': index + 1, 'active_play': True,
                    'gsi_snapshot': {'map_name': 'de_dust2',
                                     'round_id': 'de_dust2:1'},
                    'radar_pose': {'x': 10 + index, 'y': 20,
                                   'yaw_degrees': 0},
                    'local_clearances': [.02, .04, .5, 1],
                    'primary_visible_target': target,
                })
            perception.write_text(
                '\n'.join(json.dumps(row) for row in rows) + '\n')
            output = root / 'frames.jsonl'
            summary = assemble(perception, output)
            frames = read_frames(output)
            self.assertEqual(summary['emitted_frames'], 3)
            self.assertEqual(summary['fire_allowed_frames'], 1)
            self.assertEqual(frames[0].action_mask,
                             (True, False, True, True, True, True, True, True))
            self.assertFalse(frames[1].action_mask[7])
            self.assertFalse(frames[2].action_mask[7])
            self.assertIsNone(frames[2].target)

    def test_gsi_recorder_drops_auth_and_unrequested_privileged_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'gsi.jsonl'
            recorder = GSIRecorder(output, token='secret')
            payload = {
                'auth': {'token': 'secret'}, 'provider': {'timestamp': 1},
                'map': {'name': 'de_dust2', 'round': 1},
                'player': {'position': '1,2,3', 'forward': '1,0,0'},
                'allplayers': {'enemy': {'position': '99,99,99'}},
            }
            recorder.append(payload); recorder.close()
            row = json.loads(output.read_text())
            self.assertNotIn('auth', row['retained_payload'])
            self.assertNotIn('allplayers', row['retained_payload'])
            self.assertEqual(row['discarded_top_level_fields'], ['allplayers'])

    def test_loopback_gsi_http_endpoint_records_authenticated_post(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'gsi.jsonl'
            recorder = GSIRecorder(output, token='secret')
            server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(recorder))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
                body = json.dumps({'auth': {'token': 'secret'},
                                   'map': {'name': 'de_dust2', 'round': 2},
                                   'player': {'position': '1,2,3', 'forward': '1,0,0'}})
                connection.request('POST', '/', body=body,
                                   headers={'Content-Type': 'application/json'})
                response = connection.getresponse(); response.read()
                self.assertEqual(response.status, 200)
                connection.close()
            finally:
                server.shutdown(); server.server_close(); thread.join(); recorder.close()
            row = json.loads(output.read_text())
            self.assertEqual(row['snapshot']['round_id'], 'de_dust2:2')

    def test_calibration_uses_dust2_positions_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / 'capture.jsonl'
            rows = []
            for index in range(10):
                rows.append({'snapshot': {'map_name': 'de_dust2',
                                           'pose': {'x': index * 10, 'y': index * 20}}})
            capture.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            output = Path(temporary) / 'calibration.json'
            result = calibrate(capture, output, margin=10)
            self.assertEqual(result['positions'], 10)
            saved = Dust2Calibration.from_dict(json.loads(output.read_text()))
            self.assertEqual((saved.min_x, saved.max_x), (-10, 100))
            self.assertEqual((saved.min_y, saved.max_y), (-10, 190))
            with self.assertRaises(FileExistsError):
                calibrate(capture, output, margin=10)

    def test_visible_radar_player_marker_produces_pose_and_heading(self):
        image = Image.new('RGB', (240, 180), (65, 65, 65))
        draw = ImageDraw.Draw(image)
        draw.ellipse((97, 77, 113, 91), fill=(245, 210, 40))
        draw.polygon(((105, 68), (99, 76), (111, 76)), fill=(245, 245, 245))
        draw.rectangle((20, 20, 23, 23), fill=(240, 190, 20))
        pose = detect_player_pose(image, origin=(10, 20))
        self.assertAlmostEqual(pose.x, 115, delta=1)
        self.assertAlmostEqual(pose.y, 104, delta=1)
        self.assertAlmostEqual(pose.yaw_degrees, -90, delta=8)
        self.assertGreater(pose.confidence, 0.8)
        self.assertEqual(pose.heading_color, 'white')

        red = Image.new('RGB', (240, 180), (65, 65, 65))
        draw = ImageDraw.Draw(red)
        draw.ellipse((97, 77, 113, 91), fill=(245, 210, 40))
        draw.polygon(((88, 84), (97, 77), (97, 91)), fill=(235, 35, 35))
        red_pose = detect_player_pose(red)
        self.assertAlmostEqual(abs(red_pose.yaw_degrees), 180, delta=8)
        self.assertEqual(red_pose.heading_color, 'red')

    def test_nav_mask_clearance_follows_heading_and_snaps_small_pose_error(self):
        mask = Image.new('L', (100, 100), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((10, 20, 89, 79), fill=255)
        calibration = ClearanceCalibration(
            map_name='de_dust2', screen_scale_x=2, screen_scale_y=2,
            screen_offset_x=5, screen_offset_y=7,
            overview_units_per_pixel=10, max_distance_world=500,
            max_snap_world=30, mask_file='unused.png')
        estimator = Dust2ClearanceEstimator(calibration, mask)
        pose = PlayerPose(5 + 50 * 2, 7 + 50 * 2, 0)
        forward, backward, left, right = estimator.estimate(pose)
        self.assertAlmostEqual(forward, 39 / 50, delta=.02)
        self.assertAlmostEqual(backward, 40 / 50, delta=.02)
        self.assertAlmostEqual(left, 30 / 50, delta=.02)
        self.assertAlmostEqual(right, 29 / 50, delta=.02)

        snapped = estimator.estimate_with_details(PlayerPose(
            5 + 50 * 2, 7 + 18 * 2, 90))
        self.assertLessEqual(snapped['snap_world'], 30)
        with self.assertRaisesRegex(ValueError, 'too far'):
            estimator.estimate(PlayerPose(5 + 50 * 2, 7 + 5 * 2, 90))
        with self.assertRaisesRegex(ValueError, 'schema version'):
            ClearanceCalibration.from_dict({'schema_version': 2})

    def test_radar_search_bounds_and_temporal_support_reject_false_pose(self):
        image = Image.new('RGB', (240, 180), (65, 65, 65))
        draw = ImageDraw.Draw(image)
        draw.ellipse((97, 77, 113, 91), fill=(245, 210, 40))
        draw.polygon(((105, 68), (99, 76), (111, 76)), fill=(245, 245, 245))
        draw.ellipse((197, 137, 219, 159), fill=(245, 210, 40))
        draw.rectangle((202, 128, 214, 136), fill=(245, 245, 245))
        pose = detect_player_pose(image, search_bounds=(80, 50, 140, 120))
        self.assertAlmostEqual(pose.x, 105, delta=1)

        competing = Image.new('RGB', (240, 180), (65, 65, 65))
        draw = ImageDraw.Draw(competing)
        draw.ellipse((97, 77, 113, 91), fill=(245, 210, 40))
        draw.polygon(((105, 68), (99, 76), (111, 76)), fill=(245, 245, 245))
        draw.ellipse((147, 107, 171, 131), fill=(245, 210, 40))
        draw.rectangle((152, 96, 166, 106), fill=(235, 35, 35))
        pose = detect_player_pose(competing)
        self.assertAlmostEqual(pose.x, 105, delta=1)
        self.assertEqual(pose.heading_color, 'white')

        rows = []
        for frame_id, x in ((0, 10), (1, 11), (2, 200), (3, 12), (4, 13)):
            rows.append({'frame_id': frame_id,
                         'radar_pose': {'x': x, 'y': 10}})
        kept, rejected = _temporally_supported(rows, 20, 2)
        self.assertEqual([row['frame_id'] for row in kept], [0, 1, 3, 4])
        self.assertEqual([row['frame_id'] for row in rejected], [2])

    def test_radar_enemy_diamond_and_last_known_question_mark_are_distinct(self):
        image = Image.new('RGB', (240, 180), (90, 90, 90))
        draw = ImageDraw.Draw(image)
        draw.polygon(((80, 50), (92, 59), (80, 68), (68, 59)),
                     fill=(240, 28, 28))
        draw.rectangle((130, 50, 152, 65), fill=(240, 28, 28))
        draw.arc((170, 48, 186, 64), 190, 520, fill=(190, 110, 100), width=4)
        draw.line(((178, 61), (178, 66)), fill=(190, 110, 100), width=4)
        draw.rectangle((176, 70, 180, 74), fill=(190, 110, 100))

        markers = detect_enemy_markers(image)
        self.assertEqual([marker.state for marker in markers],
                         ['confirmed', 'last_known'])
        confirmed = next(marker for marker in markers
                         if marker.state == 'confirmed')
        last_known = next(marker for marker in markers
                          if marker.state == 'last_known')
        self.assertAlmostEqual(confirmed.x, 80, delta=1)
        self.assertAlmostEqual(last_known.x, 178, delta=2)
        self.assertTrue(all(abs(marker.x - 141) > 8 for marker in markers))

        bounded = detect_enemy_markers(image, origin=(10, 20),
                                       search_bounds=(65, 60, 105, 100))
        self.assertEqual(len(bounded), 1)
        self.assertEqual(bounded[0].state, 'confirmed')
        self.assertAlmostEqual(bounded[0].x, 90, delta=1)

        rows = [
            {'frame_id': 0, 'enemy_markers': [
                {'x': 80, 'y': 60, 'state': 'confirmed'}]},
            {'frame_id': 1, 'enemy_markers': [
                {'x': 81, 'y': 60, 'state': 'last_known'},
                {'x': 180, 'y': 130, 'state': 'last_known'}]},
            {'frame_id': 2, 'enemy_markers': [
                {'x': 82, 'y': 61, 'state': 'confirmed'}]},
        ]
        filtered, rejected = _temporally_supported_markers(rows, 10, 2)
        self.assertEqual(rejected, 1)
        self.assertEqual([len(row['enemy_markers']) for row in filtered], [1, 1, 1])

    def test_visible_target_calibration_uses_screen_box_only(self):
        calibration = VisibleTargetCalibration(
            image_width=2560, image_height=1440,
            horizontal_half_fov_degrees=53.13,
            inverse_height_scale=7800, range_offset=0,
            min_box_height=70, max_box_height=760,
            min_range=10, max_range=115,
        )
        centered = calibration.target_from_box((1200, 500, 1360, 700), 0.8)
        self.assertAlmostEqual(centered.forward, 39, delta=0.1)
        self.assertAlmostEqual(centered.right, 0, delta=0.1)
        self.assertEqual(centered.confidence, 0.8)
        right = calibration.target_from_box((1840, 500, 2000, 700))
        self.assertGreater(right.right, 0)
        self.assertLess(right.forward, centered.forward)
        clipped = calibration.target_from_box((1200, 100, 1360, 1400))
        self.assertAlmostEqual(clipped.forward, 7800 / 760, delta=0.1)
        with self.assertRaisesRegex(ValueError, 'inside'):
            calibration.target_from_box((-1, 0, 10, 20))

    def test_visible_player_labels_require_in_frame_positive_boxes(self):
        label = validate_frame_label({
            'annotated': True,
            'players': [{'x1': 10, 'y1': 20, 'x2': 50, 'y2': 90,
                         'team': 'enemy', 'visibility': 'partial'}],
        }, width=100, height=100)
        self.assertEqual(PlayerBox.from_dict(label['players'][0], 100, 100).team,
                         'enemy')
        self.assertEqual(validate_frame_label(
            {'annotated': True, 'players': []}, 100, 100)['players'], [])
        with self.assertRaisesRegex(ValueError, 'inside the frame'):
            validate_frame_label({
                'annotated': True,
                'players': [{'x1': 10, 'y1': 20, 'x2': 110, 'y2': 90}],
            }, width=100, height=100)

    def test_visible_player_label_audit_counts_session_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / 'capture'
            capture.mkdir()
            frames = [
                {'frame_id': 0, 'file': '0.png', 'width': 100, 'height': 80},
                {'frame_id': 1, 'file': '1.png', 'width': 100, 'height': 80},
            ]
            (capture / 'frames.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in frames) + '\n')
            labels = Path(temporary) / 'labels.json'
            labels.write_text(json.dumps({
                'schema_version': 1, 'split': 'validation', 'labels': {
                    '0': {'annotated': True, 'players': [{
                        'x1': 10, 'y1': 10, 'x2': 40, 'y2': 60,
                        'team': 'enemy', 'visibility': 'full'}]},
                    '1': {'annotated': True, 'players': []},
                }}) + '\n')
            result = audit(capture, labels)
            self.assertEqual(result['split'], 'validation')
            self.assertEqual(result['positive_frames'], 1)
            self.assertEqual(result['negative_frames'], 1)
            self.assertEqual(result['player_boxes'], 1)
            filtered = LabelSet(capture, labels, 'validation', exclude_frames=[1])
            self.assertEqual([frame['frame_id'] for frame in filtered.frames], [0])

    def test_detector_dataset_export_preserves_session_splits_and_yolo_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def make_session(name, split, boxes):
                capture = root / name
                capture.mkdir()
                rows = []
                labels = {}
                for frame_id, frame_boxes in enumerate(boxes):
                    file_name = f'{frame_id:07d}.png'
                    Image.new('RGB', (100, 80), (frame_id, 0, 0)).save(
                        capture / file_name)
                    rows.append({'frame_id': frame_id, 'file': file_name,
                                 'width': 100, 'height': 80})
                    labels[str(frame_id)] = {
                        'annotated': True, 'players': frame_boxes,
                    }
                (capture / 'frames.jsonl').write_text(
                    '\n'.join(json.dumps(row) for row in rows) + '\n')
                labels_path = root / f'{name}_labels.json'
                labels_path.write_text(json.dumps({
                    'schema_version': 1, 'split': split, 'labels': labels,
                }) + '\n')
                return capture, labels_path

            train = make_session('train_capture', 'train', [[{
                'x1': 10, 'y1': 10, 'x2': 40, 'y2': 60,
                'team': 'enemy', 'visibility': 'full',
            }], []])
            validation = make_session('validation_capture', 'validation', [[{
                'x1': 20, 'y1': 20, 'x2': 60, 'y2': 70,
                'team': 'friendly', 'visibility': 'partial',
            }]])
            output = root / 'dataset'
            summary = export_dataset([train, validation], output,
                                     class_mode='team', transfer='copy')
            self.assertEqual(summary['splits']['train']['frames'], 2)
            self.assertEqual(summary['splits']['validation']['boxes'], 1)
            yolo = (output / 'labels' / 'train' /
                    'train_capture__0000000.txt').read_text().strip()
            self.assertEqual(yolo, '0 0.25000000 0.43750000 0.30000000 0.62500000')
            self.assertEqual((output / 'labels' / 'train' /
                              'train_capture__0000001.txt').read_text(), '')
            config = yaml.safe_load((output / 'dataset.yaml').read_text())
            self.assertEqual(config['names'], {0: 'enemy', 1: 'friendly', 2: 'unknown'})
            with self.assertRaises(FileExistsError):
                export_dataset([train], output)

    def test_detector_metrics_count_grouped_recall_and_negative_false_positives(self):
        records = [{
            'ground_truth': [[10, 10, 30, 30], [50, 50, 80, 80]],
            'boxes_metadata': [
                {'team': 'enemy', 'visibility': 'full'},
                {'team': 'friendly', 'visibility': 'partial'},
            ],
            'predicted_boxes': [[9, 9, 31, 31], [0, 0, 5, 5]],
            'scores': [0.9, 0.8],
        }, {
            'ground_truth': [], 'boxes_metadata': [],
            'predicted_boxes': [[1, 1, 9, 9]], 'scores': [0.7],
        }]
        metrics = evaluate_detection_records(records)
        self.assertEqual((metrics['true_positives'], metrics['false_positives'],
                          metrics['false_negatives']), (1, 2, 1))
        self.assertAlmostEqual(metrics['precision'], 1 / 3)
        self.assertAlmostEqual(metrics['recall'], 1 / 2)
        self.assertEqual(metrics['false_positives_per_negative_frame'], 1)
        self.assertEqual(metrics['grouped_recall']['team:enemy']['recall'], 1)
        self.assertEqual(metrics['grouped_recall']['visibility:partial']['recall'], 0)
        self.assertAlmostEqual(metrics['per_class']['1']['precision'], 1 / 3)

    def test_player_ssdlite_has_background_and_player_outputs(self):
        model = build_player_ssdlite(
            pretrained=False, image_size=640, foreground_classes=3)
        self.assertEqual(model.transform.fixed_size, (640, 640))
        anchors = model.anchor_generator.num_anchors_per_location()
        for block, anchor_count in zip(
                model.head.classification_head.module_list, anchors):
            self.assertEqual(block[1].out_channels, anchor_count * 4)
        with self.assertRaisesRegex(ValueError, 'multiple of 32'):
            build_player_ssdlite(pretrained=False, image_size=321)
        with self.assertRaisesRegex(ValueError, 'foreground_classes'):
            build_player_ssdlite(pretrained=False, foreground_classes=0)


if __name__ == '__main__':
    unittest.main()
