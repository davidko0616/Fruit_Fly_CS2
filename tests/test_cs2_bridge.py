import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
import torch
from PIL import Image, ImageDraw

from cs2_bridge.encoder import Dust2ObservationEncoder
from cs2_bridge.gsi import parse_gsi_payload
from cs2_bridge.radar import detect_player_pose
from cs2_bridge.replay import read_frames, replay_frames, write_jsonl
from cs2_bridge.schema import BridgeFrame, Dust2Calibration, PlayerPose, VisibleTarget
from tools.calibrate_dust2_bridge import calibrate
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


if __name__ == '__main__':
    unittest.main()
