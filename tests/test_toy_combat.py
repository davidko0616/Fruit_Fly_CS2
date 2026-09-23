import tempfile
from pathlib import Path
import unittest

import numpy as np

from toy_combat.env import REWARD_NAMES, ToyCombatEnv, scripted_action
from training.run_toy_combat import run
from training.train_toy_combat import train
from tools.verify_combat_recording import verify


class ToyCombatTests(unittest.TestCase):
    def test_reset_and_transitions_are_deterministic(self):
        first, second = ToyCombatEnv(), ToyCombatEnv()
        a, ai = first.reset(123); b, bi = second.reset(123)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ai['privileged_state'], bi['privileged_state'])
        for action in (5, 1, 7, 3, 6):
            observed_a = first.step(action, repeat=2)
            observed_b = second.step(action, repeat=2)
            np.testing.assert_array_equal(observed_a[0], observed_b[0])
            self.assertEqual(observed_a[1:4], observed_b[1:4])
            self.assertEqual(observed_a[4]['reward_components'], observed_b[4]['reward_components'])
            if observed_a[2] or observed_a[3]: break

    def test_reward_components_sum_and_action_repeat_stops_at_terminal(self):
        env = ToyCombatEnv(); env.reset(7)
        for _ in range(16):
            observation = env.observation()
            action = scripted_action(observation, env.action_mask())
            _, reward, terminated, truncated, info = env.step(action, repeat=2)
            self.assertAlmostEqual(reward, sum(info['reward_components'][name] for name in REWARD_NAMES))
            self.assertLessEqual(info['action_applied_ticks'], 2)
            if terminated or truncated:
                self.assertTrue(terminated)
                self.assertEqual(info['reset_reason'], 'hit')
                break
        else:
            self.fail('Scripted policy did not solve the aiming episode')

    def test_short_parallel_recording_replays_in_separate_code_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'
            metrics = run(directory, decisions=20, workers=2, seed=5, action_repeat=2)
            result = verify(directory)
            self.assertEqual(metrics['decisions'], 20)
            self.assertEqual(result['decisions'], 20)
            self.assertTrue(result['audit_passed'])

    def test_one_ppo_update_saves_replayable_policy_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'ppo'
            metrics = train(directory, seed=6, updates=1, workers=2, horizon=4,
                            action_repeat=2, ppo_epochs=1, minibatch_size=8)
            result = verify(directory)
            self.assertEqual(metrics['decisions'], 8)
            self.assertTrue((directory / 'weights/0000001.npz').is_file())
            self.assertEqual(result['decisions'], 8)


if __name__ == '__main__':
    unittest.main()
