import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from toy_combat.env import (NAVIGATION_REWARD_NAMES, REWARD_NAMES, ToyCombatEnv,
                            integrated_navigation_config, moving_target_config,
                            navigation_config, scripted_action, scripted_navigation_action)
from training.run_toy_combat import load_policy, run
from training.train_toy_combat import train
from tools.verify_combat_recording import verify


class ToyCombatTests(unittest.TestCase):
    def test_comparison_policies_match_declared_budget_and_recurrent_io_initialization(self):
        flywire, _, flywire_adjacency, _ = load_policy(42, 'flywire')
        random, _, random_adjacency, _ = load_policy(42, 'random')
        mlp, _, _, hidden_sizes = load_policy(42, 'mlp')
        self.assertEqual(hidden_sizes, (78, 81))
        self.assertEqual([sum(parameter.numel() for parameter in model.parameters())
                          for model in (flywire, random, mlp)], [7913, 7913, 7913])
        self.assertEqual(flywire_adjacency.nnz, random_adjacency.nnz)
        np.testing.assert_array_equal(np.diff(flywire_adjacency.indptr),
                                      np.diff(random_adjacency.indptr))
        self.assertGreater((flywire_adjacency != random_adjacency).nnz, 0)
        for name in ('input_proj.weight', 'input_proj.bias', 'output_proj.weight',
                     'output_proj.bias'):
            torch.testing.assert_close(dict(flywire.named_parameters())[name],
                                       dict(random.named_parameters())[name], rtol=0, atol=0)

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

    def test_navigation_layouts_start_occluded_and_privileged_oracle_solves_them(self):
        lengths = []
        for seed in range(200):
            env = ToyCombatEnv(navigation_config())
            observation, info = env.reset(seed)
            self.assertEqual(observation.shape, (14,))
            self.assertFalse(info['line_of_sight'])
            self.assertTrue(info['action_mask'][:7].all())
            self.assertFalse(info['action_mask'][7])
            for decision in range(1, env.config.max_ticks + 1):
                action = scripted_navigation_action(env)
                observation, reward, terminated, truncated, info = env.step(action, repeat=1)
                self.assertAlmostEqual(reward, sum(info['reward_components'][name]
                                                   for name in NAVIGATION_REWARD_NAMES))
                if terminated or truncated:
                    break
            self.assertTrue(info['hit'], f'Oracle failed navigation seed {seed}')
            lengths.append(decision)
        self.assertLessEqual(max(lengths), 30)

    def test_integrated_navigation_keeps_movement_aiming_and_fire_available(self):
        env = ToyCombatEnv(integrated_navigation_config())
        _, info = env.reset(123)
        self.assertFalse(info['line_of_sight'])
        self.assertTrue(info['action_mask'].all())
        self.assertFalse(env.config.navigation_phase_masking)

        while not env._line_of_sight():
            _, _, terminated, truncated, info = env.step(scripted_navigation_action(env))
            self.assertFalse(terminated or truncated)
        self.assertTrue(info['action_mask'].all())

    def test_moving_target_is_deterministic_obstacle_aware_and_fully_recorded_in_state(self):
        first = ToyCombatEnv(moving_target_config())
        second = ToyCombatEnv(moving_target_config())
        first.reset(321); second.reset(321)
        moved = 0
        for _ in range(24):
            before = first.enemy.copy()
            observed_first = first.step(0)
            observed_second = second.step(0)
            np.testing.assert_array_equal(observed_first[0], observed_second[0])
            np.testing.assert_array_equal(observed_first[4]['privileged_state'],
                                          observed_second[4]['privileged_state'])
            self.assertEqual(len(observed_first[4]['privileged_state']), 8)
            self.assertNotIn(tuple(first.enemy), first.obstacles)
            self.assertFalse(np.array_equal(first.enemy, first.agent))
            moved += int(not np.array_equal(before, first.enemy))
        self.assertGreaterEqual(moved, 20)

    def test_moving_target_oracle_and_recording_replay(self):
        moved_episodes = hits = 0
        for seed in range(200):
            env = ToyCombatEnv(moving_target_config())
            env.reset(seed)
            target_moved = False
            for _ in range(env.config.max_ticks):
                _, _, terminated, truncated, info = env.step(scripted_navigation_action(env))
                target_moved |= info['target_moved']
                if terminated or truncated:
                    break
            hits += int(info['hit'])
            moved_episodes += int(target_moved)
        self.assertGreaterEqual(hits, 198)
        self.assertGreaterEqual(moved_episodes, 100)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'moving-target'
            metrics = train(directory, seed=11, updates=1, workers=2, horizon=8,
                            action_repeat=1, ppo_epochs=1, minibatch_size=16,
                            architecture='flywire', environment_config=moving_target_config())
            result = verify(directory)
            self.assertEqual(metrics['decisions'], 16)
            self.assertEqual(result['decisions'], 16)
            self.assertTrue(result['audit_passed'])

    def test_short_parallel_recording_replays_in_separate_code_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'run'
            metrics = run(directory, decisions=20, workers=2, seed=5, action_repeat=2)
            result = verify(directory)
            self.assertEqual(metrics['decisions'], 20)
            self.assertEqual(result['decisions'], 20)
            self.assertTrue(result['audit_passed'])

    def test_one_ppo_update_saves_replayable_policy_versions_for_all_architectures(self):
        with tempfile.TemporaryDirectory() as temporary:
            parameter_counts = []
            for architecture in ('flywire', 'random', 'mlp'):
                with self.subTest(architecture=architecture):
                    directory = Path(temporary) / architecture
                    metrics = train(directory, seed=6, updates=1, workers=2, horizon=4,
                                    action_repeat=2, ppo_epochs=1, minibatch_size=8,
                                    architecture=architecture)
                    result = verify(directory)
                    parameter_counts.append(metrics['parameters'])
                    self.assertEqual(metrics['decisions'], 8)
                    self.assertTrue((directory / 'weights/0000001.npz').is_file())
                    self.assertTrue((directory / 'provenance.json').is_file())
                    self.assertEqual(result['architecture'], architecture)
                    self.assertEqual(result['decisions'], 8)
            self.assertEqual(parameter_counts, [7913, 7913, 7913])

    def test_navigation_ppo_recording_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'navigation'
            metrics = train(directory, seed=9, updates=1, workers=2, horizon=4,
                            action_repeat=1, ppo_epochs=1, minibatch_size=8,
                            architecture='flywire', environment_config=navigation_config())
            result = verify(directory)
            self.assertEqual(metrics['decisions'], 8)
            self.assertEqual(result['decisions'], 8)
            self.assertTrue(result['audit_passed'])

            integrated = Path(temporary) / 'integrated'
            train(integrated, seed=10, updates=1, workers=2, horizon=4,
                  action_repeat=1, ppo_epochs=1, minibatch_size=8,
                  architecture='flywire', environment_config=integrated_navigation_config(),
                  initial_policy_run=directory, initial_policy_version=1)
            integrated_result = verify(integrated)
            with np.load(directory / 'weights/0000001.npz') as source, \
                    np.load(integrated / 'weights/0000000.npz') as transferred:
                for index in range(6):
                    np.testing.assert_array_equal(source[f'p{index}'], transferred[f'p{index}'])
            manifest = json.loads((integrated / 'manifest.json').read_text())
            self.assertEqual(manifest['config']['initial_policy']['policy_version'], 1)
            self.assertEqual(manifest['config']['initial_policy']['run_id'],
                             json.loads((directory / 'manifest.json').read_text())['run_id'])
            self.assertTrue(integrated_result['audit_passed'])


if __name__ == '__main__':
    unittest.main()
