"""Audit and independently replay every toy-combat decision."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.flywire_network import FlyWireNetwork
from models.mlp_baseline import MLPBaseline
from toy_combat.env import REWARD_NAMES, CombatConfig, ToyCombatEnv
from toy_combat.recording import RESET_CODES, mlp_forward_with_activity, policy_forward_with_activity

MODEL_RTOL = 1e-5
MODEL_ATOL = 5e-6


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['task'] != 'toy_combat':
        raise ValueError('Not a toy-combat recording')
    chunks = []
    for path in sorted((directory / 'chunks').glob('*.npz')):
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        count = len(arrays['global_decision'])
        if not all(len(value) == count and np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f'Invalid arrays in {path.name}')
        chunks.append(arrays)
    if not chunks:
        raise ValueError('No committed decisions')
    data = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    count = len(data['global_decision'])
    np.testing.assert_array_equal(data['global_decision'], np.arange(count))
    if manifest['status'] == 'complete' and count != manifest['decisions']:
        raise ValueError('Manifest decision count disagrees with chunks')
    config = manifest['config']
    architecture = config.get('architecture', 'flywire')
    if architecture == 'mlp':
        model = MLPBaseline(config['observation_size'], len(config['action_names']),
                            tuple(config['hidden_sizes']))
    else:
        with np.load(directory / 'graph.npz') as graph:
            graph = dict(graph)
        model = FlyWireNetwork(sp.load_npz(directory / 'adjacency.npz'), graph['signs'], graph['inputs'],
                               graph['outputs'], config['observation_size'],
                               len(config['action_names']), 3, 'normalized_synapse_count')
    model.eval()
    environments = {worker: ToyCombatEnv(CombatConfig(**manifest['config']['environment']))
                    for worker in range(manifest['config']['workers'])}
    active_episode, observations, infos = {}, {}, {}
    replayed_hits, loaded_version = 0, None
    for i in range(count):
        worker, episode = int(data['worker_id'][i]), int(data['episode_id'][i])
        version = int(data['policy_version'][i])
        if version != loaded_version:
            path = directory / 'weights' / f'{version:07d}.npz'
            if not path.exists() and version == 0:
                path = directory / 'policy_weights.npz'  # schema migration for the first smoke run
            with np.load(path) as weights, torch.no_grad():
                for parameter_index, parameter in enumerate(model.parameters()):
                    parameter.copy_(torch.from_numpy(weights[f'p{parameter_index}']))
            loaded_version = version
        env = environments[worker]
        if active_episode.get(worker) != episode:
            observations[worker], infos[worker] = env.reset(int(data['episode_seed'][i]))
            active_episode[worker] = episode
        np.testing.assert_array_equal(observations[worker], data['observation'][i])
        np.testing.assert_array_equal(infos[worker]['privileged_state'], data['privileged_before'][i])
        np.testing.assert_array_equal(infos[worker]['action_mask'], data['action_mask'][i])
        x = torch.from_numpy(data['transformed_input'][i:i+1])
        mask = torch.from_numpy(data['action_mask'][i:i+1])
        with torch.no_grad():
            if architecture == 'mlp':
                logits, probabilities, hidden_pre, hidden_post = mlp_forward_with_activity(model, x, mask)
            else:
                logits, probabilities, states, pre, sensory = policy_forward_with_activity(model, x, mask)
        np.testing.assert_allclose(logits.numpy()[0], data['logits'][i],
                                   rtol=MODEL_RTOL, atol=MODEL_ATOL)
        np.testing.assert_allclose(probabilities.numpy()[0], data['probabilities'][i],
                                   rtol=MODEL_RTOL, atol=MODEL_ATOL)
        if architecture == 'mlp':
            np.testing.assert_allclose(hidden_pre[0], data['hidden_preactivations'][i],
                                       rtol=MODEL_RTOL, atol=MODEL_ATOL)
            np.testing.assert_allclose(hidden_post[0], data['hidden_activations'][i],
                                       rtol=MODEL_RTOL, atol=MODEL_ATOL)
        else:
            np.testing.assert_allclose(states[0], data['states'][i],
                                       rtol=MODEL_RTOL, atol=MODEL_ATOL)
            np.testing.assert_allclose(pre[0], data['preactivations'][i],
                                       rtol=MODEL_RTOL, atol=MODEL_ATOL)
            np.testing.assert_allclose(sensory[0], data['sensory'][i],
                                       rtol=MODEL_RTOL, atol=MODEL_ATOL)
        action = int(data['executed_action'][i])
        next_observation, reward, terminated, truncated, outcome = env.step(action, manifest['config']['action_repeat'])
        np.testing.assert_allclose(next_observation, data['next_observation'][i], rtol=0, atol=1e-7)
        np.testing.assert_allclose(reward, data['reward'][i], rtol=0, atol=1e-6)
        expected_components = [outcome['reward_components'][name] for name in REWARD_NAMES]
        np.testing.assert_allclose(expected_components, data['reward_components'][i], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(outcome['privileged_state'], data['privileged_after'][i])
        assert bool(terminated) == bool(data['terminated'][i])
        assert bool(truncated) == bool(data['truncated'][i])
        assert RESET_CODES[outcome['reset_reason']] == int(data['reset_reason'][i])
        assert int(outcome['action_applied_ticks']) == int(data['action_applied_ticks'][i])
        replayed_hits += int(outcome['hit'])
        observations[worker], infos[worker] = next_observation, outcome
    result = {'status': manifest['status'], 'architecture': architecture, 'decisions': count,
              'episodes_seen': int(len(set(zip(data['worker_id'].tolist(), data['episode_id'].tolist())))),
              'hits': replayed_hits, 'model_rtol': MODEL_RTOL, 'model_atol': MODEL_ATOL,
              'environment_atol': 1e-7, 'reward_atol': 1e-6, 'audit_passed': True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)
