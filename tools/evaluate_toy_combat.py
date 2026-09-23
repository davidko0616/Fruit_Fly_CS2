"""Evaluate saved policy versions on fixed held-out toy-combat episodes."""
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.flywire_network import FlyWireNetwork
from models.mlp_baseline import MLPBaseline
from toy_combat.env import (ACTION_NAMES, CombatConfig, ToyCombatEnv, scripted_action,
                            scripted_navigation_action)


def load_policy(run, version):
    manifest = json.loads((run / 'manifest.json').read_text())
    config = manifest['config']
    if config.get('architecture', 'flywire') == 'mlp':
        model = MLPBaseline(config['observation_size'], len(config['action_names']),
                            tuple(config['hidden_sizes']))
    else:
        with np.load(run / 'graph.npz') as graph:
            model = FlyWireNetwork(sp.load_npz(run / 'adjacency.npz'), graph['signs'], graph['inputs'],
                                   graph['outputs'], config['observation_size'],
                                   len(config['action_names']), 3, 'normalized_synapse_count')
    with np.load(run / 'weights' / f'{version:07d}.npz') as weights, torch.no_grad():
        for i, parameter in enumerate(model.parameters()):
            parameter.copy_(torch.from_numpy(weights[f'p{i}']))
    model.eval()
    return model, manifest


def evaluate_model(model, config, episode_seeds, mode, random_seed=20260923):
    environments = [ToyCombatEnv(CombatConfig(**config['environment'])) for _ in episode_seeds]
    reset = [env.reset(seed) for env, seed in zip(environments, episode_seeds)]
    observations, infos = [v[0] for v in reset], [v[1] for v in reset]
    active = np.ones(len(environments), dtype=bool)
    returns = np.zeros(len(environments)); lengths = np.zeros(len(environments), dtype=int)
    hits = np.zeros(len(environments), dtype=bool); action_counts = np.zeros(len(ACTION_NAMES), dtype=int)
    acquired_line_of_sight = np.zeros(len(environments), dtype=bool)
    reached_firing_alignment = np.zeros(len(environments), dtype=bool)
    fired = np.zeros(len(environments), dtype=bool)
    max_decisions = math.ceil(config['environment']['max_ticks'] / config['action_repeat'])
    uniforms = np.random.default_rng(random_seed).random((len(environments), max_decisions))
    for decision in range(max_decisions):
        indices = np.flatnonzero(active)
        if not len(indices): break
        x = torch.from_numpy(np.stack([observations[i] for i in indices]))
        masks = torch.from_numpy(np.stack([infos[i]['action_mask'] for i in indices]))
        with torch.no_grad():
            logits = model(x).masked_fill(~masks, torch.finfo(torch.float32).min)
            probabilities = logits.softmax(1).numpy()
        if mode == 'greedy':
            actions = probabilities.argmax(1)
        elif mode == 'stochastic':
            actions = np.asarray([np.searchsorted(np.cumsum(p), uniforms[index, decision], side='right')
                                  for index, p in zip(indices, probabilities)])
            actions = np.minimum(actions, probabilities.shape[1] - 1)
        else:
            raise ValueError(mode)
        for index, action in zip(indices, actions):
            acquired_line_of_sight[index] |= bool(infos[index]['line_of_sight'])
            reached_firing_alignment[index] |= bool(
                infos[index]['line_of_sight'] and infos[index]['aim_alignment'] + 1e-6 >=
                np.cos(np.deg2rad(config['environment']['hit_tolerance_degrees'])))
            fired[index] |= int(action) == 7
            next_observation, reward, terminated, truncated, outcome = environments[index].step(int(action), config['action_repeat'])
            observations[index], infos[index] = next_observation, outcome
            returns[index] += reward; lengths[index] += 1; action_counts[action] += 1
            if terminated or truncated:
                active[index] = False; hits[index] = outcome['hit']
    if active.any():
        raise RuntimeError('Evaluation exceeded environment time limit')
    return {'episodes': len(environments), 'hit_rate': float(hits.mean()),
            'return_mean': float(returns.mean()), 'return_sample_sd': float(returns.std(ddof=1)),
            'length_mean': float(lengths.mean()),
            'line_of_sight_acquisition_rate': float(acquired_line_of_sight.mean()),
            'firing_alignment_rate': float(reached_firing_alignment.mean()),
            'fired_rate': float(fired.mean()),
            'action_counts': {name: int(action_counts[i]) for i, name in enumerate(ACTION_NAMES)}}


def evaluate_scripted(config, episode_seeds):
    returns, hits, lengths = [], [], []
    for seed in episode_seeds:
        env = ToyCombatEnv(CombatConfig(**config['environment'])); observation, info = env.reset(seed)
        total = 0.0
        max_decisions = math.ceil(config['environment']['max_ticks'] / config['action_repeat'])
        for length in range(1, max_decisions + 1):
            action = (scripted_navigation_action(env)
                      if env.config.scenario == 'navigation_v1'
                      else scripted_action(observation, info['action_mask']))
            observation, reward, terminated, truncated, info = env.step(action, config['action_repeat'])
            total += reward
            if terminated or truncated: break
        returns.append(total); hits.append(info['hit']); lengths.append(length)
    return {'episodes': len(episode_seeds), 'hit_rate': float(np.mean(hits)),
            'return_mean': float(np.mean(returns)), 'return_sample_sd': float(np.std(returns, ddof=1)),
            'length_mean': float(np.mean(lengths))}


def evaluate(run, output, episodes=256):
    output.mkdir(parents=True, exist_ok=True)
    episode_seeds = list(range(9_000_000, 9_000_000 + episodes))
    final_version = max(int(path.stem) for path in (run / 'weights').glob('*.npz'))
    versions = list(range(0, final_version + 1, 10))
    if versions[-1] != final_version:
        versions.append(final_version)
    rows = []
    for version in versions:
        model, manifest = load_policy(run, version)
        for mode in ('stochastic', 'greedy'):
            result = evaluate_model(model, manifest['config'], episode_seeds, mode)
            rows.append({'policy_version': version, 'mode': mode, **result})
            print(f'v{version} {mode}: hit={result["hit_rate"]:.1%} return={result["return_mean"]:.3f}', flush=True)
    scripted = evaluate_scripted(manifest['config'], episode_seeds)
    payload = {'run_id': manifest['run_id'], 'evaluation_episode_seeds': [episode_seeds[0], episode_seeds[-1]],
               'episodes_per_evaluation': episodes, 'policies': rows, 'scripted_oracle': scripted,
               'selection': 'versions fixed every 10 PPO updates; held-out episode seeds not used for training'}
    (output / 'evaluation.json').write_text(json.dumps(payload, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode, color in [('stochastic', '#2776a8'), ('greedy', '#bd6127')]:
        selected = [row for row in rows if row['mode'] == mode]
        axes[0].plot(versions, [100 * r['hit_rate'] for r in selected], marker='o', label=mode, color=color)
        axes[1].plot(versions, [r['return_mean'] for r in selected], marker='o', label=mode, color=color)
    axes[0].axhline(100 * scripted['hit_rate'], color='#56833e', linestyle=':', label='scripted oracle')
    axes[1].axhline(scripted['return_mean'], color='#56833e', linestyle=':')
    axes[0].set(title='Held-out hit rate', xlabel='PPO update / policy version', ylabel='Hit rate (%)', ylim=(0, 105))
    axes[1].set(title='Held-out episode return', xlabel='PPO update / policy version', ylabel='Mean return')
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(f'{manifest["config"].get("architecture", "flywire")} aiming · '
                 f'{episodes} fixed held-out episodes')
    fig.savefig(output / 'learning.png', dpi=160); plt.close(fig)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=256)
    args = parser.parse_args(); evaluate(args.run, args.output, args.episodes)
