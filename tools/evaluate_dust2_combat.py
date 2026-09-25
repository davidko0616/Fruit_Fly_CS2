"""Evaluate a recorded Dust II NAV policy on validation and held-out routes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dust2_training.env import (Dust2CombatConfig, Dust2CombatEnv,
                                scripted_dust2_action)
from tools.evaluate_toy_combat import load_policy


def _evaluate_policy(model, config, split, episode_seeds, mode, random_seed):
    environment_config = replace(config, route_split=split)
    rng = np.random.default_rng(random_seed)
    environments = [Dust2CombatEnv(environment_config) for _ in episode_seeds]
    reset = [environment.reset(seed)
             for environment, seed in zip(environments, episode_seeds)]
    observations = [value[0] for value in reset]
    infos = [value[1] for value in reset]
    active = np.ones(len(environments), dtype=bool)
    hits = np.zeros(len(environments), dtype=bool)
    returns = np.zeros(len(environments), dtype=float)
    lengths = np.zeros(len(environments), dtype=int)
    acquired = np.zeros(len(environments), dtype=bool)
    action_counts = np.zeros(Dust2CombatEnv.action_size, dtype=int)
    for _ in range(config.max_ticks):
        indices = np.flatnonzero(active)
        if not len(indices):
            break
        x = torch.from_numpy(np.stack([observations[index] for index in indices]))
        mask = torch.from_numpy(np.stack([infos[index]['action_mask']
                                          for index in indices]))
        with torch.no_grad():
            logits = model(x).masked_fill(~mask, torch.finfo(torch.float32).min)
            probabilities = logits.softmax(1).numpy()
        if mode == 'greedy':
            actions = probabilities.argmax(1)
        elif mode == 'stochastic':
            actions = np.asarray([
                rng.choice(len(probability), p=probability)
                for probability in probabilities])
        else:
            raise ValueError(f'Unknown evaluation mode: {mode}')
        for index, action in zip(indices, actions):
            observation, reward, terminated, truncated, info = \
                environments[index].step(int(action))
            observations[index], infos[index] = observation, info
            action_counts[action] += 1
            returns[index] += reward
            lengths[index] += 1
            acquired[index] |= bool(info['line_of_sight'])
            if terminated or truncated:
                active[index] = False
                hits[index] = bool(info['hit'])
    if active.any():
        raise RuntimeError('Dust II evaluation exceeded environment time limit')
    return {
        'episodes': len(episode_seeds), 'mode': mode,
        'hit_rate': float(hits.mean()),
        'return_mean': float(returns.mean()),
        'length_mean': float(lengths.mean()),
        'line_of_sight_acquisition_rate': float(acquired.mean()),
        'action_counts': action_counts.tolist(),
    }


def _evaluate_oracle(config, split, episode_seeds):
    environment_config = replace(config, route_split=split)
    hits, lengths = [], []
    for seed in episode_seeds:
        environment = Dust2CombatEnv(environment_config)
        environment.reset(seed)
        for length in range(1, config.max_ticks + 1):
            _, _, terminated, truncated, info = environment.step(
                scripted_dust2_action(environment))
            if terminated or truncated:
                break
        hits.append(bool(info['hit']))
        lengths.append(length)
    return {
        'episodes': len(episode_seeds), 'hit_rate': float(np.mean(hits)),
        'length_mean': float(np.mean(lengths)),
        'max_length': int(max(lengths)),
    }


def evaluate(run, output, episodes=64, splits=('validation',),
             policy_version=None):
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    environment = manifest['config']['environment']
    if environment.get('scenario') != 'dust2_navigation_v1':
        raise ValueError('Run is not a Dust II navigation policy')
    config = Dust2CombatConfig(**environment)
    available_versions = {
        int(path.stem) for path in (run / 'weights').glob('*.npz')}
    if policy_version is None:
        policy_version = max(available_versions)
    if policy_version not in available_versions:
        raise ValueError(f'Policy version {policy_version} is unavailable')
    model, _ = load_policy(run, policy_version)
    payload = {
        'schema_version': 1, 'run_id': manifest['run_id'],
        'policy_version': policy_version,
        'episodes_per_split': episodes,
        'route_selection': 'hash-disjoint route pairs',
        'splits': {},
    }
    seeds_by_split = {'validation': 9_100_000, 'heldout': 9_200_000}
    for split in splits:
        if split not in seeds_by_split:
            raise ValueError(f'Unsupported evaluation split: {split}')
        seed_start = seeds_by_split[split]
        seeds = list(range(seed_start, seed_start + episodes))
        payload['splits'][split] = {
            'episode_seeds': [seeds[0], seeds[-1]],
            'stochastic': _evaluate_policy(model, config, split, seeds,
                                           'stochastic', seed_start + 99),
            'greedy': _evaluate_policy(model, config, split, seeds, 'greedy', 0),
            'scripted_oracle': _evaluate_oracle(config, split, seeds),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--split', action='append', choices=('validation', 'heldout'),
                        dest='splits')
    parser.add_argument('--policy-version', type=int)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    print(json.dumps(evaluate(
        args.run, args.output, args.episodes,
        tuple(args.splits or ('validation',)),
        args.policy_version), indent=2))


if __name__ == '__main__':
    main()
