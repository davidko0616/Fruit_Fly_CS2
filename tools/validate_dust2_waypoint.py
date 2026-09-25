"""Validate the NAV waypoint interface without a learned controller."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dust2_training.env import (Dust2CombatConfig, Dust2CombatEnv,
                                scripted_waypoint_action)


def validate(mask, split='validation', episodes=16, minimum_route_cells=24,
             maximum_route_cells=None, waypoint_lookahead_cells=6):
    config = Dust2CombatConfig(
        str(mask), split, minimum_route_cells=minimum_route_cells,
        maximum_route_cells=maximum_route_cells, waypoint_planner_enabled=True,
        waypoint_lookahead_cells=waypoint_lookahead_cells)
    seed_start = 8_100_000 if split == 'train' else 9_100_000
    hits, acquired, lengths = [], [], []
    for seed in range(seed_start, seed_start + episodes):
        environment = Dust2CombatEnv(config)
        _, info = environment.reset(seed)
        saw_target = bool(info['line_of_sight'])
        for length in range(1, config.max_ticks + 1):
            _, _, terminated, truncated, info = environment.step(
                scripted_waypoint_action(environment))
            saw_target |= bool(info['line_of_sight'])
            if terminated or truncated:
                break
        hits.append(bool(info['hit']))
        acquired.append(saw_target)
        lengths.append(length)
    return {
        'split': split,
        'episodes': episodes,
        'minimum_route_cells': minimum_route_cells,
        'maximum_route_cells': maximum_route_cells,
        'waypoint_lookahead_cells': waypoint_lookahead_cells,
        'hit_rate': float(np.mean(hits)),
        'line_of_sight_acquisition_rate': float(np.mean(acquired)),
        'length_mean': float(np.mean(lengths)),
        'length_max': int(max(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation'),
                        default='validation')
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--minimum-route-cells', type=int, default=24)
    parser.add_argument('--maximum-route-cells', type=int)
    parser.add_argument('--waypoint-lookahead-cells', type=int, default=6)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    print(json.dumps(validate(
        args.mask, args.split, args.episodes, args.minimum_route_cells,
        args.maximum_route_cells, args.waypoint_lookahead_cells), indent=2))


if __name__ == '__main__':
    main()
