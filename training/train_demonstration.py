"""CPU behavior cloning from compact Dust II human demonstrations."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.flywire_behavior import (
    BUTTON_NAMES, DIRECTION_NAMES, FlyWireBehaviorCloner)
from tools.evaluate_toy_combat import load_policy


def hindsight_goal_observations(rows, horizon_seconds=1.0,
                                distance_scale=127.0,
                                maximum_slack_seconds=.5):
    """Replace target channels with a causal-at-inference future-route goal.

    Training derives intent from a later human pose. Live inference supplies the
    same channels from the waypoint planner; future state is never used live.
    """
    if horizon_seconds <= 0 or distance_scale <= 0:
        raise ValueError('Hindsight horizon and distance scale must be positive')
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['round_id']].append(index)
    observations, selected = [], []
    target_delta_ns = int(horizon_seconds * 1e9)
    maximum_ns = int((horizon_seconds + maximum_slack_seconds) * 1e9)
    for indices in groups.values():
        timestamps = [int(rows[index]['observation_monotonic_ns'])
                      for index in indices]
        for local_index, row_index in enumerate(indices):
            start_ns = timestamps[local_index]
            future_local = bisect_left(
                timestamps, start_ns + target_delta_ns,
                lo=local_index + 1)
            if future_local >= len(indices):
                continue
            if timestamps[future_local] - start_ns > maximum_ns:
                continue
            row, future = rows[row_index], rows[indices[future_local]]
            x, y = float(row['pose']['x']), float(row['pose']['y'])
            target_x = float(future['pose']['x'])
            target_y = float(future['pose']['y'])
            delta_x, delta_y = target_x - x, target_y - y
            distance = math.hypot(delta_x, delta_y)
            if distance < 1.0:
                future_yaw = math.radians(float(future['pose']['yaw_degrees']))
                delta_x = math.cos(future_yaw) * 8.0
                delta_y = math.sin(future_yaw) * 8.0
                distance = 8.0
            yaw = math.radians(float(row['pose']['yaw_degrees']))
            forward_x, forward_y = math.cos(yaw), math.sin(yaw)
            right_x, right_y = -forward_y, forward_x
            local_forward = delta_x * forward_x + delta_y * forward_y
            local_right = delta_x * right_x + delta_y * right_y
            observation = list(row['observation'])
            observation[4] = local_forward / distance_scale
            observation[5] = local_right / distance_scale
            observation[6] = distance / (math.sqrt(2) * distance_scale)
            observation[7] = local_forward / distance
            observation[8] = 0.0
            observations.append(observation)
            selected.append(row_index)
    return selected, np.asarray(observations, dtype=np.float32)


def load_demonstration(path, hindsight_goal_seconds=None):
    rows = []
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            if row.get('training_valid'):
                rows.append(row)
    if len(rows) < 10:
        raise ValueError('At least ten valid demonstration rows are required')
    if hindsight_goal_seconds is None:
        selected = list(range(len(rows)))
        observations = np.asarray(
            [row['observation'] for row in rows], dtype=np.float32)
    else:
        selected, observations = hindsight_goal_observations(
            rows, hindsight_goal_seconds)
        rows = [rows[index] for index in selected]
    directions = np.asarray([[
        row['controls']['forward_duty'], row['controls']['backward_duty'],
        row['controls']['strafe_left_duty'],
        row['controls']['strafe_right_duty']]
        for row in rows], dtype=np.float32)
    yaw = np.asarray([
        row['controls']['turn_yaw_delta_degrees'] for row in rows],
        dtype=np.float32)
    buttons = np.asarray([[
        row['controls'][name] for name in BUTTON_NAMES]
        for row in rows], dtype=np.float32)
    if observations.shape[1] != 14:
        raise ValueError('Demonstrations must contain 14-value observations')
    for name, values in (
            ('observations', observations), ('directions', directions),
            ('yaw', yaw), ('buttons', buttons)):
        if not np.isfinite(values).all():
            raise ValueError(f'{name} contain non-finite values')
    return rows, observations, directions, yaw, buttons


def _metrics(prediction, directions, yaw, buttons, yaw_scale):
    direction_predictions = prediction['direction_logits'].sigmoid()
    button_predictions = prediction['button_logits'].sigmoid()
    yaw_predictions = prediction['yaw_normalized'] * yaw_scale
    active = yaw.abs() > .25
    return {
        'direction_mae': float((direction_predictions - directions).abs().mean()),
        'button_mae': float((button_predictions - buttons).abs().mean()),
        'yaw_mae_degrees': float((yaw_predictions - yaw).abs().mean()),
        'yaw_median_absolute_error_degrees': float(
            (yaw_predictions - yaw).abs().median()),
        'yaw_sign_accuracy_on_turns': (float(
            (torch.sign(yaw_predictions[active]) ==
             torch.sign(yaw[active])).float().mean()) if active.any() else None),
        'turn_rows': int(active.sum()),
    }


def _baseline_metrics(train_directions, train_yaw, train_buttons,
                      directions, yaw, buttons):
    direction = train_directions.mean(0).expand_as(directions)
    button = train_buttons.mean(0).expand_as(buttons)
    yaw_prediction = train_yaw.median().expand_as(yaw)
    active = yaw.abs() > .25
    return {
        'direction_mae': float((direction - directions).abs().mean()),
        'button_mae': float((button - buttons).abs().mean()),
        'yaw_mae_degrees': float((yaw_prediction - yaw).abs().mean()),
        'yaw_median_absolute_error_degrees': float(
            (yaw_prediction - yaw).abs().median()),
        'yaw_sign_accuracy_on_turns': (float(
            (torch.sign(yaw_prediction[active]) ==
             torch.sign(yaw[active])).float().mean()) if active.any() else None),
        'turn_rows': int(active.sum()),
    }


def train(demonstration, policy_run, output, policy_version=80, seed=20261005,
          epochs=60, batch_size=256, learning_rate=1e-3,
          validation_fraction=.2, yaw_scale_degrees=90.0,
          hindsight_goal_seconds=None):
    demonstration, policy_run, output = map(Path, (demonstration, policy_run, output))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if not 0 < validation_fraction < .5 or epochs < 1 or batch_size < 1:
        raise ValueError('Invalid split or training configuration')
    rows, observations_np, directions_np, yaw_np, buttons_np = \
        load_demonstration(demonstration, hindsight_goal_seconds)
    split = int(len(rows) * (1 - validation_fraction))
    if split < 1 or split >= len(rows):
        raise ValueError('Validation split leaves an empty partition')
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    backbone, source_manifest = load_policy(policy_run, policy_version)
    if source_manifest['config'].get('architecture') != 'flywire':
        raise ValueError('Behavior cloning requires the FlyWire backbone')
    model = FlyWireBehaviorCloner(backbone, yaw_scale_degrees)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate, foreach=False)
    observations = torch.from_numpy(observations_np)
    directions = torch.from_numpy(directions_np)
    yaw = torch.from_numpy(yaw_np).clamp(
        -yaw_scale_degrees, yaw_scale_degrees)
    buttons = torch.from_numpy(buttons_np)
    train_indices = torch.arange(split)
    validation_indices = torch.arange(split, len(rows))
    generator = torch.Generator().manual_seed(seed + 1)
    fixed_indices = model.backbone.connectome_layer.indices.detach().clone()
    fixed_signs = model.backbone.connectome_layer.edge_signs.detach().clone()
    history = []
    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for indices in order.split(batch_size):
            prediction = model(observations[indices])
            direction_loss = F.binary_cross_entropy_with_logits(
                prediction['direction_logits'], directions[indices])
            button_loss = F.binary_cross_entropy_with_logits(
                prediction['button_logits'], buttons[indices])
            yaw_target = yaw[indices] / yaw_scale_degrees
            yaw_loss = F.smooth_l1_loss(
                prediction['yaw_normalized'], yaw_target, beta=.1)
            loss = direction_loss + .25 * button_loss + yaw_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters()
                 if parameter.requires_grad], 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError('Non-finite behavior-cloning gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
    if (not torch.equal(model.backbone.connectome_layer.indices, fixed_indices) or
            not torch.equal(model.backbone.connectome_layer.edge_signs, fixed_signs)):
        raise RuntimeError('Connectome topology or neurotransmitter signs changed')
    model.eval()
    with torch.no_grad():
        train_prediction = model(observations[train_indices])
        validation_prediction = model(observations[validation_indices])
    metrics = {
        'schema_version': 1,
        'status': 'complete',
        'architecture': 'flywire_multi_head_behavior_cloning',
        'source_policy_run_id': source_manifest['run_id'],
        'source_policy_version': policy_version,
        'seed': seed,
        'rows': len(rows),
        'train_rows': len(train_indices),
        'validation_rows': len(validation_indices),
        'split': 'chronological_tail_holdout_provisional',
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'yaw_scale_degrees': yaw_scale_degrees,
        'hindsight_goal_seconds': hindsight_goal_seconds,
        'elapsed_seconds': time.perf_counter() - started,
        'topology_and_signs_preserved': True,
        'train': _metrics(
            train_prediction, directions[train_indices], yaw[train_indices],
            buttons[train_indices], yaw_scale_degrees),
        'validation': _metrics(
            validation_prediction, directions[validation_indices],
            yaw[validation_indices], buttons[validation_indices],
            yaw_scale_degrees),
        'validation_mean_baseline': _baseline_metrics(
            directions[train_indices], yaw[train_indices], buttons[train_indices],
            directions[validation_indices], yaw[validation_indices],
            buttons[validation_indices]),
        'history': history,
    }
    output.mkdir(parents=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {key: metrics[key] for key in (
            'architecture', 'source_policy_run_id', 'source_policy_version',
            'seed', 'yaw_scale_degrees', 'hindsight_goal_seconds')},
    }, output / 'model.pt')
    shutil.copyfile(policy_run / 'graph.npz', output / 'graph.npz')
    shutil.copyfile(policy_run / 'adjacency.npz', output / 'adjacency.npz')
    metrics['demonstration_sha256'] = hashlib.sha256(
        demonstration.read_bytes()).hexdigest()
    metrics['model_sha256'] = hashlib.sha256(
        (output / 'model.pt').read_bytes()).hexdigest()
    (output / 'metrics.json').write_text(
        json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demonstration', type=Path, required=True)
    parser.add_argument('--policy-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-version', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--yaw-scale-degrees', type=float, default=90)
    parser.add_argument('--hindsight-goal-seconds', type=float)
    args = parser.parse_args()
    print(json.dumps(train(
        args.demonstration, args.policy_run, args.output,
        args.policy_version, args.seed, args.epochs, args.batch_size,
        args.learning_rate, args.validation_fraction,
        args.yaw_scale_degrees, args.hindsight_goal_seconds), indent=2))


if __name__ == '__main__':
    main()
