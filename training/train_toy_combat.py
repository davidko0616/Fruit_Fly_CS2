"""PPO training for matched aiming policies with complete rollout recording."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from toy_combat.env import (ACTION_NAMES, CombatConfig, ToyCombatEnv,
                            integrated_navigation_config, moving_target_config,
                            navigation_config, partial_observability_config)
from toy_combat.recording import (CombatRecorder, RESET_CODES, mlp_forward_with_activity,
                                  policy_forward_with_activity,
                                  policy_forward_with_memory_activity)
from training.recording import array, atomic_json
from training.run_toy_combat import load_policy

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments/cpu_spiral_100/seed_42'


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def train(output, seed=42, updates=80, workers=8, horizon=64, action_repeat=2,
          learning_rate=3e-4, ppo_epochs=4, minibatch_size=256, architecture='flywire',
          environment_config=None, entropy_coefficient=0.01,
          final_entropy_coefficient=None, initial_policy_run=None,
          initial_policy_version=None, temporal_state_decay=None):
    if updates < 1 or workers < 1 or horizon < 2:
        raise ValueError('Positive updates/workers and horizon >=2 required')
    if entropy_coefficient < 0 or (final_entropy_coefficient is not None and
                                   final_entropy_coefficient < 0):
        raise ValueError('Entropy coefficients must be nonnegative')
    if temporal_state_decay is not None and not 0 <= temporal_state_decay <= 1:
        raise ValueError('Temporal state decay must be between zero and one')
    if temporal_state_decay is not None and architecture == 'mlp':
        raise ValueError('Temporal connectome state is unavailable for the MLP')
    final_entropy_coefficient = (entropy_coefficient if final_entropy_coefficient is None
                                 else final_entropy_coefficient)
    torch.manual_seed(seed); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    environment_config = environment_config or CombatConfig()
    if (environment_config.scenario in
            ('navigation_v1', 'moving_target_v1', 'partial_observability_v1') and
            action_repeat != 1):
        raise ValueError('Navigation scenarios require action_repeat=1 for cell-level control')
    observation_size = ToyCombatEnv.observation_size_for(environment_config)
    policy, source_graph, run_adjacency, hidden_sizes = load_policy(
        seed, architecture, observation_size, ToyCombatEnv.action_size)
    initial_policy = None
    if initial_policy_run is not None:
        initial_policy_run = Path(initial_policy_run)
        source_manifest = json.loads((initial_policy_run / 'manifest.json').read_text())
        source_config = source_manifest['config']
        if (source_config.get('architecture', 'flywire') != architecture or
                source_config['observation_size'] != observation_size or
                len(source_config['action_names']) != ToyCombatEnv.action_size):
            raise ValueError('Initial policy architecture or interface does not match this run')
        if initial_policy_version is None:
            initial_policy_version = max(
                int(path.stem) for path in (initial_policy_run / 'weights').glob('*.npz'))
        weights_path = initial_policy_run / 'weights' / f'{initial_policy_version:07d}.npz'
        with np.load(weights_path) as weights, torch.no_grad():
            parameters = list(policy.parameters())
            policy_keys = {key for key in weights.files if key.startswith('p')}
            if policy_keys != {f'p{i}' for i in range(len(parameters))}:
                raise ValueError('Initial policy parameter set does not match this model')
            for index, parameter in enumerate(parameters):
                value = torch.from_numpy(weights[f'p{index}'])
                if value.shape != parameter.shape:
                    raise ValueError('Initial policy parameter shape does not match this model')
                parameter.copy_(value)
        initial_policy = {
            'run': str(initial_policy_run), 'run_id': source_manifest['run_id'],
            'policy_version': int(initial_policy_version),
            'weights_sha256': hashlib.sha256(weights_path.read_bytes()).hexdigest()}
    torch.manual_seed(seed + 3)
    value_model = ValueNetwork(observation_size)
    optimizer = torch.optim.Adam([*policy.parameters(), *value_model.parameters()], lr=learning_rate, foreach=False)
    sample_generator = torch.Generator().manual_seed(seed + 1)
    batch_generator = torch.Generator().manual_seed(seed + 2)
    environments = [ToyCombatEnv(environment_config) for _ in range(workers)]
    reward_names = environments[0].reward_names
    episode_ids, episode_steps = list(range(workers)), [0] * workers
    episode_seeds = [seed * 100000 + i for i in range(workers)]
    reset = [env.reset(s) for env, s in zip(environments, episode_seeds)]
    observations, infos = [v[0] for v in reset], [v[1] for v in reset]
    temporal_states = (torch.zeros(workers, policy.num_neurons)
                       if temporal_state_decay is not None else None)
    episode_returns, episode_return = [], [0.0] * workers
    episode_hits, completed_lengths = [], []
    if architecture == 'mlp':
        graph = {'architecture': architecture, 'hidden_sizes': list(hidden_sizes),
                 'parameter_names': [name for name, _ in policy.named_parameters()]}
    else:
        graph = {'architecture': architecture,
                 'root_ids': [str(i) for i in source_graph['root_ids']],
                 'inputs': source_graph['inputs'].tolist(), 'outputs': source_graph['outputs'].tolist(),
                 'sources': policy.connectome_layer.indices[1].tolist(),
                 'targets': policy.connectome_layer.indices[0].tolist(),
                 'edge_signs': policy.connectome_layer.edge_signs.tolist()}
    parameter_budget = sum(parameter.numel() for parameter in policy.parameters())
    config = {'seed': seed, 'workers': workers, 'updates': updates, 'horizon': horizon,
              'action_repeat': action_repeat, 'algorithm': 'PPO', 'learning_rate': learning_rate,
              'ppo_epochs': ppo_epochs, 'minibatch_size': minibatch_size, 'gamma': .99,
              'gae_lambda': .95, 'clip_ratio': .2,
              'entropy_coefficient': entropy_coefficient,
              'final_entropy_coefficient': final_entropy_coefficient,
              'value_coefficient': .5, 'max_gradient_norm': .5,
              'architecture': architecture, 'parameters': sum(p.numel() for p in policy.parameters()),
              'parameter_budget': parameter_budget,
              'hidden_sizes': list(hidden_sizes) if architecture == 'mlp' else None,
              'topology_seed': seed if architecture == 'random' else None,
              'random_control': ('source degree, self-loops, source sign, and per-source count '
                                 'multiset preserved') if architecture == 'random' else None,
              'value_seed': seed + 3,
              'observation_size': observation_size, 'action_names': ACTION_NAMES,
              'reward_names': reward_names, 'environment': environment_config.__dict__}
    config['initial_policy'] = initial_policy
    config['temporal_state_decay'] = temporal_state_decay
    update_rows, policy_version, global_decision = [], 0, 0
    fixed_structure = [(name, tuple(parameter.shape)) for name, parameter in policy.named_parameters()]
    fixed_indices = (policy.connectome_layer.indices.detach().clone()
                     if architecture != 'mlp' else None)
    fixed_edge_signs = (policy.connectome_layer.edge_signs.detach().clone()
                        if architecture != 'mlp' else None)
    start = time.perf_counter()
    with CombatRecorder(output, policy, graph, config, max_buffer_decisions=256) as recorder:
        shutil.copyfile(ARCHIVE / 'graph.npz', Path(output) / 'graph.npz')
        sp.save_npz(Path(output) / 'adjacency.npz', run_adjacency)
        provenance_paths = (Path(__file__), ROOT / 'training/run_toy_combat.py',
                            ROOT / 'toy_combat/env.py',
                            ROOT / 'toy_combat/recording.py', ROOT / 'models/flywire_network.py',
                            ROOT / 'models/sparse_layer.py',
                            ROOT / 'models/random_sparse.py', ROOT / 'models/mlp_baseline.py',
                            ARCHIVE / 'graph.npz', ARCHIVE / 'adjacency.npz')
        atomic_json(Path(output) / 'provenance.json', {
            'python': platform.python_version(), 'platform': platform.platform(),
            'processor': platform.processor(), 'torch': torch.__version__,
            'numpy': np.__version__, 'architecture': architecture,
            'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in provenance_paths},
            'run_adjacency_sha256': hashlib.sha256((Path(output) / 'adjacency.npz').read_bytes()).hexdigest()})
        for update in range(1, updates + 1):
            schedule_fraction = 0.0 if updates == 1 else (update - 1) / (updates - 1)
            current_entropy_coefficient = (entropy_coefficient + schedule_fraction *
                                           (final_entropy_coefficient - entropy_coefficient))
            rollout = {key: [] for key in ('observations', 'masks', 'actions', 'old_log_probabilities',
                                            'rewards', 'dones', 'values')}
            if temporal_states is not None:
                rollout['temporal_states'] = []
            for rollout_step in range(horizon):
                obs_tensor = torch.from_numpy(np.stack(observations))
                mask_tensor = torch.from_numpy(np.stack([info['action_mask'] for info in infos]))
                with torch.no_grad():
                    if architecture == 'mlp':
                        logits, probabilities, hidden_pre, hidden_post = mlp_forward_with_activity(
                            policy, obs_tensor, mask_tensor)
                    elif temporal_states is None:
                        logits, probabilities, states, pre, sensory = policy_forward_with_activity(
                            policy, obs_tensor, mask_tensor)
                    else:
                        temporal_before = temporal_states.clone()
                        logits, probabilities, states, pre, sensory, temporal_after = (
                            policy_forward_with_memory_activity(
                                policy, obs_tensor, mask_tensor, temporal_before,
                                temporal_state_decay))
                        temporal_states = temporal_after.detach()
                    values = value_model(obs_tensor)
                    actions = torch.multinomial(probabilities, 1, generator=sample_generator).squeeze(1)
                    selected = probabilities.gather(1, actions[:, None]).squeeze(1)
                    log_probabilities = selected.log()
                    entropy = -(probabilities.clamp_min(1e-30) * probabilities.clamp_min(1e-30).log()).sum(1)
                step_rewards, step_dones = [], []
                for worker, env in enumerate(environments):
                    before = infos[worker]['privileged_state'].copy()
                    next_obs, reward, terminated, truncated, outcome = env.step(int(actions[worker]), action_repeat)
                    done = terminated or truncated
                    components = np.asarray([outcome['reward_components'][name] for name in reward_names],
                                            dtype=np.float32)
                    activity = ({'hidden_preactivations': hidden_pre[worker],
                                 'hidden_activations': hidden_post[worker]}
                                if architecture == 'mlp' else
                                {'sensory': sensory[worker], 'states': states[worker],
                                 'preactivations': pre[worker]})
                    if temporal_states is not None:
                        activity.update(temporal_state_before=array(temporal_before[worker]),
                                        temporal_state_after=array(temporal_states[worker]))
                    recorder.append(worker_id=worker, episode_id=episode_ids[worker],
                        episode_seed=episode_seeds[worker], decision_index=episode_steps[worker],
                        global_decision=global_decision, policy_version=policy_version,
                        tick_before=before[-1], tick_after=outcome['tick'], observation=observations[worker],
                        transformed_input=observations[worker], action_mask=infos[worker]['action_mask'],
                        logits=array(logits[worker]), probabilities=array(probabilities[worker]),
                        chosen_action=int(actions[worker]), executed_action=int(actions[worker]),
                        log_probability=float(log_probabilities[worker]), entropy=float(entropy[worker]),
                        value_estimate=float(values[worker]), action_applied_ticks=outcome['action_applied_ticks'],
                        action_rejected=outcome['action_rejected'], reward=reward, reward_components=components,
                        next_observation=next_obs, terminated=terminated, truncated=truncated,
                        reset_reason=RESET_CODES[outcome['reset_reason']], hit=outcome['hit'], miss=outcome['miss'],
                        collision=outcome['collision'], privileged_before=before,
                        privileged_after=outcome['privileged_state'], **activity)
                    global_decision += 1; episode_steps[worker] += 1; episode_return[worker] += reward
                    step_rewards.append(reward); step_dones.append(done)
                    observations[worker], infos[worker] = next_obs, outcome
                    if done:
                        episode_returns.append(episode_return[worker]); episode_return[worker] = 0.0
                        episode_hits.append(int(outcome['hit'])); completed_lengths.append(episode_steps[worker])
                        episode_ids[worker] += workers; episode_steps[worker] = 0
                        episode_seeds[worker] = seed * 100000 + episode_ids[worker]
                        observations[worker], infos[worker] = env.reset(episode_seeds[worker])
                        if temporal_states is not None:
                            temporal_states[worker].zero_()
                rollout['observations'].append(obs_tensor); rollout['masks'].append(mask_tensor)
                rollout['actions'].append(actions); rollout['old_log_probabilities'].append(log_probabilities)
                rollout['rewards'].append(torch.tensor(step_rewards, dtype=torch.float32))
                rollout['dones'].append(torch.tensor(step_dones, dtype=torch.float32)); rollout['values'].append(values)
                if temporal_states is not None:
                    rollout['temporal_states'].append(temporal_before)
            with torch.no_grad():
                bootstrap = value_model(torch.from_numpy(np.stack(observations)))
            rewards = torch.stack(rollout['rewards']); dones = torch.stack(rollout['dones'])
            values = torch.stack(rollout['values']); advantages = torch.zeros_like(rewards)
            gae = torch.zeros(workers)
            for step in reversed(range(horizon)):
                next_value = bootstrap if step == horizon - 1 else values[step + 1]
                not_done = 1 - dones[step]
                delta = rewards[step] + .99 * next_value * not_done - values[step]
                gae = delta + .99 * .95 * not_done * gae
                advantages[step] = gae
            returns = advantages + values
            flat = {key: torch.stack(rollout[key]).reshape(horizon * workers, *torch.stack(rollout[key]).shape[2:])
                    for key in ('observations', 'masks', 'actions', 'old_log_probabilities')}
            if temporal_states is not None:
                flat['temporal_states'] = torch.stack(rollout['temporal_states']).reshape(
                    horizon * workers, policy.num_neurons)
            flat_advantages = advantages.flatten(); flat_returns = returns.flatten()
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
            diagnostics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': [], 'clip_fraction': []}
            before = [p.detach().clone() for p in policy.parameters()]
            for _ in range(ppo_epochs):
                for indices in torch.randperm(horizon * workers, generator=batch_generator).split(minibatch_size):
                    if temporal_states is None:
                        new_logits = policy(flat['observations'][indices])
                    else:
                        new_logits, _ = policy.forward_with_state(
                            flat['observations'][indices], flat['temporal_states'][indices],
                            temporal_state_decay)
                    new_logits = new_logits.masked_fill(~flat['masks'][indices],
                                                        torch.finfo(torch.float32).min)
                    new_log_probs = new_logits.log_softmax(1).gather(1, flat['actions'][indices, None]).squeeze(1)
                    probs = new_logits.softmax(1); entropy = -(probs.clamp_min(1e-30) * probs.clamp_min(1e-30).log()).sum(1).mean()
                    ratio = (new_log_probs - flat['old_log_probabilities'][indices]).exp()
                    unclipped = ratio * flat_advantages[indices]
                    clipped = ratio.clamp(.8, 1.2) * flat_advantages[indices]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = (value_model(flat['observations'][indices]) - flat_returns[indices]).square().mean()
                    loss = policy_loss + .5 * value_loss - current_entropy_coefficient * entropy
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_([*policy.parameters(), *value_model.parameters()], .5)
                    if not torch.isfinite(gradient_norm): raise RuntimeError('Nonfinite gradient')
                    optimizer.step()
                    diagnostics['policy_loss'].append(float(policy_loss)); diagnostics['value_loss'].append(float(value_loss))
                    diagnostics['entropy'].append(float(entropy)); diagnostics['approx_kl'].append(float((flat['old_log_probabilities'][indices] - new_log_probs).mean()))
                    diagnostics['clip_fraction'].append(float((abs(ratio - 1) > .2).float().mean()))
            policy_version += 1
            model_structure_preserved = (fixed_structure ==
                                         [(name, tuple(parameter.shape))
                                          for name, parameter in policy.named_parameters()])
            topology_and_signs_preserved = (None if architecture == 'mlp' else
                (torch.equal(policy.connectome_layer.indices, fixed_indices)
                 and torch.equal(policy.connectome_layer.edge_signs, fixed_edge_signs)))
            if not model_structure_preserved or topology_and_signs_preserved is False:
                raise RuntimeError('Policy structure, connectome topology, or edge signs changed')
            update_summary = {'update': update, 'policy_version': policy_version,
                **{key: float(np.mean(value)) for key, value in diagnostics.items()},
                'entropy_coefficient': current_entropy_coefficient,
                'gradient_norm': float(gradient_norm),
                'policy_update_norm': float(torch.sqrt(sum((p - old).square().sum() for p, old in zip(policy.parameters(), before)))),
                'model_structure_preserved': model_structure_preserved,
                'topology_and_signs_preserved': topology_and_signs_preserved,
                'rollout_mean_reward': float(rewards.mean()), 'episodes_completed_total': len(episode_returns),
                'recent_100_hit_rate': float(np.mean(episode_hits[-100:])) if episode_hits else None,
                'recent_100_return': float(np.mean(episode_returns[-100:])) if episode_returns else None,
                'elapsed_seconds': time.perf_counter() - start}
            update_rows.append(update_summary); recorder.write_updates(update_rows)
            recorder.save_version(policy_version, value_model, optimizer, update_summary)
            if update == 1 or update % 10 == 0 or update == updates:
                print(f"Update {update}/{updates}: episodes={len(episode_returns)} "
                      f"hit100={update_summary['recent_100_hit_rate']} reward={update_summary['recent_100_return']}", flush=True)
        metrics = {'architecture': architecture, 'seed': seed,
                   'parameters': sum(p.numel() for p in policy.parameters()),
                   'updates': updates, 'decisions': global_decision, 'episodes': len(episode_returns),
                   'hits': int(sum(episode_hits)), 'overall_hit_rate': float(np.mean(episode_hits)) if episode_hits else None,
                   'recent_100_hit_rate': float(np.mean(episode_hits[-100:])) if episode_hits else None,
                   'recent_100_return': float(np.mean(episode_returns[-100:])) if episode_returns else None,
                   'mean_episode_length': float(np.mean(completed_lengths)) if completed_lengths else None,
                   'model_structure_preserved': True,
                   'topology_and_signs_preserved': None if architecture == 'mlp' else True,
                   'training_seconds_including_recording': time.perf_counter() - start}
        atomic_json(Path(output) / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2)); return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--updates', type=int, default=80); parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=64); parser.add_argument('--action-repeat', type=int, default=2)
    parser.add_argument('--architecture', choices=('flywire', 'random', 'mlp'), default='flywire')
    parser.add_argument('--scenario',
                        choices=('aiming', 'navigation', 'integrated-navigation', 'moving-target',
                                 'partial-observability'),
                        default='aiming')
    parser.add_argument('--entropy-coefficient', type=float, default=0.01)
    parser.add_argument('--final-entropy-coefficient', type=float)
    parser.add_argument('--initial-policy-run', type=Path)
    parser.add_argument('--initial-policy-version', type=int)
    parser.add_argument('--temporal-state-decay', type=float)
    args = parser.parse_args()
    environment = (navigation_config() if args.scenario == 'navigation' else
                   integrated_navigation_config() if args.scenario == 'integrated-navigation' else
                   moving_target_config() if args.scenario == 'moving-target' else
                   partial_observability_config() if args.scenario == 'partial-observability' else
                   CombatConfig())
    train(args.output, args.seed, args.updates, args.workers,
                                      args.horizon, args.action_repeat, architecture=args.architecture,
                                      environment_config=environment,
                                      entropy_coefficient=args.entropy_coefficient,
                                      final_entropy_coefficient=args.final_entropy_coefficient,
                                      initial_policy_run=args.initial_policy_run,
                                      initial_policy_version=args.initial_policy_version,
                                      temporal_state_decay=args.temporal_state_decay)
