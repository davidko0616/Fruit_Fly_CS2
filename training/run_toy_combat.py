"""Run a fully recorded untrained connectome policy in parallel toy environments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import scipy.sparse as sp
import torch

from models.flywire_network import FlyWireNetwork
from models.mlp_baseline import MLPBaseline, matched_hidden_sizes
from models.random_sparse import randomize_destinations
from toy_combat.env import ACTION_NAMES, REWARD_NAMES, CombatConfig, ToyCombatEnv
from toy_combat.recording import CombatRecorder, RESET_CODES, policy_forward_with_activity
from training.recording import atomic_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments/cpu_spiral_100/seed_42'


def load_policy(seed, architecture='flywire', observation_size=ToyCombatEnv.observation_size,
                action_size=ToyCombatEnv.action_size):
    if architecture not in ('flywire', 'random', 'mlp'):
        raise ValueError(f'Unknown architecture: {architecture}')
    with np.load(ARCHIVE / 'graph.npz', allow_pickle=False) as file:
        graph = dict(file)
    reference_adjacency = sp.load_npz(ARCHIVE / 'adjacency.npz')
    torch.manual_seed(seed)
    reference = FlyWireNetwork(reference_adjacency, graph['signs'], graph['inputs'], graph['outputs'],
                               observation_size, action_size, 3,
                               'normalized_synapse_count')
    parameter_budget = sum(parameter.numel() for parameter in reference.parameters())
    hidden_sizes = matched_hidden_sizes(parameter_budget, observation_size, action_size)
    if architecture == 'flywire':
        model, adjacency = reference, reference_adjacency
    elif architecture == 'random':
        adjacency = randomize_destinations(reference_adjacency, seed)
        torch.manual_seed(seed)
        model = FlyWireNetwork(adjacency, graph['signs'], graph['inputs'], graph['outputs'],
                               observation_size, action_size, 3,
                               'normalized_synapse_count')
    else:
        adjacency = reference_adjacency
        torch.manual_seed(seed)
        model = MLPBaseline(observation_size, action_size, hidden_sizes)
    model.eval()
    return model, graph, adjacency, hidden_sizes


def run(output, decisions=128, workers=2, seed=42, action_repeat=2):
    if decisions < workers or workers < 1:
        raise ValueError('Decisions must cover at least one decision per worker')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    model, source_graph, _, _ = load_policy(seed)
    environments = [ToyCombatEnv() for _ in range(workers)]
    episode_ids = list(range(workers)); episode_counters = [0] * workers
    episode_seeds = [seed * 100000 + worker for worker in range(workers)]
    observations, infos = zip(*(env.reset(s) for env, s in zip(environments, episode_seeds)))
    observations, infos = list(observations), list(infos)
    generator = torch.Generator().manual_seed(seed + 1)
    graph = {'root_ids': [str(i) for i in source_graph['root_ids']],
             'inputs': source_graph['inputs'].tolist(), 'outputs': source_graph['outputs'].tolist(),
             'sources': model.connectome_layer.indices[1].tolist(),
             'targets': model.connectome_layer.indices[0].tolist(),
             'edge_signs': model.connectome_layer.edge_signs.tolist()}
    config = {'seed': seed, 'workers': workers, 'decisions': decisions, 'action_repeat': action_repeat,
              'policy_version': 0, 'policy': 'untrained_connectome_smoke',
              'observation_size': ToyCombatEnv.observation_size, 'action_names': ACTION_NAMES,
              'reward_names': REWARD_NAMES, 'environment': CombatConfig().__dict__}
    with CombatRecorder(output, model, graph, config) as recorder:
        for name in ('graph.npz', 'adjacency.npz'):
            shutil.copyfile(ARCHIVE / name, Path(output) / name)
        atomic_json(Path(output) / 'provenance.json', {'torch': torch.__version__, 'numpy': np.__version__,
            'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__), ROOT / 'toy_combat/env.py',
                                           ROOT / 'toy_combat/recording.py', ROOT / 'models/flywire_network.py')}})
        totals = {'episodes': 0, 'hits': 0, 'reward': 0.0}
        for global_decision in range(decisions):
            worker = global_decision % workers
            env = environments[worker]
            obs = torch.from_numpy(np.stack([observations[worker]]))
            mask = torch.from_numpy(np.stack([infos[worker]['action_mask']]))
            before_state = infos[worker]['privileged_state'].copy()
            with torch.no_grad():
                logits, probabilities, states, pre, sensory = policy_forward_with_activity(model, obs, mask)
                action = int(torch.multinomial(probabilities[0], 1, generator=generator))
            log_probability = float(torch.log(probabilities[0, action]))
            positive = probabilities[0] > 0
            entropy = float(-(probabilities[0, positive] * probabilities[0, positive].log()).sum())
            next_observation, reward, terminated, truncated, outcome = env.step(action, action_repeat)
            components = np.asarray([outcome['reward_components'][name] for name in REWARD_NAMES], dtype=np.float32)
            recorder.append(worker_id=worker, episode_id=episode_ids[worker],
                episode_seed=episode_seeds[worker], decision_index=episode_counters[worker],
                global_decision=global_decision, policy_version=0, tick_before=before_state[-1],
                tick_after=outcome['tick'], observation=observations[worker], transformed_input=observations[worker],
                action_mask=infos[worker]['action_mask'], sensory=sensory[0], states=states[0],
                preactivations=pre[0], logits=array(logits[0]), probabilities=array(probabilities[0]),
                chosen_action=action, executed_action=action, log_probability=log_probability, entropy=entropy,
                action_applied_ticks=outcome['action_applied_ticks'], action_rejected=outcome['action_rejected'],
                reward=reward, reward_components=components, next_observation=next_observation,
                terminated=terminated, truncated=truncated, reset_reason=RESET_CODES[outcome['reset_reason']],
                hit=outcome['hit'], miss=outcome['miss'], collision=outcome['collision'],
                privileged_before=before_state, privileged_after=outcome['privileged_state'])
            totals['reward'] += reward; totals['hits'] += int(outcome['hit'])
            observations[worker], infos[worker] = next_observation, outcome
            episode_counters[worker] += 1
            if terminated or truncated:
                totals['episodes'] += 1
                episode_ids[worker] += workers
                episode_counters[worker] = 0
                episode_seeds[worker] = seed * 100000 + episode_ids[worker]
                observations[worker], infos[worker] = env.reset(episode_seeds[worker])
        metrics = {**totals, 'decisions': decisions, 'mean_reward_per_decision': totals['reward'] / decisions,
                   'completed_episode_hit_rate': totals['hits'] / totals['episodes'] if totals['episodes'] else None}
        atomic_json(Path(output) / 'metrics.json', metrics)
        torch.save({'state_dict': model.state_dict(), 'config': config}, Path(output) / 'model.pt')
    print(json.dumps(metrics, indent=2))
    return metrics


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--action-repeat', type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.decisions, args.workers, args.seed, args.action_repeat)
