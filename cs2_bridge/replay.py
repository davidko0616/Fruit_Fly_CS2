"""Offline bridge replay through a fixed policy; no game input is emitted."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from toy_combat.env import ACTION_NAMES
from toy_combat.recording import policy_forward_with_activity

from .encoder import Dust2ObservationEncoder
from .schema import BridgeFrame


def read_frames(path):
    frames = []
    with Path(path).open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                frames.append(BridgeFrame.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f'Invalid bridge frame at line {line_number}: {error}') from error
    if not frames:
        raise ValueError('Replay contains no bridge frames')
    return frames


class PolicyRunner:
    """Stateful, read-only policy inference shared by replay and live shadow."""

    def __init__(self, encoder: Dust2ObservationEncoder, model,
                 mode='greedy', seed=0, execution='offline_replay_only'):
        if mode not in ('greedy', 'stochastic'):
            raise ValueError('Replay mode must be greedy or stochastic')
        self.encoder = encoder
        self.model = model
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.execution = execution
        self.model.eval()

    def decide(self, frame: BridgeFrame):
        encoded = self.encoder.encode(frame)
        observation = encoded.observation
        mask = torch.tensor(frame.action_mask, dtype=torch.bool)
        with torch.no_grad():
            observation_tensor = torch.from_numpy(observation)[None]
            if all(hasattr(self.model, name) for name in
                   ('input_proj', 'connectome_layer', 'activation', 'num_neurons')):
                logits_batch, probabilities_batch, states, preactivations, sensory = (
                    policy_forward_with_activity(
                        self.model, observation_tensor, mask[None]))
                logits = logits_batch[0]
                probabilities = probabilities_batch[0].cpu().numpy()
                activity = {
                    'sensory': sensory[0].tolist(),
                    'neuron_states': states[0].tolist(),
                    'neuron_preactivations': preactivations[0].tolist(),
                }
            else:
                logits = self.model(observation_tensor)[0]
                masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
                probabilities = masked.softmax(0).cpu().numpy()
                activity = None
        if self.mode == 'greedy':
            action = int(probabilities.argmax())
        else:
            action = int(self.rng.choice(len(probabilities), p=probabilities))
        digest = hashlib.sha256(observation.tobytes()).hexdigest()
        return {
            'schema_version': 1,
            'sequence': frame.sequence,
            'monotonic_ns': frame.monotonic_ns,
            'source_tick': frame.source_tick,
            'round_id': frame.round_id,
            'observation': observation.tolist(),
            'observation_sha256': digest,
            'target_observation_is_live': encoded.target_observation_is_live,
            'target_memory_in_observation': encoded.target_memory_in_observation,
            'has_last_seen_target': encoded.has_last_seen_target,
            'last_seen_age_ns': encoded.last_seen_age_ns,
            'waypoint_planner_active': encoded.waypoint_planner_active,
            'waypoint_world': encoded.waypoint_world,
            'waypoint_path_remaining': encoded.waypoint_path_remaining,
            'waypoint_target_snap_world': encoded.waypoint_target_snap_world,
            'waypoint_planner_rejection': encoded.waypoint_planner_rejection,
            'waypoint_source': encoded.waypoint_source,
            'patrol_active': encoded.patrol_active,
            'patrol_goal_world': encoded.patrol_goal_world,
            'patrol_goal_index': encoded.patrol_goal_index,
            'action_mask': list(frame.action_mask),
            'logits': logits.cpu().numpy().tolist(),
            'probabilities': probabilities.tolist(),
            'activity': activity,
            'action': action,
            'action_name': ACTION_NAMES[action],
            'execution': self.execution,
        }


def replay_frames(frames, encoder: Dust2ObservationEncoder, model, mode='greedy', seed=0):
    """Return fully inspectable decisions for synchronized recorded frames."""
    runner = PolicyRunner(encoder, model, mode, seed)
    return [runner.decide(frame) for frame in frames]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    with path.open('x', encoding='utf-8', newline='\n') as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
