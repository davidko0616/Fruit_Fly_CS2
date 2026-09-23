"""Lossless decision-to-outcome recording for toy combat."""
import json
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from training.recording import array, atomic_json, atomic_npz, replace_with_retry


RESET_CODES = {None: 0, 'hit': 1, 'time_limit': 2}


def policy_forward_with_activity(model, observations, action_masks):
    sensory, pre, post, hooks = [], [], [], []
    hooks.append(model.input_proj.register_forward_hook(lambda m, a, out: sensory.append(array(out))))
    hooks.append(model.connectome_layer.register_forward_hook(lambda m, a, out: pre.append(array(out))))
    hooks.append(model.activation.register_forward_hook(lambda m, a, out: post.append(array(out))))
    try:
        logits = model(observations)
    finally:
        for hook in hooks:
            hook.remove()
    if len(sensory) != 1 or len(pre) != model.num_steps or len(post) != model.num_steps:
        raise RuntimeError('Unexpected activity sequence')
    masked = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
    probabilities = masked.softmax(1)
    initial = np.zeros((len(observations), model.num_neurons), dtype=np.float32)
    initial[:, model.input_idx] = sensory[0]
    return logits, probabilities, np.stack([initial, *post], axis=1), np.stack(pre, axis=1), sensory[0]


class CombatRecorder:
    def __init__(self, directory, model, graph, config, max_buffer_decisions=256):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'chunks').mkdir()
        (self.directory / 'weights').mkdir()
        (self.directory / 'checkpoints').mkdir()
        self.model, self.limit = model, max_buffer_decisions
        self.pending, self.count, self.chunks = [], 0, 0
        self.started, self.closed = time.monotonic(), False
        self.run_id = str(uuid.uuid4())
        self.manifest = {'schema_version': 1, 'task': 'toy_combat', 'run_id': self.run_id,
                         'status': 'running', 'config': config, 'decisions': 0, 'committed_chunks': 0,
                         'capture': 'every decision, full model activity, action, reward and outcome',
                         'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        atomic_json(self.directory / 'manifest.json', self.manifest)
        atomic_json(self.directory / 'graph.json', graph)
        self.save_version(0)

    def __enter__(self): return self

    def __exit__(self, kind, error, traceback):
        self.close('complete' if kind is None else 'interrupted',
                   None if error is None else f'{kind.__name__}: {error}')
        return False

    def append(self, **values):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        arrays = {key: np.asarray(value) for key, value in values.items()}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('Nonfinite decision data')
        self.pending.append(arrays); self.count += 1
        if len(self.pending) >= self.limit:
            self.flush()

    def save_version(self, version, value_model=None, optimizer=None, diagnostics=None):
        destination = self.directory / 'weights' / f'{int(version):07d}.npz'
        if destination.exists():
            raise FileExistsError(destination)
        arrays = {f'p{i}': array(p) for i, p in enumerate(self.model.parameters())}
        if value_model is not None:
            arrays.update({f'v{i}': array(p) for i, p in enumerate(value_model.parameters())})
        arrays['metadata'] = np.asarray(json.dumps({'version': int(version), **(diagnostics or {})}))
        atomic_npz(destination, **arrays)
        if optimizer is not None:
            temporary = self.directory / 'checkpoints' / f'{int(version):07d}.pt.tmp'
            torch.save({'policy': self.model.state_dict(), 'value': value_model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'version': int(version)}, temporary)
            replace_with_retry(temporary, temporary.with_suffix(''))

    def write_updates(self, updates):
        atomic_json(self.directory / 'updates.json', updates)

    def flush(self):
        if not self.pending: return
        keys = self.pending[0].keys()
        if any(row.keys() != keys for row in self.pending):
            raise RuntimeError('Decision schema changed')
        atomic_npz(self.directory / 'chunks' / f'{self.chunks:06d}.npz',
                   **{key: np.stack([row[key] for row in self.pending]) for key in keys})
        self.pending = []; self.chunks += 1
        self.manifest.update(decisions=self.count, committed_chunks=self.chunks)
        atomic_json(self.directory / 'manifest.json', self.manifest)

    def close(self, status='complete', error=None):
        if self.closed: return
        self.flush()
        self.manifest.update(status=status, error=error, decisions=self.count,
                             committed_chunks=self.chunks, elapsed_seconds=time.monotonic() - self.started,
                             bytes_on_disk=sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file()))
        atomic_json(self.directory / 'manifest.json', self.manifest)
        self.closed = True
