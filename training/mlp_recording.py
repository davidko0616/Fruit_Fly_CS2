"""Full feedforward traces using shared durable recording machinery (schema 2).

Hidden layers are concatenated with explicit offsets, not depicted as recurrent
steps. The existing recurrent-network viewer deliberately rejects this schema.
"""
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.recording import ActivityRecorder, array, atomic_json


class MLPActivityRecorder(ActivityRecorder):
    def __init__(self, directory, model, root_ids, config, max_buffer_samples=1024):
        if max_buffer_samples < 1:
            raise ValueError('max_buffer_samples must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        for name in ('chunks', 'weights', 'checkpoints'):
            (self.directory / name).mkdir()
        self.model, self.limit = model, max_buffer_samples
        self.pending, self.pending_samples = [], 0
        self.event_count = self.sample_count = self.chunk_count = 0
        self.seconds, self.closed, self.started = 0.0, False, time.monotonic()
        self.parameter_names = [n for n, _ in model.named_parameters()]
        self.parameter_shapes = [list(p.shape) for p in model.parameters()]
        self.manifest = {
            'schema_version': 2, 'run_id': str(uuid.uuid4()), 'status': 'running',
            'task': 'spiral_classification', 'architecture': 'mlp', 'config': config,
            'capture': 'every forward and sample; all hidden pre/post-ReLU values and logits; float32',
            'state_semantics': 'hidden layers concatenated in feedforward order; no recurrent state',
            'hidden_sizes': list(model.hidden_sizes),
            'hidden_offsets': np.cumsum([0, *model.hidden_sizes]).tolist(),
            'input_transform': 'identity', 'normalization': None,
            'device': str(next(model.parameters()).device),
            'parameter_names': self.parameter_names, 'parameter_shapes': self.parameter_shapes,
            'sign_constraints': 'none; dense signed weights train freely',
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        atomic_json(self.directory / 'architecture.json', {
            'kind': 'mlp', 'input_dim': model.input_dim, 'output_dim': model.output_dim,
            'hidden_sizes': list(model.hidden_sizes)})
        self._manifest()

    def _check_constraints(self):
        if self.parameter_names != [n for n, _ in self.model.named_parameters()] or self.parameter_shapes != [list(p.shape) for p in self.model.parameters()]:
            raise RuntimeError('MLP architecture changed')

    def forward(self, x, labels, sample_ids, *, split, phase, epoch, version):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        if not (self.directory / 'weights' / f'{version:07d}.npz').exists():
            raise ValueError('Weight version was not saved')
        if not (0 < len(x) <= self.limit and len(x) == len(labels) == len(sample_ids)):
            raise ValueError('Invalid batch length')
        if self.pending_samples + len(x) > self.limit:
            self.flush()
        pre, post, hooks = [], [], []
        for module in list(self.model.net)[:-1]:
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(lambda m, a, out: pre.append(array(out))))
            elif isinstance(module, torch.nn.ReLU):
                hooks.append(module.register_forward_hook(lambda m, a, out: post.append(array(out))))
        started = time.monotonic()
        try:
            logits = self.model(x)
        finally:
            for hook in hooks:
                hook.remove()
        finished = time.monotonic()
        if len(pre) != len(self.model.hidden_sizes) or len(post) != len(pre):
            raise RuntimeError('Unexpected MLP capture sequence')
        with torch.no_grad():
            probabilities = array(logits.softmax(1))
            losses = array(F.cross_entropy(logits, labels, reduction='none'))
        values = {'inputs': array(x), 'labels': array(labels), 'sample_ids': np.asarray(sample_ids, dtype=np.int64),
                  'hidden_preactivations': np.concatenate(pre, axis=1),
                  'hidden_activations': np.concatenate(post, axis=1), 'logits': array(logits),
                  'probabilities': probabilities, 'predictions': probabilities.argmax(1), 'losses': losses}
        self._append_event(values, split, phase, epoch, version, started, finished,
                           values['hidden_activations'], values['hidden_activations'])
        return logits
