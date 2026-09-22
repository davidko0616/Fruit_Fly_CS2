"""Lossless, bounded recording of classifier forwards and optimizer updates.

Each atomically committed chunk is independently readable. The event log is
rebuilt from chunks, so even a process killed before its manifest update does
not hide completed data. No pickle is used for traces or parameter versions.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F


def replace_with_retry(source, destination):
    """OneDrive/antivirus may briefly hold Windows file handles during replacement."""
    delays = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def array(tensor):
    return tensor.detach().cpu().numpy().copy()


class ActivityRecorder:
    """Use as a context manager; any recording error must propagate to training."""

    def __init__(self, directory, model, root_ids, config, max_buffer_samples=1024):
        if max_buffer_samples < 1:
            raise ValueError('max_buffer_samples must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'chunks').mkdir()
        (self.directory / 'weights').mkdir()
        (self.directory / 'checkpoints').mkdir()
        self.model = model
        self.limit = max_buffer_samples
        self.pending, self.pending_samples = [], 0
        self.event_count = self.sample_count = self.chunk_count = 0
        self.seconds = 0.0
        self.closed = False
        self.started = time.monotonic()
        self.fixed_indices = array(model.connectome_layer.indices)
        self.fixed_signs = array(model.connectome_layer.edge_signs)
        self.parameter_names = [name for name, _ in model.named_parameters()]
        self.manifest = {
            'schema_version': 1, 'run_id': str(uuid.uuid4()), 'status': 'running',
            'task': 'spiral_classification', 'config': config,
            'capture': 'every forward, every sample, every internal step; float32 without quantization',
            'state_semantics': 'step 0: injected input; steps 1..K: post-ReLU; reset every forward',
            'input_transform': 'identity', 'normalization': None,
            'activation_kind': 'artificial model activation, not biological measurement',
            'device': str(next(model.parameters()).device),
            'parameter_names': self.parameter_names,
            'parameter_shapes': [list(p.shape) for p in model.parameters()],
            'num_steps': model.num_steps, 'num_neurons': model.num_neurons,
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }
        inputs, outputs = list(map(int, model.input_idx)), list(map(int, model.output_idx))
        if len(root_ids) != model.num_neurons:
            raise ValueError('Neuron ID count does not match model')
        graph = {
            # FlyWire IDs exceed JavaScript's safe integer range: always strings.
            'root_ids': [str(i) for i in root_ids], 'inputs': inputs, 'outputs': outputs,
            'roles': ['input' if i in inputs else 'output' if i in outputs else 'internal'
                      for i in range(model.num_neurons)],
            'sources': self.fixed_indices[1].tolist(), 'targets': self.fixed_indices[0].tolist(),
            'edge_signs': self.fixed_signs.tolist(), 'layout': 'computational; not anatomical',
        }
        atomic_json(self.directory / 'graph.json', graph)
        self._manifest()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close('complete' if exc_type is None else 'interrupted',
                   None if exc is None else f'{exc_type.__name__}: {exc}')
        return False

    def _manifest(self):
        self.manifest.update(events=self.event_count, samples=self.sample_count,
                             committed_chunks=self.chunk_count,
                             recording_seconds=self.seconds)
        atomic_json(self.directory / 'manifest.json', self.manifest)

    @contextmanager
    def _capture(self):
        sensory, pre, post = [], [], []
        hooks = [self.model.input_proj.register_forward_hook(lambda m, a, out: sensory.append(array(out))),
                 self.model.connectome_layer.register_forward_hook(lambda m, a, out: pre.append(array(out))),
                 self.model.activation.register_forward_hook(lambda m, a, out: post.append(array(out)))]
        try:
            yield sensory, pre, post
        finally:
            for hook in hooks:
                hook.remove()

    def forward(self, x, labels, sample_ids, *, split, phase, epoch, version):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        if not (self.directory / 'weights' / f'{version:07d}.npz').exists():
            raise ValueError(f'Weight version {version} was not saved')
        if len(x) != len(labels) or len(x) != len(sample_ids) or len(x) == 0:
            raise ValueError('Inputs, labels, and sample IDs must be nonempty and aligned')
        if len(x) > self.limit:
            raise ValueError('Forward batch exceeds recorder buffer limit; increase limit or split batch')
        if self.pending_samples + len(x) > self.limit:
            self.flush()
        started = time.monotonic()
        with self._capture() as (sensory, pre, post):
            logits = self.model(x)
        forward_finished = time.monotonic()
        if len(sensory) != 1 or len(post) != self.model.num_steps or len(pre) != len(post):
            raise RuntimeError('Unexpected model activation sequence')
        initial = np.zeros((len(x), self.model.num_neurons), dtype=sensory[0].dtype)
        initial[:, self.model.input_idx] = sensory[0]
        with torch.no_grad():
            probabilities = array(logits.softmax(dim=1))
            losses = array(F.cross_entropy(logits, labels, reduction='none'))
        values = {
            'inputs': array(x), 'labels': array(labels),
            'sample_ids': np.asarray(sample_ids, dtype=np.int64),
            'sensory': sensory[0], 'states': np.stack([initial, *post], axis=1),
            'preactivations': np.stack(pre, axis=1), 'logits': array(logits),
            'probabilities': probabilities, 'predictions': probabilities.argmax(axis=1), 'losses': losses,
        }
        self._append_event(values, split, phase, epoch, version, started, forward_finished,
                           values['states'], values['states'][:, 1:])
        return logits

    def _append_event(self, values, split, phase, epoch, version, started, forward_finished,
                      activity, post_activity):
        """Shared bounded storage for recurrent and feedforward activity schemas."""
        if not all(np.isfinite(v).all() for v in values.values()):
            raise RuntimeError('Nonfinite recorded values')
        event = {'event_id': self.event_count, 'split': split, 'phase': phase,
                 'epoch': int(epoch), 'version': int(version), 'count': len(values['inputs']),
                 'training_mode': self.model.training, 'grad_enabled': torch.is_grad_enabled(),
                 'monotonic_seconds': started - self.started,
                 'forward_with_capture_seconds': forward_finished - started,
                 'mean_loss': float(values['losses'].mean()),
                 'accuracy': float((values['predictions'] == values['labels']).mean()),
                 'activation_min': float(activity.min()),
                 'activation_max': float(activity.max()),
                 'activation_mean': float(activity.mean()),
                 'inactive_fraction': float((post_activity == 0).mean())}
        self.pending.append((event, values))
        self.pending_samples += len(values['inputs'])
        self.event_count += 1
        self.sample_count += len(values['inputs'])
        # Capture time is included in the forward measurement; this is post-forward I/O preparation.
        self.seconds += time.monotonic() - forward_finished
        if self.pending_samples >= self.limit:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        started = time.monotonic()
        offset, events = 0, []
        for event, values in self.pending:
            events.append({**event, 'offset': offset})
            offset += event['count']
        arrays = {key: np.concatenate([v[key] for _, v in self.pending])
                  for key in self.pending[0][1]}
        atomic_npz(self.directory / 'chunks' / f'{self.chunk_count:06d}.npz',
                   events=np.asarray(json.dumps(events)), **arrays)
        self.chunk_count += 1
        self.pending, self.pending_samples = [], 0
        self.seconds += time.monotonic() - started
        self._manifest()

    def save_weights(self, version, diagnostics=None):
        started = time.monotonic()
        destination = self.directory / 'weights' / f'{version:07d}.npz'
        if destination.exists():
            raise FileExistsError(destination)
        self._check_constraints()
        values = {f'p{i}': array(p) for i, p in enumerate(self.model.parameters())}
        if not all(np.isfinite(v).all() for v in values.values()):
            raise RuntimeError('Nonfinite parameters')
        atomic_npz(destination, metadata=np.asarray(json.dumps({
            'version': version, 'topology_and_signs_preserved': True if hasattr(self.model, 'connectome_layer') else None,
            **(diagnostics or {})})), **values)
        self.seconds += time.monotonic() - started

    def _check_constraints(self):
        core = self.model.connectome_layer
        if not np.array_equal(array(core.indices), self.fixed_indices) or not np.array_equal(array(core.edge_signs), self.fixed_signs):
            raise RuntimeError('Topology or edge signs changed')

    def checkpoint(self, name, optimizer, generator, progress):
        """Full continuation state; parameter versions are separately stored for every update."""
        started = time.monotonic()
        destination = self.directory / 'checkpoints' / f'{name}.pt'
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.with_suffix('.pt.tmp')
        payload = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'torch_rng': torch.get_rng_state(), 'batch_rng': generator.get_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if next(self.model.parameters()).is_cuda else [],
                   'progress': progress, 'config': self.manifest['config'],
                   'scheduler': None, 'normalization': None}
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, destination)
        self.seconds += time.monotonic() - started

    def close(self, status='complete', error=None):
        if self.closed:
            return
        try:
            self.flush()
        except BaseException as failure:
            self.manifest.update(status='interrupted', error=f'Recording flush failed: {failure}')
            try:
                self._manifest()
            except OSError:
                pass  # A full disk may prevent even an error marker. Complete chunks remain readable.
            raise
        self.manifest.update(status=status, error=error,
                             elapsed_seconds=time.monotonic() - self.started)
        self.manifest['bytes_on_disk_before_final_manifest'] = sum(
            p.stat().st_size for p in self.directory.rglob('*') if p.is_file())
        self._manifest()
        self.closed = True


def read_events(directory):
    """Recover index from committed chunks, including chunks missing from stale manifests."""
    events = []
    for path in sorted((Path(directory) / 'chunks').glob('*.npz')):
        with np.load(path, allow_pickle=False) as chunk:
            for event in json.loads(str(chunk['events'])):
                events.append({**event, 'chunk': path.name})
    ids = [event['event_id'] for event in events]
    if ids != list(range(len(events))):
        raise ValueError('Recording has missing or duplicate events')
    return events
