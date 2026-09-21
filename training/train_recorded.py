"""Train the existing archived 100-neuron circuit with complete activity recording."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from models.flywire_network import FlyWireNetwork
from training.recording import ActivityRecorder, atomic_json

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments/cpu_spiral_100/seed_42'


def load_version(model, directory, version):
    with np.load(Path(directory) / 'weights' / f'{version:07d}.npz', allow_pickle=False) as weights:
        with torch.no_grad():
            for i, parameter in enumerate(model.parameters()):
                parameter.copy_(torch.from_numpy(weights[f'p{i}']).to(parameter.device))


def train(output, device='cpu', epochs=200, seed=42):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    with np.load(ARCHIVE / 'graph.npz', allow_pickle=False) as file:
        graph = dict(file)
    with np.load(ARCHIVE / 'dataset.npz', allow_pickle=False) as file:
        dataset = dict(file)
    adjacency = sp.load_npz(ARCHIVE / 'adjacency.npz')
    model = FlyWireNetwork(adjacency, graph['signs'], graph['inputs'], graph['outputs'],
                           2, 3, 3, 'normalized_synapse_count').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    generator = torch.Generator().manual_seed(seed + 2)
    tensors = {s: (torch.from_numpy(dataset['x'][dataset[s]]).to(device),
                   torch.from_numpy(dataset['y'][dataset[s]]).to(device))
               for s in ('train', 'validation', 'test')}
    config = {'epochs': epochs, 'model_seed': seed, 'data_seed': 42, 'batch_seed': seed + 2,
              'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'Adam', 'cpu_threads': 4,
              'num_steps': 3, 'weight_init': 'normalized_synapse_count',
              'dataset': 'archived seed_42 spiral, fixed splits',
              'checkpoint_selection': 'minimum validation loss; test once afterward'}
    expected_samples = 704 + epochs * (704 + 704 + 148) + 1000
    print(f'Full capture: {expected_samples:,} sample-forwards; approximately '
          f'{expected_samples * 700 * 4 / 2**20:.0f} MiB uncompressed state/preactivation arrays, '
          'plus inputs, outputs and parameter snapshots.', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    version = epoch = 0
    history, best_loss, best_version, best_epoch = [], float('inf'), 0, 0
    core_received_gradients = False
    with ActivityRecorder(output, model, graph['root_ids'], config) as recorder:
        for name in ('graph.npz', 'adjacency.npz', 'dataset.npz'):
            shutil.copyfile(ARCHIVE / name, Path(output) / name)
        sources = [Path(__file__), ROOT / 'training/recording.py', ROOT / 'models/flywire_network.py',
                   ROOT / 'models/sparse_layer.py', ARCHIVE / 'graph.npz',
                   ARCHIVE / 'adjacency.npz', ARCHIVE / 'dataset.npz']
        atomic_json(Path(output) / 'provenance.json', {
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'platform': platform.platform(), 'device': str(device),
            'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'cuda_runtime': torch.version.cuda, 'deterministic_algorithms': True,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            'biological_caveats': 'AL_R; computational IO fallbacks; sign assumptions; dense small circuit'})
        recorder.save_weights(0)
        recorder.checkpoint('initial', optimizer, generator, {'epoch': 0, 'version': 0})

        def evaluate(split, phase):
            model.eval()
            with torch.no_grad():
                x, y = tensors[split]
                logits = recorder.forward(x, y, dataset[split], split=split, phase=phase,
                                          epoch=epoch, version=version)
                return {'loss': F.cross_entropy(logits, y).item(),
                        'accuracy': (logits.argmax(1) == y).float().mean().item()}

        initial = evaluate('train', 'initial')
        start = time.perf_counter()
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                x, y = tensors['train']
                permutation = torch.randperm(len(x), generator=generator)
                for batch_index, batch in enumerate(permutation.split(64)):
                    batch_device = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = recorder.forward(x[batch_device], y[batch_device], dataset['train'][batch.numpy()],
                                              split='train', phase='optimization', epoch=epoch, version=version)
                    loss = F.cross_entropy(logits, y[batch_device])
                    backward_start = time.perf_counter()
                    loss.backward()
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    backward_seconds = time.perf_counter() - backward_start
                    diagnostics, before = {}, []
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise RuntimeError('Missing or nonfinite gradient')
                        diagnostics[name] = {'gradient_norm': parameter.grad.norm().item()}
                        before.append(parameter.detach().clone())
                    core_received_gradients |= bool(model.connectome_layer.weight_magnitudes.grad.abs().sum() > 0)
                    optimizer.step()
                    version += 1
                    for (name, parameter), old in zip(model.named_parameters(), before):
                        diagnostics[name].update(weight_norm=parameter.norm().item(),
                                                 update_norm=(parameter - old).norm().item())
                    recorder.save_weights(version, {
                        'from_version': version - 1, 'epoch': epoch, 'batch_index': batch_index,
                        'forward_event': recorder.event_count - 1, 'loss': loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'finite_gradients': True, 'backward_seconds': backward_seconds,
                        'parameters': diagnostics})
                train_metrics, validation = evaluate('train', 'epoch'), evaluate('validation', 'epoch')
                history.append({'epoch': epoch, 'version': version, 'train': train_metrics, 'validation': validation})
                if validation['loss'] < best_loss:
                    best_loss, best_version, best_epoch = validation['loss'], version, epoch
                if epoch % 25 == 0 or epoch == epochs:
                    recorder.flush()
                    recorder.checkpoint(f'epoch_{epoch:04d}', optimizer, generator,
                                        {'epoch': epoch, 'version': version, 'best_version': best_version,
                                         'best_loss': best_loss, 'best_epoch': best_epoch})
                if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
                    print(f'Epoch {epoch}: train={train_metrics["accuracy"]:.1%} '
                          f'validation={validation["accuracy"]:.1%}; '
                          f'{recorder.sample_count:,} sample-forwards recorded', flush=True)
        except BaseException:
            try:
                recorder.checkpoint('interrupted', optimizer, generator,
                                    {'epoch': epoch, 'version': version,
                                     'best_version': best_version, 'best_loss': best_loss,
                                     'note': 'Mid-epoch recovery; batch sequence is in recorded events'})
            except OSError:
                pass
            raise
        seconds = time.perf_counter() - start
        load_version(model, output, best_version)
        version = best_version
        selected = {split: evaluate(split, 'selected') for split in ('train', 'validation', 'test')}
        torch.save({'state_dict': model.state_dict(), 'best_version': best_version,
                    'best_epoch': best_epoch, 'config': config}, Path(output) / 'model.pt')
        metrics = {'initial_train': initial, **selected, 'best_epoch': best_epoch,
                   'best_version': best_version, 'training_seconds_including_recording': seconds,
                   'core_received_nonzero_gradients': core_received_gradients,
                   'topology_and_signs_preserved': True, 'expected_sample_forwards': expected_samples,
                   'recorded_sample_forwards': recorder.sample_count,
                   'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None}
        atomic_json(Path(output) / 'history.json', history)
        atomic_json(Path(output) / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'Viewer: python -m visualization.serve --run "{Path(output).resolve()}"', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42, help='Model seed; archived dataset remains fixed')
    args = parser.parse_args()
    train(args.output, args.device, args.epochs, args.seed)


if __name__ == '__main__':
    main()
