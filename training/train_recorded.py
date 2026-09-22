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
from models.random_sparse import randomize_destinations
from models.mlp_baseline import MLPBaseline, matched_hidden_sizes
from training.recording import ActivityRecorder, atomic_json
from training.mlp_recording import MLPActivityRecorder

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments/cpu_spiral_100/seed_42'


def peak_working_set_mib():
    """Windows process peak resident working set, including runtime and recorder."""
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    *[(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                      'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                      'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / 2**20


def load_version(model, directory, version):
    with np.load(Path(directory) / 'weights' / f'{version:07d}.npz', allow_pickle=False) as weights:
        with torch.no_grad():
            for i, parameter in enumerate(model.parameters()):
                parameter.copy_(torch.from_numpy(weights[f'p{i}']).to(parameter.device))


def train(output, device='cpu', epochs=200, seed=42, architecture='flywire'):
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
    if architecture not in ('flywire', 'random', 'mlp'):
        raise ValueError('Unknown architecture')
    if architecture == 'random':
        adjacency = randomize_destinations(adjacency, seed)
    model = FlyWireNetwork(adjacency, graph['signs'], graph['inputs'], graph['outputs'],
                           2, 3, 3, 'normalized_synapse_count').to(device)
    parameter_budget = sum(p.numel() for p in model.parameters())
    hidden_sizes = matched_hidden_sizes(parameter_budget)
    if architecture == 'mlp':
        torch.manual_seed(seed)
        model = MLPBaseline(2, 3, hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    generator = torch.Generator().manual_seed(seed + 2)
    tensors = {s: (torch.from_numpy(dataset['x'][dataset[s]]).to(device),
                   torch.from_numpy(dataset['y'][dataset[s]]).to(device))
               for s in ('train', 'validation', 'test')}
    config = {'epochs': epochs, 'model_seed': seed, 'data_seed': 42, 'batch_seed': seed + 2,
              'batch_size': 64, 'learning_rate': 0.001, 'optimizer': 'Adam', 'cpu_threads': 4,
              'num_steps': 3 if architecture != 'mlp' else None,
              'weight_init': 'normalized_synapse_count' if architecture != 'mlp' else 'pytorch_linear_default',
              'architecture': architecture, 'hidden_sizes': list(hidden_sizes) if architecture == 'mlp' else None,
              'parameter_budget': parameter_budget, 'parameters': sum(p.numel() for p in model.parameters()),
              'topology_seed': seed if architecture == 'random' else None,
              'random_control': 'source degree, self-loops, source sign, per-source count multiset preserved' if architecture == 'random' else None,
              'dataset': 'archived seed_42 spiral, fixed splits',
              'checkpoint_selection': 'minimum validation loss; test once afterward'}
    expected_samples = 704 + epochs * (704 + 704 + 148) + 1000
    activity_values = 700 if architecture != 'mlp' else 2 * sum(hidden_sizes)
    print(f'Full capture: {expected_samples:,} sample-forwards; approximately '
          f'{expected_samples * activity_values * 4 / 2**20:.0f} MiB uncompressed activity arrays, '
          'plus inputs, outputs and parameter snapshots.', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    version = epoch = 0
    history, best_loss, best_version, best_epoch = [], float('inf'), 0, 0
    core_received_gradients = False
    recorder_type = MLPActivityRecorder if architecture == 'mlp' else ActivityRecorder
    with recorder_type(output, model, graph['root_ids'], config) as recorder:
        for name in ('graph.npz', 'adjacency.npz', 'dataset.npz'):
            shutil.copyfile(ARCHIVE / name, Path(output) / name)
        if architecture == 'random':
            sp.save_npz(Path(output) / 'adjacency.npz', adjacency)
        sources = [Path(__file__), ROOT / 'training/recording.py', ROOT / 'models/flywire_network.py',
                   ROOT / 'models/sparse_layer.py', ARCHIVE / 'graph.npz',
                   ARCHIVE / 'adjacency.npz', ARCHIVE / 'dataset.npz', ROOT / 'models/random_sparse.py',
                   ROOT / 'models/mlp_baseline.py', ROOT / 'training/mlp_recording.py']
        atomic_json(Path(output) / 'provenance.json', {
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'platform': platform.platform(), 'device': str(device),
            'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None,
            'cuda_runtime': torch.version.cuda, 'deterministic_algorithms': True,
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
            'biological_caveats': 'AL_R; computational IO fallbacks; sign assumptions; dense small circuit',
            'architecture': architecture,
            'run_adjacency_sha256': hashlib.sha256((Path(output) / 'adjacency.npz').read_bytes()).hexdigest()})
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
                    if architecture != 'mlp':
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
                history.append({'epoch': epoch, 'version': version, 'train': train_metrics, 'validation': validation,
                                'elapsed_seconds_including_recording': time.perf_counter() - start})
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
                   'architecture': architecture, 'seed': seed, 'parameters': config['parameters'],
                   'first_validation_95_epoch': next((r['epoch'] for r in history if r['validation']['accuracy'] >= .95), None),
                   'first_validation_95_seconds_including_recording': next((r['elapsed_seconds_including_recording'] for r in history if r['validation']['accuracy'] >= .95), None),
                   'core_received_nonzero_gradients': core_received_gradients if architecture != 'mlp' else None,
                   'topology_and_signs_preserved': True if architecture != 'mlp' else None, 'expected_sample_forwards': expected_samples,
                   'recorded_sample_forwards': recorder.sample_count,
                   'peak_process_working_set_mib': peak_working_set_mib(),
                   'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None}
        atomic_json(Path(output) / 'history.json', history)
        atomic_json(Path(output) / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    if architecture == 'mlp':
        print(f'MLP schema 2 recording: {Path(output).resolve()} (numerical audit supported; recurrent viewer not applicable)', flush=True)
    else:
        print(f'Viewer: python -m visualization.serve --run "{Path(output).resolve()}"', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42, help='Model seed; archived dataset remains fixed')
    parser.add_argument('--architecture', choices=['flywire', 'random', 'mlp'], default='flywire')
    args = parser.parse_args()
    train(args.output, args.device, args.epochs, args.seed, args.architecture)


if __name__ == '__main__':
    main()
