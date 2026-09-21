"""Train the archived 100-neuron experiment on CPU or CUDA without raw data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.flywire_network import FlyWireNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate', type=Path, help='Reload a checkpoint in a fresh process')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    archive = ROOT / 'experiments/cpu_spiral_100/seed_42'
    graph = np.load(archive / 'graph.npz')
    data = np.load(archive / 'dataset.npz')
    adjacency = sp.load_npz(archive / 'adjacency.npz')
    model = FlyWireNetwork(adjacency, graph['signs'], graph['inputs'], graph['outputs'],
                           2, 3, 3, 'normalized_synapse_count').to(device)
    splits = {s: (torch.from_numpy(data['x'][data[s]]).to(device),
                  torch.from_numpy(data['y'][data[s]]).to(device))
              for s in ('train', 'validation', 'test')}

    def evaluate(split):
        model.eval()
        with torch.no_grad():
            x, y = splits[split]
            logits = model(x)
            return {'loss': F.cross_entropy(logits, y).item(),
                    'accuracy': (logits.argmax(1) == y).float().mean().item()}

    if args.evaluate:
        model.load_state_dict(torch.load(args.evaluate, map_location=device, weights_only=True)['state_dict'])
        result = {'device': str(device), 'checkpoint': str(args.evaluate), 'test': evaluate('test')}
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
    generator = torch.Generator().manual_seed(44)
    indices = model.connectome_layer.indices.clone()
    signs = model.connectome_layer.edge_signs.clone()
    initial = evaluate('train')
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history, best_state, best_loss, best_epoch = [], None, float('inf'), 0
    core_gradients = False
    for epoch in range(1, 201):
        model.train()
        x, y = splits['train']
        for batch in torch.randperm(len(x), generator=generator).split(64):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            for p in model.parameters():
                if p.grad is None or not torch.isfinite(p.grad).all():
                    raise RuntimeError('Missing or nonfinite gradient')
            core = model.connectome_layer
            core_gradients |= bool(core.weight_magnitudes.grad.abs().sum() > 0)
            optimizer.step()
            if not torch.equal(core.indices, indices) or not torch.equal(core.edge_signs, signs):
                raise RuntimeError('Topology or signs changed')
            weights = core.weight_magnitudes.abs() * core.edge_signs
            if not torch.isfinite(weights).all() or not (weights * signs >= 0).all():
                raise RuntimeError('Invalid effective weights')
        train, validation = evaluate('train'), evaluate('validation')
        history.append({'epoch': epoch, 'train': train, 'validation': validation})
        if validation['loss'] < best_loss:
            best_loss, best_epoch = validation['loss'], epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            print(f"{device} epoch {epoch}: train={train['accuracy']:.1%} validation={validation['accuracy']:.1%}", flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    result = {'device': str(device), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'python': sys.version, 'platform': platform.platform(), 'seed': 42,
              'epochs': 200, 'batch_size': 64, 'cpu_threads': 4, 'deterministic_algorithms': True,
              'neurons': adjacency.shape[0], 'edges': adjacency.nnz,
              'parameters': sum(p.numel() for p in model.parameters()),
              'initial_train': initial, 'train': evaluate('train'),
              'validation': evaluate('validation'), 'test': evaluate('test'),
              'best_epoch': best_epoch, 'training_seconds': seconds,
              'core_received_nonzero_gradients': core_gradients, 'topology_and_signs_preserved': True,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / 'models/flywire_network.py',
                                          ROOT / 'models/sparse_layer.py', archive / 'graph.npz',
                                          archive / 'adjacency.npz', archive / 'dataset.npz']}}
    if device.type == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(),
                      total_vram_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
                      peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
    result['passed'] = bool(core_gradients and result['test']['accuracy'] >= 0.8
                            and result['validation']['accuracy'] >= 0.8
                            and result['train']['loss'] < initial['loss'])
    torch.save({'state_dict': {k: v.cpu() for k, v in best_state.items()},
                'best_epoch': best_epoch}, args.output / 'model.pt')
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
