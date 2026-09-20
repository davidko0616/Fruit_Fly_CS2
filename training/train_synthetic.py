"""Train a small connectome network; choose checkpoints using validation only."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from connectome.graph import ConnectomeGraph
from models.flywire_network import FlyWireNetwork
from training.synthetic import make_spiral, stratified_split


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def fingerprint(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plot_results(model, x, y, splits, history, run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    epochs = [row['epoch'] for row in history]
    for split in ('train', 'validation'):
        axes[0].plot(epochs, [row[f'{split}_loss'] for row in history], label=split)
        axes[1].plot(epochs, [row[f'{split}_accuracy'] for row in history], label=split)
    axes[0].set(title='Cross-entropy loss', xlabel='Epoch', ylabel='Loss')
    axes[1].set(title='Accuracy (test held out)', xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.03))
    axes[1].axhline(1 / len(np.unique(y)), color='gray', linestyle=':', label='Chance')
    for axis in axes[:2]:
        axis.legend()
        axis.grid(alpha=0.2)
    grid_x, grid_y = np.meshgrid(np.linspace(-1.1, 1.1, 160), np.linspace(-1.1, 1.1, 160))
    grid = torch.from_numpy(np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32))
    with torch.no_grad():
        predictions = torch.cat([model(batch).argmax(1) for batch in grid.split(1024)]).numpy()
    axes[2].contourf(grid_x, grid_y, predictions.reshape(grid_x.shape),
                     levels=np.arange(len(np.unique(y)) + 1) - 0.5, cmap='viridis', alpha=0.22)
    idx = splits['test']
    axes[2].scatter(x[idx, 0], x[idx, 1], c=y[idx], cmap='viridis', s=14, edgecolors='none')
    axes[2].set(title='Decision regions and held-out test points', xlabel='x1', ylabel='x2', aspect='equal')
    fig.suptitle('FlyWire AL_R · 100 neurons · three-class spiral')
    fig.tight_layout()
    fig.savefig(run_dir / 'training.png', dpi=160)
    plt.close(fig)


def train(config_path, seed=None):
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if seed is not None:
        config['experiment']['seed'] = seed
    seed = config['experiment']['seed']
    options = config['training']
    if options['device'] != 'cpu' or options['optimizer'] != 'adam' or config['data']['task'] != 'spiral':
        raise ValueError('This milestone supports CPU, Adam, and spiral classification')
    if options['epochs'] < 1 or options['batch_size'] < 1 or options['learning_rate'] <= 0:
        raise ValueError('Training epochs, batch size, and learning rate must be positive')
    if config['connectome']['version'] != '783':
        raise ValueError('Only the downloaded v783 dataset is configured')
    if config['connectome']['io_strategy'] != 'flow_with_computational_fallback':
        raise ValueError('Unknown IO strategy')
    torch.manual_seed(seed)
    torch.set_num_threads(options['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    run_dir = ROOT / config['experiment']['output_dir'] / config['experiment']['name'] / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    start = time.monotonic()
    graph = ConnectomeGraph(ROOT / 'data/raw', ROOT / 'data/metadata')
    selection = config['connectome']['subgraph']
    subgraph = graph.extract_subgraph_by_neuropil(selection['region'], selection['max_neurons'])
    inputs, outputs, hidden = graph.assign_io_neurons(subgraph['metadata'])
    adjacency = subgraph['adjacency']
    components = connected_components(adjacency, directed=True, connection='weak', return_labels=False)
    if components != 1:
        raise ValueError(f'Selected graph has {components} weak components; revise selection before training')
    reachable = np.zeros(adjacency.shape[0], dtype=np.int64)
    reachable[inputs] = 1
    binary = adjacency.copy()
    binary.data[:] = 1
    for _ in range(config['model']['num_steps']):
        reachable = np.asarray(binary.T @ reachable > 0, dtype=np.int64)
    if not reachable[outputs].all():
        raise ValueError('Some outputs cannot receive inputs in the configured message-passing steps')
    sp.save_npz(run_dir / 'adjacency.npz', adjacency)
    np.savez(run_dir / 'graph.npz', root_ids=np.asarray(subgraph['root_ids'], dtype=np.int64),
             signs=subgraph['nt_signs'], inputs=inputs, outputs=outputs, hidden=hidden)
    flow_counts = subgraph['metadata']['flow'].value_counts().to_dict()
    stats = {'neurons': adjacency.shape[0], 'edges': adjacency.nnz,
             'density_including_self_pairs': adjacency.nnz / adjacency.shape[0] ** 2,
             'weak_components': int(components), 'inputs': len(inputs), 'outputs': len(outputs),
             'hidden': len(hidden), 'flow_counts': flow_counts,
             'computational_input_fallback': flow_counts.get('afferent', 0) == 0,
             'computational_output_fallback': flow_counts.get('efferent', 0) == 0,
             'all_outputs_reachable_at_num_steps': True,
             'excitatory_neurons': int((subgraph['nt_signs'] > 0).sum()),
             'inhibitory_neurons': int((subgraph['nt_signs'] < 0).sum())}
    del graph
    data_config = config['data']
    x, y = make_spiral(data_config['n_samples'], data_config['n_classes'], data_config['noise'], seed)
    splits = stratified_split(y, data_config['validation_fraction'], data_config['test_fraction'], seed + 1)
    np.savez(run_dir / 'dataset.npz', x=x, y=y, **splits)
    tensors = {name: (torch.from_numpy(x[idx]), torch.from_numpy(y[idx])) for name, idx in splits.items()}
    model = FlyWireNetwork(adjacency, subgraph['nt_signs'], inputs, outputs, 2,
                           data_config['n_classes'], config['model']['num_steps'],
                           config['connectome']['weight_init'])
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'], foreach=False)
    generator = torch.Generator().manual_seed(seed + 2)
    fixed_indices = model.connectome_layer.indices.clone()
    fixed_signs = model.connectome_layer.edge_signs.clone()
    history, best_state, best_epoch, best_val = [], None, 0, float('inf')
    core_received_gradients = False

    def evaluate(split):
        with torch.no_grad():
            values, labels = tensors[split]
            logits = model(values)
            return float(F.cross_entropy(logits, labels)), float((logits.argmax(1) == labels).float().mean())

    initial_train_loss, initial_train_accuracy = evaluate('train')
    training_start = time.monotonic()
    for epoch in range(1, options['epochs'] + 1):
        model.train()
        tx, ty = tensors['train']
        for batch in torch.randperm(len(tx), generator=generator).split(options['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(tx[batch]), ty[batch])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError('Missing or nonfinite parameter gradient')
            core_received_gradients |= bool(model.connectome_layer.weight_magnitudes.grad.abs().sum() > 0)
            optimizer.step()
            core = model.connectome_layer
            if not torch.equal(core.indices, fixed_indices) or not torch.equal(core.edge_signs, fixed_signs):
                raise RuntimeError('Connectivity or sign assignments changed during training')
            weights = core.weight_magnitudes.abs() * core.edge_signs
            if not torch.isfinite(weights).all() or not torch.all(weights * fixed_signs >= 0):
                raise RuntimeError('Invalid effective weights')
        model.eval()
        train_loss, train_acc = evaluate('train')
        val_loss, val_acc = evaluate('validation')
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                        'validation_loss': val_loss, 'validation_accuracy': val_acc})
        if val_loss < best_val:
            best_val, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0:
            print(f'Epoch {epoch:3d}: loss={train_loss:.4f} train={train_acc:.1%} val={val_acc:.1%}', flush=True)
    training_seconds = time.monotonic() - training_start
    model.load_state_dict(best_state)
    model.eval()
    # Test data is evaluated only once, after selecting the checkpoint on validation loss.
    test_loss, test_accuracy = evaluate('test')
    selected_train_loss, selected_train_accuracy = evaluate('train')
    selected_val_loss, selected_val_accuracy = evaluate('validation')
    passed = (core_received_gradients and selected_train_loss < initial_train_loss
              and selected_val_accuracy >= config['evaluation']['minimum_validation_accuracy']
              and test_accuracy >= config['evaluation']['minimum_test_accuracy'])
    metrics = {'milestone_passed': passed, 'seed': seed, 'best_epoch': best_epoch,
               'checkpoint_selection': 'minimum validation cross-entropy; test evaluated once afterward',
               'initial_train_loss': initial_train_loss, 'initial_train_accuracy': initial_train_accuracy,
               'train_loss': selected_train_loss, 'train_accuracy': selected_train_accuracy,
               'validation_loss': selected_val_loss, 'validation_accuracy': selected_val_accuracy,
               'test_loss': test_loss, 'test_accuracy': test_accuracy,
               'chance_accuracy': 1 / data_config['n_classes'], 'parameters': sum(p.numel() for p in model.parameters()),
               'split_sizes': {name: len(idx) for name, idx in splits.items()},
               'core_received_nonzero_gradients': core_received_gradients,
               'topology_and_signs_preserved': True, 'training_seconds': training_seconds, 'graph': stats}
    sources = [ROOT / 'data/raw/proofread_connections_783.feather',
               ROOT / 'data/raw/proofread_root_ids_783.npy', ROOT / 'data/metadata/neuron_annotations.tsv']
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    source_code = [ROOT / 'connectome/graph.py', ROOT / 'models/sparse_layer.py',
                   ROOT / 'models/flywire_network.py', Path(__file__), ROOT / 'training/synthetic.py']
    provenance = {'dataset_version': '783', 'files_sha256': {str(p.relative_to(ROOT)): fingerprint(p) for p in sources},
                  'source_code_sha256': {str(p.relative_to(ROOT)): fingerprint(p) for p in source_code},
                  'git_commit': commit, 'git_dirty': dirty, 'python': sys.version,
                  'platform': platform.platform(), 'processor': platform.processor(),
                  'device': 'cpu', 'threads': torch.get_num_threads(),
                  'packages': {name: importlib.metadata.version(name) for name in
                               ('torch', 'numpy', 'scipy', 'pandas', 'pyarrow', 'matplotlib', 'pyyaml')},
                  'sign_assumption': 'GABA/glutamate negative; others and missing predictions positive',
                  'selection': 'Top degree within neuropil; ties broken by ascending root ID',
                  'weight_init': config['connectome']['weight_init']}
    torch.save({'state_dict': best_state, 'config': config, 'best_epoch': best_epoch}, run_dir / 'model.pt')
    write_json(run_dir / 'history.json', history)
    write_json(run_dir / 'provenance.json', provenance)
    plot_results(model, x, y, splits, history, run_dir)
    metrics['total_seconds'] = time.monotonic() - start
    write_json(run_dir / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f'Artifacts: {run_dir}', flush=True)
    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'training/configs/synthetic_100.yaml')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    raise SystemExit(0 if train(args.config, args.seed) else 1)
