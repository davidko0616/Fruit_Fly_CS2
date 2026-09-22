"""Descriptive training/validation analysis; excludes held-out test events."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def analyze(run, output):
    if output.resolve() == run.resolve() or run.resolve() in output.resolve().parents:
        raise ValueError('Keep analysis outputs outside the source recording')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / 'manifest.json').read_text())
    graph = json.loads((run / 'graph.json').read_text())
    with np.load(run / 'dataset.npz', allow_pickle=False) as data:
        split_ids = {s: np.sort(data[s]) for s in ('train', 'validation')}
    rows, snapshots = [], {}
    for path in sorted((run / 'chunks').glob('*.npz')):
        with np.load(path, allow_pickle=False) as chunk:
            selected = [e for e in json.loads(str(chunk['events']))
                        if e['split'] in split_ids and e['phase'] in ('initial', 'epoch')]
            if not selected:
                continue
            arrays = {k: chunk[k] for k in ('states', 'sample_ids', 'labels', 'predictions',
                                           'losses', 'probabilities', 'inputs')}
            for event in selected:
                sl = slice(event['offset'], event['offset'] + event['count'])
                a = {k: v[sl] for k, v in arrays.items()}
                np.testing.assert_array_equal(np.sort(a['sample_ids']), split_ids[event['split']])
                assert np.isfinite(a['states']).all()
                states = a['states'][:, 1:]
                assert (states >= 0).all()
                row = dict(split=event['split'], epoch=event['epoch'], version=event['version'],
                           samples=event['count'], accuracy=float(np.mean(a['predictions'] == a['labels'])),
                           loss=float(a['losses'].mean()))
                for step in range(states.shape[1]):
                    values = states[:, step]
                    row.update({f'step{step+1}_mean': float(values.mean()),
                                f'step{step+1}_zero_fraction': float(np.mean(values == 0)),
                                f'step{step+1}_inactive_neurons': int(np.sum(np.all(values == 0, axis=0)))})
                rows.append(row)
                if event['epoch'] in (0, 1, 10, 50, 100, 200):
                    snapshots[event['split'], event['epoch']] = a
    rows.sort(key=lambda r: (r['split'], r['epoch']))
    for split in split_ids:
        assert len([r for r in rows if r['split'] == split and r['epoch'] > 0]) == manifest['config']['epochs']
    history = json.loads((run / 'history.json').read_text())
    for row in rows:
        if row['epoch']:
            for key in ('accuracy', 'loss'):
                np.testing.assert_allclose(row[key], history[row['epoch'] - 1][row['split']][key],
                                           rtol=1e-6, atol=1e-7)
    with (output / 'epoch_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for split, color in [('train', '#2471a3'), ('validation', '#c05621')]:
        r = [v for v in rows if v['split'] == split]
        x = [v['epoch'] for v in r]
        for ax, key, title in zip(axes.flat,
                ['accuracy', 'loss', 'step3_zero_fraction', 'step3_mean'],
                ['Classification accuracy', 'Cross-entropy loss', 'Final-step zero activation fraction', 'Final-step mean activation']):
            ax.plot(x, [v[key] for v in r], label=split, color=color)
            ax.set(xlabel='Epoch', title=title)
            ax.grid(alpha=.2)
    axes[0, 1].set_yscale('log')
    axes[0, 0].legend()
    fig.suptitle('Recorded seed 42: learning and activity (fixed evaluation samples)')
    fig.savefig(output / 'learning.png'); plt.close(fig)

    summary = {'run_id': manifest['run_id'], 'scope': 'Train and validation evaluations only; no test analysis',
               'definitions': {'inactive': 'Exactly zero for every sample in this split at this step; not proof of causal irrelevance',
                               'eta_squared': 'Between-class sum of squares / total sum of squares, final-step activation; descriptive, no significance test'},
               'splits': {}}
    neuron_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, split in zip(axes, split_ids):
        a = snapshots[split, 200]
        values = a['states'][:, -1].astype(float)
        labels = a['labels']
        means = np.array([values[labels == c].mean(axis=0) for c in range(3)])
        total = ((values - values.mean(axis=0)) ** 2).sum(axis=0)
        between = sum(np.sum(labels == c) * (means[c] - values.mean(axis=0)) ** 2 for c in range(3))
        eta = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
        inactive = [np.flatnonzero(np.all(a['states'][:, step] == 0, axis=0)).tolist() for step in range(1, 4)]
        r = [v for v in rows if v['split'] == split]
        summary['splits'][split] = {'first': r[0], 'last': r[-1],
            'first_epoch_95_percent': next((v['epoch'] for v in r if v['accuracy'] >= .95), None),
            'first_epoch_perfect': next((v['epoch'] for v in r if v['accuracy'] == 1), None),
            'inactive_by_step_final': inactive,
            'inactive_all_processing_steps_final': sorted(set.intersection(*(set(v) for v in inactive))),
            'top_class_associated_neurons': [{'index': int(i), 'role': graph['roles'][i], 'eta_squared': float(eta[i])}
                for i in np.argsort(-eta)[:10]],
            'mistakes_at_snapshots': {str(epoch): int(np.sum(v['labels'] != v['predictions']))
                for (s, epoch), v in snapshots.items() if s == split}}
        # Within-neuron standardized class means avoid equating brightness with selectivity.
        std = values.std(axis=0)
        standardized = np.divide(means - values.mean(axis=0), std, out=np.zeros_like(means), where=std > 0)
        im = ax.imshow(standardized, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set(title=f'{split.capitalize()}: epoch 200', xlabel='Neuron index', ylabel='Class', yticks=[0, 1, 2])
        for i in range(values.shape[1]):
            neuron_rows.append({'split': split, 'index': i, 'root_id': graph['root_ids'][i],
                'role': graph['roles'][i], 'active_fraction': float(np.mean(values[:, i] > 0)),
                'mean': float(values[:, i].mean()), 'eta_squared': float(eta[i]),
                **{f'class{c}_mean': float(means[c, i]) for c in range(3)}})
    fig.colorbar(im, ax=axes, label='Standardized class mean (SD)')
    fig.suptitle('Class-associated responses at the final processing step')
    fig.savefig(output / 'class_responses.png'); plt.close(fig)
    with (output / 'neurons.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(neuron_rows[0])); writer.writeheader(); writer.writerows(neuron_rows)

    epochs = [1, 10, 50, 200]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    for ax, epoch in zip(axes, epochs):
        a = snapshots['validation', epoch]
        wrong = a['labels'] != a['predictions']
        ax.scatter(*a['inputs'].T, c=a['labels'], cmap='viridis', s=14, vmin=0, vmax=2)
        ax.scatter(*a['inputs'][wrong].T, facecolors='none', edgecolors='red', s=45, linewidths=1)
        ax.set(title=f'Epoch {epoch}: {wrong.sum()}/148 errors', xlabel='x', ylabel='y', aspect='equal')
    fig.suptitle('Same validation points across training; red rings mark errors')
    fig.savefig(output / 'validation_errors.png'); plt.close(fig)

    # Same-input early/late comparison with explicit ID alignment.
    early, late = snapshots['train', 0], snapshots['train', 200]
    ei, li = np.argsort(early['sample_ids']), np.argsort(late['sample_ids'])
    np.testing.assert_array_equal(early['sample_ids'][ei], late['sample_ids'][li])
    summary['same_training_inputs'] = {'count': len(ei), 'mean_absolute_final_step_change': float(
        np.abs(late['states'][li, -1] - early['states'][ei, -1]).mean())}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run, args.output)
