"""Summarize the predeclared five-seed experiment without selecting runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


NAMES = {'flywire': 'FlyWire', 'random': 'Random destinations', 'mlp': 'MLP'}


def summarize(results):
    records = [json.loads((results / f'{architecture}_seed_{seed}.json').read_text())
               for architecture in NAMES for seed in range(42, 47)]
    rows = []
    for record in records:
        m, config = record['metrics'], record['config']
        assert config['epochs'] == 200 and record['verification']['audit_passed']
        assert m['recorded_sample_forwards'] == m['expected_sample_forwards'] == 312904
        best = min(record['history'], key=lambda r: r['validation']['loss'])
        assert best['epoch'] == m['best_epoch']
        np.testing.assert_allclose(best['validation']['loss'], m['validation']['loss'], rtol=1e-5)
        rows.append({'architecture': m['architecture'], 'seed': m['seed'], 'parameters': m['parameters'],
                     'test_accuracy_percent': 100 * m['test']['accuracy'], 'test_loss': m['test']['loss'],
                     'validation_accuracy_percent': 100 * m['validation']['accuracy'], 'best_epoch': m['best_epoch'],
                     'validation_95_epoch': m['first_validation_95_epoch'],
                     'validation_95_seconds': m['first_validation_95_seconds_including_recording'],
                     'recorded_training_seconds': m['training_seconds_including_recording'],
                     'peak_working_set_mib': m['peak_process_working_set_mib'],
                     'recording_mib': record['bytes_on_disk'] / 2**20})
    with (results.parent / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped = {architecture: [r for r in rows if r['architecture'] == architecture] for architecture in NAMES}
    summary = {}
    for architecture, group in grouped.items():
        summary[architecture] = {'seeds': [r['seed'] for r in group], 'runs': len(group)}
        for key in list(rows[0])[3:]:
            values = np.array([r[key] for r in group if r[key] is not None], dtype=float)
            summary[architecture][key] = {'n': len(values), 'mean': float(values.mean()) if len(values) else None,
                'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None,
                'min': float(values.min()) if len(values) else None, 'max': float(values.max()) if len(values) else None}
    pairs = {}
    for other in ('random', 'mlp'):
        differences = [f['test_accuracy_percent'] - o['test_accuracy_percent']
                       for f, o in zip(grouped['flywire'], grouped[other])]
        pairs[other] = {'flywire_minus_baseline_test_percentage_points': differences,
                       'mean_difference': float(np.mean(differences)),
                       'sample_sd_difference': float(np.std(differences, ddof=1)),
                       'flywire_wins_ties_losses': [sum(d > 1e-6 for d in differences),
                                                  sum(abs(d) <= 1e-6 for d in differences),
                                                  sum(d < -1e-6 for d in differences)]}
    (results.parent / 'summary.json').write_text(json.dumps({'architectures': summary, 'paired': pairs}, indent=2) + '\n')
    plt.rcParams.update({'figure.dpi': 150, 'font.size': 10})
    colors = {'flywire': '#2776a8', 'random': '#bd6127', 'mlp': '#56833e'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for architecture in NAMES:
        subset = [r for r in records if r['metrics']['architecture'] == architecture]
        accuracy = np.array([[h['validation']['accuracy'] * 100 for h in r['history']] for r in subset])
        loss = np.array([[h['validation']['loss'] for h in r['history']] for r in subset])
        for ax, values in [(axes[0, 0], accuracy), (axes[0, 1], loss)]:
            mean, sd = values.mean(0), values.std(0, ddof=1)
            ax.plot(np.arange(1, 201), mean, label=NAMES[architecture], color=colors[architecture])
            if ax is axes[0, 0]:
                ax.fill_between(np.arange(1, 201), np.maximum(mean - sd, 0), np.minimum(mean + sd, 100),
                                alpha=.13, color=colors[architecture])
            else:
                for seed_values in values:
                    ax.plot(np.arange(1, 201), seed_values, alpha=.15, linewidth=.8, color=colors[architecture])
    axes[0, 0].set(title='Validation accuracy: mean ± sample SD', xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 102))
    axes[0, 1].set(title='Validation loss: individual seeds and mean', xlabel='Epoch', ylabel='Cross-entropy', yscale='log')
    axes[0, 0].legend()
    for ax, key, title, ylabel in [(axes[1, 0], 'test_accuracy_percent', 'Held-out accuracy: all five seeds', 'Accuracy (%)'),
                                   (axes[1, 1], 'recorded_training_seconds', 'CPU training with full recording', 'Seconds')]:
        for i, (architecture, group) in enumerate(grouped.items()):
            values = [r[key] for r in group]
            ax.scatter(i + np.linspace(-.12, .12, 5), values, color=colors[architecture], s=35)
            ax.errorbar(i, np.mean(values), yerr=np.std(values, ddof=1), fmt='_', color='black', capsize=5)
        ax.set(xticks=range(3), xticklabels=list(NAMES.values()), title=title, ylabel=ylabel)
    axes[1, 0].set_ylim(min(r['test_accuracy_percent'] for r in rows) - 1, 100.5)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Fixed spiral split · seeds 42–46 · 200 epochs · ~7,778 parameters')
    fig.savefig(results.parent / 'comparison.png'); plt.close(fig)

    def fmt(architecture, key, digits=2):
        v = summary[architecture][key]
        return f"{v['mean']:.{digits}f} ± {v['sample_sd']:.{digits}f}" if v['n'] == 5 else f"{v['n']}/5 reached threshold"
    lines = ['# Five-seed CPU baseline comparison', '',
             'Completed all 15 predeclared runs (three architectures, seeds 42–46). All',
             'recordings passed independent audits. No runs were excluded and no settings',
             'were tuned after observing these results. See [protocol](PROTOCOL.md).', '',
             '## Results', '', 'Values are mean ± sample standard deviation across five seeds.', '',
             '| Architecture | Parameters | Test accuracy (%) | Test loss | First validation ≥95% epoch | Recorded training (s) |',
             '|---|---:|---:|---:|---:|---:|']
    for architecture in NAMES:
        lines.append(f"| {NAMES[architecture]} | {grouped[architecture][0]['parameters']:,} | {fmt(architecture, 'test_accuracy_percent')} | {fmt(architecture, 'test_loss', 4)} | {fmt(architecture, 'validation_95_epoch', 1)} | {fmt(architecture, 'recorded_training_seconds', 1)} |")
    lines += ['', '![Comparison curves and individual-seed outcomes](comparison.png)', '', '## Paired differences', '']
    for other, values in pairs.items():
        w, t, l = values['flywire_wins_ties_losses']
        lines.append(f"- FlyWire minus {NAMES[other]} test accuracy: {values['mean_difference']:+.3f} percentage points on average; {w} wins, {t} ties, {l} losses across paired seeds.")
    lines += ['', 'The MLP reached 95% validation accuracy before FlyWire and the random',
              'control in all five paired seeds. Its mean recorded runtime and mean test',
              'cross-entropy were also lower. FlyWire\'s small accuracy lead consists of one',
              'additional correct prediction at seed 46; it is not a consistent accuracy',
              'separation across seeds. The evidence does not establish a fly-wiring advantage.', '',
              'Peak process working set (mean ± sample SD, MiB): ' + '; '.join(
                  f"{NAMES[a]} {fmt(a, 'peak_working_set_mib', 1)}" for a in NAMES) + '.', '',
              'The independent [matching audit](matching_audit.json) confirmed identical',
              'paired minibatch sequences, dataset bytes, recurrent IO initialization,',
              'edge budgets and source-sign assignments. The random graphs retain about',
              '81.5–82.0% of original edge positions, as expected for this dense circuit',
              'under source-degree constraints; this limits how strongly this control',
              'perturbs the topology. Signed edge totals match: 2,596 inhibitory and',
              '5,019 excitatory edges per recurrent network.']
    lines += ['', '## Interpretation and limits', '',
              'This is a small, dense connectome circuit on one synthetic classification',
              'task. Five runs share the same train/validation/test split; the error bars',
              'measure training variability, not uncertainty across tasks or data splits.',
              'With 148 test points, one changed prediction moves a run by about 0.676',
              'percentage points. Small differences near the accuracy ceiling do not',
              'establish a biological advantage.', '',
              'The random control preserves source degree, source sign, self-loop policy',
              'and per-source synapse-count multiset; incoming degrees and motifs differ.',
              'The MLP matches parameter count within one parameter, but uses different',
              'depth, initialization, unconstrained signs and computation. None of these',
              'results alone proves a mechanism or transfers to reinforcement learning.', '',
              'Runtime includes full activity/parameter recording and excludes startup,',
              'initial evaluation and post-training audit. Captured activity sizes differ;',
              'background load and disk writes affect timing. It is not a pure model-speed',
              'benchmark. Peak process working set includes the Python runtime and recorder.', '',
              '## Artifacts and verification', '',
              f"- Total captured sample-forwards: {sum(r['samples'] for r in records):,}.",
              f"- Full run artifacts: {sum(r['bytes_on_disk'] for r in records) / 2**30:.2f} GiB, retained locally under `artifacts/synthetic/baseline_comparison_100/`.",
              '- [Per-seed metrics](per_seed.csv), [aggregate statistics](summary.json), and `results/` contain compact results, histories, configuration and source hashes.',
              '- Every audit checks committed arrays, sample/split assignments, parameter-version availability and three reproduced forward passes.',
              '- The selected checkpoint is the minimum-validation-loss epoch; test is evaluated once per full run afterward.',
              '- Two-epoch smoke runs preceded the full experiment and are excluded from these results.',
              '- MLP traces use schema 2 (hidden layers with offsets); the existing recurrent viewer accepts schema 1 only.', '',
              '## Next work', '',
              'Keep all three architectures for the next stage. Before claiming a wiring',
              'benefit, compare predeclared harder tasks and independent dataset splits,',
              'or a less dense circuit where topology controls differ more substantially.',
              'A minimal toy-combat environment can now be developed as the next task,',
              'with synchronized observation/action/reward/activity recording and the same',
              'baseline comparisons. Larger models or GPU setup are not required to begin.', '',
              '## Reproduce', '', 'From the repository root, using a fresh output directory:', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe tools/run_comparison.py --output artifacts/synthetic/baseline_comparison_100_repeat --results experiments/baseline_comparison_100_repeat/results',
              '.\\.venv\\Scripts\\python.exe tools/summarize_comparison.py --results experiments/baseline_comparison_100_repeat/results', '```', '']
    (results.parent / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'architectures': summary, 'paired': pairs}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('experiments/baseline_comparison_100/results'))
    summarize(parser.parse_args().results)
