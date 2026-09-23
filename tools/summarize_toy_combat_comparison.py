"""Create compact tables and figures for the fixed toy-combat comparison."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARCHITECTURES = ('flywire', 'random', 'mlp')
NAMES = {'flywire': 'FlyWire', 'random': 'Random destinations', 'mlp': 'MLP'}
COLORS = {'flywire': '#2776a8', 'random': '#bd6127', 'mlp': '#56833e'}


def policy_row(record, version, mode):
    return next(row for row in record['evaluation']['policies']
                if row['policy_version'] == version and row['mode'] == mode)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {'n': int(len(values)), 'mean': float(values.mean()),
            'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'min': float(values.min()), 'max': float(values.max())}


def summarize(results, seeds):
    records = {(architecture, seed): json.loads(
        (results / f'{architecture}_seed_{seed}.json').read_text())
        for architecture in ARCHITECTURES for seed in seeds}
    rows = []
    for architecture in ARCHITECTURES:
        for seed in seeds:
            record = records[architecture, seed]
            assert record['audit']['audit_passed'] and record['audit']['decisions'] == 51_200
            assert record['config']['parameters'] == 7_913
            initial = policy_row(record, 0, 'stochastic')
            final_stochastic = policy_row(record, 100, 'stochastic')
            final_greedy = policy_row(record, 100, 'greedy')
            learning = sorted((row for row in record['evaluation']['policies']
                               if row['mode'] == 'stochastic'), key=lambda row: row['policy_version'])
            versions = np.asarray([row['policy_version'] for row in learning])
            hit_rates = 100 * np.asarray([row['hit_rate'] for row in learning])
            rows.append({'architecture': architecture, 'seed': seed, 'parameters': 7913,
                         'initial_stochastic_hit_percent': 100 * initial['hit_rate'],
                         'final_stochastic_hit_percent': 100 * final_stochastic['hit_rate'],
                         'final_stochastic_return': final_stochastic['return_mean'],
                         'final_stochastic_length': final_stochastic['length_mean'],
                         'final_greedy_hit_percent': 100 * final_greedy['hit_rate'],
                         'hit_curve_auc_percent': float(np.trapezoid(hit_rates, versions) /
                                                        (versions[-1] - versions[0])),
                         'training_overall_hit_percent': 100 * record['metrics']['overall_hit_rate'],
                         'recorded_training_seconds': record['metrics']['training_seconds_including_recording'],
                         'recording_mib': record['bytes_on_disk'] / 2**20})
    with (results.parent / 'per_seed.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    grouped = {architecture: [row for row in rows if row['architecture'] == architecture]
               for architecture in ARCHITECTURES}
    summary = {'seeds': seeds, 'architectures': {}, 'paired_flywire_differences': {}}
    measures = list(rows[0])[3:]
    for architecture, group in grouped.items():
        summary['architectures'][architecture] = {
            measure: mean_sd([row[measure] for row in group]) for measure in measures}
    for baseline in ('random', 'mlp'):
        comparisons = {}
        for measure in ('final_stochastic_hit_percent', 'final_stochastic_return',
                        'hit_curve_auc_percent', 'final_greedy_hit_percent'):
            differences = [grouped['flywire'][index][measure] - grouped[baseline][index][measure]
                           for index in range(len(seeds))]
            comparisons[measure] = {**mean_sd(differences), 'per_seed': differences,
                                    'wins_ties_losses': [sum(value > 1e-9 for value in differences),
                                                         sum(abs(value) <= 1e-9 for value in differences),
                                                         sum(value < -1e-9 for value in differences)]}
        summary['paired_flywire_differences'][baseline] = comparisons
    (results.parent / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    plt.rcParams.update({'figure.dpi': 150, 'font.size': 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    versions = np.arange(0, 101, 10)
    for architecture in ARCHITECTURES:
        learning = []
        returns = []
        for seed in seeds:
            record = records[architecture, seed]
            learning.append([100 * policy_row(record, int(version), 'stochastic')['hit_rate']
                             for version in versions])
            returns.append([policy_row(record, int(version), 'stochastic')['return_mean']
                            for version in versions])
        for axis, values, label in ((axes[0, 0], np.asarray(learning), 'Hit rate (%)'),
                                    (axes[0, 1], np.asarray(returns), 'Mean return')):
            mean, sd = values.mean(0), values.std(0, ddof=1)
            axis.plot(versions, mean, marker='o', markersize=3, color=COLORS[architecture],
                      label=NAMES[architecture])
            axis.fill_between(versions, mean - sd, mean + sd, color=COLORS[architecture], alpha=.12)
            axis.set(xlabel='PPO update', ylabel=label)
            axis.grid(alpha=.2)
    axes[0, 0].set_title('Held-out stochastic hit learning')
    axes[0, 1].set_title('Held-out stochastic return learning')
    axes[0, 0].legend()
    x = np.arange(len(seeds))
    for architecture in ARCHITECTURES:
        axes[1, 0].plot(x, [row['final_stochastic_hit_percent'] for row in grouped[architecture]],
                        marker='o', color=COLORS[architecture], label=NAMES[architecture])
    axes[1, 0].set(title='Final held-out hit rate by seed', xlabel='Training seed',
                   ylabel='Hit rate (%)', xticks=x, xticklabels=seeds, ylim=(0, 105))
    runtime_means = [summary['architectures'][architecture]['recorded_training_seconds']['mean']
                     for architecture in ARCHITECTURES]
    runtime_sd = [summary['architectures'][architecture]['recorded_training_seconds']['sample_sd']
                  for architecture in ARCHITECTURES]
    axes[1, 1].bar([NAMES[value] for value in ARCHITECTURES], runtime_means, yerr=runtime_sd,
                   color=[COLORS[value] for value in ARCHITECTURES], capsize=4)
    axes[1, 1].set(title='Recorded CPU training time', ylabel='Seconds')
    axes[1, 1].grid(axis='y', alpha=.2)
    fig.suptitle(f'Toy-combat aiming baseline comparison · {len(seeds)} seeds')
    fig.savefig(results.parent / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path('experiments/toy_combat_baseline_comparison/results'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    arguments = parser.parse_args()
    summarize(arguments.results, arguments.seeds)
