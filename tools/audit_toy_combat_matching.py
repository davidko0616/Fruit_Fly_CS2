"""Audit actor budgets and topology-control constraints for the aiming comparison."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.recording import atomic_json
from training.run_toy_combat import load_policy


def audit(output, seeds):
    rows = []
    for seed in seeds:
        flywire, graph, reference, _ = load_policy(seed, 'flywire')
        random, _, randomized, _ = load_policy(seed, 'random')
        mlp, _, _, hidden_sizes = load_policy(seed, 'mlp')
        parameter_counts = {name: sum(parameter.numel() for parameter in model.parameters())
                            for name, model in (('flywire', flywire), ('random', random), ('mlp', mlp))}
        source_degrees_match = np.array_equal(np.diff(reference.indptr), np.diff(randomized.indptr))
        self_loops_match = np.array_equal(reference.diagonal(), randomized.diagonal())
        count_multisets_match = all(
            np.array_equal(np.sort(reference.getrow(source).data),
                           np.sort(randomized.getrow(source).data))
            for source in range(reference.shape[0]))
        io_parameters_match = all(torch.equal(dict(flywire.named_parameters())[name],
                                              dict(random.named_parameters())[name])
                                  for name in ('input_proj.weight', 'input_proj.bias',
                                               'output_proj.weight', 'output_proj.bias'))
        reference_positions = reference.copy(); reference_positions.data[:] = 1
        randomized_positions = randomized.copy(); randomized_positions.data[:] = 1
        overlap = reference_positions.multiply(randomized_positions).nnz
        signs = np.asarray(graph['signs'])
        flywire_sources = reference.tocoo().row
        random_sources = randomized.tocoo().row
        signed_edge_counts = {
            'flywire_excitatory': int((signs[flywire_sources] > 0).sum()),
            'flywire_inhibitory': int((signs[flywire_sources] < 0).sum()),
            'random_excitatory': int((signs[random_sources] > 0).sum()),
            'random_inhibitory': int((signs[random_sources] < 0).sum())}
        checks = {'parameter_counts_exact': parameter_counts == {'flywire': 7913, 'random': 7913,
                                                                  'mlp': 7913},
                  'mlp_hidden_sizes_exact': tuple(hidden_sizes) == (78, 81),
                  'edge_count_match': reference.nnz == randomized.nnz == 7615,
                  'source_degrees_match': bool(source_degrees_match),
                  'self_loops_match': bool(self_loops_match),
                  'per_source_count_multisets_match': bool(count_multisets_match),
                  'recurrent_io_initialization_match': bool(io_parameters_match),
                  'signed_edge_counts_match': (signed_edge_counts['flywire_excitatory'] ==
                                               signed_edge_counts['random_excitatory'] and
                                               signed_edge_counts['flywire_inhibitory'] ==
                                               signed_edge_counts['random_inhibitory'])}
        if not all(checks.values()):
            raise RuntimeError(f'Matching audit failed for seed {seed}: {checks}')
        rows.append({'seed': seed, 'parameter_counts': parameter_counts,
                     'mlp_hidden_sizes': list(hidden_sizes), 'edges': reference.nnz,
                     'randomized_edge_overlap_count': overlap,
                     'randomized_edge_overlap_percent': 100 * overlap / reference.nnz,
                     'signed_edge_counts': signed_edge_counts, 'checks': checks})
    payload = {'seeds': seeds, 'all_checks_passed': True, 'runs': rows}
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('experiments/toy_combat_baseline_comparison/matching_audit.json'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    arguments = parser.parse_args()
    audit(arguments.output, arguments.seeds)
