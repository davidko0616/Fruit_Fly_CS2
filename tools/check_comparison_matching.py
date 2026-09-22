"""Independently verify the actual saved graphs, batch orders and initial weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def check(root, output):
    findings = []
    for seed in range(42, 47):
        dirs = {name: root / f'{name}_seed_{seed}' for name in ('flywire', 'random', 'mlp')}
        manifests = {name: json.loads((p / 'manifest.json').read_text()) for name, p in dirs.items()}
        assert all(m['status'] == 'complete' and m['samples'] == 312904 for m in manifests.values())
        datasets = {hashlib.sha256((p / 'dataset.npz').read_bytes()).hexdigest() for p in dirs.values()}
        assert len(datasets) == 1
        for key in ('epochs', 'batch_size', 'learning_rate', 'optimizer', 'cpu_threads', 'batch_seed', 'model_seed'):
            assert len({m['config'][key] for m in manifests.values()}) == 1
        original = sp.load_npz(dirs['flywire'] / 'adjacency.npz')
        random = sp.load_npz(dirs['random'] / 'adjacency.npz')
        assert original.nnz == random.nnz == 7615 and random.has_canonical_format
        np.testing.assert_array_equal(original.diagonal(), random.diagonal())
        np.testing.assert_array_equal(np.diff(original.indptr), np.diff(random.indptr))
        for i in range(100):
            np.testing.assert_array_equal(np.sort(original.getrow(i).data), np.sort(random.getrow(i).data))
        with np.load(dirs['flywire'] / 'graph.npz') as a, np.load(dirs['random'] / 'graph.npz') as b:
            for key in ('signs', 'inputs', 'outputs'):
                np.testing.assert_array_equal(a[key], b[key])
            signs = a['signs']
            expected_negative = int(np.sum(np.diff(original.indptr)[signs < 0]))
            assert expected_negative == int(np.sum(np.diff(random.indptr)[signs < 0]))
        with np.load(dirs['flywire'] / 'weights/0000000.npz') as a, np.load(dirs['random'] / 'weights/0000000.npz') as b:
            for key in ('p0', 'p1', 'p4', 'p5'):
                np.testing.assert_array_equal(a[key], b[key])
        batch_hashes = {}
        for name, directory in dirs.items():
            digest, batches, tests = hashlib.sha256(), 0, []
            for path in sorted((directory / 'chunks').glob('*.npz')):
                with np.load(path, allow_pickle=False) as chunk:
                    events = json.loads(str(chunk['events']))
                    ids = chunk['sample_ids']
                    for event in events:
                        if event['phase'] == 'optimization':
                            digest.update(np.asarray([event['epoch'], event['count']], dtype='<i8').tobytes())
                            digest.update(ids[event['offset']:event['offset']+event['count']].astype('<i8').tobytes())
                            batches += 1
                        if event['split'] == 'test':
                            tests.append(event)
            assert batches == 2200 and len(tests) == 1 and tests[0]['phase'] == 'selected'
            batch_hashes[name] = digest.hexdigest()
        assert len(set(batch_hashes.values())) == 1
        binary_original = original.copy(); binary_original.data[:] = 1
        binary_random = random.copy(); binary_random.data[:] = 1
        overlap = binary_original.multiply(binary_random).nnz / original.nnz
        findings.append({'seed': seed, 'matching_passed': True, 'negative_edges': expected_negative,
                         'edge_overlap_fraction': overlap, 'batch_order_sha256': batch_hashes['flywire']})
    output.write_text(json.dumps(findings, indent=2) + '\n')
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/synthetic/baseline_comparison_100'))
    parser.add_argument('--output', type=Path, default=Path('experiments/baseline_comparison_100/matching_audit.json'))
    args = parser.parse_args()
    check(args.root, args.output)
