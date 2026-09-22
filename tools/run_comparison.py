"""Run the fixed CPU protocol sequentially, auditing every completed recording."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.recording import atomic_json


def run(output, results, epochs, seeds):
    output.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=True)
    architectures = ['flywire', 'random', 'mlp']
    atomic_json(output / 'schedule.json', {'epochs': epochs, 'seeds': seeds,
        'architectures': architectures, 'order': 'rotate by seed index', 'device': 'cpu',
        'protocol_sha256': hashlib.sha256((ROOT / 'experiments/baseline_comparison_100/PROTOCOL.md').read_bytes()).hexdigest()})
    completed = []
    for i, seed in enumerate(seeds):
        order = architectures[i % 3:] + architectures[:i % 3]
        for architecture in order:
            name = f'{architecture}_seed_{seed}'
            directory = output / name
            print(f'START {name}', flush=True)
            with (output / f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'training.train_recorded', '--output', str(directory),
                    '--architecture', architecture, '--epochs', str(epochs), '--seed', str(seed)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            audit = subprocess.run([sys.executable, 'tools/verify_recording.py', '--run', str(directory)],
                                   cwd=ROOT, capture_output=True, text=True, check=True)
            verification = json.loads(audit.stdout)
            atomic_json(directory / 'audit.json', verification)
            metrics = json.loads((directory / 'metrics.json').read_text())
            manifest = json.loads((directory / 'manifest.json').read_text())
            history = json.loads((directory / 'history.json').read_text())
            if manifest['status'] != 'complete' or metrics['expected_sample_forwards'] != manifest['samples']:
                raise RuntimeError('Incomplete run')
            compact = {'name': name, 'metrics': metrics, 'config': manifest['config'],
                'run_id': manifest['run_id'], 'events': manifest['events'], 'samples': manifest['samples'],
                'bytes_on_disk': sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
                'verification': verification, 'history': history,
                'provenance': json.loads((directory / 'provenance.json').read_text())}
            atomic_json(results / f'{name}.json', compact)
            completed.append({'name': name, **metrics})
            atomic_json(results / 'runs.json', completed)
            print(f'DONE {name}: test={metrics["test"]["accuracy"]:.2%}; '
                  f'{metrics["training_seconds_including_recording"]:.1f}s; audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    run(args.output, args.results, args.epochs, args.seeds)
