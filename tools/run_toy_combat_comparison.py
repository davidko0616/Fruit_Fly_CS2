"""Run and audit the fixed five-seed toy-combat architecture comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.recording import atomic_json

ARCHITECTURES = ('flywire', 'random', 'mlp')
PROTOCOL = ROOT / 'experiments/toy_combat_baseline_comparison/PROTOCOL.md'


def run(output, results, seeds):
    schedule = {'seeds': seeds, 'architectures': list(ARCHITECTURES),
                'order': 'rotate by seed index', 'device': 'cpu', 'updates': 100,
                'workers': 8, 'horizon': 64, 'action_repeat': 2,
                'evaluation_episodes': 256,
                'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    output.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, 'tools/audit_toy_combat_matching.py', '--output',
                    str(results.parent / 'matching_audit.json'), '--seeds',
                    *[str(seed) for seed in seeds]], cwd=ROOT, check=True)
    schedule_path = output / 'schedule.json'
    if schedule_path.exists() and json.loads(schedule_path.read_text()) != schedule:
        raise RuntimeError('Existing output directory uses a different schedule')
    atomic_json(schedule_path, schedule)
    completed_path = results / 'runs.json'
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else []
    completed_names = {row['name'] for row in completed}
    for seed_index, seed in enumerate(seeds):
        order = ARCHITECTURES[seed_index % len(ARCHITECTURES):] + ARCHITECTURES[:seed_index % len(ARCHITECTURES)]
        for architecture in order:
            name = f'{architecture}_seed_{seed}'
            compact_path = results / f'{name}.json'
            if name in completed_names and compact_path.is_file():
                print(f'SKIP {name}: compact result already exists', flush=True)
                continue
            directory = output / name
            recover_complete_run = False
            if directory.exists():
                manifest_path = directory / 'manifest.json'
                if manifest_path.is_file():
                    existing_manifest = json.loads(manifest_path.read_text())
                    recover_complete_run = (existing_manifest.get('status') == 'complete' and
                                            existing_manifest.get('decisions') == 51_200)
                if not recover_complete_run:
                    raise RuntimeError(f'Incomplete run directory already exists: {directory}')
            log_path = output / f'{name}.log'
            print(f'{"RECOVER" if recover_complete_run else "START"} {name}', flush=True)
            with log_path.open('a' if recover_complete_run else 'w', encoding='utf-8') as log:
                if not recover_complete_run:
                    subprocess.run([sys.executable, '-u', '-m', 'training.train_toy_combat',
                                    '--output', str(directory), '--architecture', architecture,
                                    '--seed', str(seed), '--updates', '100', '--workers', '8',
                                    '--horizon', '64', '--action-repeat', '2'], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                audit_process = subprocess.run([sys.executable, 'tools/verify_combat_recording.py',
                                                '--run', str(directory)], cwd=ROOT,
                                               capture_output=True, text=True)
                log.write(audit_process.stdout); log.write(audit_process.stderr)
                if audit_process.returncode:
                    raise RuntimeError(f'Audit failed for {name}: {audit_process.stderr[-2000:]}')
                audit = json.loads(audit_process.stdout)
                atomic_json(directory / 'audit.json', audit)
                evaluation_directory = directory / 'evaluation'
                evaluation_process = subprocess.run([sys.executable, 'tools/evaluate_toy_combat.py',
                                                      '--run', str(directory), '--output',
                                                      str(evaluation_directory), '--episodes', '256'],
                                                     cwd=ROOT, capture_output=True, text=True)
                log.write(evaluation_process.stdout); log.write(evaluation_process.stderr)
                if evaluation_process.returncode:
                    raise RuntimeError(f'Evaluation failed for {name}: '
                                       f'{evaluation_process.stderr[-2000:]}')
            manifest = json.loads((directory / 'manifest.json').read_text())
            metrics = json.loads((directory / 'metrics.json').read_text())
            if manifest['status'] != 'complete' or manifest['decisions'] != 51_200:
                raise RuntimeError(f'Incomplete recording: {name}')
            if not audit['audit_passed'] or audit['decisions'] != 51_200:
                raise RuntimeError(f'Failed audit: {name}')
            compact = {'name': name, 'architecture': architecture, 'seed': seed,
                       'run_id': manifest['run_id'], 'config': manifest['config'],
                       'metrics': metrics, 'updates': json.loads((directory / 'updates.json').read_text()),
                       'audit': audit,
                       'evaluation': json.loads((evaluation_directory / 'evaluation.json').read_text()),
                       'provenance': json.loads((directory / 'provenance.json').read_text()),
                       'bytes_on_disk': sum(path.stat().st_size for path in directory.rglob('*')
                                            if path.is_file())}
            atomic_json(compact_path, compact)
            completed.append({'name': name, 'architecture': architecture, 'seed': seed,
                              'final_stochastic_hit_rate': compact['evaluation']['policies'][-2]['hit_rate'],
                              'training_seconds': metrics['training_seconds_including_recording']})
            atomic_json(completed_path, completed)
            completed_names.add(name)
            print(f'DONE {name}: final stochastic hit='
                  f'{compact["evaluation"]["policies"][-2]["hit_rate"]:.1%}; '
                  f'{metrics["training_seconds_including_recording"]:.1f}s; audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    arguments = parser.parse_args()
    run(arguments.output, arguments.results, arguments.seeds)
