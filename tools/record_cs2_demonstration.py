"""Record compact, read-only human CS2 demonstrations on Dust II."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.demonstration import (
    WindowsInputStateSampler, build_demonstration_rows)
from tools.run_cs2_shadow import build_parser, run, validate_args


def _write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def record(args, sampler_factory=WindowsInputStateSampler):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    sampler = sampler_factory(sample_hz=args.input_hz)
    sampler.start()
    error = None
    summary = None
    try:
        summary = run(args, action_executor=None)
    except Exception as caught:
        error = caught
    finally:
        input_samples = sampler.stop()

    if output.exists():
        input_path = output / 'input.jsonl'
        if not input_path.exists():
            _write_jsonl(input_path, input_samples)
        shadow_records = _read_jsonl(output / 'shadow.jsonl')
        rows, demonstration_summary = build_demonstration_rows(
            shadow_records, input_samples, sample_hz=args.input_hz)
        demonstration_path = output / 'demonstration.jsonl'
        if not demonstration_path.exists():
            _write_jsonl(demonstration_path, rows)
        demonstration_summary.update({
            'schema_version': 1,
            'session_kind': args.session_kind,
            'input_sample_hz': args.input_hz,
            'control_bit_mapping': {
                'forward': 0, 'backward': 1, 'strafe_left': 2,
                'strafe_right': 3, 'fire': 4, 'walk': 5,
                'crouch': 6, 'jump': 7, 'secondary_fire': 8,
            },
            'input_samples': len(input_samples),
            'foreground_input_samples': sum(
                bool(row['foreground_cs2']) for row in input_samples),
            'input_control_nonzero_samples': sum(
                bool(row['control_bits']) for row in input_samples),
            'input_sha256': _sha256(input_path),
            'demonstration_sha256': _sha256(demonstration_path),
            'no_images_retained': args.audit_every == 0,
            'read_only_capture': True,
        })
        if summary is None:
            summary_path = output / 'summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding='utf-8'))
            else:
                summary = {'schema_version': 1, 'status': 'failed'}
        summary['human_demonstration'] = demonstration_summary
        (output / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if error is not None:
        raise error
    return summary


def main():
    parser = build_parser(__doc__)
    parser.set_defaults(
        duration_seconds=600, hz=8, audit_every=0, sound_cues=True,
        spoken_prompt=(
            'Dust Two demonstration recording starts after the countdown. '
            'Play naturally until the completion sound.'))
    parser.add_argument('--input-hz', type=float, default=100)
    parser.add_argument(
        '--session-kind', choices=('navigation', 'combat', 'mixed'),
        default='navigation')
    args = parser.parse_args()
    validate_args(parser, args)
    if args.input_hz <= 0:
        parser.error('--input-hz must be positive')
    if args.audit_every != 0:
        parser.error('Demonstration recording requires --audit-every 0')
    print(json.dumps(record(args), indent=2))


if __name__ == '__main__':
    main()
