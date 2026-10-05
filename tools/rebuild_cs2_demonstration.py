"""Rebuild derived demonstration rows from an existing shadow/input capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.demonstration import build_demonstration_rows


def _read_jsonl(path):
    with Path(path).open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(output, input_hz=100.0):
    output = Path(output)
    paths = {name: output / name for name in (
        'input.jsonl', 'shadow.jsonl', 'demonstration.jsonl', 'summary.json')}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing capture files: {missing}')
    summary = json.loads(paths['summary.json'].read_text(encoding='utf-8'))
    previous = summary.get('human_demonstration') or {}
    rows, rebuilt = build_demonstration_rows(
        _read_jsonl(paths['shadow.jsonl']),
        _read_jsonl(paths['input.jsonl']), input_hz)
    temporary = output / 'demonstration.jsonl.rebuild'
    with temporary.open('w', encoding='utf-8', newline='\n') as destination:
        for row in rows:
            destination.write(json.dumps(row, separators=(',', ':')) + '\n')
    temporary.replace(paths['demonstration.jsonl'])
    rebuilt.update({
        key: value for key, value in previous.items()
        if key not in rebuilt and key != 'demonstration_sha256'
    })
    rebuilt['demonstration_sha256'] = _sha256(paths['demonstration.jsonl'])
    rebuilt['derived_rows_rebuilt'] = True
    summary['human_demonstration'] = rebuilt
    temporary_summary = output / 'summary.json.rebuild'
    temporary_summary.write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    temporary_summary.replace(paths['summary.json'])
    return rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-hz', type=float, default=100)
    args = parser.parse_args()
    if args.input_hz <= 0:
        parser.error('--input-hz must be positive')
    print(json.dumps(rebuild(args.output, args.input_hz), indent=2))


if __name__ == '__main__':
    main()
