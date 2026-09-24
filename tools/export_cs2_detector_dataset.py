"""Export audited CS2 sessions into a CPU-detector training dataset."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.detector_dataset import export_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--session', action='append', nargs=2, required=True,
        metavar=('CAPTURE', 'LABELS'),
        help='Capture directory and its label JSON; repeat for each session')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--class-mode', choices=('team', 'player'), default='team')
    parser.add_argument('--transfer', choices=('hardlink', 'copy'), default='hardlink')
    args = parser.parse_args()
    summary = export_dataset(args.session, args.output, args.class_mode, args.transfer)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
