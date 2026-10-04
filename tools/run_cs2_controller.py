"""Run the Dust II policy with a guarded local-practice input executor."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.executor import (
    GuardedActionExecutor, InputSafetyConfig, WindowsSendInputBackend)
from tools.run_cs2_shadow import build_parser, run, validate_args


CONFIRMATION = 'LOCAL_PRACTICE_ONLY'


def main():
    parser = build_parser(description=__doc__)
    parser.set_defaults(spoken_prompt=(
        'Dust Two guarded controller starting. Press F twelve to stop.'))
    parser.add_argument(
        '--enable-input', action='store_true',
        help='Allow bounded movement and turn input after all guards pass.')
    parser.add_argument(
        '--confirm-local-practice', metavar=CONFIRMATION,
        help='Required exact phrase when --enable-input is used.')
    parser.add_argument(
        '--enable-fire', action='store_true',
        help='Allow fire actions. Off by default for validation runs.')
    parser.add_argument('--execution-max-frame-age-ms', type=float, default=250)
    parser.add_argument('--execution-max-gsi-age-ms', type=float, default=1000)
    parser.add_argument('--key-hold-ms', type=float, default=60)
    parser.add_argument('--turn-pixels', type=int, default=32)
    parser.add_argument('--required-window-title', default='Counter-Strike 2')
    args = parser.parse_args()
    validate_args(parser, args)
    if not args.enable_input:
        parser.error(
            'This controller requires --enable-input; use run_cs2_shadow.py '
            'for a read-only run')
    if args.confirm_local_practice != CONFIRMATION:
        parser.error(
            f'--enable-input requires --confirm-local-practice {CONFIRMATION}')
    config = InputSafetyConfig(
        max_frame_age_ms=args.execution_max_frame_age_ms,
        max_gsi_age_ms=args.execution_max_gsi_age_ms,
        key_hold_ms=args.key_hold_ms,
        turn_pixels=args.turn_pixels,
        required_window_title=args.required_window_title,
        fire_enabled=args.enable_fire)
    executor = GuardedActionExecutor(
        WindowsSendInputBackend(), enabled=True, config=config)
    print(json.dumps(run(args, action_executor=executor), indent=2))


if __name__ == '__main__':
    main()
