"""Capture timestamped lossless CS2 screen regions for offline perception work."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from PIL import ImageGrab
from PIL import Image


def _beep(pattern):
    if os.name != 'nt':
        return
    try:
        import winsound
        for frequency, duration_ms, pause_seconds in pattern:
            winsound.Beep(frequency, duration_ms)
            if pause_seconds:
                time.sleep(pause_seconds)
    except (ImportError, RuntimeError):
        pass


def _speak(message):
    if not message or os.name != 'nt':
        return
    try:
        import comtypes.client
        voice = comtypes.client.CreateObject('SAPI.SpVoice')
        voice.Speak(message)
    except Exception:
        pass


def wait_for_capture(delay, sound_cues=False, spoken_prompt=None):
    if spoken_prompt:
        _speak(spoken_prompt)
    delay = float(delay)
    if delay <= 0:
        if sound_cues:
            _beep(((1100, 250, 0),))
        return
    print(f'Capture starts in {delay:g} seconds', flush=True)
    deadline = time.monotonic() + delay
    quiet_wait = max(0.0, deadline - time.monotonic() - 3.0)
    if quiet_wait:
        time.sleep(quiet_wait)
    remaining = min(3, int(round(delay)))
    for count in range(remaining, 0, -1):
        print(f'{count}...', flush=True)
        if sound_cues:
            _beep(((700 + (3 - count) * 150, 180, 0),))
        next_tick = deadline - (count - 1)
        time.sleep(max(0.0, next_tick - time.monotonic()))
    print('Capture started', flush=True)
    if sound_cues:
        _beep(((1200, 300, 0),))


def notify_capture_complete(sound_cues=False):
    if sound_cues:
        _beep(((900, 160, 0.08), (1250, 160, 0.08), (1700, 350, 0),))
        _speak('Capture complete. You can return to Codex.')


def notify_capture_failed(sound_cues=False):
    if sound_cues:
        _beep(((900, 220, 0.08), (650, 220, 0.08), (400, 450, 0),))
        _speak('Capture failed. Please return to Codex.')


class ScreenGrabber:
    """Capture with Pillow, falling back to DXGI for Direct3D full-screen apps."""

    def __init__(self, backend='auto'):
        if backend not in ('auto', 'pillow', 'dxcam'):
            raise ValueError('Backend must be auto, pillow, or dxcam')
        self.requested_backend = backend
        self.backend = 'pillow' if backend in ('auto', 'pillow') else 'dxcam'
        self.camera = None

    def _open_dxcam(self):
        if self.camera is None:
            try:
                import dxcam
            except ImportError as error:
                raise RuntimeError(
                    'DXcam is required when Pillow cannot capture the display') from error
            self.camera = dxcam.create(output_color='BGRA')
        self.backend = 'dxcam'

    def grab(self, region):
        if self.backend == 'pillow':
            try:
                return ImageGrab.grab(bbox=region, all_screens=region is None)
            except OSError:
                if self.requested_backend != 'auto':
                    raise
                self._open_dxcam()
        if self.camera is None:
            self._open_dxcam()
        frame = self.camera.grab(region=region, new_frame_only=False)
        if frame is None:
            raise RuntimeError('DXcam returned no display frame')
        rgb = frame[:, :, 2::-1].copy()
        return Image.fromarray(rgb)

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None


def parse_region(value):
    if value is None:
        return None
    try:
        region = tuple(int(part) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Region must contain four integers') from error
    if len(region) != 4 or region[2] <= region[0] or region[3] <= region[1]:
        raise argparse.ArgumentTypeError('Region must be left,top,right,bottom')
    return region


def capture(output, frames=1, interval=0.2, region=None, backend='auto'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    rows = []
    deadline = time.monotonic()
    grabber = ScreenGrabber(backend)
    try:
        with (output / 'frames.jsonl').open('x', encoding='utf-8', newline='\n') as manifest:
            for index in range(frames):
                start_ns = time.monotonic_ns()
                image = grabber.grab(region)
                end_ns = time.monotonic_ns()
                filename = f'{index:07d}.png'
                image.save(output / filename)
                row = {
                    'schema_version': 1, 'frame_id': index, 'file': filename,
                    'capture_start_monotonic_ns': start_ns,
                    'capture_end_monotonic_ns': end_ns,
                    'capture_midpoint_monotonic_ns': (start_ns + end_ns) // 2,
                    'captured_utc': datetime.now(timezone.utc).isoformat(),
                    'region': region, 'width': image.width, 'height': image.height,
                    'capture_backend': grabber.backend,
                }
                rows.append(row)
                manifest.write(json.dumps(row, separators=(',', ':')) + '\n')
                manifest.flush()
                deadline += interval
                if index + 1 < frames:
                    time.sleep(max(0, deadline - time.monotonic()))
    finally:
        grabber.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--interval', type=float, default=0.2)
    parser.add_argument('--region', type=parse_region,
                        help='Optional left,top,right,bottom screen crop')
    parser.add_argument('--backend', choices=('auto', 'pillow', 'dxcam'),
                        default='auto')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Seconds before capture, with final countdown')
    parser.add_argument('--sound-cues', action='store_true',
                        help='Play countdown, start, and completion tones on Windows')
    parser.add_argument('--spoken-prompt',
                        help='Optional Windows speech prompt before the countdown')
    args = parser.parse_args()
    if args.frames < 1 or args.interval <= 0 or args.delay < 0:
        raise ValueError('Frames and interval must be positive and delay nonnegative')
    wait_for_capture(args.delay, args.sound_cues, args.spoken_prompt)
    try:
        rows = capture(args.output, args.frames, args.interval, args.region, args.backend)
    except Exception:
        notify_capture_failed(args.sound_cues)
        raise
    notify_capture_complete(args.sound_cues)
    print(json.dumps({'status': 'complete', 'frames': len(rows),
                      'output': str(args.output), 'region': args.region,
                      'backend': rows[-1]['capture_backend']}, indent=2))


if __name__ == '__main__':
    main()
