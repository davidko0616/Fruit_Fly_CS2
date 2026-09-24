"""Capture timestamped lossless CS2 screen regions for offline perception work."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from PIL import ImageGrab
from PIL import Image


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
    args = parser.parse_args()
    if args.frames < 1 or args.interval <= 0:
        raise ValueError('Frames and interval must be positive')
    rows = capture(args.output, args.frames, args.interval, args.region, args.backend)
    print(json.dumps({'status': 'complete', 'frames': len(rows),
                      'output': str(args.output), 'region': args.region,
                      'backend': rows[-1]['capture_backend']}, indent=2))


if __name__ == '__main__':
    main()
