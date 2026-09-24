"""Serve a loopback-only browser UI for labeling visible players in captures."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.labels import validate_frame_label


class LabelSet:
    def __init__(self, capture, output, split, start_frame=0, end_frame=None,
                 stride=1, exclude_frames=()):
        self.capture = Path(capture).resolve()
        self.output = Path(output).resolve()
        manifest_path = self.capture / 'frames.jsonl'
        self.frames = [json.loads(line) for line in manifest_path.read_text().splitlines()
                       if line.strip()]
        excluded = {int(frame_id) for frame_id in exclude_frames}
        self.frames = [frame for frame in self.frames
                       if int(frame['frame_id']) >= start_frame and
                       (end_frame is None or int(frame['frame_id']) <= end_frame) and
                       (int(frame['frame_id']) - start_frame) % stride == 0 and
                       int(frame['frame_id']) not in excluded]
        if not self.frames:
            raise ValueError('Capture manifest has no frames')
        self.by_id = {int(frame['frame_id']): frame for frame in self.frames}
        if len(self.by_id) != len(self.frames):
            raise ValueError('Capture manifest has duplicate frame IDs')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Split must be train, validation, or test')
        self.split = split
        self.labels = {}
        if self.output.exists():
            saved = json.loads(self.output.read_text(encoding='utf-8'))
            if saved.get('schema_version') != 1:
                raise ValueError('Unsupported label schema')
            if saved.get('split') != split:
                raise ValueError('Existing label file uses a different session split')
            self.labels = saved.get('labels', {})

    def info(self):
        return {
            'schema_version': 1,
            'split': self.split,
            'frames': [{key: frame[key] for key in
                        ('frame_id', 'file', 'width', 'height',
                         'capture_midpoint_monotonic_ns')}
                       for frame in self.frames],
            'labels': self.labels,
        }

    def image_path(self, frame_id):
        frame = self.by_id.get(frame_id)
        if frame is None:
            raise ValueError('Unknown frame ID')
        path = (self.capture / frame['file']).resolve()
        if path.parent != self.capture or not path.is_file():
            raise ValueError('Invalid frame path')
        return path

    def save(self, frame_id, value):
        frame = self.by_id.get(frame_id)
        if frame is None:
            raise ValueError('Unknown frame ID')
        label = validate_frame_label(value, frame['width'], frame['height'])
        self.labels[str(frame_id)] = label
        payload = {
            'schema_version': 1,
            'split': self.split,
            'capture_frames': len(self.frames),
            'labels': self.labels,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(self.output.name + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.output)
        return label


def make_server(capture, output, split='train', port=8766, start_frame=0,
                end_frame=None, stride=1, exclude_frames=()):
    labels = LabelSet(capture, output, split, start_frame, end_frame, stride,
                      exclude_frames)
    html = Path(__file__).with_name('cs2_labeler.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; script-src 'unsafe-inline'; "
                             "style-src 'unsafe-inline'; img-src 'self'; "
                             "connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlparse(self.path)
            try:
                if route.path == '/':
                    self._send(200, html, 'text/html; charset=utf-8')
                    return
                if route.path == '/api/info':
                    payload = json.dumps(labels.info()).encode('utf-8')
                    self._send(200, payload, 'application/json; charset=utf-8')
                    return
                if route.path.startswith('/frame/'):
                    frame_id = int(unquote(route.path.removeprefix('/frame/')))
                    payload = labels.image_path(frame_id).read_bytes()
                    self._send(200, payload, 'image/png')
                    return
                self.send_error(404)
            except (ValueError, KeyError, OSError) as error:
                self.send_error(400, str(error))

        def do_POST(self):
            if urlparse(self.path).path != '/api/label':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1_000_000:
                    raise ValueError('Invalid request length')
                value = json.loads(self.rfile.read(length))
                label = labels.save(int(value['frame_id']), value['label'])
                payload = json.dumps({'status': 'saved', 'label': label}).encode('utf-8')
                self._send(200, payload, 'application/json; charset=utf-8')
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                self.send_error(400, str(error))

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--exclude-frame', type=int, action='append', default=[],
                        help='Frame ID to omit; repeat for each rejected frame')
    args = parser.parse_args()
    if args.start_frame < 0 or args.stride < 1:
        raise ValueError('Start frame must be nonnegative and stride positive')
    if args.end_frame is not None and args.end_frame < args.start_frame:
        raise ValueError('End frame must not precede start frame')
    with make_server(args.capture, args.output, args.split, args.port,
                     args.start_frame, args.end_frame, args.stride,
                     args.exclude_frame) as server:
        print(f'CS2 visible-player labeler: http://127.0.0.1:{server.server_port}',
              flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
