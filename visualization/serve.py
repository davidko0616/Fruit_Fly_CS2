"""Serve a saved recording on loopback only; no training or internet required."""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from training.recording import read_events


class Recording:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['schema_version'] != 1:
            raise ValueError('Unsupported recording schema')
        self.graph = json.loads((self.directory / 'graph.json').read_text(encoding='utf-8'))
        self.events = read_events(self.directory)
        if self.manifest['status'] == 'complete' and (
                len(self.events) != self.manifest['events'] or
                sum(e['count'] for e in self.events) != self.manifest['samples']):
            raise ValueError('Finalized recording is missing committed events or samples')
        self.metrics = (json.loads((self.directory / 'metrics.json').read_text(encoding='utf-8'))
                        if (self.directory / 'metrics.json').exists() else None)

    @lru_cache(maxsize=2)
    def chunk(self, filename):
        with np.load(self.directory / 'chunks' / filename, allow_pickle=False) as data:
            return {key: data[key] for key in data.files if key != 'events'}

    def frame(self, event_id, sample):
        if event_id < 0 or event_id >= len(self.events):
            raise ValueError('Event out of range')
        event = self.events[event_id]
        if sample < 0 or sample >= event['count']:
            raise ValueError('Sample out of range')
        data = self.chunk(event['chunk'])
        offset = event['offset'] + sample
        return {'event': event, 'sample_index': sample,
                **{key: value[offset].tolist() for key, value in data.items()}}

    def dataset(self, split):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Invalid split')
        with np.load(self.directory / 'dataset.npz', allow_pickle=False) as data:
            ids = data[split]
            return {'ids': ids.tolist(), 'inputs': data['x'][ids].tolist(), 'labels': data['y'][ids].tolist()}


def make_server(directory, port=8765):
    recording = Recording(directory)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlparse(self.path)
            query = parse_qs(route.query)
            try:
                if route.path == '/':
                    payload = Path(__file__).with_name('viewer.html').read_bytes()
                    content_type = 'text/html; charset=utf-8'
                else:
                    if route.path == '/api/info':
                        value = {'manifest': recording.manifest, 'graph': recording.graph,
                                 'events': recording.events, 'metrics': recording.metrics}
                    elif route.path == '/api/frame':
                        value = recording.frame(int(query.get('event', ['0'])[0]), int(query.get('sample', ['0'])[0]))
                    elif route.path == '/api/dataset':
                        value = recording.dataset(query.get('split', ['validation'])[0])
                    else:
                        self.send_error(404)
                        return
                    payload = json.dumps(value, allow_nan=False).encode('utf-8')
                    content_type = 'application/json; charset=utf-8'
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, KeyError, IndexError) as error:
                self.send_error(400, str(error))

        def log_message(self, fmt, *args):
            pass

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    with make_server(args.run, args.port) as server:
        print(f'Offline activation viewer: http://127.0.0.1:{server.server_port}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
