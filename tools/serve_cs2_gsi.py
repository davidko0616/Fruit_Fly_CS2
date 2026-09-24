"""Record read-only CS2 Game State Integration POSTs on loopback."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs2_bridge.gsi import parse_gsi_payload


class GSIRecorder:
    def __init__(self, output, token=None):
        self.output = Path(output)
        self.token = token
        self.sequence = 0
        self.lock = threading.Lock()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output.open('x', encoding='utf-8', newline='\n')

    def close(self):
        self.file.close()

    def append(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('GSI payload must be an object')
        with self.lock:
            return self._append_locked(payload)

    def _append_locked(self, payload):
        supplied = (payload.get('auth') or {}).get('token')
        if self.token is not None and supplied != self.token:
            raise PermissionError('GSI token mismatch')
        snapshot = parse_gsi_payload(payload)
        retained_payload = {key: payload[key] for key in ('provider', 'map', 'round', 'player')
                            if key in payload}
        discarded_fields = sorted(set(payload) - set(retained_payload) - {'auth'})
        row = {
            'schema_version': 1,
            'sequence': self.sequence,
            'received_monotonic_ns': time.monotonic_ns(),
            'received_utc': datetime.now(timezone.utc).isoformat(),
            'snapshot': {
                'map_name': snapshot.map_name,
                'round_number': snapshot.round_number,
                'round_id': snapshot.round_id,
                'player_activity': snapshot.player_activity,
                'health': snapshot.health,
                'pose': (None if snapshot.pose is None else {
                    'x': snapshot.pose.x, 'y': snapshot.pose.y,
                    'yaw_degrees': snapshot.pose.yaw_degrees,
                }),
                'provider_timestamp': snapshot.provider_timestamp,
            },
            'retained_payload': retained_payload,
            'discarded_top_level_fields': discarded_fields,
        }
        self.sequence += 1
        self.file.write(json.dumps(row, separators=(',', ':')) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        return row


def make_handler(recorder):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length <= 0 or length > 2_000_000:
                    raise ValueError('Invalid GSI body length')
                payload = json.loads(self.rfile.read(length))
                recorder.append(payload)
            except PermissionError as error:
                self.send_error(403, str(error)); return
            except (ValueError, json.JSONDecodeError) as error:
                self.send_error(400, str(error)); return
            self.send_response(200); self.end_headers()

        def log_message(self, format, *args):
            return
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--token')
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('The initial recorder is restricted to loopback')
    recorder = GSIRecorder(args.output, args.token)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(recorder))
    print(f'Recording read-only GSI at http://{args.host}:{args.port}/ to {args.output}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close(); recorder.close()


if __name__ == '__main__':
    main()
