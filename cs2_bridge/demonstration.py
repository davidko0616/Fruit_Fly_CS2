"""Compact Windows input tracing and causal demonstration-row assembly."""
from bisect import bisect_right
from collections import Counter
import math
import os
import threading
import time


CONTROL_NAMES = (
    'forward', 'backward', 'strafe_left', 'strafe_right', 'fire',
    'walk', 'crouch', 'jump', 'secondary_fire')
CONTROL_BITS = {name: 1 << index for index, name in enumerate(CONTROL_NAMES)}


def control_active(sample, name):
    """Return one compact trace sample's named control state."""
    if 'control_bits' in sample:
        return bool(int(sample['control_bits']) & CONTROL_BITS[name])
    return bool(sample.get(name, False))


class WindowsInputStateSampler:
    """Sample CS2 controls without installing hooks or emitting input."""

    _VIRTUAL_KEYS = {
        'forward': 0x57, 'backward': 0x53,
        'strafe_left': 0x41, 'strafe_right': 0x44,
        'fire': 0x01, 'walk': 0x10, 'crouch': 0x11,
        'jump': 0x20, 'secondary_fire': 0x02,
    }

    def __init__(self, sample_hz=100.0, foreground_marker='Counter-Strike 2'):
        if os.name != 'nt':
            raise RuntimeError('Windows input-state sampling requires Windows')
        if not math.isfinite(float(sample_hz)) or float(sample_hz) <= 0:
            raise ValueError('sample_hz must be positive and finite')
        import ctypes
        self.user32 = ctypes.windll.user32
        self.sample_hz = float(sample_hz)
        self.foreground_marker = str(foreground_marker)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _foreground_is_cs2(self):
        handle = self.user32.GetForegroundWindow()
        if not handle:
            return False
        length = self.user32.GetWindowTextLengthW(handle)
        if length <= 0:
            return False
        import ctypes
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(handle, buffer, length + 1)
        return self.foreground_marker.lower() in buffer.value.lower()

    def sample_once(self):
        foreground = self._foreground_is_cs2()
        bits = 0
        if foreground:
            for name, virtual_key in self._VIRTUAL_KEYS.items():
                if self.user32.GetAsyncKeyState(virtual_key) & 0x8000:
                    bits |= CONTROL_BITS[name]
        return {
            'monotonic_ns': time.monotonic_ns(),
            'foreground_cs2': foreground,
            'control_bits': bits,
        }

    def _run(self):
        period = 1.0 / self.sample_hz
        deadline = time.perf_counter()
        while not self._stop.is_set():
            self.samples.append(self.sample_once())
            deadline += period
            self._stop.wait(max(0.0, deadline - time.perf_counter()))

    def start(self):
        if self._thread is not None:
            raise RuntimeError('Input sampler has already been started')
        self._thread = threading.Thread(
            target=self._run, name='cs2-input-state-sampler', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return list(self.samples)


def _round_id(record):
    return (record.get('gsi_snapshot') or {}).get('round_id')


def _normalize_yaw_delta(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def _interval_samples(samples, timestamps, start_ns, end_ns):
    left = bisect_right(timestamps, int(start_ns))
    right = bisect_right(timestamps, int(end_ns))
    return samples[left:right]


def build_demonstration_rows(shadow_records, input_samples, sample_hz=100.0,
                             minimum_coverage=.6, minimum_foreground=.8,
                             minimum_interval_seconds=.05,
                             maximum_interval_seconds=.5):
    """Pair each previous observation with controls until the next frame.

    The label describes what the player did after seeing the previous frame.
    This avoids pairing an action with state that already includes its effect.
    """
    samples = sorted(input_samples, key=lambda row: int(row['monotonic_ns']))
    timestamps = [int(row['monotonic_ns']) for row in samples]
    accepted = [row for row in shadow_records
                if row.get('status') == 'accepted' and
                isinstance((row.get('decision') or {}).get('observation'), list)]
    rows = []
    invalid_counts = Counter()
    for previous, current in zip(accepted, accepted[1:]):
        start_ns = int(previous['monotonic_ns'])
        end_ns = int(current['monotonic_ns'])
        interval_seconds = (end_ns - start_ns) / 1e9
        interval = _interval_samples(samples, timestamps, start_ns, end_ns)
        expected_samples = max(1.0, interval_seconds * float(sample_hz))
        coverage = min(1.0, len(interval) / expected_samples)
        foreground_fraction = (sum(bool(row.get('foreground_cs2')) for row in interval) /
                               len(interval) if interval else 0.0)
        duties = {
            name: (sum(control_active(row, name) for row in interval) /
                   len(interval) if interval else 0.0)
            for name in CONTROL_NAMES
        }
        opposing_longitudinal = (sum(
            control_active(row, 'forward') and control_active(row, 'backward')
            for row in interval) / len(interval) if interval else 0.0)
        opposing_strafe = (sum(
            control_active(row, 'strafe_left') and
            control_active(row, 'strafe_right')
            for row in interval) / len(interval) if interval else 0.0)

        reasons = []
        if _round_id(previous) != _round_id(current):
            reasons.append('round_discontinuity')
        if not minimum_interval_seconds <= interval_seconds <= maximum_interval_seconds:
            reasons.append('invalid_frame_interval')
        if coverage < minimum_coverage:
            reasons.append('insufficient_input_coverage')
        if foreground_fraction < minimum_foreground:
            reasons.append('foreground_interruption')
        previous_pose = previous.get('radar_pose') or {}
        current_pose = current.get('radar_pose') or {}
        try:
            yaw_delta = _normalize_yaw_delta(
                float(current_pose['yaw_degrees']) -
                float(previous_pose['yaw_degrees']))
            yaw_rate = yaw_delta / interval_seconds
            if not math.isfinite(yaw_rate):
                raise ValueError
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            yaw_delta = None
            yaw_rate = None
            reasons.append('invalid_yaw_label')

        for reason in set(reasons):
            invalid_counts[reason] += 1
        decision = previous['decision']
        rows.append({
            'schema_version': 1,
            'source_frame_index': previous.get('frame_index'),
            'next_frame_index': current.get('frame_index'),
            'round_id': _round_id(previous),
            'observation_monotonic_ns': start_ns,
            'label_end_monotonic_ns': end_ns,
            'interval_seconds': interval_seconds,
            'observation': list(decision['observation']),
            'action_mask': list(decision.get('action_mask', [])),
            'pose': previous_pose,
            'next_pose': current_pose,
            'target_visible': previous.get('primary_visible_target') is not None,
            'target_memory_active': bool(
                decision.get('target_memory_in_observation')),
            'patrol_active': bool(decision.get('patrol_active')),
            'controls': {
                'forward_duty': duties['forward'],
                'backward_duty': duties['backward'],
                'forward': duties['forward'] - duties['backward'],
                'strafe_left_duty': duties['strafe_left'],
                'strafe_right_duty': duties['strafe_right'],
                'strafe': duties['strafe_right'] - duties['strafe_left'],
                'turn_yaw_delta_degrees': yaw_delta,
                'turn_yaw_rate_degrees_per_second': yaw_rate,
                'fire': duties['fire'],
                'walk': duties['walk'],
                'crouch': duties['crouch'],
                'jump': duties['jump'],
                'secondary_fire': duties['secondary_fire'],
            },
            'input_quality': {
                'sample_count': len(interval),
                'expected_sample_count': expected_samples,
                'coverage': coverage,
                'foreground_fraction': foreground_fraction,
                'opposing_longitudinal_fraction': opposing_longitudinal,
                'opposing_strafe_fraction': opposing_strafe,
            },
            'training_valid': not reasons,
            'invalid_reasons': sorted(set(reasons)),
        })
    summary = {
        'candidate_rows': len(rows),
        'training_valid_rows': sum(row['training_valid'] for row in rows),
        'training_invalid_rows': sum(not row['training_valid'] for row in rows),
        'invalid_reason_counts': dict(sorted(invalid_counts.items())),
    }
    valid_rows = [row for row in rows if row['training_valid']]
    summary['valid_label_activity'] = {
        'forward_rows': sum(row['controls']['forward'] > .1 for row in valid_rows),
        'backward_rows': sum(row['controls']['forward'] < -.1 for row in valid_rows),
        'strafe_left_rows': sum(row['controls']['strafe'] < -.1 for row in valid_rows),
        'strafe_right_rows': sum(row['controls']['strafe'] > .1 for row in valid_rows),
        'turn_rows': sum(abs(row['controls']['turn_yaw_delta_degrees']) > .25
                         for row in valid_rows),
        'fire_rows': sum(row['controls']['fire'] > 0 for row in valid_rows),
        'simultaneous_move_and_turn_rows': sum(
            (abs(row['controls']['forward']) > .1 or
             abs(row['controls']['strafe']) > .1) and
            abs(row['controls']['turn_yaw_delta_degrees']) > .25
            for row in valid_rows),
    }
    yaw_deltas = [abs(row['controls']['turn_yaw_delta_degrees'])
                  for row in valid_rows]
    summary['valid_yaw_delta_degrees'] = {
        'mean_absolute': (sum(yaw_deltas) / len(yaw_deltas)
                          if yaw_deltas else None),
        'maximum_absolute': max(yaw_deltas) if yaw_deltas else None,
    }
    return rows, summary
