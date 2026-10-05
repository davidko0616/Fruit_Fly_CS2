"""Guarded, local-practice-only execution of CS2 policy actions."""
from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import threading
import sys
import time

from toy_combat.env import ACTION_NAMES


@dataclass(frozen=True)
class InputSafetyConfig:
    """Hard bounds applied after perception and before native input."""

    max_frame_age_ms: float = 250.0
    max_gsi_age_ms: float = 1_000.0
    key_hold_ms: float = 60.0
    sustain_movement: bool = False
    turn_pixels: int = 32
    turn_duration_ms: float = 110.0
    turn_substeps: int = 8
    required_window_title: str = 'Counter-Strike 2'
    fire_enabled: bool = False

    def __post_init__(self):
        if not 0 < float(self.max_frame_age_ms) <= 1_000:
            raise ValueError('max_frame_age_ms must be in (0, 1000]')
        if not 0 < float(self.max_gsi_age_ms) <= 5_000:
            raise ValueError('max_gsi_age_ms must be in (0, 5000]')
        if not 0 < float(self.key_hold_ms) <= 250:
            raise ValueError('key_hold_ms must be in (0, 250]')
        if not 0 < int(self.turn_pixels) <= 200:
            raise ValueError('turn_pixels must be in (0, 200]')
        if not 0 < float(self.turn_duration_ms) <= 250:
            raise ValueError('turn_duration_ms must be in (0, 250]')
        if not 1 <= int(self.turn_substeps) <= int(self.turn_pixels):
            raise ValueError('turn_substeps must be in [1, turn_pixels]')
        if not str(self.required_window_title).strip():
            raise ValueError('required_window_title is required')

    def to_dict(self):
        return asdict(self)


class WindowsSendInputBackend:
    """Minimal Windows input backend. Construct only after explicit enablement."""

    VK_F12 = 0x7B
    KEYEVENTF_KEYUP = 0x0002
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    VIRTUAL_KEYS = {'w': 0x57, 's': 0x53, 'a': 0x41, 'd': 0x44}

    def __init__(self):
        if sys.platform != 'win32':
            raise RuntimeError('Native CS2 input is supported on Windows only')
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)

        ulong_ptr = (ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8
                     else ctypes.c_ulong)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ('dx', wintypes.LONG), ('dy', wintypes.LONG),
                ('mouseData', wintypes.DWORD), ('dwFlags', wintypes.DWORD),
                ('time', wintypes.DWORD), ('dwExtraInfo', ulong_ptr),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ('wVk', wintypes.WORD), ('wScan', wintypes.WORD),
                ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD),
                ('dwExtraInfo', ulong_ptr),
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [('mi', MOUSEINPUT), ('ki', KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ('union',)
            _fields_ = [('type', wintypes.DWORD), ('union', INPUT_UNION)]

        self._mouse_input = MOUSEINPUT
        self._keyboard_input = KEYBDINPUT
        self._input_union = INPUT_UNION
        self._input = INPUT
        self.user32.SendInput.argtypes = (
            wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.GetForegroundWindow.argtypes = ()
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
        self.user32.GetWindowTextLengthW.restype = ctypes.c_int
        self.user32.GetWindowTextW.argtypes = (
            wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
        self.user32.GetWindowTextW.restype = ctypes.c_int
        self.user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
        self.user32.GetAsyncKeyState.restype = wintypes.SHORT

    def _send(self, input_value):
        sent = self.user32.SendInput(
            1, ctypes.byref(input_value), ctypes.sizeof(self._input))
        if sent != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def key_down(self, key):
        self._key(key, 0)

    def key_up(self, key):
        self._key(key, self.KEYEVENTF_KEYUP)

    def _key(self, key, flags):
        if key not in self.VIRTUAL_KEYS:
            raise ValueError(f'Unsupported key: {key}')
        value = self._input(
            type=self.INPUT_KEYBOARD,
            union=self._input_union(ki=self._keyboard_input(
                wVk=self.VIRTUAL_KEYS[key], dwFlags=flags)))
        self._send(value)

    def move_mouse(self, dx, dy=0):
        value = self._input(
            type=self.INPUT_MOUSE,
            union=self._input_union(mi=self._mouse_input(
                dx=int(dx), dy=int(dy), dwFlags=self.MOUSEEVENTF_MOVE)))
        self._send(value)

    def click_left(self):
        for flags in (self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP):
            value = self._input(
                type=self.INPUT_MOUSE,
                union=self._input_union(mi=self._mouse_input(dwFlags=flags)))
            self._send(value)

    def foreground_title(self):
        window = self.user32.GetForegroundWindow()
        if not window:
            return ''
        length = self.user32.GetWindowTextLengthW(window)
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(window, buffer, len(buffer))
        return buffer.value

    def emergency_stop_pressed(self):
        return bool(self.user32.GetAsyncKeyState(self.VK_F12) & 0x8000)


class GuardedActionExecutor:
    """Validate a live decision and emit at most one bounded action."""

    KEY_ACTIONS = {
        'forward': 'w', 'backward': 's',
        'strafe_left': 'a', 'strafe_right': 'd',
    }

    def __init__(self, backend, enabled=False, config=None, sleep=time.sleep,
                 turn_wait=None):
        self.backend = backend
        self.enabled = bool(enabled)
        self.config = config or InputSafetyConfig()
        self.sleep = sleep
        self.turn_wait = (turn_wait if turn_wait is not None else
                          lambda event, seconds: event.wait(seconds))
        self.emergency_stop_latched = False
        self.held_keys = set()
        self._turn_lock = threading.Lock()
        self._turn_thread = None
        self._turn_stop = None
        self._turn_direction = None
        self._turn_pending_direction = None
        self.last_turn_stop_reason = None
        self._turn_commands_received = 0
        self._turn_commands_queued = 0
        self._turn_commands_replaced = 0
        self._turns_started = 0
        self._turn_substeps_emitted = 0
        self._turn_pixels_emitted_signed = 0
        self._turn_pixels_emitted_absolute = 0
        self._turn_stop_counts = {}
        self._movement_lock = threading.Lock()
        self._movement_thread = None
        self._movement_stop = None
        self._movement_key = None
        self._movement_deadline = None
        self.last_movement_stop_reason = None
        self._movement_commands_received = 0
        self._movement_commands_renewed = 0
        self._movement_holds_started = 0
        self._movement_stop_counts = {}

    def _result(self, reason, action_name=None, input_emitted=False, **extra):
        result = {
            'enabled': self.enabled,
            'input_emitted': bool(input_emitted),
            'reason': reason,
            'action_name': action_name,
            'emergency_stop_latched': self.emergency_stop_latched,
        }
        result.update(extra)
        return result

    def poll_emergency_stop(self):
        if self.backend.emergency_stop_pressed():
            self.emergency_stop_latched = True
            self._stop_turn()
            self._stop_movement()
            self.release_all()
        return self.emergency_stop_latched

    def _run_movement(self, key, stop_event):
        reason = 'expired'
        try:
            self.backend.key_down(key)
            self.held_keys.add(key)
            while True:
                if stop_event.is_set():
                    reason = 'superseded'
                    break
                if self.backend.emergency_stop_pressed():
                    self.emergency_stop_latched = True
                    reason = 'emergency_stop'
                    break
                title = self.backend.foreground_title()
                if self.config.required_window_title.casefold() not in title.casefold():
                    reason = 'foreground_window_mismatch'
                    break
                with self._movement_lock:
                    deadline = self._movement_deadline
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                stop_event.wait(min(.01, remaining))
        finally:
            try:
                self.backend.key_up(key)
            finally:
                self.held_keys.discard(key)
            self.last_movement_stop_reason = reason
            with self._movement_lock:
                self._movement_stop_counts[reason] = (
                    self._movement_stop_counts.get(reason, 0) + 1)
                if self._movement_thread is threading.current_thread():
                    self._movement_thread = None
                    self._movement_stop = None
                    self._movement_key = None
                    self._movement_deadline = None

    def _stop_movement(self):
        with self._movement_lock:
            thread = self._movement_thread
            stop_event = self._movement_stop
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=.5)
        with self._movement_lock:
            if self._movement_thread is thread:
                self._movement_thread = None
                self._movement_stop = None
                self._movement_key = None
                self._movement_deadline = None

    def _start_movement(self, key):
        deadline = time.monotonic() + self.config.key_hold_ms / 1_000
        with self._movement_lock:
            self._movement_commands_received += 1
            active = (self._movement_thread is not None and
                      self._movement_thread.is_alive() and
                      self._movement_stop is not None and
                      not self._movement_stop.is_set())
            if active and key == self._movement_key:
                self._movement_deadline = deadline
                self._movement_commands_renewed += 1
                return 'renewed'
        self._stop_movement()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_movement, args=(key, stop_event),
            name='cs2-sustained-movement', daemon=True)
        with self._movement_lock:
            self._movement_stop = stop_event
            self._movement_thread = thread
            self._movement_key = key
            self._movement_deadline = deadline
            self._movement_holds_started += 1
        thread.start()
        return 'started'

    def wait_for_movement(self, timeout=None):
        with self._movement_lock:
            thread = self._movement_thread
        if thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def _blocked_result(self, reason, action_name=None, **extra):
        self._stop_turn()
        self._stop_movement()
        return self._result(reason, action_name, **extra)

    def _turn_deltas(self, direction):
        """Return integer mouse deltas following a smoothstep curve."""
        cumulative = 0
        deltas = []
        for index in range(1, self.config.turn_substeps + 1):
            progress = index / self.config.turn_substeps
            eased = progress * progress * (3.0 - 2.0 * progress)
            next_cumulative = round(self.config.turn_pixels * eased)
            delta = next_cumulative - cumulative
            if delta:
                deltas.append(direction * delta)
            cumulative = next_cumulative
        return tuple(deltas)

    def _run_turn(self, direction, stop_event):
        try:
            self._run_turn_loop(direction, stop_event)
        finally:
            with self._turn_lock:
                if self._turn_thread is threading.current_thread():
                    self._turn_thread = None
                    self._turn_stop = None
                    self._turn_direction = None
                    self._turn_pending_direction = None

    def _run_turn_loop(self, direction, stop_event):
        while True:
            deltas = self._turn_deltas(direction)
            interval = (0.0 if len(deltas) <= 1 else
                        self.config.turn_duration_ms / 1_000 /
                        (len(deltas) - 1))
            reason = 'completed'
            for index, delta in enumerate(deltas):
                if stop_event.is_set():
                    reason = 'superseded'
                    break
                if self.backend.emergency_stop_pressed():
                    self.emergency_stop_latched = True
                    reason = 'emergency_stop'
                    self.release_all()
                    break
                title = self.backend.foreground_title()
                if self.config.required_window_title.casefold() not in title.casefold():
                    reason = 'foreground_window_mismatch'
                    self.release_all()
                    break
                self.backend.move_mouse(delta, 0)
                with self._turn_lock:
                    self._turn_substeps_emitted += 1
                    self._turn_pixels_emitted_signed += delta
                    self._turn_pixels_emitted_absolute += abs(delta)
                if (index + 1 < len(deltas) and
                        self.turn_wait(stop_event, interval)):
                    reason = 'superseded'
                    break
            self.last_turn_stop_reason = reason
            with self._turn_lock:
                self._turn_stop_counts[reason] = (
                    self._turn_stop_counts.get(reason, 0) + 1)
                pending = self._turn_pending_direction
                if (reason == 'completed' and not stop_event.is_set() and
                        pending is not None):
                    direction = pending
                    self._turn_pending_direction = None
                    self._turn_direction = direction
                    self._turns_started += 1
                    continue
                if self._turn_thread is threading.current_thread():
                    self._turn_thread = None
                    self._turn_stop = None
                    self._turn_direction = None
                    self._turn_pending_direction = None
                return

    def _stop_turn(self):
        with self._turn_lock:
            thread = self._turn_thread
            stop_event = self._turn_stop
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=.5)
        with self._turn_lock:
            if self._turn_thread is thread:
                self._turn_thread = None
                self._turn_stop = None
                self._turn_direction = None
                self._turn_pending_direction = None

    def _start_turn(self, direction):
        with self._turn_lock:
            self._turn_commands_received += 1
            active = (self._turn_thread is not None and
                      self._turn_thread.is_alive() and
                      self._turn_stop is not None and
                      not self._turn_stop.is_set())
            if active and direction == self._turn_direction:
                dispatch = ('replaced' if self._turn_pending_direction is not None
                            else 'queued')
                if dispatch == 'replaced':
                    self._turn_commands_replaced += 1
                self._turn_pending_direction = direction
                self._turn_commands_queued += 1
                return dispatch
        self._stop_turn()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_turn, args=(direction, stop_event),
            name='cs2-smooth-turn', daemon=True)
        with self._turn_lock:
            self._turn_stop = stop_event
            self._turn_thread = thread
            self._turn_direction = direction
            self._turns_started += 1
        thread.start()
        return 'started'

    def wait_for_turn(self, timeout=None):
        with self._turn_lock:
            thread = self._turn_thread
        if thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def actuation_summary(self):
        with self._turn_lock:
            result = {
                'smooth_turn_commands_received': self._turn_commands_received,
                'smooth_turn_commands_queued': self._turn_commands_queued,
                'smooth_turn_commands_replaced': self._turn_commands_replaced,
                'smooth_turns_started': self._turns_started,
                'smooth_turn_substeps_emitted': self._turn_substeps_emitted,
                'smooth_turn_pixels_emitted_signed': (
                    self._turn_pixels_emitted_signed),
                'smooth_turn_pixels_emitted_absolute': (
                    self._turn_pixels_emitted_absolute),
                'smooth_turn_stop_counts': dict(sorted(
                    self._turn_stop_counts.items())),
                'smooth_turn_active': (
                    self._turn_thread is not None and
                    self._turn_thread.is_alive()),
            }
        with self._movement_lock:
            result.update({
                'sustained_movement_commands_received': (
                    self._movement_commands_received),
                'sustained_movement_commands_renewed': (
                    self._movement_commands_renewed),
                'sustained_movement_holds_started': (
                    self._movement_holds_started),
                'sustained_movement_stop_counts': dict(sorted(
                    self._movement_stop_counts.items())),
                'sustained_movement_active': (
                    self._movement_thread is not None and
                    self._movement_thread.is_alive()),
            })
        return result

    def execute(self, record, now_ns=None):
        """Execute one accepted record after every safety condition passes."""
        if not self.enabled:
            return self._blocked_result('executor_disabled')
        if self.poll_emergency_stop():
            return self._result('emergency_stop')
        if record.get('status') != 'accepted':
            return self._blocked_result('record_not_accepted')

        decision = record.get('decision')
        if not isinstance(decision, dict):
            return self._blocked_result('missing_decision')
        action = decision.get('action')
        action_name = decision.get('action_name')
        mask = decision.get('action_mask')
        if (not isinstance(action, int) or not 0 <= action < len(ACTION_NAMES) or
                action_name != ACTION_NAMES[action]):
            return self._blocked_result('invalid_action', action_name)
        if (not isinstance(mask, (list, tuple)) or len(mask) != len(ACTION_NAMES)
                or not bool(mask[action])):
            return self._blocked_result('action_mask_blocked', action_name)

        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        frame_ns = int(record.get('monotonic_ns', -1))
        frame_age_ms = (now_ns - frame_ns) / 1e6
        if (frame_ns < 0 or frame_age_ms < 0 or
                frame_age_ms > self.config.max_frame_age_ms):
            return self._blocked_result(
                'stale_frame', action_name, frame_age_ms=frame_age_ms)

        gsi_delta_ns = record.get('gsi_delta_ns')
        if gsi_delta_ns is None or int(gsi_delta_ns) > 0:
            return self._blocked_result('invalid_gsi_time', action_name)
        gsi_age_ms = (now_ns - (frame_ns + int(gsi_delta_ns))) / 1e6
        if gsi_age_ms < 0 or gsi_age_ms > self.config.max_gsi_age_ms:
            return self._blocked_result(
                'stale_gsi', action_name, frame_age_ms=frame_age_ms,
                gsi_age_ms=gsi_age_ms)

        snapshot = record.get('gsi_snapshot') or {}
        if (snapshot.get('map_name') != 'de_dust2' or
                snapshot.get('player_activity') != 'playing' or
                snapshot.get('round_id') is None or
                snapshot.get('health') is None or int(snapshot['health']) <= 0):
            return self._blocked_result('inactive_gsi', action_name)

        title = self.backend.foreground_title()
        if self.config.required_window_title.casefold() not in title.casefold():
            self.release_all()
            return self._blocked_result(
                'foreground_window_mismatch', action_name,
                foreground_title=title)

        if action_name == 'wait':
            self._stop_turn()
            self._stop_movement()
            return self._result(
                'wait', action_name, frame_age_ms=frame_age_ms,
                gsi_age_ms=gsi_age_ms)
        if action_name == 'fire' and not self.config.fire_enabled:
            return self._blocked_result(
                'fire_disabled', action_name, frame_age_ms=frame_age_ms,
                gsi_age_ms=gsi_age_ms)

        movement_dispatch = None
        if action_name in self.KEY_ACTIONS:
            self._stop_turn()
            key = self.KEY_ACTIONS[action_name]
            if self.config.sustain_movement:
                movement_dispatch = self._start_movement(key)
            else:
                self.backend.key_down(key)
                self.held_keys.add(key)
                try:
                    self.sleep(self.config.key_hold_ms / 1_000)
                finally:
                    self.backend.key_up(key)
                    self.held_keys.discard(key)
        elif action_name in ('turn_left', 'turn_right'):
            self._stop_movement()
            direction = -1 if action_name == 'turn_left' else 1
            turn_dispatch = self._start_turn(direction)
        elif action_name == 'fire':
            self._stop_turn()
            self._stop_movement()
            self.backend.click_left()
        else:
            return self._blocked_result('unsupported_action', action_name)

        return self._result(
            'executed', action_name, input_emitted=True,
            frame_age_ms=frame_age_ms, gsi_age_ms=gsi_age_ms,
            movement_actuation=(None if action_name not in self.KEY_ACTIONS else {
                'key': self.KEY_ACTIONS[action_name],
                'hold_ms': self.config.key_hold_ms,
                'mode': ('sustained' if self.config.sustain_movement
                         else 'pulse'),
                'dispatch': movement_dispatch,
            }),
            turn_actuation=(None if action_name not in
                            ('turn_left', 'turn_right') else {
                                'pixels': self.config.turn_pixels,
                                'duration_ms': self.config.turn_duration_ms,
                                'substeps': self.config.turn_substeps,
                                'curve': 'smoothstep',
                                'dispatch': turn_dispatch,
                            }))

    def release_all(self):
        for key in ('w', 's', 'a', 'd'):
            try:
                self.backend.key_up(key)
            except Exception:
                pass
        self.held_keys.clear()

    def close(self):
        self._stop_turn()
        self._stop_movement()
        self.release_all()
