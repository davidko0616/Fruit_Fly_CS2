"""Guarded, local-practice-only execution of CS2 policy actions."""
from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import sys
import time

from toy_combat.env import ACTION_NAMES


@dataclass(frozen=True)
class InputSafetyConfig:
    """Hard bounds applied after perception and before native input."""

    max_frame_age_ms: float = 250.0
    max_gsi_age_ms: float = 1_000.0
    key_hold_ms: float = 60.0
    turn_pixels: int = 32
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

    def __init__(self, backend, enabled=False, config=None, sleep=time.sleep):
        self.backend = backend
        self.enabled = bool(enabled)
        self.config = config or InputSafetyConfig()
        self.sleep = sleep
        self.emergency_stop_latched = False
        self.held_keys = set()

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
            self.release_all()
        return self.emergency_stop_latched

    def execute(self, record, now_ns=None):
        """Execute one accepted record after every safety condition passes."""
        if not self.enabled:
            return self._result('executor_disabled')
        if self.poll_emergency_stop():
            return self._result('emergency_stop')
        if record.get('status') != 'accepted':
            return self._result('record_not_accepted')

        decision = record.get('decision')
        if not isinstance(decision, dict):
            return self._result('missing_decision')
        action = decision.get('action')
        action_name = decision.get('action_name')
        mask = decision.get('action_mask')
        if (not isinstance(action, int) or not 0 <= action < len(ACTION_NAMES) or
                action_name != ACTION_NAMES[action]):
            return self._result('invalid_action', action_name)
        if (not isinstance(mask, (list, tuple)) or len(mask) != len(ACTION_NAMES)
                or not bool(mask[action])):
            return self._result('action_mask_blocked', action_name)

        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        frame_ns = int(record.get('monotonic_ns', -1))
        frame_age_ms = (now_ns - frame_ns) / 1e6
        if (frame_ns < 0 or frame_age_ms < 0 or
                frame_age_ms > self.config.max_frame_age_ms):
            return self._result(
                'stale_frame', action_name, frame_age_ms=frame_age_ms)

        gsi_delta_ns = record.get('gsi_delta_ns')
        if gsi_delta_ns is None or int(gsi_delta_ns) > 0:
            return self._result('invalid_gsi_time', action_name)
        gsi_age_ms = (now_ns - (frame_ns + int(gsi_delta_ns))) / 1e6
        if gsi_age_ms < 0 or gsi_age_ms > self.config.max_gsi_age_ms:
            return self._result(
                'stale_gsi', action_name, frame_age_ms=frame_age_ms,
                gsi_age_ms=gsi_age_ms)

        snapshot = record.get('gsi_snapshot') or {}
        if (snapshot.get('map_name') != 'de_dust2' or
                snapshot.get('player_activity') != 'playing' or
                snapshot.get('round_id') is None or
                snapshot.get('health') is None or int(snapshot['health']) <= 0):
            return self._result('inactive_gsi', action_name)

        title = self.backend.foreground_title()
        if self.config.required_window_title.casefold() not in title.casefold():
            self.release_all()
            return self._result(
                'foreground_window_mismatch', action_name,
                foreground_title=title)

        if action_name == 'wait':
            return self._result(
                'wait', action_name, frame_age_ms=frame_age_ms,
                gsi_age_ms=gsi_age_ms)
        if action_name == 'fire' and not self.config.fire_enabled:
            return self._result(
                'fire_disabled', action_name, frame_age_ms=frame_age_ms,
                gsi_age_ms=gsi_age_ms)

        if action_name in self.KEY_ACTIONS:
            key = self.KEY_ACTIONS[action_name]
            self.backend.key_down(key)
            self.held_keys.add(key)
            try:
                self.sleep(self.config.key_hold_ms / 1_000)
            finally:
                self.backend.key_up(key)
                self.held_keys.discard(key)
        elif action_name in ('turn_left', 'turn_right'):
            direction = -1 if action_name == 'turn_left' else 1
            self.backend.move_mouse(direction * self.config.turn_pixels, 0)
        elif action_name == 'fire':
            self.backend.click_left()
        else:
            return self._result('unsupported_action', action_name)

        return self._result(
            'executed', action_name, input_emitted=True,
            frame_age_ms=frame_age_ms, gsi_age_ms=gsi_age_ms)

    def release_all(self):
        for key in ('w', 's', 'a', 'd'):
            try:
                self.backend.key_up(key)
            except Exception:
                pass
        self.held_keys.clear()

    def close(self):
        self.release_all()
