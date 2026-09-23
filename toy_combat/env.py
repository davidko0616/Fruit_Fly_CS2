"""Small deterministic 2D aiming environment with a CS2-like step boundary."""
from dataclasses import dataclass
import math

import numpy as np


ACTION_NAMES = ('wait', 'forward', 'backward', 'strafe_left', 'strafe_right',
                'turn_left', 'turn_right', 'fire')
REWARD_NAMES = ('time', 'aim_progress', 'collision', 'shot')


@dataclass(frozen=True)
class CombatConfig:
    grid_size: int = 9
    obstacle_count: int = 5
    max_ticks: int = 80
    turn_degrees: float = 11.25
    hit_tolerance_degrees: float = 11.25
    fire_cooldown_ticks: int = 2
    movement_enabled: bool = False
    time_penalty: float = -0.01
    aim_progress_scale: float = 0.2
    collision_penalty: float = -0.05
    miss_penalty: float = -0.15
    hit_reward: float = 2.0


class ToyCombatEnv:
    """One agent, one stationary target, deterministic obstacles and transitions."""
    observation_size = 10
    action_size = len(ACTION_NAMES)

    def __init__(self, config=None):
        self.config = config or CombatConfig()
        if self.config.grid_size < 5 or self.config.obstacle_count < 0:
            raise ValueError('Grid must be at least 5x5 and obstacle count nonnegative')
        self.rng = None
        self.episode_seed = None
        self.agent = self.enemy = self.obstacles = None
        self.heading = self.cooldown = self.tick = 0
        self.terminated = self.truncated = False

    def reset(self, seed):
        self.episode_seed = int(seed)
        self.rng = np.random.default_rng(self.episode_seed)
        cells = np.array([(x, y) for x in range(self.config.grid_size)
                          for y in range(self.config.grid_size)], dtype=np.int16)
        selected = self.rng.choice(len(cells), size=2, replace=False)
        self.agent = cells[selected[0]].copy()
        self.enemy = cells[selected[1]].copy()
        delta = self.enemy - self.agent
        steps = int(max(abs(int(delta[0])), abs(int(delta[1]))))
        clear = {tuple(self.agent), tuple(self.enemy)}
        clear.update(tuple(np.rint(self.agent + delta * (i / steps)).astype(int))
                     for i in range(1, steps))
        candidates = np.asarray([cell for cell in cells if tuple(cell) not in clear])
        obstacle_indices = self.rng.choice(len(candidates), size=self.config.obstacle_count, replace=False)
        self.obstacles = {tuple(v) for v in candidates[obstacle_indices]}
        self.heading = int(self.rng.integers(0, self.heading_count))
        self.cooldown = self.tick = 0
        self.terminated = self.truncated = False
        return self.observation(), self.info(reset_reason='episode_start')

    @property
    def heading_count(self):
        return round(360 / self.config.turn_degrees)

    @property
    def angle(self):
        return self.heading * math.radians(self.config.turn_degrees)

    def _relative(self):
        delta = self.enemy.astype(np.float32) - self.agent
        forward = np.array([math.cos(self.angle), math.sin(self.angle)], dtype=np.float32)
        right = np.array([-forward[1], forward[0]], dtype=np.float32)
        distance = float(np.linalg.norm(delta))
        return delta, forward, right, distance

    def _line_of_sight(self):
        # Sample cell centers along the ray; endpoints are never obstacles.
        delta = self.enemy - self.agent
        steps = int(max(abs(int(delta[0])), abs(int(delta[1]))))
        if steps <= 1:
            return True
        for i in range(1, steps):
            point = np.rint(self.agent + delta * (i / steps)).astype(int)
            if tuple(point) in self.obstacles:
                return False
        return True

    def aim_alignment(self):
        delta, forward, _, distance = self._relative()
        return 1.0 if distance == 0 else float(np.dot(delta, forward) / distance)

    def observation(self):
        delta, forward, right, distance = self._relative()
        scale = max(1, self.config.grid_size - 1)
        local_forward = float(np.dot(delta, forward) / scale)
        local_right = float(np.dot(delta, right) / scale)
        return np.asarray([self.agent[0] / scale, self.agent[1] / scale,
                           forward[0], forward[1], local_forward, local_right,
                           distance / (math.sqrt(2) * scale), self.aim_alignment(),
                           float(self._line_of_sight()),
                           self.cooldown / max(1, self.config.fire_cooldown_ticks)], dtype=np.float32)

    def action_mask(self):
        mask = np.ones(self.action_size, dtype=bool)
        if not self.config.movement_enabled:
            mask[1:5] = False
        mask[7] = self.cooldown == 0
        return mask

    def privileged_state(self):
        return np.asarray([self.agent[0], self.agent[1], self.heading,
                           self.enemy[0], self.enemy[1], self.cooldown, self.tick], dtype=np.int32)

    def info(self, **extra):
        return {'tick': self.tick, 'episode_seed': self.episode_seed,
                'privileged_state': self.privileged_state(),
                'action_mask': self.action_mask(), 'line_of_sight': self._line_of_sight(),
                'aim_alignment': self.aim_alignment(), **extra}

    def _movement_delta(self, action):
        forward = np.asarray([round(math.cos(self.angle)), round(math.sin(self.angle))], dtype=int)
        right = np.asarray([-forward[1], forward[0]], dtype=int)
        return {1: forward, 2: -forward, 3: -right, 4: right}[action]

    def _single_tick(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError('Reset is required after episode completion')
        before_aim = self.aim_alignment()
        components = dict.fromkeys(REWARD_NAMES, 0.0)
        components['time'] = self.config.time_penalty
        rejected = False
        hit = miss = collision = False
        if action in (1, 2, 3, 4):
            if not self.config.movement_enabled:
                rejected = True
            else:
                target = self.agent + self._movement_delta(action)
                outside = np.any(target < 0) or np.any(target >= self.config.grid_size)
                if outside or tuple(target) in self.obstacles or np.array_equal(target, self.enemy):
                    collision = True
                    components['collision'] = self.config.collision_penalty
                else:
                    self.agent = target
        elif action == 5:
            self.heading = (self.heading + 1) % self.heading_count
        elif action == 6:
            self.heading = (self.heading - 1) % self.heading_count
        elif action == 7:
            if self.cooldown:
                rejected = True
            else:
                tolerance = math.cos(math.radians(self.config.hit_tolerance_degrees))
                hit = self._line_of_sight() and self.aim_alignment() + 1e-6 >= tolerance
                miss = not hit
                components['shot'] = self.config.hit_reward if hit else self.config.miss_penalty
                self.cooldown = self.config.fire_cooldown_ticks + 1
        elif action != 0:
            raise ValueError(f'Invalid action {action}')
        self.cooldown = max(0, self.cooldown - 1)
        self.tick += 1
        if not hit:
            components['aim_progress'] = self.config.aim_progress_scale * (self.aim_alignment() - before_aim)
        self.terminated = hit
        self.truncated = self.tick >= self.config.max_ticks and not hit
        return components, rejected, hit, miss, collision

    def step(self, action, repeat=1):
        if repeat < 1:
            raise ValueError('Action repeat must be positive')
        totals = dict.fromkeys(REWARD_NAMES, 0.0)
        applied, rejected = 0, False
        events = {'hit': False, 'miss': False, 'collision': False}
        for _ in range(repeat):
            components, was_rejected, hit, miss, collision = self._single_tick(int(action))
            for name, value in components.items():
                totals[name] += value
            applied += 1
            rejected |= was_rejected
            events['hit'] |= hit; events['miss'] |= miss; events['collision'] |= collision
            if self.terminated or self.truncated:
                break
        reason = 'hit' if self.terminated else 'time_limit' if self.truncated else None
        info = self.info(reward_components=totals, action_rejected=rejected,
                         action_applied_ticks=applied, reset_reason=reason, **events)
        return self.observation(), float(sum(totals.values())), self.terminated, self.truncated, info


def scripted_action(observation, action_mask=None):
    """Simple oracle for environment solvability checks; never used for learning."""
    forward, right, alignment = observation[4], observation[5], observation[7]
    if observation[8] > .5 and alignment + 1e-6 >= math.cos(math.radians(11.25)) and (action_mask is None or action_mask[7]):
        return 7
    if abs(right) > .015:
        return 5 if right > 0 else 6
    return 1 if forward > 0 else 6
