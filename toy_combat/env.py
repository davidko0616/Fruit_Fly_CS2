"""Small deterministic 2D aiming environment with a CS2-like step boundary."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


ACTION_NAMES = ('wait', 'forward', 'backward', 'strafe_left', 'strafe_right',
                'turn_left', 'turn_right', 'fire')
REWARD_NAMES = ('time', 'aim_progress', 'collision', 'shot')
NAVIGATION_REWARD_NAMES = ('time', 'aim_progress', 'distance_progress',
                           'line_of_sight_progress', 'collision', 'shot')


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
    scenario: str = 'aiming_v1'
    require_initial_occlusion: bool = False
    distance_progress_scale: float = 0.0
    line_of_sight_progress_scale: float = 0.0
    navigation_phase_masking: bool = False


def navigation_config():
    """First movement curriculum: occluded target, local clearance sensing, and shooting."""
    return CombatConfig(max_ticks=120, movement_enabled=True, scenario='navigation_v1',
                        require_initial_occlusion=True, aim_progress_scale=0.2,
                        distance_progress_scale=0.1, line_of_sight_progress_scale=0.2,
                        navigation_phase_masking=True, miss_penalty=-0.1, hit_reward=3.0)


def integrated_navigation_config():
    """Second curriculum: navigation and combat actions remain available together."""
    return CombatConfig(max_ticks=120, movement_enabled=True, scenario='navigation_v1',
                        require_initial_occlusion=True, aim_progress_scale=0.2,
                        distance_progress_scale=0.1, line_of_sight_progress_scale=0.2,
                        navigation_phase_masking=False, miss_penalty=-0.1, hit_reward=3.0)


class ToyCombatEnv:
    """One agent, one stationary target, deterministic obstacles and transitions."""
    observation_size = 10
    navigation_observation_size = 14
    action_size = len(ACTION_NAMES)

    def __init__(self, config=None):
        self.config = config or CombatConfig()
        if self.config.grid_size < 5 or self.config.obstacle_count < 0:
            raise ValueError('Grid must be at least 5x5 and obstacle count nonnegative')
        if self.config.scenario not in ('aiming_v1', 'navigation_v1'):
            raise ValueError(f'Unknown scenario: {self.config.scenario}')
        if self.config.require_initial_occlusion and self.config.obstacle_count < 1:
            raise ValueError('Initial occlusion requires at least one obstacle')
        if self.config.scenario == 'navigation_v1' and not self.config.movement_enabled:
            raise ValueError('Navigation scenario requires movement')
        self.rng = None
        self.episode_seed = None
        self.agent = self.enemy = self.obstacles = None
        self.heading = self.cooldown = self.tick = 0
        self.terminated = self.truncated = False

    @staticmethod
    def observation_size_for(config):
        return (ToyCombatEnv.navigation_observation_size
                if config.scenario == 'navigation_v1' else ToyCombatEnv.observation_size)

    @property
    def reward_names(self):
        return NAVIGATION_REWARD_NAMES if self.config.scenario == 'navigation_v1' else REWARD_NAMES

    def reset(self, seed):
        self.episode_seed = int(seed)
        self.rng = np.random.default_rng(self.episode_seed)
        cells = np.array([(x, y) for x in range(self.config.grid_size)
                          for y in range(self.config.grid_size)], dtype=np.int16)
        if self.config.require_initial_occlusion:
            self._reset_occluded_layout(cells)
        else:
            selected = self.rng.choice(len(cells), size=2, replace=False)
            self.agent = cells[selected[0]].copy()
            self.enemy = cells[selected[1]].copy()
            delta = self.enemy - self.agent
            steps = int(max(abs(int(delta[0])), abs(int(delta[1]))))
            clear = {tuple(self.agent), tuple(self.enemy)}
            clear.update(tuple(np.rint(self.agent + delta * (i / steps)).astype(int))
                         for i in range(1, steps))
            candidates = np.asarray([cell for cell in cells if tuple(cell) not in clear])
            obstacle_indices = self.rng.choice(len(candidates), size=self.config.obstacle_count,
                                               replace=False)
            self.obstacles = {tuple(v) for v in candidates[obstacle_indices]}
        self.heading = int(self.rng.integers(0, self.heading_count))
        self.cooldown = self.tick = 0
        self.terminated = self.truncated = False
        return self.observation(), self.info(reset_reason='episode_start')

    def _reset_occluded_layout(self, cells):
        for _ in range(1000):
            selected = self.rng.choice(len(cells), size=2, replace=False)
            agent, enemy = cells[selected[0]].copy(), cells[selected[1]].copy()
            delta = enemy - agent
            steps = int(max(abs(int(delta[0])), abs(int(delta[1]))))
            if steps < 3:
                continue
            interior = sorted({tuple(np.rint(agent + delta * (i / steps)).astype(int))
                               for i in range(1, steps)} - {tuple(agent), tuple(enemy)})
            if not interior:
                continue
            blocker = interior[int(self.rng.integers(0, len(interior)))]
            excluded = {tuple(agent), tuple(enemy), blocker}
            candidates = np.asarray([cell for cell in cells if tuple(cell) not in excluded])
            extra_count = self.config.obstacle_count - 1
            if extra_count > len(candidates):
                raise ValueError('Too many obstacles for the grid')
            extras = (self.rng.choice(len(candidates), size=extra_count, replace=False)
                      if extra_count else [])
            self.agent, self.enemy = agent, enemy
            self.obstacles = {blocker, *(tuple(candidates[index]) for index in extras)}
            if not self._line_of_sight() and self._navigation_path_to_line_of_sight() is not None:
                return
        raise RuntimeError('Could not generate a solvable occluded layout')

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

    def _line_of_sight_from(self, position):
        # Sample cell centers along the ray; endpoints are never obstacles.
        position = np.asarray(position, dtype=np.int16)
        delta = self.enemy - position
        steps = int(max(abs(int(delta[0])), abs(int(delta[1]))))
        if steps <= 1:
            return True
        for i in range(1, steps):
            point = np.rint(position + delta * (i / steps)).astype(int)
            if tuple(point) in self.obstacles:
                return False
        return True

    def _line_of_sight(self):
        return self._line_of_sight_from(self.agent)

    def _navigation_path_to_line_of_sight(self):
        """Shortest privileged path to any free cell that can see the target."""
        start = tuple(int(value) for value in self.agent)
        queue, parents = deque([start]), {start: None}
        neighbors = ((1, 0), (0, 1), (-1, 0), (0, -1),
                     (1, 1), (-1, 1), (-1, -1), (1, -1))
        goal = None
        while queue:
            current = queue.popleft()
            if self._line_of_sight_from(current):
                goal = current
                break
            for dx, dy in neighbors:
                candidate = (current[0] + dx, current[1] + dy)
                if (candidate in parents or candidate in self.obstacles or
                        candidate == tuple(self.enemy) or min(candidate) < 0 or
                        max(candidate) >= self.config.grid_size):
                    continue
                parents[candidate] = current
                queue.append(candidate)
        if goal is None:
            return None
        path = []
        while goal is not None:
            path.append(goal)
            goal = parents[goal]
        return list(reversed(path))

    def aim_alignment(self):
        delta, forward, _, distance = self._relative()
        return 1.0 if distance == 0 else float(np.dot(delta, forward) / distance)

    def observation(self):
        delta, forward, right, distance = self._relative()
        scale = max(1, self.config.grid_size - 1)
        local_forward = float(np.dot(delta, forward) / scale)
        local_right = float(np.dot(delta, right) / scale)
        values = [self.agent[0] / scale, self.agent[1] / scale,
                  forward[0], forward[1], local_forward, local_right,
                  distance / (math.sqrt(2) * scale), self.aim_alignment(),
                  float(self._line_of_sight()),
                  self.cooldown / max(1, self.config.fire_cooldown_ticks)]
        if self.config.scenario == 'navigation_v1':
            directions = self._movement_directions()
            values.extend(self._clearance(directions[action]) / scale for action in (1, 2, 3, 4))
        return np.asarray(values, dtype=np.float32)

    def action_mask(self):
        mask = np.ones(self.action_size, dtype=bool)
        if not self.config.movement_enabled:
            mask[1:5] = False
        if self.config.navigation_phase_masking:
            if self._line_of_sight():
                mask[1:5] = False
            else:
                mask[7] = False
        mask[7] &= self.cooldown == 0
        return mask

    def privileged_state(self):
        return np.asarray([self.agent[0], self.agent[1], self.heading,
                           self.enemy[0], self.enemy[1], self.cooldown, self.tick], dtype=np.int32)

    def info(self, **extra):
        return {'tick': self.tick, 'episode_seed': self.episode_seed,
                'privileged_state': self.privileged_state(),
                'action_mask': self.action_mask(), 'line_of_sight': self._line_of_sight(),
                'aim_alignment': self.aim_alignment(), 'distance': self._relative()[3], **extra}

    def _movement_directions(self):
        forward = np.asarray([round(math.cos(self.angle)), round(math.sin(self.angle))], dtype=int)
        right = np.asarray([-forward[1], forward[0]], dtype=int)
        return {1: forward, 2: -forward, 3: -right, 4: right}

    def _movement_delta(self, action):
        return self._movement_directions()[action]

    def _clearance(self, direction):
        position, steps = self.agent.copy(), 0
        for _ in range(self.config.grid_size - 1):
            candidate = position + direction
            if (np.any(candidate < 0) or np.any(candidate >= self.config.grid_size) or
                    tuple(candidate) in self.obstacles or np.array_equal(candidate, self.enemy)):
                break
            position, steps = candidate, steps + 1
        return steps

    def _single_tick(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError('Reset is required after episode completion')
        before_aim = self.aim_alignment()
        before_distance = self._relative()[3]
        before_line_of_sight = self._line_of_sight()
        components = dict.fromkeys(self.reward_names, 0.0)
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
            if self.config.scenario == 'navigation_v1':
                components['distance_progress'] = self.config.distance_progress_scale * (
                    before_distance - self._relative()[3])
                components['line_of_sight_progress'] = self.config.line_of_sight_progress_scale * (
                    float(self._line_of_sight()) - float(before_line_of_sight))
        self.terminated = hit
        self.truncated = self.tick >= self.config.max_ticks and not hit
        return components, rejected, hit, miss, collision

    def step(self, action, repeat=1):
        if repeat < 1:
            raise ValueError('Action repeat must be positive')
        totals = dict.fromkeys(self.reward_names, 0.0)
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


def scripted_navigation_action(env):
    """Privileged shortest-path oracle for solvability checks, never for learning."""
    if env._line_of_sight():
        return scripted_action(env.observation(), env.action_mask())
    path = env._navigation_path_to_line_of_sight()
    if path is None or len(path) < 2:
        raise RuntimeError('Navigation layout is not solvable')
    desired = np.asarray(path[1], dtype=int) - env.agent
    for action, movement in env._movement_directions().items():
        if np.array_equal(movement, desired):
            return action
    target_angle = math.atan2(float(desired[1]), float(desired[0]))
    difference = (target_angle - env.angle + math.pi) % (2 * math.pi) - math.pi
    return 5 if difference > 0 else 6
