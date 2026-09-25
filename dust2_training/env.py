"""Deterministic Dust II navigation training on a registered NAV mask."""
from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from toy_combat.env import ACTION_NAMES, NAVIGATION_REWARD_NAMES


ROUTE_SPLITS = ('train', 'validation', 'heldout')


@dataclass(frozen=True)
class Dust2CombatConfig:
    mask_path: str
    route_split: str = 'train'
    grid_step: int = 8
    max_ticks: int = 512
    turn_degrees: float = 11.25
    hit_tolerance_degrees: float = 11.25
    fire_cooldown_ticks: int = 2
    memory_ticks: int = 512
    initialize_target_memory: bool = True
    minimum_route_cells: int = 24
    maximum_route_cells: int | None = None
    time_penalty: float = -0.01
    aim_progress_scale: float = 0.2
    distance_progress_scale: float = 0.1
    line_of_sight_progress_scale: float = 0.2
    collision_penalty: float = -0.05
    miss_penalty: float = -0.1
    hit_reward: float = 3.0
    scenario: str = 'dust2_navigation_v1'
    movement_enabled: bool = True
    require_initial_occlusion: bool = True
    navigation_phase_masking: bool = False
    target_movement_enabled: bool = False
    hide_target_when_occluded: bool = True
    provide_last_seen_target: bool = True
    waypoint_planner_enabled: bool = False
    waypoint_lookahead_cells: int = 6

    def __post_init__(self):
        if self.scenario != 'dust2_navigation_v1':
            raise ValueError('Dust II environment requires dust2_navigation_v1')
        if self.route_split not in ROUTE_SPLITS:
            raise ValueError(f'Route split must be one of {ROUTE_SPLITS}')
        if self.grid_step < 2 or self.max_ticks < 1:
            raise ValueError('Grid step and maximum ticks must be positive')
        if self.turn_degrees <= 0 or 360 % self.turn_degrees != 0:
            raise ValueError('Turn degrees must divide 360')
        if self.fire_cooldown_ticks < 0 or self.memory_ticks < 1:
            raise ValueError('Cooldown must be nonnegative and memory positive')
        if self.minimum_route_cells < 1:
            raise ValueError('Minimum route length must be positive')
        if self.waypoint_lookahead_cells < 1:
            raise ValueError('Waypoint lookahead must be positive')
        if (self.maximum_route_cells is not None and
                self.maximum_route_cells < self.minimum_route_cells):
            raise ValueError('Maximum route length must not be below minimum')
        if not Path(self.mask_path).is_file():
            raise FileNotFoundError(f'Dust II mask does not exist: {self.mask_path}')


class Dust2CombatEnv:
    """One player and one stationary target on the Dust II NAV mask."""

    action_size = len(ACTION_NAMES)
    observation_size = 14
    reward_names = NAVIGATION_REWARD_NAMES

    def __init__(self, config: Dust2CombatConfig):
        self.config = config
        mask = np.asarray(Image.open(config.mask_path).convert('L')) > 0
        offset = config.grid_step // 2
        self.grid = mask[offset::config.grid_step, offset::config.grid_step]
        labels, count = ndimage.label(self.grid)
        if count < 1:
            raise ValueError('Dust II mask has no connected walkable cells')
        sizes = np.bincount(labels.ravel())
        largest = int(np.argmax(sizes[1:]) + 1)
        self.grid = labels == largest
        self.cells = np.argwhere(self.grid)
        if len(self.cells) < 100:
            raise ValueError('Dust II walkability grid is too small')
        self.height, self.width = self.grid.shape
        self.rng = None
        self.agent = self.enemy = None
        self.yaw_degrees = self.cooldown = self.tick = 0
        self.last_seen_enemy = None
        self.last_seen_tick = None
        self.route_id = None
        self._scripted_path = None
        self._planner_path = None
        self._planner_target = None

    @staticmethod
    def observation_size_for(config):
        return Dust2CombatEnv.observation_size

    @staticmethod
    def _route_bucket(start, target):
        payload = np.asarray([*start, *target], dtype='<i2').tobytes()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=2).digest(),
                              'little') % 10

    def _route_matches_split(self, start, target):
        bucket = self._route_bucket(start, target)
        return ((self.config.route_split == 'train' and bucket < 8) or
                (self.config.route_split == 'validation' and bucket == 8) or
                (self.config.route_split == 'heldout' and bucket == 9))

    def _route_hash(self, start, target):
        payload = np.asarray([*start, *target], dtype='<i2').tobytes()
        return hashlib.sha256(payload).hexdigest()[:16]

    def reset(self, seed):
        self.rng = np.random.default_rng(int(seed))
        for _ in range(50_000):
            indices = self.rng.choice(len(self.cells), size=2, replace=False)
            start = self.cells[indices[0]].astype(np.int16)
            target = self.cells[indices[1]].astype(np.int16)
            direct_distance = np.linalg.norm(target.astype(float) - start)
            if direct_distance < self.config.minimum_route_cells:
                continue
            if (self.config.maximum_route_cells is not None and
                    direct_distance > self.config.maximum_route_cells):
                continue
            if not self._route_matches_split(start, target):
                continue
            self.agent, self.enemy = start, target
            if self.config.require_initial_occlusion and self._line_of_sight():
                continue
            break
        else:
            raise RuntimeError('Could not sample a split-matched occluded route')
        turns = round(360 / self.config.turn_degrees)
        self.yaw_degrees = float(self.rng.integers(turns) *
                                 self.config.turn_degrees)
        self.cooldown = self.tick = 0
        self.last_seen_enemy = (self.enemy.copy()
                                if self.config.initialize_target_memory else None)
        self.last_seen_tick = (0 if self.config.initialize_target_memory else None)
        self.route_id = self._route_hash(self.agent, self.enemy)
        self._scripted_path = None
        self._planner_path = None
        self._planner_target = None
        return self.observation(), self.info()

    def _cell_is_walkable(self, point):
        y, x = (int(point[0]), int(point[1]))
        return 0 <= y < self.height and 0 <= x < self.width and self.grid[y, x]

    def _line_is_walkable(self, start, target):
        start = np.asarray(start, dtype=float)
        delta = np.asarray(target, dtype=float) - start
        steps = max(1, int(math.ceil(np.max(np.abs(delta)) * 2)))
        samples = np.rint(start[None] +
                          np.linspace(0, 1, steps + 1)[:, None] * delta).astype(int)
        return all(self._cell_is_walkable(point) for point in samples)

    def _line_of_sight_from(self, position):
        return self._line_is_walkable(position, self.enemy)

    def _line_of_sight(self):
        return self._line_of_sight_from(self.agent)

    def _axes(self):
        angle = math.radians(self.yaw_degrees)
        forward = np.asarray((math.sin(angle), math.cos(angle)), dtype=float)
        right = np.asarray((math.cos(angle), -math.sin(angle)), dtype=float)
        return forward, right

    def _relative_to(self, target):
        delta = np.asarray(target, dtype=float) - self.agent.astype(float)
        forward, right = self._axes()
        distance = float(np.linalg.norm(delta))
        return delta, forward, right, distance

    def aim_alignment(self):
        delta, forward, _, distance = self._relative_to(self.enemy)
        return 1.0 if distance == 0 else float(np.dot(delta, forward) / distance)

    @staticmethod
    def _neighbors(point):
        y, x = point
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            yield (y + dy, x + dx), dy, dx

    def _shortest_path(self, target):
        start = tuple(int(value) for value in self.agent)
        goal = tuple(int(value) for value in target)
        frontier = [(0.0, start)]
        parents = {start: None}
        costs = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            current_cost = costs[current]
            for candidate, dy, dx in self._neighbors(current):
                if not self._cell_is_walkable(candidate):
                    continue
                diagonal = abs(dy) + abs(dx) == 2
                if diagonal and (not self._cell_is_walkable((current[0] + dy, current[1])) or
                                 not self._cell_is_walkable((current[0], current[1] + dx))):
                    continue
                new_cost = current_cost + (math.sqrt(2) if diagonal else 1.0)
                if candidate in costs and new_cost >= costs[candidate]:
                    continue
                costs[candidate] = new_cost
                parents[candidate] = current
                delta_y, delta_x = goal[0] - candidate[0], goal[1] - candidate[1]
                heuristic = math.hypot(delta_y, delta_x)
                heapq.heappush(frontier, (new_cost + heuristic, candidate))
        if goal not in parents:
            return None
        path, current = [], goal
        while current is not None:
            path.append(current)
            current = parents[current]
        return list(reversed(path))

    def _path_to_remembered_target(self, target):
        target_tuple = tuple(int(value) for value in target)
        agent_tuple = tuple(int(value) for value in self.agent)
        if self._planner_target != target_tuple or self._planner_path is None:
            self._planner_path = self._shortest_path(target_tuple)
            self._planner_target = target_tuple
        elif agent_tuple in self._planner_path:
            self._planner_path = self._planner_path[
                self._planner_path.index(agent_tuple):]
        else:
            self._planner_path = self._shortest_path(target_tuple)
        return self._planner_path

    def _planner_waypoint(self, target):
        path = self._path_to_remembered_target(target)
        if not path:
            return np.asarray(target, dtype=np.int16), 0.0
        maximum_index = min(len(path) - 1, self.config.waypoint_lookahead_cells)
        waypoint_index = 1 if len(path) > 1 else 0
        for index in range(maximum_index, 0, -1):
            if self._line_is_walkable(self.agent, path[index]):
                waypoint_index = index
                break
        return (np.asarray(path[waypoint_index], dtype=np.int16),
                float(len(path) - 1))

    def _visible_or_remembered_target(self):
        if self._line_of_sight():
            self.last_seen_enemy = self.enemy.copy()
            self.last_seen_tick = self.tick
            return self.enemy, True, False
        if (self.config.provide_last_seen_target and
                self.last_seen_enemy is not None and
                self.tick - self.last_seen_tick <= self.config.memory_ticks):
            return self.last_seen_enemy, False, True
        return None, False, False

    def _control_target(self):
        target, live, memory = self._visible_or_remembered_target()
        planner_active = bool(self.config.waypoint_planner_enabled and memory)
        remaining = None
        if planner_active:
            target, remaining = self._planner_waypoint(target)
        return target, live, memory, planner_active, remaining

    def _control_geometry(self):
        target, live, memory, planner_active, remaining = self._control_target()
        if target is None:
            return target, live, memory, planner_active, remaining, 0.0, 0.0
        delta, forward, _, distance = self._relative_to(target)
        alignment = 1.0 if distance == 0 else float(np.dot(delta, forward) / distance)
        progress_distance = remaining if planner_active else distance
        return (target, live, memory, planner_active, remaining,
                alignment, float(progress_distance))

    def _clearance(self, direction, maximum=24):
        direction = np.asarray(direction, dtype=float)
        length = np.linalg.norm(direction)
        direction = direction / length
        distance = 0.0
        while distance < maximum:
            candidate_distance = min(maximum, distance + .25)
            point = np.rint(self.agent + direction * candidate_distance).astype(int)
            if not self._cell_is_walkable(point):
                break
            distance = candidate_distance
        return float(distance / maximum)

    def clearances(self):
        forward, right = self._axes()
        return (self._clearance(forward), self._clearance(-forward),
                self._clearance(-right), self._clearance(right))

    def observation(self):
        target, live, memory, _, _ = self._control_target()
        forward_axis, _ = self._axes()
        if target is None:
            local_forward = local_right = distance = alignment = 0.0
        else:
            delta, forward, right, distance = self._relative_to(target)
            local_forward = float(np.dot(delta, forward))
            local_right = float(np.dot(delta, right))
            alignment = 1.0 if distance == 0 else local_forward / distance
        scale = max(self.height - 1, self.width - 1)
        return np.asarray([
            self.agent[1] / (self.width - 1),
            self.agent[0] / (self.height - 1),
            forward_axis[1], forward_axis[0],
            local_forward / scale, local_right / scale,
            distance / (math.sqrt(2) * scale), alignment,
            float(live), self.cooldown / max(1, self.config.fire_cooldown_ticks),
            *self.clearances(),
        ], dtype=np.float32)

    def action_mask(self):
        mask = np.ones(self.action_size, dtype=bool)
        clearances = self.clearances()
        mask[1:5] = np.asarray(clearances) >= (1 / 24)
        mask[7] = (self.cooldown == 0 and self._line_of_sight() and
                   self.aim_alignment() >= math.cos(math.radians(
                       self.config.hit_tolerance_degrees)))
        return mask

    def privileged_state(self):
        last = (-1, -1) if self.last_seen_enemy is None else self.last_seen_enemy
        last_tick = -1 if self.last_seen_tick is None else self.last_seen_tick
        return np.asarray([
            *self.agent, *self.enemy, self.yaw_degrees, self.cooldown, self.tick,
            *last, last_tick,
        ], dtype=np.float32)

    def info(self, **updates):
        line_of_sight = self._line_of_sight()
        age = (None if self.last_seen_tick is None else
               self.tick - self.last_seen_tick)
        memory = (not line_of_sight and self.last_seen_enemy is not None and
                  age <= self.config.memory_ticks)
        control_target, _, _, planner_active, remaining = self._control_target()
        value = {
            'action_mask': self.action_mask(),
            'privileged_state': self.privileged_state(),
            'line_of_sight': line_of_sight,
            'aim_alignment': self.aim_alignment(),
            'target_observation_is_live': line_of_sight,
            'target_memory_in_observation': memory,
            'waypoint_planner_active': planner_active,
            'waypoint': (None if not planner_active else
                         [int(value) for value in control_target]),
            'waypoint_path_remaining': (-1.0 if remaining is None else remaining),
            'has_last_seen_target': self.last_seen_enemy is not None,
            'last_seen_age': -1 if age is None else age,
            'route_id': self.route_id,
            'route_split': self.config.route_split,
            'tick': self.tick,
            'target_moved': False,
            'hit': False, 'miss': False, 'collision': False,
            'reset_reason': None, 'action_applied_ticks': 0,
            'action_rejected': False,
        }
        value.update(updates)
        return value

    @staticmethod
    def _rounded_direction(vector):
        return np.rint(vector).astype(np.int16)

    def _movement_target(self, action):
        forward, right = self._axes()
        direction = {1: forward, 2: -forward, 3: -right, 4: right}[action]
        step = self._rounded_direction(direction)
        candidate = self.agent + step
        if not self._cell_is_walkable(candidate):
            return None
        if abs(int(step[0])) + abs(int(step[1])) == 2:
            if (not self._cell_is_walkable(self.agent + (step[0], 0)) or
                    not self._cell_is_walkable(self.agent + (0, step[1]))):
                return None
        return candidate

    def step(self, action, repeat=1):
        action = int(action)
        if not 0 <= action < self.action_size or repeat < 1:
            raise ValueError('Invalid action or repeat')
        total_reward = 0.0
        aggregate = {name: 0.0 for name in self.reward_names}
        terminated = truncated = False
        hit = miss = collision = False
        applied = 0
        for _ in range(repeat):
            (_, _, _, _, _, before_alignment,
             before_distance) = self._control_geometry()
            before_los = self._line_of_sight()
            before_memory = (self.last_seen_tick is not None and
                             self.tick - self.last_seen_tick <=
                             self.config.memory_ticks)
            if self.cooldown > 0:
                self.cooldown -= 1
            if action in (1, 2, 3, 4):
                candidate = self._movement_target(action)
                if candidate is None:
                    collision = True
                else:
                    self.agent = candidate
            elif action == 5:
                self.yaw_degrees = (self.yaw_degrees -
                                    self.config.turn_degrees) % 360
            elif action == 6:
                self.yaw_degrees = (self.yaw_degrees +
                                    self.config.turn_degrees) % 360
            elif action == 7 and self.cooldown == 0:
                tolerance = math.cos(math.radians(
                    self.config.hit_tolerance_degrees))
                hit = self._line_of_sight() and self.aim_alignment() >= tolerance
                miss = not hit
                self.cooldown = self.config.fire_cooldown_ticks + 1
            self.tick += 1
            applied += 1
            (_, _, _, _, _, after_alignment,
             after_distance) = self._control_geometry()
            after_los = self._line_of_sight()
            after_memory = (self.last_seen_tick is not None and
                            self.tick - self.last_seen_tick <=
                            self.config.memory_ticks)
            target_shaping_available = ((before_los or before_memory) and
                                        (after_los or after_memory))
            components = {
                'time': self.config.time_penalty,
                'aim_progress': self.config.aim_progress_scale *
                    (after_alignment - before_alignment)
                    if target_shaping_available else 0.0,
                'distance_progress': self.config.distance_progress_scale *
                    (before_distance - after_distance)
                    if target_shaping_available else 0.0,
                'line_of_sight_progress': self.config.line_of_sight_progress_scale *
                    (float(after_los) - float(before_los)),
                'collision': self.config.collision_penalty if collision else 0.0,
                'shot': self.config.hit_reward if hit else
                        self.config.miss_penalty if miss else 0.0,
            }
            for name, value in components.items():
                aggregate[name] += value
            total_reward += sum(components.values())
            terminated = hit
            truncated = not terminated and self.tick >= self.config.max_ticks
            if terminated or truncated:
                break
        observation = self.observation()
        return observation, total_reward, terminated, truncated, self.info(
            reward_components=aggregate, hit=hit, miss=miss,
            collision=collision,
            reset_reason='hit' if hit else 'time_limit' if truncated else None,
            action_applied_ticks=applied)

    def _path_to_line_of_sight(self):
        start = tuple(int(value) for value in self.agent)
        target = tuple(int(value) for value in self.enemy)
        parents = {start: None}
        queue = deque([start])
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1))
        found = None
        while queue:
            current = queue.popleft()
            if current == target:
                found = current
                break
            for dy, dx in directions:
                candidate = (current[0] + dy, current[1] + dx)
                if candidate in parents or not self._cell_is_walkable(candidate):
                    continue
                if abs(dy) + abs(dx) == 2:
                    if (not self._cell_is_walkable((current[0] + dy, current[1])) or
                            not self._cell_is_walkable((current[0], current[1] + dx))):
                        continue
                parents[candidate] = current
                queue.append(candidate)
        if found is None:
            return None
        path = []
        while found is not None:
            path.append(found)
            found = parents[found]
        return list(reversed(path))


def _turn_toward(env, delta):
    desired = math.degrees(math.atan2(delta[0], delta[1])) % 360
    difference = (desired - env.yaw_degrees + 180) % 360 - 180
    if abs(difference) <= env.config.turn_degrees / 2 + 1e-9:
        return 1
    return 6 if difference > 0 else 5


def scripted_dust2_action(env: Dust2CombatEnv):
    """Privileged benchmark that navigates to sight, aims, and fires."""
    if env._line_of_sight():
        env._scripted_path = None
        delta = env.enemy.astype(float) - env.agent.astype(float)
        action = _turn_toward(env, delta)
        return 7 if action == 1 else action
    path = env._scripted_path
    if path is not None and len(path) > 1 and tuple(env.agent) == path[1]:
        path.pop(0)
    if path is None or not path or tuple(env.agent) != path[0]:
        path = env._path_to_line_of_sight()
        env._scripted_path = path
    if path is None or len(path) < 2:
        raise RuntimeError('Dust II route has no path to target visibility')
    delta = np.asarray(path[1]) - env.agent
    return _turn_toward(env, delta)


def scripted_waypoint_action(env: Dust2CombatEnv):
    """Reference controller that follows only the exposed waypoint interface."""
    target, live, _, planner_active, _ = env._control_target()
    if target is None:
        return 0
    delta = np.asarray(target, dtype=float) - env.agent.astype(float)
    if live and not planner_active:
        action = _turn_toward(env, delta)
        if action != 1:
            return action
        tolerance = math.cos(math.radians(env.config.hit_tolerance_degrees))
        if env.aim_alignment() >= tolerance:
            return 7
    current_distance = float(np.linalg.norm(delta))
    candidates = []
    mask = env.action_mask()
    for action in range(1, 5):
        candidate = env._movement_target(action) if mask[action] else None
        if candidate is not None:
            candidates.append((float(np.linalg.norm(
                np.asarray(target, dtype=float) - candidate)), action))
    if candidates:
        distance, action = min(candidates)
        if distance + 1e-9 < current_distance:
            return action
    return _turn_toward(env, delta)
