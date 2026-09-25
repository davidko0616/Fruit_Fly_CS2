"""Dust II NAV waypoints for remembered targets in bridge coordinates."""
from dataclasses import dataclass
import heapq
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from .clearance import ClearanceCalibration, Dust2ClearanceEstimator


@dataclass(frozen=True)
class WaypointPlan:
    world: tuple[float, float]
    grid: tuple[int, int]
    target_grid: tuple[int, int]
    path_remaining_cells: float
    player_snap_world: float
    target_snap_world: float


class Dust2WaypointPlanner:
    """A* planner on the same downsampled NAV mask used during training."""

    def __init__(self, calibration: ClearanceCalibration, mask,
                 grid_step=8, lookahead_cells=6,
                 target_max_snap_world=90.0):
        if grid_step < 2 or lookahead_cells < 1 or target_max_snap_world <= 0:
            raise ValueError('Grid step and waypoint lookahead must be positive')
        self.calibration = calibration
        self.grid_step = int(grid_step)
        self.lookahead_cells = int(lookahead_cells)
        self.target_max_snap_world = float(target_max_snap_world)
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask)
        self.full_mask = np.asarray(
            mask.convert('L') if isinstance(mask, Image.Image) else mask) > 0
        self.estimator = Dust2ClearanceEstimator(calibration, self.full_mask)
        self.offset = self.grid_step // 2
        sampled = self.full_mask[
            self.offset::self.grid_step, self.offset::self.grid_step]
        labels, count = ndimage.label(sampled)
        if count < 1:
            raise ValueError('Waypoint mask has no connected walkable cells')
        sizes = np.bincount(labels.ravel())
        self.grid = labels == int(np.argmax(sizes[1:]) + 1)
        _, self.nearest_grid = ndimage.distance_transform_edt(
            ~self.grid, return_indices=True)
        self.path = None
        self.target_grid = None

    @classmethod
    def from_json(cls, path, grid_step=8, lookahead_cells=6,
                  target_max_snap_world=90.0):
        calibration, mask_path = ClearanceCalibration.from_json(path)
        return cls(calibration, mask_path, grid_step, lookahead_cells,
                   target_max_snap_world)

    def reset(self):
        self.path = None
        self.target_grid = None

    def _world_to_grid(self, world, max_snap_world):
        mask_x, mask_y = self.estimator.screen_to_mask(*world)
        ix, iy = int(round(mask_x)), int(round(mask_y))
        height, width = self.full_mask.shape
        if not (0 <= ix < width and 0 <= iy < height):
            raise ValueError('Position falls outside the walkability mask')
        snap_pixels = 0.0
        if not self.full_mask[iy, ix]:
            nearest_y = int(self.estimator.nearest_inside[0, iy, ix])
            nearest_x = int(self.estimator.nearest_inside[1, iy, ix])
            snap_pixels = math.hypot(nearest_x - mask_x, nearest_y - mask_y)
            if snap_pixels * self.calibration.overview_units_per_pixel > max_snap_world:
                raise ValueError('Position is too far from walkable navigation space')
            mask_x, mask_y = float(nearest_x), float(nearest_y)
        grid_x = int(round((mask_x - self.offset) / self.grid_step))
        grid_y = int(round((mask_y - self.offset) / self.grid_step))
        height, width = self.grid.shape
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            raise ValueError('Position falls outside the waypoint grid')
        if not self.grid[grid_y, grid_x]:
            nearest_y = int(self.nearest_grid[0, grid_y, grid_x])
            nearest_x = int(self.nearest_grid[1, grid_y, grid_x])
            grid_snap_pixels = self.grid_step * math.hypot(
                nearest_x - grid_x, nearest_y - grid_y)
            if grid_snap_pixels * self.calibration.overview_units_per_pixel > \
                    max_snap_world + \
                    self.grid_step * self.calibration.overview_units_per_pixel:
                raise ValueError('Position is too far from the waypoint grid')
            grid_y, grid_x = nearest_y, nearest_x
        return ((grid_y, grid_x),
                float(snap_pixels * self.calibration.overview_units_per_pixel))

    def _grid_to_world(self, point):
        grid_y, grid_x = point
        mask_x = self.offset + grid_x * self.grid_step
        mask_y = self.offset + grid_y * self.grid_step
        calibration = self.calibration
        return (
            float(mask_x * calibration.screen_scale_x +
                  calibration.screen_offset_x),
            float(mask_y * calibration.screen_scale_y +
                  calibration.screen_offset_y),
        )

    def _walkable(self, point):
        y, x = point
        return (0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]
                and self.grid[y, x])

    @staticmethod
    def _neighbors(point):
        y, x = point
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            yield (y + dy, x + dx), dy, dx

    def _shortest_path(self, start, target):
        frontier = [(0.0, start)]
        costs, parents = {start: 0.0}, {start: None}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == target:
                break
            for candidate, dy, dx in self._neighbors(current):
                if not self._walkable(candidate):
                    continue
                diagonal = abs(dy) + abs(dx) == 2
                if diagonal and (not self._walkable((current[0] + dy, current[1])) or
                                 not self._walkable((current[0], current[1] + dx))):
                    continue
                cost = costs[current] + (math.sqrt(2) if diagonal else 1.0)
                if candidate in costs and cost >= costs[candidate]:
                    continue
                costs[candidate], parents[candidate] = cost, current
                heuristic = math.hypot(
                    target[0] - candidate[0], target[1] - candidate[1])
                heapq.heappush(frontier, (cost + heuristic, candidate))
        if target not in parents:
            raise ValueError('No NAV path connects player and remembered target')
        path, current = [], target
        while current is not None:
            path.append(current)
            current = parents[current]
        return list(reversed(path))

    def _line_is_walkable(self, start, target):
        start = np.asarray(start, dtype=float)
        delta = np.asarray(target, dtype=float) - start
        steps = max(1, int(math.ceil(np.max(np.abs(delta)) * 2)))
        samples = np.rint(
            start[None] + np.linspace(0, 1, steps + 1)[:, None] * delta
        ).astype(int)
        return all(self._walkable(tuple(point)) for point in samples)

    def plan(self, player_world, target_world):
        player, player_snap = self._world_to_grid(
            player_world, self.calibration.max_snap_world)
        target, target_snap = self._world_to_grid(
            target_world, self.target_max_snap_world)
        if self.target_grid != target or self.path is None:
            self.path = self._shortest_path(player, target)
            self.target_grid = target
        elif player in self.path:
            self.path = self.path[self.path.index(player):]
        else:
            self.path = self._shortest_path(player, target)
        maximum = min(len(self.path) - 1, self.lookahead_cells)
        waypoint_index = 1 if len(self.path) > 1 else 0
        for index in range(maximum, 0, -1):
            if self._line_is_walkable(player, self.path[index]):
                waypoint_index = index
                break
        waypoint = self.path[waypoint_index]
        return WaypointPlan(
            world=self._grid_to_world(waypoint), grid=waypoint,
            target_grid=target, path_remaining_cells=float(len(self.path) - 1),
            player_snap_world=player_snap, target_snap_world=target_snap)
