from terrain import FIELDS
from typing import List


class Map:
    """Represents a hexagonal map with terrain for the wargame."""

    def __init__(self, width: int, height: int, default_terrain=FIELDS):
        self.width = width
        self.height = height
        self.grid: List[List] = [[default_terrain for _ in range(width)] for _ in range(height)]

    def set_terrain(self, x: int, y: int, terrain):
        """Set terrain at a specific coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = terrain

    def get_terrain(self, x: int, y: int):
        """Return the terrain at a specific coordinate."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def terrain_cost(self, x: int, y: int) -> float:
        """Return movement cost for terrain at a coordinate."""
        t = self.get_terrain(x, y)
        return t.move_cost if t else 999  # impassable if out of bounds
