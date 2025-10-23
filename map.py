from terrain import Terrain
from terrain import FIELDS, FOREST, RIVER, HILL
from unit import Unit

import heapq
from typing import Optional

class Hex:
    """A single tile on the hex map."""
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.unit: Optional[Unit] = None
        self.victory_points: Optional[int] = 0


class Map:
    """Hexagonal grid storing terrain and units, with pathfinding and combat logic."""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[Hex(FIELDS) for _ in range(width)] for _ in range(height)]

    # --- Terrain setup ---
    def set_terrain(self, x: int, y: int, terrain: Terrain):
        self.grid[y][x].terrain = terrain

    def get_terrain(self, x: int, y: int) -> Terrain:
        return self.grid[y][x].terrain
    
    def get_hex(self, x: int, y: int) -> Optional[Hex]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None
    # --- Unit management ---
    def place_unit(self, unit: Unit, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        tile = self.grid[y][x]
        if tile.unit:
            return False
        tile.unit = unit
        unit.x, unit.y = x, y
        return True

    def move_unit(self, unit: Unit, target_x: int, target_y: int) -> bool:
        """Move a unit if reachable within its mobility using Dijkstra’s algorithm."""
        reachable = self.find_reachable_hexes(unit)
        if (target_x, target_y) not in reachable:
            return False  # too far

        self.grid[unit.y][unit.x].unit = None
        self.grid[target_y][target_x].unit = unit
        unit.move(target_x, target_y, reachable[(target_x, target_y)])
        return True

    # --- Dijkstra Pathfinding ---
    def find_reachable_hexes(self, unit: Unit):
        """Find all reachable hexes within unit.mobility using terrain move_cost."""
        start = (unit.x, unit.y)
        max_cost = unit.mobility

        cost_so_far = {start: 0}
        pq = [(0, start)]
        reachable = {}

        while pq:
            current_cost, (x, y) = heapq.heappop(pq)
            if current_cost > max_cost:
                continue
            reachable[(x, y)] = current_cost

            for nx, ny in self.get_neighbors(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                neighbor_tile = self.grid[ny][nx]
                if neighbor_tile.unit and neighbor_tile.unit.faction != unit.faction:
                    continue  # enemy blocks path
                new_cost = current_cost + neighbor_tile.terrain.move_cost
                if new_cost <= max_cost and ((nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]):
                    cost_so_far[(nx, ny)] = new_cost
                    heapq.heappush(pq, (new_cost, (nx, ny)))

        return reachable

    # --- Adjacency and Combat ---
    def get_neighbors(self, x: int, y: int):
        """Return hex neighbors using offset coordinates."""
        offsets_even = [(+1, 0), (-1, 0), (0, +1), (0, -1), (-1, +1), (-1, -1)]
        offsets_odd = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (+1, -1)]
        offsets = offsets_odd if y % 2 else offsets_even
        for dx, dy in offsets:
            yield x + dx, y + dy

    def resolve_adjacent_combat(self):
        """Make adjacent enemy units engage in combat."""
        processed = set()
        for y in range(self.height):
            for x in range(self.width):
                unit = self.grid[y][x].unit
                if not unit or (x, y) in processed:
                    continue
                for nx, ny in self.get_neighbors(x, y):
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    enemy = self.grid[ny][nx].unit
                    if enemy and enemy.faction != unit.faction:
                        print(f"{unit.name} engages {enemy.name} in combat!")
                        self.combat(unit, enemy)
                        processed.add((x, y))
                        processed.add((nx, ny))

    def combat(self, attacker: Unit, defender: Unit):
        """Simple mutual combat."""
        att_power = attacker.combat_power(self.get_terrain(attacker.x, attacker.y))
        def_power = defender.combat_power(self.get_terrain(defender.x, defender.y))

        damage_to_def = max(1, int(att_power / 10))
        damage_to_att = max(1, int(def_power / 10))

        defender.take_damage(damage_to_def)
        attacker.take_damage(damage_to_att)

        # Remove routed units
        if defender.is_routed():
            self.grid[defender.y][defender.x].unit = None
        if attacker.is_routed():
            self.grid[attacker.y][attacker.x].unit = None