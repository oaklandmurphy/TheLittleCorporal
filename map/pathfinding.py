"""Pathfinding utilities for hexagonal grid navigation."""

import heapq
from typing import Tuple, Dict, List, Generator
from unit import Unit


def get_neighbors(x: int, y: int) -> Generator[Tuple[int, int], None, None]:
    """Return hex neighbors using offset coordinates.
    
    Args:
        x: Column coordinate
        y: Row coordinate
        
    Yields:
        Tuples of (x, y) coordinates for each neighbor
    """

    offsets = direction_offsets(y)
    for dx, dy in offsets:
        yield x + dx, y + dy


def hex_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate hex distance using cube coordinates for even-q (column offset) layout.
    
    Args:
        x1, y1: First hex coordinates
        x2, y2: Second hex coordinates
        
    Returns:
        Distance between the two hexes
    """
    def offset_to_cube(x: int, y: int) -> Tuple[int, int, int]:
        q = x
        r = y - (x - (x & 1)) // 2
        return q, r, -q - r

    q1, r1, s1 = offset_to_cube(x1, y1)
    q2, r2, s2 = offset_to_cube(x2, y2)
    return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2


def direction_offsets(y: int) -> List[Tuple[int, int]]:
    """Return the 6 neighbor direction offsets in order.
    
    Args:
        y: Row coordinate (used to determine even/odd row)
        
    Returns:
        List of (dx, dy) offset tuples
    """
    offsets_even = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, -1)]
    offsets_odd = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (-1, +1)]
    return offsets_odd if y % 2 else offsets_even


def find_reachable_hexes(unit: Unit, grid, width: int, height: int) -> Dict[Tuple[int, int], int]:
    """Find all reachable hexes within unit.mobility using terrain move_cost and enemy zone of control (ZOC).
    
    Entering a hex adjacent to an enemy unit sets remaining movement to 0 for the turn.
    
    Args:
        unit: The unit that is moving
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        
    Returns:
        Dictionary mapping (x, y) coordinates to movement cost
    """
    start = (unit.x, unit.y)
    max_cost = unit.mobility

    cost_so_far = {start: 0}
    pq = [(0, start)]
    reachable = {}

    def is_adjacent_to_enemy(x: int, y: int) -> bool:
        """Check if a position is adjacent to an enemy unit."""
        for nx, ny in get_neighbors(x, y):
            if 0 <= nx < width and 0 <= ny < height:
                neighbor_unit = grid[ny][nx].unit
                if neighbor_unit and neighbor_unit.faction != unit.faction:
                    return True
        return False

    while pq:
        current_cost, (x, y) = heapq.heappop(pq)
        if current_cost > max_cost:
            continue

        # Mark current hex as reachable only if it's the start position or unoccupied
        current_tile = grid[y][x]
        if (x, y) == start or current_tile.unit is None:
            reachable[(x, y)] = current_cost

        # If this hex is adjacent to an enemy (ZOC), do not allow further movement from here
        if (x, y) != start and is_adjacent_to_enemy(x, y):
            continue

        for nx, ny in get_neighbors(x, y):
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            neighbor_tile = grid[ny][nx]
            # Enemy blocks path; friendly units also block destination but allow path-through
            if neighbor_tile.unit and neighbor_tile.unit.faction != unit.faction:
                continue  # enemy blocks path
            new_cost = current_cost + neighbor_tile.terrain.move_cost
            if new_cost <= max_cost and ((nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]):
                cost_so_far[(nx, ny)] = new_cost
                heapq.heappush(pq, (new_cost, (nx, ny)))

    return reachable


def best_reachable_toward(unit: Unit, target: Tuple[int, int], grid, width: int, height: int, 
                          max_cost: int = None) -> Tuple[int, int] | None:
    """Find the best reachable hex closest to the target.
    
    Args:
        unit: The unit that is moving
        target: Target (x, y) coordinates
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        max_cost: Optional maximum movement cost to consider
        
    Returns:
        (x, y) coordinates of the best reachable hex, or None if no valid hex found
    """
    reachable = find_reachable_hexes(unit, grid, width, height)
    if max_cost is not None:
        reachable = {pos: cost for pos, cost in reachable.items() if cost <= max_cost}
    if not reachable:
        return None
    best_hex = None
    best_cost = 0
    best_dist = float("inf")
    for (rx, ry), cost in reachable.items():
        if (rx, ry) == (unit.x, unit.y):
            continue
        dist = hex_distance(rx, ry, target[0], target[1])
        if dist < best_dist or (dist == best_dist and cost < best_cost):
            best_cost = cost
            best_dist = dist
            best_hex = (rx, ry)
    return best_hex
