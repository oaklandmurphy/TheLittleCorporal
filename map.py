from terrain import Terrain
from terrain import FIELDS, FOREST, RIVER, HILL
from unit import Unit

import heapq
import random
from typing import Optional, List, Tuple, Set

class Hex:
    """A single tile on the hex map."""
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.unit: Optional[Unit] = None
        self.victory_points: Optional[int] = 0
        self.features: list[str] = [] 


class Map:
    def get_combat_advantage(self, x1, y1, x2, y2) -> float:
        """Calculate combat advantage for unit at (x1,y1) against unit at (x2,y2)."""
        hex1 = self.get_hex(x1, y1)
        hex2 = self.get_hex(x2, y2)
        if not hex1 or not hex2:
            return 0.0  # No advantage if hexes or units are missing

        terrain1 = hex1.terrain
        terrain2 = hex2.terrain

        offense_mod = terrain1.getOffenseModifier(terrain2.elevation)
        defense_mod = terrain2.getDefenseModifier()

        # Simple formula: offense modifier minus defense modifier
        advantage = offense_mod - defense_mod
        return advantage

    def get_frontline(self, feature_name: str, direction: str) -> List[Tuple[int, int]]:
        """Compute the best continuous frontline near a feature facing a direction.

        This version allows irregular (non-straight) but continuous frontlines, and enforces
        that the chosen line spans at least from edge to edge of the feature measured along the
        axis perpendicular to the facing direction.

        Args:
            feature_name: A feature label present in tile.features.
            direction: One of "E", "W", "NE", "NW", "SE", "SW" indicating enemy-facing.

        Returns:
            A list of (x, y) hex coordinates forming a single continuous line within 2 hexes of the
            feature that maximizes the sum of get_combat_advantage(tile, tile+direction) and whose
            span (perpendicular to the facing direction) covers the feature's span. If no such path
            exists, returns an empty list.
        """

        # Only these 6 directions are supported. Note: 0,-1 is North; 0,+1 is South
        valid_dirs = {"N", "S", "NE", "NW", "SE", "SW"}
        if direction not in valid_dirs:
            return []

        feature_tiles = set(self.get_feature_coordinates(feature_name))
        if not feature_tiles:
            return []

        # Neighbor in hex direction using even-q (column offset) layout with N,S present.
        # Enforces N=(0,-1), S=(0,+1) across both parities.
        def hex_neighbors_dir(x: int, y: int, dir_name: str) -> Tuple[int, int]:
            if x % 2 == 0:  # even column
                deltas = {
                    "N":  (0, -1),
                    "S":  (0, +1),
                    "NE": (+1, -1),
                    "NW": (-1, -1),
                    "SE": (+1, 0),
                    "SW": (-1, 0),
                }
            else:  # odd column
                deltas = {
                    "N":  (0, -1),
                    "S":  (0, +1),
                    "NE": (+1, 0),
                    "NW": (-1, 0),
                    "SE": (+1, +1),
                    "SW": (-1, +1),
                }
            dx, dy = deltas[dir_name]
            return x + dx, y + dy

        def to_cube(x: int, y: int) -> Tuple[int, int, int]:
            # Consistent with _hex_distance conversion (even-q, column offset)
            q = x
            r = y - (x - (x & 1)) // 2
            s = -q - r
            return q, r, s

        # Choose span (perpendicular) coordinate and a lateral tiebreaker to build a DAG
        # Mapping based on cube coordinates:
        # - Lines parallel to E/W have constant s -> perpendicular span uses s
        # - Lines parallel to NE/SW have constant r -> perpendicular span uses r
        # - Lines parallel to NW/SE have constant q -> perpendicular span uses q
        def span_and_lateral(x: int, y: int) -> Tuple[int, int]:
            q, r, s = to_cube(x, y)
            # Perpendicular span coordinate to the facing direction:
            # - N/S  : span by q (columns; left-right across the front)
            # - NE/SW: span by s (labels NE-parallel lines)
            # - NW/SE: span by r (labels E-W rows, perpendicular to NW/SE axis)
            if direction in ("N", "S"):
                return q, r  # span by q, tie-break by r
            if direction in ("NE", "SW"):
                return s, r  # span by s, tie-break by r
            # direction in ("NW", "SE")
            return r, q      # span by r, tie-break by q

        def hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return self._hex_distance(a[0], a[1], b[0], b[1])

        # Allowed tiles: within 2 hexes of any feature tile
        allowed: Set[Tuple[int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                here = (x, y)
                for fx, fy in feature_tiles:
                    if hex_distance(here, (fx, fy)) <= 2:
                        allowed.add(here)
                        break

        if not allowed:
            return []

        # Compute tile weights = combat advantage when facing the given direction
        weight: dict[Tuple[int, int], float] = {}
        for (x, y) in allowed:
            nx, ny = hex_neighbors_dir(x, y, direction)
            weight[(x, y)] = float(self.get_combat_advantage(x, y, nx, ny))

        # Compute feature span along the perpendicular coordinate
        feature_span_vals = [span_and_lateral(x, y)[0] for (x, y) in feature_tiles]
        min_span = min(feature_span_vals)
        max_span = max(feature_span_vals)

        # Precompute span/lateral for allowed tiles
        span_coord: dict[Tuple[int, int], int] = {}
        lateral_coord: dict[Tuple[int, int], int] = {}
        for p in allowed:
            sc, lc = span_and_lateral(p[0], p[1])
            span_coord[p] = sc
            lateral_coord[p] = lc

        # Start and end sets must cover the feature span
        start_nodes = [p for p in allowed if span_coord[p] <= min_span]
        end_nodes = set(p for p in allowed if span_coord[p] >= max_span)
        if not start_nodes or not end_nodes:
            return []

        # Build adjacency restricted to allowed tiles
        neighbors: dict[Tuple[int, int], List[Tuple[int, int]]] = {p: [] for p in allowed}
        for (x, y) in allowed:
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) in allowed:
                    neighbors[(x, y)].append((nx, ny))

        # Dynamic programming over a DAG induced by ordering (span, lateral)
        nodes_sorted = sorted(allowed, key=lambda p: (span_coord[p], lateral_coord[p]))
        neg_inf = float("-inf")
        best_score: dict[Tuple[int, int], float] = {p: neg_inf for p in allowed}
        parent: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {p: None for p in allowed}

        for p in nodes_sorted:
            if p in start_nodes:
                # starting at p
                if weight[p] > best_score[p]:
                    best_score[p] = weight[p]
                    parent[p] = None

            # Relax edges forward only if they increase (span, lateral)
            for v in neighbors[p]:
                sp, sl = span_coord[p], lateral_coord[p]
                sv, lv = span_coord[v], lateral_coord[v]
                if (sv > sp) or (sv == sp and lv > sl):
                    if best_score[p] + weight[v] > best_score[v]:
                        best_score[v] = best_score[p] + weight[v]
                        parent[v] = p

        # Pick best end node
        best_end = None
        best_end_score = neg_inf
        for e in end_nodes:
            if best_score[e] > best_end_score:
                best_end_score = best_score[e]
                best_end = e

        if best_end is None or best_end_score == neg_inf:
            return []

        # Reconstruct path from best_end
        path: List[Tuple[int, int]] = []
        cur = best_end
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        return path

    def list_feature_names(self) -> list[str]:
        """Return a sorted list of all unique feature names present on the map."""
        features = set()
        for y in range(self.height):
            for x in range(self.width):
                for fname in self.grid[y][x].features:
                    features.add(fname)
        return sorted(features)
    def get_feature_coordinates(self, feature_name: str) -> list[tuple[int, int]]:
        """Return a list of (x, y) coordinates that make up the given feature name."""
        coords = []
        for y in range(self.height):
            for x in range(self.width):
                if feature_name in self.grid[y][x].features:
                    coords.append((x, y))
        return coords

    def describe_feature(self, feature_name: str) -> str:
        """Return an English description of the feature: terrain type, units present, nearby units, and coordinates."""
        coords = self.get_feature_coordinates(feature_name)
        if not coords:
            return f"No feature named '{feature_name}' found."

        # Determine terrain type (majority terrain among feature hexes)
        terrain_count = {}
        for x, y in coords:
            tname = self.grid[y][x].terrain.name
            terrain_count[tname] = terrain_count.get(tname, 0) + 1
        terrain_type = max(terrain_count, key=terrain_count.get)

        # Units present on the feature
        units_on = []
        for x, y in coords:
            unit = self.grid[y][x].unit
            if unit:
                units_on.append(f"{unit.name} ({unit.faction}) at ({x},{y})")

        # Units near the feature (adjacent to any feature hex, but not on it)
        units_near = set()
        for x, y in coords:
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in coords and 0 <= nx < self.width and 0 <= ny < self.height:
                    nunit = self.grid[ny][nx].unit
                    if nunit:
                        units_near.add(f"{nunit.name} ({nunit.faction}) at ({nx},{ny})")

        desc = f"Feature '{feature_name}':\n"
        desc += f"  Terrain type: {terrain_type}\n"
        desc += f"  Hexes: {coords}\n"
        if units_on:
            desc += f"  Units present: {', '.join(units_on)}\n"
        else:
            desc += "  Units present: None\n"
        if units_near:
            desc += f"  Units nearby: {', '.join(units_near)}\n"
        else:
            desc += "  Units nearby: None\n"
        return desc
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
    
    def get_units_by_faction(self, faction: str) -> list[Unit]:
        """Return a list of unit objects on the map that match the given faction."""
        units = []
        for row in self.grid:
            for hex in row:
                if hex.unit and hex.unit.faction == faction:
                    units.append(hex.unit)
        return units
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
        """Move a unit if reachable within its mobility using Dijkstra's algorithm."""
        reachable = self.find_reachable_hexes(unit)
        if (target_x, target_y) not in reachable:
            return False  # too far or occupied

        self.grid[unit.y][unit.x].unit = None
        self.grid[target_y][target_x].unit = unit
        unit.move(target_x, target_y, reachable[(target_x, target_y)])
        return True

    def check_and_engage_combat(self, unit: Unit):
        """Check if a unit is adjacent to enemy units and engage them in combat."""
        for nx, ny in self.get_neighbors(unit.x, unit.y):
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            enemy = self.grid[ny][nx].unit
            if enemy and enemy.faction != unit.faction:
                # Both units become engaged
                unit.engaged = True
                enemy.engaged = True
                print(f"{unit.name} engages {enemy.name}!")

    def check_all_engagements(self):
        """Check all units on the map and engage adjacent enemies."""
        processed = set()
        for y in range(self.height):
            for x in range(self.width):
                unit = self.grid[y][x].unit
                if not unit or (x, y) in processed:
                    continue
                
                # Check for adjacent enemies
                for nx, ny in self.get_neighbors(x, y):
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    enemy = self.grid[ny][nx].unit
                    if enemy and enemy.faction != unit.faction:
                        # Both units become engaged
                        unit.engaged = True
                        enemy.engaged = True
                        if (nx, ny) not in processed:
                            print(f"{unit.name} engages {enemy.name}!")
                        processed.add((x, y))
                        processed.add((nx, ny))

    def apply_engagement_damage(self):
        """Apply combat damage to all engaged units."""
        processed = set()
        for y in range(self.height):
            for x in range(self.width):
                unit = self.grid[y][x].unit
                if not unit or not unit.engaged or (x, y) in processed:
                    continue
                
                # Find adjacent engaged enemies
                for nx, ny in self.get_neighbors(x, y):
                    if not (0 <= nx < self.width and 0 <= ny < self.height):
                        continue
                    enemy = self.grid[ny][nx].unit
                    if enemy and enemy.faction != unit.faction and enemy.engaged:
                        # Avoid processing the same pair twice
                        if (nx, ny) in processed:
                            continue
                        
                        # Calculate combat power for both units
                        unit_power = 1 # unit.combat_power(self.get_terrain(unit.x, unit.y))
                        enemy_power = 1 # enemy.combat_power(self.get_terrain(enemy.x, enemy.y))
                        
                        # Apply damage
                        damage_to_enemy = max(1, int(unit_power / 10))
                        damage_to_unit = max(1, int(enemy_power / 10))
                        
                        print(f"Combat: {unit.name} (power {unit_power:.1f}) vs {enemy.name} (power {enemy_power:.1f})")
                        
                        enemy.take_damage(damage_to_enemy)
                        unit.take_damage(damage_to_unit)
                        
                        # Remove routed units
                        if enemy.is_routed():
                            self.grid[ny][nx].unit = None
                        if unit.is_routed():
                            self.grid[y][x].unit = None
                        
                        # Mark both as processed
                        processed.add((x, y))
                        processed.add((nx, ny))
                        break  # Only process one enemy per unit per turn

    def move_units_toward_destinations(self, movement_orders: dict) -> dict:
        """Move units toward their destination hexes in order.
        
        Args:
            movement_orders: Dictionary with structure:
                {
                    "orders": [
                        {"unit_name": "1st Line", "destination": [8, 5]},
                        {"unit_name": "2nd Line", "destination": [9, 4]},
                        ...
                    ]
                }
        
        Returns:
            Dictionary with results for each unit:
                {
                    "results": [
                        {"unit_name": "1st Line", "moved": True, "final_position": [7, 5], "distance_to_goal": 1},
                        ...
                    ]
                }
        """
        
        def hex_distance(x1, y1, x2, y2):
            """Calculate hex distance using cube coordinates."""
            # Convert offset to cube coordinates
            def offset_to_cube(x, y):
                q = x - (y - (y & 1)) // 2
                r = y
                return q, r, -q - r
            
            q1, r1, s1 = offset_to_cube(x1, y1)
            q2, r2, s2 = offset_to_cube(x2, y2)
            return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2
        
        def find_unit_by_name(name: str) -> Optional[Unit]:
            """Find a unit on the map by name."""
            for row in self.grid:
                for hex in row:
                    if hex.unit and hex.unit.name == name:
                        return hex.unit
            return None
        
        results = []
        
        # Process orders sequentially
        for order in movement_orders.get("orders", []):
            unit_name = order.get("unit_name")
            dest = order.get("destination")
            
            if not unit_name or not dest or len(dest) != 2:
                results.append({
                    "unit_name": unit_name,
                    "moved": False,
                    "error": "Invalid order format"
                })
                continue
            
            dest_x, dest_y = dest
            unit = find_unit_by_name(unit_name)
            
            if not unit:
                results.append({
                    "unit_name": unit_name,
                    "moved": False,
                    "error": "Unit not found on map"
                })
                continue
            
            # Get all reachable hexes for this unit
            reachable = self.find_reachable_hexes(unit)
            
            if not reachable:
                results.append({
                    "unit_name": unit_name,
                    "moved": False,
                    "final_position": [unit.x, unit.y],
                    "distance_to_goal": hex_distance(unit.x, unit.y, dest_x, dest_y),
                    "error": "No reachable hexes"
                })
                continue
            
            # If destination is directly reachable, move there
            if (dest_x, dest_y) in reachable:
                success = self.move_unit(unit, dest_x, dest_y)
                results.append({
                    "unit_name": unit_name,
                    "moved": success,
                    "final_position": [unit.x, unit.y],
                    "distance_to_goal": 0 if success else hex_distance(unit.x, unit.y, dest_x, dest_y)
                })
            else:
                # Find the reachable hex closest to the destination
                best_hex = None
                best_distance = float('inf')
                
                for (rx, ry) in reachable:
                    if (rx, ry) == (unit.x, unit.y):
                        continue  # Skip current position
                    dist = hex_distance(rx, ry, dest_x, dest_y)
                    if dist < best_distance:
                        best_distance = dist
                        best_hex = (rx, ry)
                
                if best_hex:
                    success = self.move_unit(unit, best_hex[0], best_hex[1])
                    results.append({
                        "unit_name": unit_name,
                        "moved": success,
                        "final_position": [unit.x, unit.y],
                        "distance_to_goal": hex_distance(unit.x, unit.y, dest_x, dest_y)
                    })
                else:
                    # No better position available, stay put
                    results.append({
                        "unit_name": unit_name,
                        "moved": False,
                        "final_position": [unit.x, unit.y],
                        "distance_to_goal": hex_distance(unit.x, unit.y, dest_x, dest_y),
                        "error": "No closer position available"
                    })
        
        return {"results": results}

    # --- Staff Officer movement helpers and actions ---
    def _hex_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate hex distance using cube coordinates for even-q (column offset) layout."""
        def offset_to_cube(x: int, y: int) -> Tuple[int, int, int]:
            q = x
            r = y - (x - (x & 1)) // 2
            return q, r, -q - r

        q1, r1, s1 = offset_to_cube(x1, y1)
        q2, r2, s2 = offset_to_cube(x2, y2)
        return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2

    def _find_unit_by_name(self, name: str) -> Optional[Unit]:
        for row in self.grid:
            for hex in row:
                if hex.unit and hex.unit.name == name:
                    return hex.unit
        return None

    def _nearest_enemy_pos(self, unit: Unit) -> Optional[Tuple[int, int]]:
        best = None
        best_d = float("inf")
        for y in range(self.height):
            for x in range(self.width):
                other = self.grid[y][x].unit
                if other and other.faction != unit.faction:
                    d = self._hex_distance(unit.x, unit.y, x, y)
                    if d < best_d:
                        best_d, best = d, (x, y)
        return best

    def _direction_offsets(self, y: int) -> List[Tuple[int, int]]:
        """Return the 6 neighbor direction offsets in order, matching get_neighbors order."""
        offsets_even = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, -1)]
        offsets_odd = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (-1, +1)]
        return offsets_odd if y % 2 else offsets_even

    def _desired_along_direction(self, x: int, y: int, dir_index: int, steps: int = 1) -> Tuple[int, int]:
        dx, dy = self._direction_offsets(y)[dir_index % 6]
        tx, ty = x, y
        for _ in range(max(1, steps)):
            tx += dx
            ty += dy
        return tx, ty

    def _best_reachable_toward(self, unit: Unit, target: Tuple[int, int], max_cost: Optional[int] = None) -> Optional[Tuple[int, int]]:
        reachable = self.find_reachable_hexes(unit)
        if max_cost is not None:
            reachable = {pos: cost for pos, cost in reachable.items() if cost <= max_cost}
        if not reachable:
            return None
        best_hex = None
        best_dist = float("inf")
        for (rx, ry), _cost in reachable.items():
            if (rx, ry) == (unit.x, unit.y):
                continue
            dist = self._hex_distance(rx, ry, target[0], target[1])
            if dist < best_dist:
                best_dist = dist
                best_hex = (rx, ry)
        return best_hex

    # def advance(self, unit_name: str, destination: Tuple[int, int]) -> dict:
    #     """Advance the named unit toward a destination hex using pathfinding."""
    #     unit = self._find_unit_by_name(unit_name)
    #     if not unit:
    #         return {"ok": False, "error": "Unit not found"}
    #     target = (int(destination[0]), int(destination[1]))
    #     # Move to destination if in range, otherwise get closest reachable toward it
    #     reachable = self.find_reachable_hexes(unit)
    #     if target in reachable:
    #         moved = self.move_unit(unit, *target)
    #         return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}
    #     best = self._best_reachable_toward(unit, target)
    #     if not best:
    #         return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], "reason": "No reachable hex"}
    #     moved = self.move_unit(unit, best[0], best[1])
    #     return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}

    # def retreat(self, unit_name: str, destination: Tuple[int, int]) -> dict:
    #     """Fall back toward a designated destination, preferring to increase distance from the nearest enemy."""
    #     unit = self._find_unit_by_name(unit_name)
    #     if not unit:
    #         return {"ok": False, "error": "Unit not found"}
    #     dest = (int(destination[0]), int(destination[1]))
    #     enemy_pos = self._nearest_enemy_pos(unit)
    #     reachable = self.find_reachable_hexes(unit)
    #     if not reachable:
    #         return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], "reason": "No reachable hex"}
    #     # Score: maximize distance to enemy (if known), then minimize distance to destination
    #     best_hex = None
    #     best_score = None
    #     for (rx, ry) in reachable.keys():
    #         if (rx, ry) == (unit.x, unit.y):
    #             continue
    #         dist_enemy = self._hex_distance(rx, ry, enemy_pos[0], enemy_pos[1]) if enemy_pos else 0
    #         dist_dest = self._hex_distance(rx, ry, dest[0], dest[1])
    #         score = (dist_enemy, -dist_dest)
    #         if best_score is None or score > best_score:
    #             best_score = score
    #             best_hex = (rx, ry)
    #     if not best_hex:
    #         return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], "reason": "No better hex"}
    #     moved = self.move_unit(unit, best_hex[0], best_hex[1])
    #     return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}

    # def flank_left(self, unit_name: str, destination: Tuple[int, int]) -> dict:
    #     """Flank left toward a destination by biasing the path to the left of the main approach."""
    #     unit = self._find_unit_by_name(unit_name)
    #     if not unit:
    #         return {"ok": False, "error": "Unit not found"}
    #     dest = (int(destination[0]), int(destination[1]))
    #     # determine forward direction as neighbor reducing distance to destination most
    #     dirs = self._direction_offsets(unit.y)
    #     best_dir = 0
    #     best_reduction = -1
    #     curd = self._hex_distance(unit.x, unit.y, dest[0], dest[1])
    #     for i, (dx, dy) in enumerate(dirs):
    #         nx, ny = unit.x + dx, unit.y + dy
    #         if not (0 <= nx < self.width and 0 <= ny < self.height):
    #             continue
    #         nd = self._hex_distance(nx, ny, dest[0], dest[1])
    #         red = curd - nd
    #         if nd < curd and red > best_reduction and self.grid[ny][nx].unit is None:
    #             best_reduction = red
    #             best_dir = i
    #     left_dir = (best_dir - 1) % 6
    #     desired = self._desired_along_direction(unit.x, unit.y, left_dir, 1)
    #     best = self._best_reachable_toward(unit, desired)
    #     if not best:
    #         # fallback toward destination normally
    #         best = self._best_reachable_toward(unit, dest)
    #         if not best:
    #             return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], "reason": "No flank path"}
    #     moved = self.move_unit(unit, best[0], best[1])
    #     return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}

    # def flank_right(self, unit_name: str, destination: Tuple[int, int]) -> dict:
    #     """Flank right toward a destination by biasing the path to the right of the main approach."""
    #     unit = self._find_unit_by_name(unit_name)
    #     if not unit:
    #         return {"ok": False, "error": "Unit not found"}
    #     dest = (int(destination[0]), int(destination[1]))
    #     dirs = self._direction_offsets(unit.y)
    #     best_dir = 0
    #     best_reduction = -1
    #     curd = self._hex_distance(unit.x, unit.y, dest[0], dest[1])
    #     for i, (dx, dy) in enumerate(dirs):
    #         nx, ny = unit.x + dx, unit.y + dy
    #         if not (0 <= nx < self.width and 0 <= ny < self.height):
    #             continue
    #         nd = self._hex_distance(nx, ny, dest[0], dest[1])
    #         red = curd - nd
    #         if nd < curd and red > best_reduction and self.grid[ny][nx].unit is None:
    #             best_reduction = red
    #             best_dir = i
    #     right_dir = (best_dir + 1) % 6
    #     desired = self._desired_along_direction(unit.x, unit.y, right_dir, 1)
    #     best = self._best_reachable_toward(unit, desired)
    #     if not best:
    #         best = self._best_reachable_toward(unit, dest)
    #         if not best:
    #             return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], "reason": "No flank path"}
    #     moved = self.move_unit(unit, best[0], best[1])
    #     return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}

    # def hold(self, unit_name: str, destination: Tuple[int, int]) -> dict:
    #     """Order the unit to hold position (no movement). Destination is accepted but not used."""
    #     unit = self._find_unit_by_name(unit_name)
    #     if not unit:
    #         return {"ok": False, "error": "Unit not found"}
    #     return {"ok": True, "unit": unit.name, "position": [unit.x, unit.y]}

    def march(self, unit_name: str, destination: Tuple[int, int]) -> dict:
        """Order the unit to march toward a destination (x, y)."""
        unit = self._find_unit_by_name(unit_name)
        if not unit:
            return {"ok": False, "error": "Unit not found"}
        dx, dy = int(destination[0]), int(destination[1])
        reachable = self.find_reachable_hexes(unit)
        if (dx, dy) in reachable:
            moved = self.move_unit(unit, dx, dy)
            return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}
        # Otherwise move as close as possible
        best = self._best_reachable_toward(unit, (dx, dy))
        if not best:
            return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], "reason": "No path"}
        moved = self.move_unit(unit, best[0], best[1])
        return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}

    # --- Dijkstra Pathfinding ---
    def find_reachable_hexes(self, unit: Unit):
        """Find all reachable hexes within unit.mobility using terrain move_cost and enemy zone of control (ZOC).
        Entering a hex adjacent to an enemy unit sets remaining movement to 0 for the turn."""
        start = (unit.x, unit.y)
        max_cost = unit.mobility

        cost_so_far = {start: 0}
        pq = [(0, start)]
        reachable = {}

        def is_adjacent_to_enemy(x, y):
            for nx, ny in self.get_neighbors(x, y):
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_unit = self.grid[ny][nx].unit
                    if neighbor_unit and neighbor_unit.faction != unit.faction:
                        return True
            return False

        while pq:
            current_cost, (x, y) = heapq.heappop(pq)
            if current_cost > max_cost:
                continue

            # Mark current hex as reachable only if it's the start position or unoccupied
            current_tile = self.grid[y][x]
            if (x, y) == start or current_tile.unit is None:
                reachable[(x, y)] = current_cost

            # If this hex is adjacent to an enemy (ZOC), do not allow further movement from here
            if (x, y) != start and is_adjacent_to_enemy(x, y):
                continue

            for nx, ny in self.get_neighbors(x, y):
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                neighbor_tile = self.grid[ny][nx]
                # Enemy blocks path; friendly units also block destination but allow path-through
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
        offsets_even = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, -1)]
        offsets_odd = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (-1, +1)]
        offsets = offsets_odd if x % 2 else offsets_even
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
        att_power = 1 # attacker.combat_power(self.get_terrain(attacker.x, attacker.y))
        def_power = 1 # defender.combat_power(self.get_terrain(defender.x, defender.y))

        damage_to_def = max(1, int(att_power / 10))
        damage_to_att = max(1, int(def_power / 10))

        defender.take_damage(damage_to_def)
        attacker.take_damage(damage_to_att)

        # Remove routed units
        if defender.is_routed():
            self.grid[defender.y][defender.x].unit = None
        if attacker.is_routed():
            self.grid[attacker.y][attacker.x].unit = None

    # --- Feature labeling ---
    def label_terrain_features(self, seed: Optional[int] = None,
                               min_sizes: Optional[dict] = None) -> None:
        """Populate each hex's `features` with names for clusters of Hills, Rivers, Forests, and Valleys.

        Valleys are approximated as contiguous Fields tiles adjacent to any River tile.

        Args:
            seed: Optional RNG seed for reproducible names.
            min_sizes: Optional per-type minimum cluster size to label, e.g.,
                {"Hill": 3, "Forest": 3, "River": 2, "Valley": 3}
        """
        if seed is not None:
            random.seed(seed)

        # Default thresholds to avoid labeling tiny clusters
        thresholds = {"Hill": 3, "Forest": 3, "River": 2, "Valley": 3}
        if min_sizes:
            thresholds.update(min_sizes)

        # Helper: cluster by a predicate
        def cluster_by(predicate) -> List[List[Tuple[int, int]]]:
            seen: Set[Tuple[int, int]] = set()
            clusters: List[List[Tuple[int, int]]] = []
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) in seen or not predicate(x, y):
                        continue
                    comp: List[Tuple[int, int]] = []
                    stack = [(x, y)]
                    seen.add((x, y))
                    while stack:
                        cx, cy = stack.pop()
                        comp.append((cx, cy))
                        for nx, ny in self.get_neighbors(cx, cy):
                            if 0 <= nx < self.width and 0 <= ny < self.height:
                                if (nx, ny) not in seen and predicate(nx, ny):
                                    seen.add((nx, ny))
                                    stack.append((nx, ny))
                    clusters.append(comp)
            return clusters

        # Identify basic type membership
        from terrain import HILL_ELEVATION_THRESHOLD, FOREST_TREE_COVER_THRESHOLD
        
        def is_hill(x, y):
            # Use elevation threshold to classify hills instead of terrain name
            return self.grid[y][x].terrain.elevation >= HILL_ELEVATION_THRESHOLD

        def is_forest(x, y):
            # Use tree_cover threshold to classify forests instead of terrain name
            return self.grid[y][x].terrain.tree_cover >= FOREST_TREE_COVER_THRESHOLD

        def is_river(x, y):
            return self.grid[y][x].terrain.name == "River"

        # Valley candidates: Fields adjacent to any River
        river_adjacent_fields: Set[Tuple[int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x].terrain.name != "Fields":
                    continue
                for nx, ny in self.get_neighbors(x, y):
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if self.grid[ny][nx].terrain.name == "River":
                            river_adjacent_fields.add((x, y))
                            break

        def is_valley(x, y):
            return (x, y) in river_adjacent_fields

        hill_clusters = [c for c in cluster_by(is_hill) if len(c) >= thresholds["Hill"]]
        forest_clusters = [c for c in cluster_by(is_forest) if len(c) >= thresholds["Forest"]]
        river_clusters = [c for c in cluster_by(is_river) if len(c) >= thresholds["River"]]
        # For valleys, cluster only over the candidate set
        def valley_pred(x, y):
            return (x, y) in river_adjacent_fields
        valley_clusters = [c for c in cluster_by(valley_pred) if len(c) >= thresholds["Valley"]]

        # Name generation — 19th-century Italian terms
        roots = [
            # Major Italian rivers
            "Adda", "Po", "Ticino", "Mincio", "Brenta", "Isonzo", "Adige", "Arno", "Liri",
            "Sesia", "Oglio", "Tanaro", "Taro", "Nera", "Volturno", "Piave", "Trebbia", "Chiese",
            "Ofanto", "Bormida", "Scrivia", "Secchia", "Panaro", "Reno", "Tagliamento", "Livenza",
            "Metauro", "Esino", "Pescara", "Sangro", "Trigno", "Biferno", "Fortore", "Sele",
            "Bacchiglione", "Brembana", "Serio", "Lambro", "Olona", "Toce", "Dora", "Orco",
            # Italian cities and towns from Napoleonic era
            "Mantua", "Verona", "Brescia", "Bergamo", "Cremona", "Pavia", "Lodi", "Milan",
            "Vicenza", "Padua", "Treviso", "Udine", "Piacenza", "Parma", "Modena", "Ferrara",
            "Bologna", "Ravenna", "Rimini", "Pesaro", "Ancona", "Macerata", "Ascoli", "Teramo",
            "Aquila", "Chieti", "Perugia", "Spoleto", "Terni", "Viterbo", "Rieti", "Frosinone",
            "Latina", "Caserta", "Benevento", "Avellino", "Salerno", "Potenza", "Matera", "Foggia",
            # Historical Italian regions and provinces
            "Lombardy", "Venetia", "Piedmont", "Liguria", "Emilia", "Romagna", "Tuscany", "Umbria",
            "Marche", "Abruzzo", "Molise", "Campania", "Apulia", "Basilicata", "Calabria"
        ]
        saints = [
            "San Marco", "San Pietro", "San Giorgio", "San Luca", "San Paolo", "Santa Maria",
            "Sant'Anna", "San Michele", "San Carlo", "San Giovanni", "San Giacomo", "San Matteo",
            "San Filippo", "San Simone", "San Tommaso", "San Bartolomeo", "San Andrea", "San Stefano",
            "Santa Lucia", "Santa Caterina", "Santa Teresa", "Sant'Antonio", "San Francesco",
            "San Domenico", "San Bernardo", "San Giuseppe", "Santa Chiara", "Sant'Agostino",
            "San Benedetto", "San Lorenzo", "San Martino", "San Nicola", "San Rocco", "San Sebastiano",
            "Santa Monica", "Sant'Ambrogio", "San Gennaro", "San Vittore", "Santa Barbara",
            "San Cristoforo", "San Donato", "San Fedele", "San Gregorio", "Sant'Eufemia",
            "San Zeno", "San Biagio", "Sant'Eusebio", "San Protaso", "San Gervaso"
        ]
        adjs = [
            # Italian adjectives - colors, sizes, qualities
            "Grande", "Piccolo", "Alto", "Basso", "Lungo", "Corto", "Nero", "Verde", "Rosso",
            "Bianco", "Azzurro", "Bruno", "Grigio", "Dorato", "Argento", "Chiaro", "Scuro",
            "Nuovo", "Vecchio", "Primo", "Secondo", "Terzo", "Ultimo", "Centrale", "Orientale",
            "Occidentale", "Settentrionale", "Meridionale", "Superiore", "Inferiore", "Esterno",
            "Interno", "Maggiore", "Minore", "Estremo", "Medio", "Largo", "Stretto", "Profondo",
            "Piano", "Ripido", "Dolce", "Aspro", "Fertile", "Arido", "Folto", "Rado"
        ]

        used_names: Set[str] = set()

        def unique(name: str) -> str:
            if name not in used_names:
                used_names.add(name)
                return name
            i = 2
            base = name
            while True:
                cand = f"{base} {i}"
                if cand not in used_names:
                    used_names.add(cand)
                    return cand
                i += 1

        def name_hill() -> str:
            choice = random.randint(0, 2)
            if choice == 0:
                return unique(f"{random.choice(roots)} Hill")
            elif choice == 1:
                return unique(f"{random.choice(saints)} Heights")
            else:
                return unique(f"{random.choice(adjs)} Ridge")

        def name_forest() -> str:
            choice = random.randint(0, 2)
            if choice == 0:
                return unique(f"{random.choice(saints)} Wood")
            elif choice == 1:
                return unique(f"{random.choice(roots)} Forest")
            else:
                return unique(f"{random.choice(adjs)} Thicket")

        def name_river() -> str:
            if random.random() < 0.5:
                return unique(f"{random.choice(roots)} River")
            else:
                return unique(f"{random.choice(roots)} Stream")

        def name_valley() -> str:
            choice = random.randint(0, 2)
            if choice == 0:
                return unique(f"{random.choice(roots)} Valley")
            elif choice == 1:
                return unique(f"{random.choice(saints)} Vale")
            else:
                return unique(f"{random.choice(adjs)} Plain")

        # Helper to subdivide large clusters into sub-features
        def subdivide_cluster(cluster: List[Tuple[int, int]], min_subcluster_size: int = 2) -> List[List[Tuple[int, int]]]:
            """Break a large cluster into smaller sub-clusters using spatial proximity.
            Aims to cover most/all hexes with sub-features."""
            if len(cluster) < min_subcluster_size:
                return []  # Too small to subdivide
            
            subclusters = []
            remaining = set(cluster)
            
            # Limit iterations to prevent infinite loops
            max_iterations = len(cluster) * 2
            iteration_count = 0
            
            # Greedy clustering: pick seeds and grow sub-clusters
            # Keep going until we've covered almost all tiles
            while len(remaining) >= min_subcluster_size and iteration_count < max_iterations:
                iteration_count += 1
                
                # Pick a random seed from remaining
                seed = random.choice(list(remaining))
                subcluster = [seed]
                remaining.remove(seed)
                
                # Create small sub-features (2-3 hexes) to maximize coverage
                target_size = random.randint(2, 3)
                
                # BFS-like growth but stay compact
                frontier = [seed]
                visited = {seed}
                growth_iterations = 0
                max_growth = target_size * 4
                
                while len(subcluster) < target_size and frontier and growth_iterations < max_growth:
                    growth_iterations += 1
                    
                    if not frontier:
                        break
                        
                    current = frontier.pop(0)
                    neighbors = list(self.get_neighbors(current[0], current[1]))
                    random.shuffle(neighbors)
                    
                    for nx, ny in neighbors:
                        if (nx, ny) in remaining and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            if random.random() < 0.75:  # High probability for maximum coverage
                                subcluster.append((nx, ny))
                                remaining.remove((nx, ny))
                                frontier.append((nx, ny))
                                if len(subcluster) >= target_size:
                                    break
                
                if len(subcluster) >= min_subcluster_size:
                    subclusters.append(subcluster)
                elif len(subcluster) == 1 and len(remaining) < min_subcluster_size:
                    # For isolated single hexes at the end, create a single-hex sub-feature
                    subclusters.append(subcluster)
            
            # Handle any remaining isolated hexes by adding them as single-hex sub-features
            for hex_pos in remaining:
                subclusters.append([hex_pos])
            
            return subclusters

        # Helper to generate sub-feature names
        def name_hill_sub() -> str:
            """Generate sub-feature names for hills (individual peaks, crests, etc.)"""
            choice = random.randint(0, 3)
            if choice == 0:
                return unique(f"{random.choice(adjs)} Rise")
            elif choice == 1:
                return unique(f"{random.choice(roots)} Peak")
            elif choice == 2:
                return unique(f"{random.choice(saints)} Crest")
            else:
                return unique(f"{random.choice(roots)} Knoll")

        def name_forest_sub() -> str:
            """Generate sub-feature names for forests (groves, thickets, etc.)"""
            choice = random.randint(0, 3)
            if choice == 0:
                return unique(f"{random.choice(adjs)} Grove")
            elif choice == 1:
                return unique(f"{random.choice(saints)} Copse")
            elif choice == 2:
                return unique(f"{random.choice(roots)} Glade")
            else:
                return unique(f"{random.choice(adjs)} Stand")

        def name_river_sub() -> str:
            """Generate sub-feature names for rivers (bends, fords, etc.)"""
            choice = random.randint(0, 3)
            if choice == 0:
                return unique(f"{random.choice(roots)} Bend")
            elif choice == 1:
                return unique(f"{random.choice(adjs)} Ford")
            elif choice == 2:
                return unique(f"{random.choice(saints)} Rapids")
            else:
                return unique(f"{random.choice(roots)} Meander")

        def name_valley_sub() -> str:
            """Generate sub-feature names for valleys (meadows, flats, etc.)"""
            choice = random.randint(0, 3)
            if choice == 0:
                return unique(f"{random.choice(adjs)} Meadow")
            elif choice == 1:
                return unique(f"{random.choice(saints)} Field")
            elif choice == 2:
                return unique(f"{random.choice(roots)} Flat")
            else:
                return unique(f"{random.choice(adjs)} Dell")

        # Apply names to clusters by populating each hex.features with both major and sub-feature names
        for cluster in hill_clusters:
            major_label = name_hill()
            # All hexes get the major feature name
            for x, y in cluster:
                self.grid[y][x].features.append(major_label)
            
            # Subdivide into sub-features for nearly all clusters (threshold = 3)
            if len(cluster) >= 3:
                subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
                for subcluster in subclusters:
                    sub_label = name_hill_sub()
                    for x, y in subcluster:
                        self.grid[y][x].features.append(sub_label)

        for cluster in forest_clusters:
            major_label = name_forest()
            for x, y in cluster:
                self.grid[y][x].features.append(major_label)
            
            # Subdivide into groves/thickets for nearly all clusters (threshold = 3)
            if len(cluster) >= 3:
                subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
                for subcluster in subclusters:
                    sub_label = name_forest_sub()
                    for x, y in subcluster:
                        self.grid[y][x].features.append(sub_label)

        for cluster in river_clusters:
            major_label = name_river()
            for x, y in cluster:
                self.grid[y][x].features.append(major_label)
            
            # Subdivide into bends/sections for nearly all rivers (threshold = 2 since rivers are linear)
            if len(cluster) >= 2:
                subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
                for subcluster in subclusters:
                    sub_label = name_river_sub()
                    for x, y in subcluster:
                        self.grid[y][x].features.append(sub_label)

        for cluster in valley_clusters:
            major_label = name_valley()
            for x, y in cluster:
                self.grid[y][x].features.append(major_label)
            
            # Subdivide into meadows/fields for nearly all valleys (threshold = 3)
            if len(cluster) >= 3:
                subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
                for subcluster in subclusters:
                    sub_label = name_valley_sub()
                    for x, y in subcluster:
                        self.grid[y][x].features.append(sub_label)
            # Subdivide into meadows/fields if large enough (lowered threshold)
            if len(cluster) >= 4:
                subclusters = subdivide_cluster(cluster, min_subcluster_size=2)
                for subcluster in subclusters:
                    sub_label = name_valley_sub()
                    for x, y in subcluster:
                        self.grid[y][x].features.append(sub_label)