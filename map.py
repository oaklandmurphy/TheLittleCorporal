from terrain import Terrain
from terrain import FIELDS
from unit import Unit
from hex import Hex

import math
from typing import Optional, List, Tuple, Dict, Any

# Import the refactored modules
import pathfinding
import combat
import terrain_features
import frontline
import orders 


class Map:

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
        combat.check_and_engage_combat(unit, self.grid, self.width, self.height)

    def check_all_engagements(self):
        """Check all units on the map and engage adjacent enemies."""
        combat.check_all_engagements(self.grid, self.width, self.height)

    def apply_engagement_damage(self):
        """Apply combat damage to all engaged units."""
        combat.apply_engagement_damage(self.grid, self.width, self.height)

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
                    "distance_to_goal": pathfinding.hex_distance(unit.x, unit.y, dest_x, dest_y),
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
                    "distance_to_goal": 0 if success else pathfinding.hex_distance(unit.x, unit.y, dest_x, dest_y)
                })
            else:
                # Find the reachable hex closest to the destination
                best_hex = None
                best_distance = float('inf')
                
                for (rx, ry) in reachable:
                    if (rx, ry) == (unit.x, unit.y):
                        continue  # Skip current position
                    dist = pathfinding.hex_distance(rx, ry, dest_x, dest_y)
                    if dist < best_distance:
                        best_distance = dist
                        best_hex = (rx, ry)
                
                if best_hex:
                    success = self.move_unit(unit, best_hex[0], best_hex[1])
                    results.append({
                        "unit_name": unit_name,
                        "moved": success,
                        "final_position": [unit.x, unit.y],
                        "distance_to_goal": pathfinding.hex_distance(unit.x, unit.y, dest_x, dest_y)
                    })
                else:
                    # No better position available, stay put
                    results.append({
                        "unit_name": unit_name,
                        "moved": False,
                        "final_position": [unit.x, unit.y],
                        "distance_to_goal": pathfinding.hex_distance(unit.x, unit.y, dest_x, dest_y),
                        "error": "No closer position available"
                    })
        
        return {"results": results}
        return {"results": results}

    # --- Staff Officer movement helpers and actions ---
    def _hex_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate hex distance using cube coordinates for even-q (column offset) layout."""
        return pathfinding.hex_distance(x1, y1, x2, y2)

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
        return pathfinding.direction_offsets(y)

    def _desired_along_direction(self, x: int, y: int, dir_index: int, steps: int = 1) -> Tuple[int, int]:
        dx, dy = self._direction_offsets(y)[dir_index % 6]
        tx, ty = x, y
        for _ in range(max(1, steps)):
            tx += dx
            ty += dy
        return tx, ty

    def _best_reachable_toward(self, unit: Unit, target: Tuple[int, int], max_cost: Optional[int] = None) -> Optional[Tuple[int, int]]:
        return pathfinding.best_reachable_toward(unit, target, self.grid, self.width, self.height, max_cost)

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
        """Find all reachable hexes within unit.mobility using terrain move_cost and enemy zone of control (ZOC)."""
        return pathfinding.find_reachable_hexes(unit, self.grid, self.width, self.height)

    # --- Adjacency and Combat ---
    def get_neighbors(self, x: int, y: int):
        """Return hex neighbors using offset coordinates."""
        return pathfinding.get_neighbors(x, y)

    def resolve_adjacent_combat(self):
        """Make adjacent enemy units engage in combat."""
        combat.resolve_adjacent_combat(self.grid, self.width, self.height)

    def combat(self, attacker: Unit, defender: Unit):
        """Simple mutual combat."""
        combat.combat(attacker, defender, self.grid)

    # --- Feature labeling ---
    def label_terrain_features(self, seed: Optional[int] = None,
                               min_sizes: Optional[dict] = None) -> None:
        """Populate each hex's `features` with names for clusters of Hills, Rivers, Forests, and Valleys."""
        terrain_features.label_terrain_features(self.grid, self.width, self.height, seed, min_sizes)

    def get_combat_advantage(self, x1, y1, x2, y2) -> float:
        """Calculate combat advantage for unit at (x1,y1) against unit at (x2,y2)."""
        return combat.get_combat_advantage(self.grid, self.width, self.height, x1, y1, x2, y2)

    def get_enemy_approach_angle(self, faction: str, feature_name: str) -> Optional[float]:
        """Calculate the angle from a feature to the weighted average position of enemy units.
        
        Enemy units are weighted inversely by their distance from the feature, so closer
        enemies have more influence on the calculated angle than distant ones.
        
        Args:
            faction: The faction for which to calculate the approach (enemies are other factions)
            feature_name: The name of the terrain feature to calculate from
            
        Returns:
            Angle in degrees clockwise from north (0° = North, 90° = East, 180° = South, 270° = West)
            Returns None if the feature doesn't exist or no enemy units are found.
        """
        import math
        
        # Get feature coordinates
        feature_coords = self.get_feature_coordinates(feature_name)
        if not feature_coords:
            return None
        
        # Calculate center of the feature
        feature_x = sum(x for x, y in feature_coords) / len(feature_coords)
        feature_y = sum(y for x, y in feature_coords) / len(feature_coords)
        
        # Find all enemy units and their distances from the feature
        enemy_data = []
        for y in range(self.height):
            for x in range(self.width):
                unit = self.grid[y][x].unit
                if unit and unit.faction != faction:
                    # Calculate distance from feature center to enemy unit
                    distance = self._hex_distance(int(feature_x), int(feature_y), x, y)
                    if distance > 0:  # Avoid division by zero
                        enemy_data.append({
                            'x': x,
                            'y': y,
                            'distance': distance,
                            'weight': 1.0 / math.sqrt(distance)  # Inverse square root distance weighting
                        })
        
        if not enemy_data:
            return None
        
        # Calculate weighted average position of enemies
        total_weight = sum(e['weight'] for e in enemy_data)
        weighted_x = sum(e['x'] * e['weight'] for e in enemy_data) / total_weight
        weighted_y = sum(e['y'] * e['weight'] for e in enemy_data) / total_weight
        
        # Convert hex coordinates to Cartesian for angle calculation
        def hex_to_cart(col: int, row: int) -> Tuple[float, float]:
            """Convert odd-q offset hex coordinates to Cartesian."""
            cx = col * 1.5
            cy = row * (3**0.5) + (col % 2) * (3**0.5) / 2
            return cx, cy
        
        feature_cart_x, feature_cart_y = hex_to_cart(feature_x, feature_y)
        enemy_cart_x, enemy_cart_y = hex_to_cart(weighted_x, weighted_y)
        
        # Calculate angle from feature to enemy weighted average
        dx = enemy_cart_x - feature_cart_x
        dy = enemy_cart_y - feature_cart_y
        
        # Calculate angle (math.atan2 returns angle from positive x-axis)
        # Convert to degrees clockwise from north
        angle_rad = math.atan2(dx, -dy)  # -dy because screen Y is inverted
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to 0-360
        return angle_deg % 360
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
    
    def get_weighted_front_arc_advantage(self, x: int, y: int, angle_degrees: int, arc_width_degrees: float = 180.0) -> float:
        """Calculate weighted combat advantage for all tiles within a front arc."""
        return frontline.get_weighted_front_arc_advantage(self.grid, self.width, self.height, x, y, angle_degrees, arc_width_degrees)

    def get_frontline_endpoints(self, feature_name: str, angle_degrees: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find two endpoint coordinates for a frontline across a feature."""
        feature_coords = self.get_feature_coordinates(feature_name)
        return frontline.get_frontline_endpoints(self.grid, self.width, self.height, feature_coords, angle_degrees)

    def get_frontline_for_feature(self, feature_name: str, angle_degrees: int) -> List[Tuple[int, int]]:
        """Compute the best frontline for a feature facing a direction."""
        endpoints = self.get_frontline_endpoints(feature_name, angle_degrees)
        if not endpoints:
            return []
        start, goal = endpoints
        return self.get_frontline(start, goal, angle_degrees)

    def distribute_units_along_frontline(self, coordinates: List[Tuple[int, int]], num_units: int) -> List[Tuple[int, int]]:
        """Take a list of coordinates and return evenly spaced points along them."""
        return frontline.distribute_units_along_frontline(coordinates, num_units)

    def get_frontline(self, start: Tuple[int, int], goal: Tuple[int, int], angle_degrees: int) -> List[Tuple[int, int]]:
        """Compute the best frontline path between two endpoints using pathfinding."""
        return frontline.get_frontline(self.grid, self.width, self.height, start, goal, angle_degrees)

    def execute_orders(self, orders_data: Dict[str, Any], faction: str) -> None:
        """Execute orders from a general by routing to specific action handlers."""
        orders.execute_orders(self, orders_data, faction)
