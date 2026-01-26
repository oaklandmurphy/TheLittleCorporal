from .terrain import Terrain, FIELDS
from unit import Unit
from .hex import Hex

import math
from typing import Optional, List, Tuple, Dict, Any

# Import the refactored modules
from . import pathfinding, terrain_features, frontline
import combat
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
    
    def get_units(self) -> list[Unit]:
        """Return a list of all unit objects on the map."""
        units = []
        for row in self.grid:
            for hex in row:
                if hex.unit:
                    units.append(hex.unit)
        return units
    
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

    def teleport_unit(self, unit: Unit, x: int, y: int, cost: int) -> bool:
        """
        Instantly move a unit to (x, y) without pathfinding checks.
        Args:
            unit: The unit to move
            x: Target x coordinate
            y: Target y coordinate
            cost: Movement cost to deduct from unit's mobility
        """
        self.grid[unit.y][unit.x].unit = None
        self.grid[y][x].unit = unit
        unit.move(x, y, cost)

    def move_unit(self, unit: Unit, target_x: int, target_y: int, allow_engaged: bool = False) -> bool:
        """Move a unit if reachable within its mobility using Dijkstra's algorithm.
        
        Args:
            unit: The unit to move
            target_x: Target x coordinate
            target_y: Target y coordinate
            allow_engaged: If True, allow engaged units to move (for retreat orders)
            
        Returns:
            True if the unit was successfully moved, False otherwise
        """
        # Engaged units cannot move unless explicitly allowed (e.g., for retreat)
        if unit.engagement and not allow_engaged:
            print(f"{unit.name} is engaged in combat and cannot move!")
            return False
            
        reachable = self.find_reachable_hexes(unit)
        if (target_x, target_y) not in reachable:
            return False  # too far or occupied

        self.teleport_unit(unit, target_x, target_y, reachable[(target_x, target_y)])

        # Handle stances
        if unit.stance == "aggressive":
            aggressive_move = self._best_reachable_toward(unit, self._nearest_enemy_pos(unit), max_cost=unit.mobility//2)
            if aggressive_move:
                self.teleport_unit(unit, aggressive_move[0], aggressive_move[1], reachable[(target_x, target_y)])

        elif unit.stance == "evasive":
            x = 1
            # TODO

        return True

    def check_and_engage_combat(self, unit: Unit):
        """Check if a unit is adjacent to enemy units and engage them in combat."""
        combat.check_and_engage_combat(unit, self.grid, self.width, self.height)

    def check_all_engagements(self):
        """Check all units on the map and engage adjacent enemies."""

        combat.check_all_engagements(self.get_units(), self.grid, self.width, self.height)

    # --- Staff Officer movement helpers and actions ---
    def _hex_distance(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Calculate hex distance using cube coordinates for even-q (column offset) layout."""
        return pathfinding.hex_distance(x1, y1, x2, y2)

    def _find_unit_by_name(self, name: str) -> Optional[Unit]:
        for unit in self.get_units():
            if unit.name == name:
                return unit
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

    def _best_reachable_toward(self, unit: Unit, target: Tuple[int, int], max_cost: Optional[int] = None) -> Optional[Tuple[int, int]]:
        return pathfinding.best_reachable_toward(unit, target, self.grid, self.width, self.height, max_cost)

    def retreat(self, unit_name: str, destination: Tuple[int, int]) -> dict:
        """Order the unit to retreat toward a destination, moving away from enemies.
        
        Retreat is the only order that engaged units can execute, allowing them to 
        disengage from combat.
        
        Args:
            unit_name: Name of the unit to retreat
            destination: Target (x, y) coordinates to retreat toward
            
        Returns:
            Dictionary with 'ok' status, unit name, and position
        """
        unit = self._find_unit_by_name(unit_name)
        if not unit:
            return {"ok": False, "error": "Unit not found"}
        
        dx, dy = int(destination[0]), int(destination[1])
        
        # Find nearest enemy position to prioritize moving away from them
        enemy_pos = self._nearest_enemy_pos(unit)
        reachable = self.find_reachable_hexes(unit)
        
        if not reachable:
            return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], 
                    "reason": "No reachable hexes"}
        
        # Score reachable hexes: prioritize distance from enemy, then closeness to destination
        best_hex = None
        best_score = None
        
        for (rx, ry) in reachable.keys():
            if (rx, ry) == (unit.x, unit.y):
                continue
            
            # Calculate distance from enemy (maximize) and distance to destination (minimize)
            dist_enemy = pathfinding.hex_distance(rx, ry, enemy_pos[0], enemy_pos[1]) if enemy_pos else 0
            dist_dest = pathfinding.hex_distance(rx, ry, dx, dy)
            
            # Score heavily favors moving away from enemy
            score = (dist_enemy * 10, -dist_dest)
            
            if best_score is None or score > best_score:
                best_score = score
                best_hex = (rx, ry)
        
        if not best_hex:
            return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], 
                    "reason": "No better hex to retreat to"}
        
        # Use allow_engaged=True to permit engaged units to move during retreat
        moved = self.move_unit(unit, best_hex[0], best_hex[1], allow_engaged=True)
        
        if moved and unit.engagement:
            print(f"{unit.name} retreats from combat!")
        
        return {"ok": bool(moved), "unit": unit.name, "position": [unit.x, unit.y]}

    def march(self, unit_name: str, destination: Tuple[int, int]) -> dict:
        """Order the unit to march toward a destination (x, y)."""
        unit = self._find_unit_by_name(unit_name)
        if not unit:
            return {"ok": False, "error": "Unit not found"}
        
        # Check if unit is engaged - engaged units cannot execute march orders
        if unit.engagement:
            return {"ok": False, "unit": unit.name, "position": [unit.x, unit.y], 
                    "reason": "Unit is engaged in combat and cannot march"}
        
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
    
    def get_weighted_front_arc_advantage(self, x: int, y: int, angle_degrees: int, arc_width_degrees: float = 180.0) -> float:
        """Calculate weighted combat advantage for all tiles within a front arc."""
        return frontline.get_weighted_front_arc_advantage(self.grid, self.width, self.height, x, y, angle_degrees, arc_width_degrees)

    def get_frontline_endpoints(self, feature_name: str, angle_degrees: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find two endpoint coordinates for a frontline across a feature."""
        feature_coords = self.get_feature_coordinates(feature_name)
        return frontline.get_frontline_endpoints(self.grid, self.width, self.height, feature_coords, angle_degrees)
    
    def assign_units_to_destinations_optimally(self, unit_names: List[str], destinations: List[Tuple[int, int]]) -> dict:
        """Assign units to destinations minimizing total movement cost using the Hungarian algorithm."""
        return frontline.assign_units_to_destinations_optimally(self, unit_names, destinations)

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

    def get_tactical_situation(self, faction: str) -> dict:
        """Get comprehensive tactical information for a faction.
        
        Args:
            faction: The faction to analyze
            
        Returns:
            Dictionary containing friendly units, enemy units, and terrain features
        """
        friendly_units = self.get_units_by_faction(faction)
        enemy_units = []
        for row in self.grid:
            for hex in row:
                if hex.unit and hex.unit.faction != faction:
                    enemy_units.append(hex.unit)
        
        # Calculate frontline positions
        frontline_units = []
        for unit in friendly_units:
            for nx, ny in self.get_neighbors(unit.x, unit.y):
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_unit = self.grid[ny][nx].unit
                    if neighbor_unit and neighbor_unit.faction != faction:
                        frontline_units.append(unit)
                        break
        
        return {
            "friendly_units": [
                {
                    "name": u.name,
                    "position": [u.x, u.y],
                    "size": u.size,
                    "mobility": u.mobility,
                    "engaged": u.engagement,
                    "on_frontline": u in frontline_units
                }
                for u in friendly_units
            ],
            "enemy_units": [
                {
                    "name": u.name,
                    "position": [u.x, u.y],
                    "size": u.size,
                    "distance_to_nearest_friendly": min(
                        pathfinding.hex_distance(u.x, u.y, f.x, f.y) 
                        for f in friendly_units
                    ) if friendly_units else None
                }
                for u in enemy_units
            ],
            "terrain_features": self.list_feature_names()
        }

    def get_battlefield_summary(self, faction: str) -> str:
        """Get a natural language summary of the battlefield for LLM consumption.
        
        Args:
            faction: The faction to generate summary for
            
        Returns:
            Formatted string describing the tactical situation
        """
        tactical_sit = self.get_tactical_situation(faction)
        
        summary = f"=== BATTLEFIELD SITUATION FOR {faction.upper()} ===\n\n"
        
        # Calculate average friendly unit position for reference
        friendly_units = tactical_sit['friendly_units']
        if friendly_units:
            avg_friendly_x = sum(u['position'][0] for u in friendly_units) / len(friendly_units)
            avg_friendly_y = sum(u['position'][1] for u in friendly_units) / len(friendly_units)
        else:
            avg_friendly_x, avg_friendly_y = 0, 0
        
        # Friendly forces summary
        summary += f"YOUR FORCES ({len(friendly_units)} units):\n"
        for unit in friendly_units:
            status = "ENGAGED" if unit['engaged'] else "READY"
            frontline = " [FRONTLINE]" if unit['on_frontline'] else ""
            summary += f"  - {unit['name']}: Size {unit['size']}, "
            summary += f"Position ({unit['position'][0]}, {unit['position'][1]}), "
            summary += f"Status: {status}{frontline}\n"
        
        # Enemy forces summary
        summary += f"\nENEMY FORCES ({len(tactical_sit['enemy_units'])} units):\n"
        for unit in tactical_sit['enemy_units']:
            dist = unit['distance_to_nearest_friendly']
            dist_str = f"{dist} hexes away" if dist is not None else "unknown distance"
            summary += f"  - {unit['name']}: Size {unit['size']}, "
            summary += f"Position ({unit['position'][0]}, {unit['position'][1]}), {dist_str}\n"
        
        # Detailed terrain features
        summary += f"\nKEY TERRAIN FEATURES:\n"
        
        # Build a map of which features overlap with which hexes
        feature_hex_map = {}  # feature_name -> set of (x, y)
        hex_features_map = {}  # (x, y) -> list of feature names
        
        for feature_name in tactical_sit['terrain_features']:
            coords = self.get_feature_coordinates(feature_name)
            feature_hex_map[feature_name] = set(coords)
            for coord in coords:
                if coord not in hex_features_map:
                    hex_features_map[coord] = []
                hex_features_map[coord].append(feature_name)
        
        # Sort features by size (largest first)
        features_by_size = sorted(
            tactical_sit['terrain_features'],
            key=lambda f: len(feature_hex_map.get(f, [])),
            reverse=True
        )
        
        # Track which features have been mentioned as minor features
        mentioned_as_minor = set()
        
        for feature_name in features_by_size[:10]:  # Limit to first 10
            # Skip if this was already mentioned as a minor feature
            if feature_name in mentioned_as_minor:
                continue
                
            coords = self.get_feature_coordinates(feature_name)
            if not coords:
                continue
            
            # Determine terrain type (majority terrain)
            terrain_count = {}
            for x, y in coords:
                tname = self.grid[y][x].terrain.name
                terrain_count[tname] = terrain_count.get(tname, 0) + 1
            terrain_type = max(terrain_count, key=terrain_count.get)
            
            # Calculate feature center and size
            feature_x = sum(x for x, y in coords) / len(coords)
            feature_y = sum(y for x, y in coords) / len(coords)
            feature_size = len(coords)
            
            # Find overlapping features (minor features that share hexes with this one)
            overlapping_features = set()
            for coord in coords:
                for other_feature in hex_features_map.get(coord, []):
                    if other_feature != feature_name:
                        # Only include as minor if it's smaller
                        if len(feature_hex_map[other_feature]) < feature_size:
                            overlapping_features.add(other_feature)
                            mentioned_as_minor.add(other_feature)
            
            # Calculate distance and direction from friendly units
            if friendly_units:
                dist_from_friendly = pathfinding.hex_distance(
                    int(avg_friendly_x), int(avg_friendly_y),
                    int(feature_x), int(feature_y)
                )
                
                # Calculate direction using simple compass logic
                dx = feature_x - avg_friendly_x
                dy = feature_y - avg_friendly_y
                
                # Determine cardinal/intercardinal direction
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    direction = "at your position"
                else:
                    angle = math.atan2(dx, -dy)  # -dy because screen Y is inverted
                    angle_deg = (math.degrees(angle) + 360) % 360
                    
                    if angle_deg < 22.5 or angle_deg >= 337.5:
                        direction = "to the north"
                    elif angle_deg < 67.5:
                        direction = "to the northeast"
                    elif angle_deg < 112.5:
                        direction = "to the east"
                    elif angle_deg < 157.5:
                        direction = "to the southeast"
                    elif angle_deg < 202.5:
                        direction = "to the south"
                    elif angle_deg < 247.5:
                        direction = "to the southwest"
                    elif angle_deg < 292.5:
                        direction = "to the west"
                    else:
                        direction = "to the northwest"
            else:
                dist_from_friendly = None
                direction = "unknown direction"
            
            # Check for units on or near the feature
            friendly_on_feature = []
            enemy_on_feature = 0
            friendly_near_feature = []
            enemy_near_feature = 0
            
            for x, y in coords:
                unit = self.grid[y][x].unit
                if unit:
                    if unit.faction == faction:
                        friendly_on_feature.append(unit.name)
                    else:
                        enemy_on_feature += 1
            
            # Check adjacent hexes for nearby units
            adjacent_checked = set()
            for x, y in coords:
                for nx, ny in self.get_neighbors(x, y):
                    if (nx, ny) not in coords and (nx, ny) not in adjacent_checked:
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            nunit = self.grid[ny][nx].unit
                            if nunit:
                                if nunit.faction == faction:
                                    friendly_near_feature.append(nunit.name)
                                else:
                                    enemy_near_feature += 1
                            adjacent_checked.add((nx, ny))
            
            # Build feature description
            summary += f"\n  {feature_name}:\n"
            summary += f"    Type: {terrain_type}\n"
            summary += f"    Size: {feature_size} hexes\n"
            if dist_from_friendly is not None:
                summary += f"    Location: {dist_from_friendly} hexes {direction} from your forces\n"
            else:
                summary += f"    Location: {direction}\n"
            summary += f"    Center position: ({int(feature_x)}, {int(feature_y)})\n"
            
            # Report units present
            units_present_parts = []
            if friendly_on_feature:
                units_present_parts.append(f"Your units: {', '.join(friendly_on_feature)}")
            if enemy_on_feature > 0:
                units_present_parts.append(f"{enemy_on_feature} enemy unit{'s' if enemy_on_feature != 1 else ''}")
            
            if units_present_parts:
                summary += f"    Units present: {'; '.join(units_present_parts)}\n"
            
            # Report nearby units
            units_nearby_parts = []
            if friendly_near_feature:
                units_nearby_parts.append(f"Your units: {', '.join(friendly_near_feature)}")
            if enemy_near_feature > 0:
                units_nearby_parts.append(f"{enemy_near_feature} enemy unit{'s' if enemy_near_feature != 1 else ''}")
            
            if units_nearby_parts:
                summary += f"    Units nearby: {'; '.join(units_nearby_parts)}\n"
            
            if not units_present_parts and not units_nearby_parts:
                summary += f"    Status: Unoccupied\n"
            
            # Add minor features if any
            if overlapping_features:
                minor_list = ', '.join(sorted(overlapping_features))
                summary += f"    Minor features: {minor_list}\n"
        
        return summary

    def execute_orders(self, orders_data: Dict[str, Any], faction: str) -> None:
        """Execute orders from a general by routing to specific action handlers."""
        orders.execute_orders(self, orders_data, faction)
