"""
TerrainAnalyzer - Handles terrain-related calculations and analysis.

Responsibilities:
- Calculate directions and distances
- Analyze terrain features and paths
- Identify flanking positions
"""

from typing import List, Dict, Any, Optional, Tuple


class TerrainAnalyzer:
    """Provides terrain analysis capabilities for battlefield reconnaissance."""
    
    def __init__(self, game_map, faction: str):
        """Initialize the terrain analyzer.
        
        Args:
            game_map: The game map instance
            faction: The faction this analyzer serves
        """
        self.game_map = game_map
        self.faction = faction
    
    def get_cardinal_direction(self, from_x: int, from_y: int, to_x: int, to_y: int) -> str:
        """Calculate cardinal direction from one point to another on hex grid.
        
        Args:
            from_x, from_y: Origin coordinates
            to_x, to_y: Target coordinates
            
        Returns:
            Cardinal direction string (e.g., 'northern', 'southeastern')
        """
        dx = to_x - from_x
        dy = to_y - from_y
        
        # Normalize for hex grid (even-q offset coordinates)
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Determine primary and secondary directions
        primary = ""
        secondary = ""
        
        # Vertical direction (north/south)
        if abs_dy > abs_dx * 0.5:  # Primarily vertical
            if dy < 0:
                primary = "north"
            else:
                primary = "south"
            # Add east/west if significant horizontal component
            if abs_dx > abs_dy * 0.3:
                if dx > 0:
                    secondary = "east"
                else:
                    secondary = "west"
        else:  # Primarily horizontal
            if dx > 0:
                primary = "east"
            else:
                primary = "west"
            # Add north/south if significant vertical component
            if abs_dy > abs_dx * 0.3:
                if dy < 0:
                    secondary = "north"
                else:
                    secondary = "south"
        
        if secondary:
            return f"{secondary}{primary}"
        else:
            return primary
    
    def get_enemy_units_near_coords(self, coord_list: List[tuple], max_distance: int = 3) -> List[Dict[str, Any]]:
        """Get enemy units near a set of coordinates.
        
        Args:
            coord_list: List of (x, y) coordinates
            max_distance: Maximum hex distance to consider
            
        Returns:
            List of enemy unit dictionaries with position, stats, and distance
        """
        enemy_units = []
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                unit = self.game_map.grid[y][x].unit
                if unit and unit.faction != self.faction:
                    # Compute nearest distance to the feature
                    min_dist = float('inf')
                    for fx, fy in coord_list:
                        d = self.game_map._hex_distance(x, y, fx, fy)
                        if d < min_dist:
                            min_dist = d
                    
                    if min_dist <= max_distance:
                        enemy_units.append({
                            "name": unit.name,
                            "position": (x, y),
                            "size": unit.size,
                            "quality": unit.quality,
                            "morale": unit.morale,
                            "distance": min_dist,
                            "unit": unit
                        })
        return enemy_units
    
    def get_units_on_feature(self, coords: List[tuple]) -> List[Any]:
        """Get all units located on a terrain feature.
        
        Args:
            coords: List of (x, y) coordinates defining the feature
            
        Returns:
            List of unit objects on the feature
        """
        units_on = []
        for x, y in coords:
            u = self.game_map.grid[y][x].unit
            if u:
                units_on.append(u)
        return units_on
    
    def get_predominant_terrain(self, coords: List[tuple]) -> str:
        """Get the most common terrain type for a set of coordinates.
        
        Args:
            coords: List of (x, y) coordinates
            
        Returns:
            Name of predominant terrain type
        """
        terrain_count = {}
        for x, y in coords:
            tname = self.game_map.grid[y][x].terrain.name
            terrain_count[tname] = terrain_count.get(tname, 0) + 1
        return max(terrain_count, key=terrain_count.get)
    
    def build_path_to_target(self, start: tuple, target_coords: List[tuple], max_steps: int = 50) -> List[tuple]:
        """Build a path from start position to target feature.
        
        Args:
            start: Starting (x, y) position
            target_coords: List of (x, y) coordinates defining target
            max_steps: Maximum path length
            
        Returns:
            List of (x, y) coordinates forming the path
        """
        current_x, current_y = start
        target_x = sum(x for x, _ in target_coords) / len(target_coords)
        target_y = sum(y for _, y in target_coords) / len(target_coords)
        target_center = (int(target_x), int(target_y))
        
        path_hexes = []
        steps = 0
        visited = set()
        visited.add((current_x, current_y))
        
        while steps < max_steps and (current_x, current_y) not in target_coords:
            steps += 1
            best_neighbor = None
            best_distance = float('inf')
            
            # Find neighbor closest to target
            for nx, ny in self.game_map.get_neighbors(current_x, current_y):
                if 0 <= nx < self.game_map.width and 0 <= ny < self.game_map.height:
                    if (nx, ny) not in visited:
                        dist = self.game_map._hex_distance(nx, ny, target_center[0], target_center[1])
                        if dist < best_distance:
                            best_distance = dist
                            best_neighbor = (nx, ny)
            
            if best_neighbor is None:
                break
            
            current_x, current_y = best_neighbor
            visited.add((current_x, current_y))
            path_hexes.append((current_x, current_y))
        
        return path_hexes
    
    def get_all_features(self) -> Dict[str, List[tuple]]:
        """Get all terrain features and their coordinates from the map.
        
        Returns:
            Dict mapping feature names to coordinate lists
        """
        all_features = {}
        for y in range(self.game_map.height):
            for x in range(self.game_map.width):
                for feature in self.game_map.grid[y][x].features:
                    if feature not in all_features:
                        all_features[feature] = []
                    all_features[feature].append((x, y))
        return all_features
    
    def identify_flanking_features(self, all_features: Dict, features_along_path: set, 
                                   target_feature: str, path_hexes: List[tuple], 
                                   target_center: tuple) -> Dict[str, Dict[str, Any]]:
        """Identify features that could serve as flanking positions.
        
        Args:
            all_features: Dict mapping feature names to coordinate lists
            features_along_path: Set of features on the direct path
            target_feature: Name of target feature
            path_hexes: List of path coordinates
            target_center: Center coordinates of target
            
        Returns:
            Dict mapping feature names to their flanking information
        """
        flanking_features = {}
        
        for feature_name, feature_coords in all_features.items():
            if feature_name == target_feature or feature_name in features_along_path:
                continue
            
            # Calculate distance to path
            min_dist_to_path = float('inf')
            if not feature_coords:
                continue
            feature_center_x = sum(x for x, y in feature_coords) / len(feature_coords)
            feature_center_y = sum(y for x, y in feature_coords) / len(feature_coords)
            
            for px, py in path_hexes:
                dist = self.game_map._hex_distance(int(feature_center_x), int(feature_center_y), px, py)
                min_dist_to_path = min(min_dist_to_path, dist)
            
            # Consider as flanking feature if within 2 hexes of path
            if min_dist_to_path <= 2:
                # Use average path position for directionality if target_center is 0,0
                if target_center == (0, 0) and path_hexes:
                    path_center_x = sum(x for x, y in path_hexes) / len(path_hexes)
                    path_center_y = sum(y for x, y in path_hexes) / len(path_hexes)
                    direction_origin = (path_center_x, path_center_y)
                else:
                    direction_origin = target_center

                direction = self.get_cardinal_direction(
                    direction_origin[0], direction_origin[1],
                    int(feature_center_x), int(feature_center_y)
                )
                
                terrain_type = self.game_map.grid[int(feature_center_y)][int(feature_center_x)].terrain.name.lower()
                
                flanking_features[feature_name] = {
                    "direction": direction,
                    "distance": min_dist_to_path,
                    "terrain": terrain_type,
                    "size": len(feature_coords),
                    "coords": feature_coords
                }
        
        return flanking_features
