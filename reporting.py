"""
Battlefield Report Generator

Generates natural language summaries of battlefield situations including:
- Terrain features (rivers, hills, forests)  
- Unit positions and deployments
- Combat proximity and engagement status
"""

from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import math

try:
    from map import Map
except Exception:
    Map = object  # type: ignore


# =====================================================
# DATA STRUCTURES
# =====================================================

@dataclass
class TerrainFeature:
    """Represents a contiguous area of the same terrain type."""
    terrain_type: str  # e.g., "River", "Hill", "Forest"
    tiles: List[Tuple[int, int]]  # All (x, y) coordinates
    size: int  # Number of tiles
    center: Tuple[float, float]  # Geometric center
    bounds: Tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y)
    map_region: str  # e.g., "the northwest", "the center"
    name: Optional[str] = None  # Optional proper name (e.g., "Shiloh Creek")


@dataclass
class UnitPosition:
    """Detailed information about a single unit's position."""
    unit: object  # The unit object
    name: str
    unit_type: str
    x: int
    y: int
    faction: str
    map_region: str  # Cardinal direction phrase
    nearest_feature: Optional[TerrainFeature] = None
    distance_to_feature: Optional[float] = None


@dataclass
class Formation:
    """A group of nearby units from the same faction."""
    faction: str
    units: List[UnitPosition]
    center: Tuple[float, float]
    map_region: str


# =====================================================
# MAIN REPORT GENERATOR
# =====================================================

class ReportGenerator:
    """Generates natural language battlefield reports."""
    
    def __init__(self, game_map):
        self.map = game_map
        self.width = game_map.width
        self.height = game_map.height
    
    # =====================================================
    # PUBLIC API
    # =====================================================
    
    def generate_general_summary(self) -> str:
        """Generate a complete battlefield report.
        
        Returns:
            Natural language description of terrain and unit positions
        """
        # Analyze battlefield
        features = self._find_terrain_features()
        major_features = self._select_notable_features(features)
        unit_positions = self._locate_all_units()
        formations = self._group_into_formations(unit_positions)
        
        # Build report sections
        sections = []
        
        # Section 1: Terrain overview
        if major_features:
            sections.append(self._describe_terrain(major_features))
        
        # Section 2: Unit deployments by faction
        if formations:
            sections.append(self._describe_deployments(formations, unit_positions))
        
        # Section 3: Combat proximity
        if len({f.faction for f in formations}) > 1:
            proximity = self._describe_proximity(formations)
            if proximity:
                sections.append(proximity)
        
        return " ".join(s.strip() for s in sections if s)
    
    # =====================================================
    # TERRAIN ANALYSIS
    # =====================================================
    
    def _find_terrain_features(self) -> List[TerrainFeature]:
        """Identify all contiguous terrain features using flood fill."""
        visited = [[False] * self.width for _ in range(self.height)]
        features = []
        
        for y in range(self.height):
            for x in range(self.width):
                if visited[y][x]:
                    continue
                
                terrain_type = self.map.grid[y][x].terrain.name
                tiles = self._flood_fill(x, y, terrain_type, visited)
                
                if tiles:
                    center = self._calculate_center(tiles)
                    bounds = self._calculate_bounds(tiles)
                    region = self._map_region_from_coords(center[0], center[1])
                    name = self._extract_feature_name(tiles)
                    
                    features.append(TerrainFeature(
                        terrain_type=terrain_type,
                        tiles=tiles,
                        size=len(tiles),
                        center=center,
                        bounds=bounds,
                        map_region=region,
                        name=name
                    ))
        
        return features
    
    def _flood_fill(self, start_x: int, start_y: int, terrain_type: str, 
                    visited: List[List[bool]]) -> List[Tuple[int, int]]:
        """Find all connected tiles of the same terrain type."""
        tiles = []
        queue = deque([(start_x, start_y)])
        visited[start_y][start_x] = True
        
        while queue:
            x, y = queue.popleft()
            tiles.append((x, y))
            
            for nx, ny in self._get_neighbors(x, y):
                if (0 <= nx < self.width and 
                    0 <= ny < self.height and 
                    not visited[ny][nx]):
                    
                    if self.map.grid[ny][nx].terrain.name == terrain_type:
                        visited[ny][nx] = True
                        queue.append((nx, ny))
        
        return tiles
    
    def _select_notable_features(self, features: List[TerrainFeature]) -> List[TerrainFeature]:
        """Filter to significant terrain features worth mentioning.
        
        Prioritizes: Rivers > Hills > Forests
        Filters out tiny patches
        """
        # Define priority and minimum sizes
        priority = {"River": 3, "Hill": 2, "Forest": 1}
        min_size = max(3, int(0.02 * self.width * self.height))
        
        # Filter by size and sort by priority
        notable = [f for f in features if f.size >= min_size]
        notable.sort(key=lambda f: (priority.get(f.terrain_type, 0), f.size), reverse=True)
        
        # Limit to avoid overwhelming reports
        result = []
        counts = {"River": 0, "Hill": 0, "Forest": 0}
        max_per_type = {"River": 2, "Hill": 3, "Forest": 3}
        
        for feature in notable:
            terrain = feature.terrain_type
            if terrain in counts and counts[terrain] < max_per_type.get(terrain, 2):
                result.append(feature)
                counts[terrain] += 1
        
        return result
    
    def _describe_terrain(self, features: List[TerrainFeature]) -> str:
        """Generate natural language terrain description."""
        descriptions = []
        
        for feature in features:
            if feature.terrain_type == "River":
                descriptions.append(self._describe_river(feature))
            elif feature.terrain_type == "Hill":
                descriptions.append(self._describe_hill(feature))
            elif feature.terrain_type == "Forest":
                descriptions.append(self._describe_forest(feature))
        
        if not descriptions:
            return ""
        
        return "The battlefield features " + self._join_with_commas(descriptions) + "."
    
    def _describe_river(self, feature: TerrainFeature) -> str:
        """Describe a river feature with shape and orientation."""
        name = f"the {feature.name}" if feature.name else "a river"
        shape = self._describe_shape(feature)
        orientation = self._describe_orientation(feature)
        location = self._soften_region(feature.map_region)
        extent = self._describe_extent(feature)
        
        return f"{name} {shape} {orientation} through {location}{extent}"
    
    def _describe_hill(self, feature: TerrainFeature) -> str:
        """Describe hills with scale and formation."""
        name = f"the {feature.name}" if feature.name else None
        scale = self._scale_word(feature.size)
        shape = self._describe_shape(feature)
        location = self._soften_region(feature.map_region)
        orientation = self._describe_orientation(feature)
        
        if name:
            return f"{name}, {shape} high ground {orientation} in {location}"
        
        # Ridgeline or scattered hills
        width = feature.bounds[2] - feature.bounds[0] + 1
        height = feature.bounds[3] - feature.bounds[1] + 1
        if max(width, height) / min(width, height) >= 2:
            return f"{scale} {shape} ridgeline {orientation} in {location}"
        else:
            return f"{scale} {shape} hills in {location}"
    
    def _describe_forest(self, feature: TerrainFeature) -> str:
        """Describe wooded areas."""
        name = f"the {feature.name}" if feature.name else None
        scale = self._scale_word(feature.size)
        shape = self._describe_shape(feature)
        location = self._soften_region(feature.map_region)
        
        if name:
            return f"{name}, {shape} woodland in {location}"
        return f"{scale} {shape} woods in {location}"
    
    # =====================================================
    # UNIT ANALYSIS
    # =====================================================
    
    def _locate_all_units(self) -> List[UnitPosition]:
        """Find and describe the position of every unit on the map."""
        unit_positions = []
        features = self._select_notable_features(self._find_terrain_features())
        
        for y in range(self.height):
            for x in range(self.width):
                unit = self.map.grid[y][x].unit
                if unit is not None and hasattr(unit, 'x') and unit.x is not None:
                    region = self._map_region_from_coords(x, y)
                    nearest, distance = self._find_nearest_feature(x, y, features)
                    
                    unit_positions.append(UnitPosition(
                        unit=unit,
                        name=getattr(unit, 'name', 'Unknown'),
                        unit_type=unit.__class__.__name__,
                        x=x,
                        y=y,
                        faction=getattr(unit, 'faction', 'Unknown'),
                        map_region=region,
                        nearest_feature=nearest,
                        distance_to_feature=distance
                    ))
        
        return unit_positions
    
    def _group_into_formations(self, positions: List[UnitPosition]) -> List[Formation]:
        """Group nearby units of the same faction into formations."""
        if not positions:
            return []
        
        formations = []
        visited: Set[int] = set()
        
        for i, pos in enumerate(positions):
            if i in visited:
                continue
            
            # Find all units within proximity of this unit
            group = [pos]
            visited.add(i)
            queue = [i]
            
            while queue:
                curr_idx = queue.pop(0)
                curr_pos = positions[curr_idx]
                
                for j, other_pos in enumerate(positions):
                    if j not in visited and other_pos.faction == curr_pos.faction:
                        dist = self._distance(
                            (curr_pos.x, curr_pos.y),
                            (other_pos.x, other_pos.y)
                        )
                        if dist <= 3.0:  # Adjacent or nearby
                            group.append(other_pos)
                            visited.add(j)
                            queue.append(j)
            
            # Create formation
            center = self._calculate_center([(p.x, p.y) for p in group])
            region = self._map_region_from_coords(center[0], center[1])
            
            formations.append(Formation(
                faction=pos.faction,
                units=group,
                center=center,
                map_region=region
            ))
        
        return formations
    
    def _describe_deployments(self, formations: List[Formation], 
                             all_positions: List[UnitPosition]) -> str:
        """Describe how each faction has deployed its units."""
        by_faction: Dict[str, List[Formation]] = defaultdict(list)
        for formation in formations:
            by_faction[formation.faction].append(formation)
        
        faction_reports = []
        
        for faction, faction_formations in by_faction.items():
            # Get all units for this faction
            faction_units = [p for p in all_positions if p.faction == faction]
            
            # Describe each unit's position
            unit_descriptions = []
            for unit_pos in faction_units:
                unit_descriptions.append(self._describe_unit_position(unit_pos))
            
            if unit_descriptions:
                joined = self._join_with_commas(unit_descriptions)
                faction_reports.append(f"{faction} forces: {joined}")
        
        return ". ".join(faction_reports) + "." if faction_reports else ""
    
    def _describe_unit_position(self, pos: UnitPosition) -> str:
        """Generate detailed position description for a single unit.
        
        Example: "the 1st Virginia (Infantry) positioned in the northwest near the Stone Bridge"
        """
        # Unit identification
        unit_id = f"the {pos.name} ({pos.unit_type})"
        
        # Location with terrain context
        location_parts = [self._detailed_location(pos.x, pos.y, pos.map_region)]
        
        # Add terrain feature if relevant
        if pos.nearest_feature and pos.distance_to_feature is not None:
            if pos.distance_to_feature <= 0.5:  # On the feature
                feature_name = self._feature_name(pos.nearest_feature)
                location_parts.append(f"on {feature_name}")
            elif pos.distance_to_feature <= 2.0:  # Adjacent
                feature_name = self._feature_name(pos.nearest_feature)
                direction = self._relative_direction(
                    (pos.x, pos.y),
                    pos.nearest_feature.center
                )
                location_parts.append(f"{direction} {feature_name}")
        
        location = " ".join(location_parts)
        
        return f"{unit_id} positioned {location}"
    
    def _describe_proximity(self, formations: List[Formation]) -> str:
        """Describe opposing forces in close proximity."""
        # Find opposing formations near each other
        contacts = []
        
        for i, form_a in enumerate(formations):
            for form_b in formations[i+1:]:
                if form_a.faction != form_b.faction:
                    dist = self._distance(form_a.center, form_b.center)
                    if dist <= 5.0:
                        contacts.append((form_a, form_b, dist))
        
        if not contacts:
            return ""
        
        # Sort by distance (closest first)
        contacts.sort(key=lambda x: x[2])
        
        # Describe most significant contacts
        descriptions = []
        for form_a, form_b, dist in contacts[:3]:
            a_desc = self._formation_identifier(form_a)
            b_desc = self._formation_identifier(form_b)
            
            # Proximity descriptor
            if dist <= 1.5:
                proximity = "in direct contact with"
            elif dist <= 2.5:
                proximity = "engaging"
            elif dist <= 3.5:
                proximity = "approaching"
            else:
                proximity = "near"
            
            # Relative direction
            dx = form_b.center[0] - form_a.center[0]
            dy = form_b.center[1] - form_a.center[1]
            direction = self._direction_between_points(dx, dy)
            
            descriptions.append(f"{a_desc} {proximity} {b_desc}{direction}")
        
        if len(descriptions) == 1:
            return descriptions[0].capitalize() + "."
        return "Combat imminent: " + self._join_with_commas(descriptions) + "."
    
    # =====================================================
    # GEOMETRY UTILITIES
    # =====================================================
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get hex neighbors using offset coordinates."""
        if y % 2 == 0:  # Even row
            offsets = [(+1, 0), (-1, 0), (0, +1), (0, -1), (-1, +1), (-1, -1)]
        else:  # Odd row
            offsets = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (+1, -1)]
        
        neighbors = []
        for dx, dy in offsets:
            neighbors.append((x + dx, y + dy))
        return neighbors
    
    def _calculate_center(self, points: List[Tuple[int, int]]) -> Tuple[float, float]:
        """Calculate geometric center of a set of points."""
        if not points:
            return (0.0, 0.0)
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (x_sum / len(points), y_sum / len(points))
    
    def _calculate_bounds(self, points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Calculate bounding box: (min_x, min_y, max_x, max_y)."""
        if not points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    # =====================================================
    # DESCRIPTION HELPERS
    # =====================================================
    
    def _describe_shape(self, feature: TerrainFeature) -> str:
        """Describe the shape of a terrain feature."""
        width = feature.bounds[2] - feature.bounds[0] + 1
        height = feature.bounds[3] - feature.bounds[1] + 1
        ratio = max(width, height) / max(1, min(width, height))
        
        # Calculate compactness (how much it fills its bounding box)
        bbox_area = width * height
        compactness = feature.size / bbox_area if bbox_area > 0 else 0
        
        if ratio >= 3.0:
            return "winding" if feature.terrain_type == "River" else "elongated"
        elif ratio >= 2.0:
            return "flowing" if feature.terrain_type == "River" else "extended"
        elif compactness >= 0.7:
            return "compact"
        elif compactness >= 0.4:
            return "irregular"
        else:
            return "sprawling"
    
    def _describe_orientation(self, feature: TerrainFeature) -> str:
        """Describe how a feature is oriented."""
        width = feature.bounds[2] - feature.bounds[0] + 1
        height = feature.bounds[3] - feature.bounds[1] + 1
        
        if width >= height * 1.5:
            return "running east-west"
        elif height >= width * 1.5:
            return "running north-south"
        else:
            return "oriented diagonally"
    
    def _describe_extent(self, feature: TerrainFeature) -> str:
        """Describe how far a feature extends across the map."""
        width = feature.bounds[2] - feature.bounds[0] + 1
        height = feature.bounds[3] - feature.bounds[1] + 1
        max_extent = max(width, height)
        map_span = max(self.width, self.height)
        
        if max_extent >= map_span * 0.7:
            return ", spanning most of the battlefield"
        elif max_extent >= map_span * 0.5:
            return ", extending across much of the area"
        elif max_extent >= map_span * 0.3:
            return ", covering considerable distance"
        return ""
    
    def _scale_word(self, size: int) -> str:
        """Convert size to descriptive word."""
        total_tiles = self.width * self.height
        fraction = size / max(1, total_tiles)
        
        if size >= 50 or fraction >= 0.18:
            return "vast"
        elif size >= 30 or fraction >= 0.12:
            return "extensive"
        elif size >= 15 or fraction >= 0.08:
            return "substantial"
        elif size >= 8 or fraction >= 0.04:
            return "notable"
        else:
            return "small"
    
    def _map_region_from_coords(self, x: float, y: float) -> str:
        """Convert coordinates to compass region (e.g., 'the northwest')."""
        norm_x = x / max(1, self.width - 1) if self.width > 1 else 0.5
        norm_y = y / max(1, self.height - 1) if self.height > 1 else 0.5
        
        col = 0 if norm_x < 0.33 else (2 if norm_x > 0.67 else 1)
        row = 0 if norm_y < 0.33 else (2 if norm_y > 0.67 else 1)
        
        regions = {
            (0, 0): "the northwest", (1, 0): "the north", (2, 0): "the northeast",
            (0, 1): "the west",      (1, 1): "the center", (2, 1): "the east",
            (0, 2): "the southwest", (1, 2): "the south",  (2, 2): "the southeast",
        }
        return regions[(col, row)]
    
    def _detailed_location(self, x: int, y: int, base_region: str) -> str:
        """Provide detailed location including edge positions."""
        norm_x = x / max(1, self.width - 1) if self.width > 1 else 0.5
        norm_y = y / max(1, self.height - 1) if self.height > 1 else 0.5
        
        modifiers = []
        if norm_x < 0.15:
            modifiers.append("western edge of")
        elif norm_x > 0.85:
            modifiers.append("eastern edge of")
        
        if norm_y < 0.15:
            modifiers.append("northern edge of")
        elif norm_y > 0.85:
            modifiers.append("southern edge of")
        
        base = self._soften_region(base_region)
        
        if modifiers:
            return f"at the {' '.join(modifiers)} {base}"
        return f"in {base}"
    
    def _soften_region(self, region: str) -> str:
        """Make 'the center' less repetitive."""
        return "the central area" if region == "the center" else region
    
    def _find_nearest_feature(self, x: int, y: int, 
                              features: List[TerrainFeature]) -> Tuple[Optional[TerrainFeature], Optional[float]]:
        """Find the nearest significant terrain feature to a position."""
        if not features:
            return None, None
        
        # Check if on a feature
        for feature in features:
            if (x, y) in feature.tiles:
                return feature, 0.0
        
        # Find closest feature by center
        closest = None
        min_dist = float('inf')
        
        for feature in features:
            dist = self._distance((x, y), feature.center)
            if dist < min_dist:
                min_dist = dist
                closest = feature
        
        return closest, min_dist
    
    def _relative_direction(self, from_pos: Tuple[int, int], 
                           to_pos: Tuple[float, float]) -> str:
        """Describe direction from one position to another."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > abs(dy) * 1.5:
            return "to the east of" if dx > 0 else "to the west of"
        elif abs(dy) > abs(dx) * 1.5:
            return "to the south of" if dy > 0 else "to the north of"
        else:
            # Diagonal
            if dx > 0 and dy > 0:
                return "to the southeast of"
            elif dx > 0 and dy < 0:
                return "to the northeast of"
            elif dx < 0 and dy > 0:
                return "to the southwest of"
            else:
                return "to the northwest of"
    
    def _direction_between_points(self, dx: float, dy: float) -> str:
        """Describe relative direction given deltas."""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return ""
        
        if abs(dx) > abs(dy) * 2:
            direction = " from the west" if dx > 0 else " from the east"
        elif abs(dy) > abs(dx) * 2:
            direction = " from the north" if dy > 0 else " from the south"
        else:
            if dx > 0 and dy > 0:
                direction = " from the northwest"
            elif dx > 0 and dy < 0:
                direction = " from the southwest"
            elif dx < 0 and dy > 0:
                direction = " from the northeast"
            else:
                direction = " from the southeast"
        
        return direction
    
    def _feature_name(self, feature: TerrainFeature) -> str:
        """Get a short reference name for a feature."""
        if feature.name:
            return f"the {feature.name}"
        
        type_names = {
            "River": "the river",
            "Hill": "the high ground",
            "Forest": "the woods"
        }
        return type_names.get(feature.terrain_type, "the terrain")
    
    def _formation_identifier(self, formation: Formation) -> str:
        """Create a short identifier for a formation."""
        if len(formation.units) == 1:
            unit = formation.units[0]
            return f"the {unit.name}"
        else:
            # Multiple units
            count = len(formation.units)
            if count == 2:
                return f"{formation.faction}'s two units"
            elif count <= 5:
                return f"{formation.faction}'s {self._number_word(count)} units"
            else:
                return f"{formation.faction}'s formation"
    
    def _number_word(self, n: int) -> str:
        """Convert small numbers to words."""
        words = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        return words[n] if 0 <= n < len(words) else str(n)
    
    def _extract_feature_name(self, tiles: List[Tuple[int, int]]) -> Optional[str]:
        """Extract proper name from feature labels if present."""
        name_counts: Dict[str, int] = {}
        
        for x, y in tiles:
            try:
                features = getattr(self.map.grid[y][x], "features", []) or []
                for feature_name in features:
                    name_counts[feature_name] = name_counts.get(feature_name, 0) + 1
            except Exception:
                continue
        
        if not name_counts:
            return None
        
        # Return most common name if it covers at least 30% of tiles
        most_common = max(name_counts.items(), key=lambda x: x[1])
        if most_common[1] / len(tiles) >= 0.30:
            return most_common[0]
        
        return None
    
    def _join_with_commas(self, items: List[str]) -> str:
        """Join list with Oxford comma."""
        items = [s for s in items if s]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"
