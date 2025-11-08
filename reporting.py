"""
Battlefield Report Generator

Generates tactical reports for LLM-based generals with:
- Terrain features
- Own unit positions with status
- Enemy positions
- Tactical situation assessment
"""

from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math


@dataclass
class TerrainFeature:
    """Represents a contiguous area of the same terrain type."""
    terrain_type: str
    tiles: List[Tuple[int, int]]
    size: int
    center: Tuple[float, float]
    bounds: Tuple[int, int, int, int]
    map_region: str
    name: Optional[str] = None


@dataclass
class UnitPosition:
    """Detailed information about a single unit's position."""
    unit: object
    name: str
    unit_type: str
    x: int
    y: int
    faction: str
    map_region: str
    
def generate_tactical_report(map, faction: str, unit_list: List = None) -> str:
    """Generate tactical report for a specific general's units.
    
    Args:
        map: The game map
        faction: The faction name (for identifying enemies)
        unit_list: List of units under this general's command. If None, includes all faction units.
    """
    # Analyze battlefield
    features = _find_terrain_features(map)
    major_features = _select_notable_features(map, features)
    unit_positions = _locate_all_units(map)

    # If unit_list is provided, filter to only those specific units
    if unit_list is not None:
        unit_ids = {id(unit) for unit in unit_list}
        friendly_units = [u for u in unit_positions if id(u.unit) in unit_ids]
        # Allied units are same faction but not under this general's command
        allied_units = [u for u in unit_positions if u.faction == faction and id(u.unit) not in unit_ids]
        # Enemy units are different faction
        enemy_units = [u for u in unit_positions if u.faction != faction]
    else:
        # Fallback to all units of the faction
        friendly_units = [u for u in unit_positions if u.faction == faction]
        allied_units = []
        enemy_units = [u for u in unit_positions if u.faction != faction]
    
    if not friendly_units:
        return f"No units under command found on the battlefield."
    
    # Build structured report
    lines = []
    lines.append(f"=== TACTICAL REPORT FOR {faction.upper()} FORCES ===\n")

    # Section 1: Terrain Overview
    lines.append("TERRAIN FEATURES:\n")

    if major_features:
        for feature in major_features:
            terrain_desc = _describe_feature_concise(feature)
            lines.append(f"  - {terrain_desc}")
    else:
        lines.append("  - Open terrain with no major features")
    lines.append("")

    # Section 2: Own Forces (detailed, unit by unit)
    lines.append(f"YOUR FORCES ({len(friendly_units)} units):")
    for unit_pos in friendly_units:
        lines.append(f"\n  [{unit_pos.name}]")
        lines.append(f"    Type: {unit_pos.unit_type}")

        # Add unit attributes (size, quality, morale)
        unit_status = _get_unit_attributes(unit_pos.unit)
        lines.append(f"    Status: {unit_status}")

        lines.append(f"    Position: {_format_region(unit_pos.map_region)}")

        # Local terrain
        local_terrain = _describe_local_terrain(unit_pos, major_features)
        if local_terrain:
            lines.append(f"    Terrain: {local_terrain}")

        # Nearby enemies
        nearby_enemies = _find_nearby_units(unit_pos, enemy_units, max_distance=5.0)
        if nearby_enemies:
            lines.append(f"    Nearby enemies:")
            for enemy, dist, direction in nearby_enemies:
                meters = dist * 200
                lines.append(f"      - {enemy.name}: {meters:.0f} meters {direction}")
        else:
            lines.append(f"    Nearby enemies: None within 1000 meters")

        # Key terrain features nearby
        nearby_features = _find_nearby_features(unit_pos, major_features, max_distance=4.0)
        if nearby_features:
            lines.append(f"    Key terrain nearby:")
            for feature, dist, direction in nearby_features:
                feature_name = feature.name if feature.name else feature.terrain_type
                meters = dist * 200
                lines.append(f"      - {feature_name}: {meters:.0f} meters {direction}")

    lines.append("")
    
    # Section 3: Allied Forces (friendly faction, not under command)
    if allied_units:
        lines.append(f"ALLIED FORCES ({len(allied_units)} units, not under your command):")
        by_region = defaultdict(list)
        for ally in allied_units:
            by_region[ally.map_region].append(ally)
        
        for region in sorted(by_region.keys()):
            units = by_region[region]
            lines.append(f"  {_format_region(region)}:")
            for u in units:
                lines.append(f"    - {u.name} ({u.unit_type})")
        
        lines.append("")
    
    # Section 4: Enemy Forces (summary with basic intel)
    lines.append(f"ENEMY FORCES ({len(enemy_units)} units):")
    if enemy_units:
        by_region = defaultdict(list)
        for enemy in enemy_units:
            by_region[enemy.map_region].append(enemy)
        
        for region in sorted(by_region.keys()):
            units = by_region[region]
            lines.append(f"  {_format_region(region)}:")
            for u in units:
                size_desc = _get_size_label(getattr(u.unit, 'size', 5))
                quality_desc = _get_quality_label(getattr(u.unit, 'quality', 3))
                lines.append(f"    - {u.name}: {size_desc}, {quality_desc} force")
    else:
        lines.append("  - No enemy units detected")
    
    lines.append("")
    
    # Section 5: Tactical Situation Summary
    lines.append("TACTICAL SITUATION:")
    situation = _assess_tactical_situation(friendly_units, enemy_units, major_features)
    for point in situation:
        lines.append(f"  - {point}")
    
    return "\n".join(lines)

# =====================================================
# TERRAIN ANALYSIS
# =====================================================

def _find_terrain_features(map) -> List[TerrainFeature]:
    """Identify all contiguous terrain features using flood fill."""
    visited = [[False] * map.width for _ in range(map.height)]
    features = []

    for y in range(map.height):
        for x in range(map.width):
            if visited[y][x]:
                continue

            terrain_type = map.grid[y][x].terrain.name
            tiles = _flood_fill(map, x, y, terrain_type, visited)

            if tiles:
                center = _calculate_center(tiles)
                bounds = _calculate_bounds(tiles)
                region = _map_region_from_coords(map, center[0], center[1])
                name = _extract_feature_name(map, tiles)

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

def _flood_fill(map, start_x: int, start_y: int, terrain_type: str, 
                visited: List[List[bool]]) -> List[Tuple[int, int]]:
    """Find all connected tiles of the same terrain type."""
    tiles = []
    queue = deque([(start_x, start_y)])
    visited[start_y][start_x] = True
    
    while queue:
        x, y = queue.popleft()
        tiles.append((x, y))

        for nx, ny in _get_neighbors(x, y):
            if (0 <= nx < map.width and 
                0 <= ny < map.height and 
                not visited[ny][nx]):
                
                if map.grid[ny][nx].terrain.name == terrain_type:
                    visited[ny][nx] = True
                    queue.append((nx, ny))
    
    return tiles

def _select_notable_features(map, features: List[TerrainFeature]) -> List[TerrainFeature]:
    """Filter to significant terrain features worth mentioning."""
    priority = {"River": 3, "Hill": 2, "Forest": 1}
    min_size = max(3, int(0.02 * map.width * map.height))

    notable = [f for f in features if f.size >= min_size]
    notable.sort(key=lambda f: (priority.get(f.terrain_type, 0), f.size), reverse=True)
    
    result = []
    counts = {"River": 0, "Hill": 0, "Forest": 0}
    max_per_type = {"River": 2, "Hill": 3, "Forest": 3}
    
    for feature in notable:
        terrain = feature.terrain_type
        if terrain in counts and counts[terrain] < max_per_type.get(terrain, 2):
            result.append(feature)
            counts[terrain] += 1
    
    return result

def _extract_feature_name(map, tiles: List[Tuple[int, int]]) -> Optional[str]:
    """Extract proper name from feature labels if present."""
    name_counts: Dict[str, int] = {}
    
    for x, y in tiles:
        try:
            features = getattr(map.grid[y][x], "features", []) or []
            for feature_name in features:
                name_counts[feature_name] = name_counts.get(feature_name, 0) + 1
        except Exception:
            continue
    
    if not name_counts:
        return None
    
    most_common = max(name_counts.items(), key=lambda x: x[1])
    if most_common[1] / len(tiles) >= 0.30:
        return most_common[0]
    
    return None

# =====================================================
# UNIT ANALYSIS
# =====================================================

def _locate_all_units(map) -> List[UnitPosition]:
    """Find and describe the position of every unit on the map."""
    unit_positions = []

    for y in range(map.height):
        for x in range(map.width):
            unit = map.grid[y][x].unit
            if unit is not None and hasattr(unit, 'x') and unit.x is not None:
                region = _map_region_from_coords(map, x, y)
                
                unit_positions.append(UnitPosition(
                    unit=unit,
                    name=getattr(unit, 'name', 'Unknown'),
                    unit_type=unit.__class__.__name__,
                    x=x,
                    y=y,
                    faction=getattr(unit, 'faction', 'Unknown'),
                    map_region=region
                ))
    
    return unit_positions

# =====================================================
# DESCRIPTION HELPERS
# =====================================================

def _describe_feature_concise(feature: TerrainFeature) -> str:
    """Concise description of a terrain feature for tactical report."""
    name = feature.name if feature.name else feature.terrain_type
    location = _format_region(feature.map_region)
    
    width = feature.bounds[2] - feature.bounds[0] + 1
    height = feature.bounds[3] - feature.bounds[1] + 1
    
    if width >= height * 1.5:
        orientation = "E-W"
    elif height >= width * 1.5:
        orientation = "N-S"
    else:
        orientation = "diagonal"
    
    return f"{name} ({feature.terrain_type}) in {location}, oriented {orientation}"

def _get_unit_attributes(unit) -> str:
    """Extract and format unit attributes (size, quality, morale)."""
    quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
    morale_labels = {range(0, 2): "broken", range(2, 4): "shaken", range(4, 7): "steady",
                        range(7, 9): "eager", range(9, 11): "fresh"}
    size_labels = {range(1, 4): "small", range(4, 7): "average-sized", 
                    range(7, 10): "large", range(10, 13): "very large"}
    
    def label_for(value, table):
        for key, label in table.items():
            if isinstance(key, range) and value in key:
                return label
            elif value == key:
                return label
        return "unknown"
    
    size = getattr(unit, 'size', 5)
    quality = getattr(unit, 'quality', 3)
    morale = getattr(unit, 'morale', 7)
    
    size_desc = label_for(size, size_labels)
    quality_desc = label_for(quality, quality_labels)
    morale_desc = label_for(morale, morale_labels)
    
    return f"{size_desc}, {quality_desc} formation; morale is {morale_desc}"

def _get_size_label(size: int) -> str:
    """Get size label for a unit."""
    size_labels = {range(1, 4): "small", range(4, 7): "average-sized", 
                    range(7, 10): "large", range(10, 13): "very large"}
    for key, label in size_labels.items():
        if isinstance(key, range) and size in key:
            return label
    return "average-sized"

def _get_quality_label(quality: int) -> str:
    """Get quality label for a unit."""
    quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
    return quality_labels.get(quality, "regular")

def _describe_local_terrain(unit_pos: UnitPosition, features: List[TerrainFeature]) -> str:
    """Describe terrain at and immediately around unit position."""
    for feature in features:
        if (unit_pos.x, unit_pos.y) in feature.tiles:
            feature_name = feature.name if feature.name else feature.terrain_type
            return f"On {feature_name}"
    
    return "On open ground"

def _find_nearby_units(unit_pos: UnitPosition, other_units: List[UnitPosition], 
                        max_distance: float) -> List[Tuple[UnitPosition, float, str]]:
    """Find units within range with distance and direction."""
    nearby = []
    
    for other in other_units:
        dist = _distance((unit_pos.x, unit_pos.y), (other.x, other.y))
        if dist <= max_distance:
            direction = _get_direction_simple(unit_pos.x, unit_pos.y, other.x, other.y)
            nearby.append((other, dist, direction))
    
    nearby.sort(key=lambda x: x[1])
    return nearby[:5]

def _find_nearby_features(unit_pos: UnitPosition, features: List[TerrainFeature],
                            max_distance: float) -> List[Tuple[TerrainFeature, float, str]]:
    """Find terrain features within range with distance and direction."""
    nearby = []
    
    for feature in features:
        dist = _distance((unit_pos.x, unit_pos.y), feature.center)
        if dist <= max_distance and (unit_pos.x, unit_pos.y) not in feature.tiles:
            direction = _get_direction_simple(unit_pos.x, unit_pos.y, 
                                                    int(feature.center[0]), int(feature.center[1]))
            nearby.append((feature, dist, direction))
    
    nearby.sort(key=lambda x: x[1])
    return nearby[:3]

def _assess_tactical_situation(friendly_units: List[UnitPosition], 
                                enemy_units: List[UnitPosition],
                                features: List[TerrainFeature]) -> List[str]:
    """Generate high-level tactical assessment points."""
    points = []
    
    if not enemy_units:
        return ["No enemy forces detected on the battlefield"]
    
    friendly_center = _calculate_center([(u.x, u.y) for u in friendly_units])
    enemy_center = _calculate_center([(u.x, u.y) for u in enemy_units])
    
    separation = _distance(friendly_center, enemy_center)
    
    if separation <= 2.0:
        points.append("Forces are in direct contact - combat is occurring")
    elif separation <= 4.0:
        points.append("Enemy forces are nearby - engagement imminent")
    elif separation <= 7.0:
        points.append("Enemy forces are in tactical range - opportunity to advance or prepare")
    else:
        points.append("Enemy forces are at distance - opportunity for maneuver")
    
    enemy_direction = _get_direction_simple(
        int(friendly_center[0]), int(friendly_center[1]),
        int(enemy_center[0]), int(enemy_center[1])
    )
    points.append(f"Enemy main body is {enemy_direction.replace('to the ', '')}")
    
    strength_ratio = len(friendly_units) / max(len(enemy_units), 1)
    if strength_ratio >= 1.5:
        points.append(f"You have numerical advantage ({len(friendly_units)} vs {len(enemy_units)} units)")
    elif strength_ratio <= 0.67:
        points.append(f"Enemy has numerical advantage ({len(enemy_units)} vs {len(friendly_units)} units)")
    else:
        points.append(f"Forces are roughly balanced ({len(friendly_units)} vs {len(enemy_units)} units)")
    
    return points

# =====================================================
# GEOMETRY UTILITIES
# =====================================================

def _get_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
    """Get hex neighbors using offset coordinates."""
    if y % 2 == 0:
        offsets = [(+1, 0), (-1, 0), (0, +1), (0, -1), (-1, +1), (-1, -1)]
    else:
        offsets = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (+1, -1)]
    
    return [(x + dx, y + dy) for dx, dy in offsets]

def _calculate_center(points: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Calculate geometric center of a set of points."""
    if not points:
        return (0.0, 0.0)
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    return (x_sum / len(points), y_sum / len(points))

def _calculate_bounds(points: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Calculate bounding box: (min_x, min_y, max_x, max_y)."""
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))

def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _get_direction_simple(from_x: int, from_y: int, to_x: int, to_y: int) -> str:
    """Get simple cardinal/intercardinal direction."""
    dx = to_x - from_x
    dy = to_y - from_y
    
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return "at same position"
    
    angle = math.atan2(dy, dx)
    degrees = math.degrees(angle)
    
    if degrees < 0:
        degrees += 360
    
    directions = ["to the East", "to the Southeast", "to the South", "to the Southwest",
                    "to the West", "to the Northwest", "to the North", "to the Northeast"]
    index = int((degrees + 22.5) / 45) % 8
    
    return directions[index]

def _map_region_from_coords(map, x: float, y: float) -> str:
    """Convert coordinates to compass region (e.g., 'the northwest')."""
    norm_x = x / max(1, map.width - 1) if map.width > 1 else 0.5
    norm_y = y / max(1, map.height - 1) if map.height > 1 else 0.5

    col = 0 if norm_x < 0.33 else (2 if norm_x > 0.67 else 1)
    row = 0 if norm_y < 0.33 else (2 if norm_y > 0.67 else 1)
    
    regions = {
        (0, 0): "the northwest", (1, 0): "the north", (2, 0): "the northeast",
        (0, 1): "the west",      (1, 1): "the center", (2, 1): "the east",
        (0, 2): "the southwest", (1, 2): "the south",  (2, 2): "the southeast",
    }
    return regions[(col, row)]

def _format_region(region: str) -> str:
    """Format region name without 'the' prefix for cleaner output."""
    return region.replace("the ", "").replace(" area", "")
