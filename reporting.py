from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

# Light type hints only; avoids import cycles
try:
    from map import Map
except Exception:
    Map = object  # type: ignore

# ------- Helper data structures -------
@dataclass
class TerrainCluster:
    terrain_name: str
    tiles: List[Tuple[int, int]]
    size: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # (minx, miny, maxx, maxy)
    orientation: str  # "east-west", "north-south", "diagonal"
    location_phrase: str  # "the north", "the center", etc.

@dataclass
class UnitGroup:
    faction: str
    members: List[Tuple[object, int, int]]  # (unit, x, y)
    centroid: Tuple[float, float]
    location_phrase: str
    composition: Dict[str, int]
    near_feature: Optional[TerrainCluster]
    relation_to_feature: Optional[str]  # "on", "near", None


class ReportGenerator:
    def __init__(self, game_map: Map):
        self.map = game_map
        self.w = game_map.width
        self.h = game_map.height

    # -------- Public API --------
    def generate_general_summary(self) -> str:
        features = self._find_terrain_clusters()
        major_features = self._select_major_features(features)
        unit_groups = self._group_units()

        parts: List[str] = []
        # Terrain overview
        if major_features:
            parts.append(self._terrain_paragraph(major_features))

        # Per-faction deployments
        by_faction: Dict[str, List[UnitGroup]] = defaultdict(list)
        for g in unit_groups:
            by_faction[g.faction].append(g)

        for faction, groups in by_faction.items():
            if groups:
                parts.append(self._faction_paragraph(faction, groups))

        # Optional proximity/contact note
        contact_line = self._contact_paragraph(unit_groups)
        if contact_line:
            parts.append(contact_line)

        return " ".join(p.strip() for p in parts if p).strip()

    # -------- Terrain analysis --------
    def _find_terrain_clusters(self) -> List[TerrainCluster]:
        seen = [[False] * self.w for _ in range(self.h)]
        clusters: List[TerrainCluster] = []

        for y in range(self.h):
            for x in range(self.w):
                if seen[y][x]:
                    continue
                tile = self.map.grid[y][x]
                tname = getattr(tile.terrain, "name", "Unknown")

                # BFS flood fill by terrain name
                q = deque([(x, y)])
                seen[y][x] = True
                comp: List[Tuple[int, int]] = []
                while q:
                    cx, cy = q.popleft()
                    comp.append((cx, cy))
                    for nx, ny in self._neighbors(cx, cy):
                        if 0 <= nx < self.w and 0 <= ny < self.h and not seen[ny][nx]:
                            if getattr(self.map.grid[ny][nx].terrain, "name", "Unknown") == tname:
                                seen[ny][nx] = True
                                q.append((nx, ny))

                # Build cluster summary
                size = len(comp)
                cx, cy = self._centroid(comp)
                minx = min(p[0] for p in comp)
                maxx = max(p[0] for p in comp)
                miny = min(p[1] for p in comp)
                maxy = max(p[1] for p in comp)
                bbox = (minx, miny, maxx, maxy)
                orientation = self._orientation(minx, miny, maxx, maxy)
                loc = self._direction_phrase(cx, cy)
                clusters.append(TerrainCluster(tname, comp, size, (cx, cy), bbox, orientation, loc))
        return clusters

    def _select_major_features(self, clusters: List[TerrainCluster]) -> List[TerrainCluster]:
        # Emphasize Rivers/Hills/Forests; ignore tiny patches
        priority = {"River": 3, "Hill": 2, "Forest": 1, "Fields": 0}
        filtered = [c for c in clusters if c.size >= max(3, int(0.02 * self.w * self.h))]
        filtered.sort(key=lambda c: (priority.get(c.terrain_name, 0), c.size), reverse=True)
        # Take top N per terrain type to keep reports concise
        top_by_type: Dict[str, List[TerrainCluster]] = defaultdict(list)
        for c in filtered:
            if len(top_by_type[c.terrain_name]) < (2 if c.terrain_name == "River" else 3):
                top_by_type[c.terrain_name].append(c)
        result: List[TerrainCluster] = []
        for t in ["River", "Hill", "Forest"]:
            result.extend(top_by_type.get(t, []))
        return result

    def _terrain_paragraph(self, features: List[TerrainCluster]) -> str:
        phrases: List[str] = []
        for c in features:
            if c.terrain_name == "River":
                phrases.append(self._river_phrase(c))
            elif c.terrain_name == "Hill":
                phrases.append(self._hill_phrase(c))
            elif c.terrain_name == "Forest":
                phrases.append(self._forest_phrase(c))
        if not phrases:
            return ""
        return "The ground features " + self._oxford_join(phrases) + "."

    def _river_phrase(self, c: TerrainCluster) -> str:
        name = self._cluster_feature_name(c)
        core = f"the {name}" if name else "a river"
        
        # Describe shape and orientation
        shape = self._describe_shape(c)
        orient_desc = self._describe_orientation_detailed(c)
        loc = self._soften_center(c.location_phrase)
        extent = self._describe_extent(c)
        
        return f"{core} {shape} {orient_desc} across {loc}{extent}"

    def _hill_phrase(self, c: TerrainCluster) -> str:
        name = self._cluster_feature_name(c)
        scale = self._scale_word(c.size)
        loc = self._soften_center(c.location_phrase)
        shape = self._describe_shape(c)
        orient = self._describe_orientation_detailed(c)
        
        if name:
            return f"the {name}, {shape} highland {orient} in {loc}"
        
        # Determine if it's a single hill or ridge
        if c.orientation in ("east-west", "north-south"):
            return f"{scale} {shape} ridge {orient} in {loc}"
        else:
            return f"{scale} {shape} hills in {loc}"

    def _forest_phrase(self, c: TerrainCluster) -> str:
        name = self._cluster_feature_name(c)
        scale = self._scale_word(c.size)
        loc = self._soften_center(c.location_phrase)
        shape = self._describe_shape(c)
        
        if name:
            return f"the {name}, {shape} woodland in {loc}"
        return f"{scale} {shape} woods in {loc}"

    # -------- Unit analysis --------
    def _group_units(self) -> List[UnitGroup]:
        occupied = {(x, y): self.map.grid[y][x].unit
                    for y in range(self.h) for x in range(self.w)
                    if self.map.grid[y][x].unit is not None and getattr(self.map.grid[y][x].unit, "x", None) is not None}

        visited: set[Tuple[int, int]] = set()
        groups: List[UnitGroup] = []
        features = self._select_major_features(self._find_terrain_clusters())

        for (x, y), unit in occupied.items():
            if (x, y) in visited:
                continue
            faction = getattr(unit, "faction", "Unknown")
            queue = deque([(x, y)])
            visited.add((x, y))
            members: List[Tuple[object, int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cu = self.map.grid[cy][cx].unit
                if cu and getattr(cu, "faction", None) == faction:
                    members.append((cu, cx, cy))
                    for nx, ny in self._neighbors(cx, cy):
                        if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in visited:
                            nu = self.map.grid[ny][nx].unit
                            if nu and getattr(nu, "faction", None) == faction:
                                visited.add((nx, ny))
                                queue.append((nx, ny))

            if not members:
                continue
            cx, cy = self._centroid([(mx, my) for _, mx, my in members])
            comp = defaultdict(int)
            for u, _, _ in members:
                comp[getattr(u, "__class__", type(u)).__name__] += 1
            loc_phrase = self._direction_phrase(cx, cy)

            near_feat, relation = self._nearest_feature((cx, cy), features, members)
            groups.append(UnitGroup(faction, members, (cx, cy), loc_phrase, dict(comp), near_feat, relation))
        return groups

    def _nearest_feature(self, centroid: Tuple[float, float], features: List[TerrainCluster],
                         members: List[Tuple[object, int, int]]) -> Tuple[Optional[TerrainCluster], Optional[str]]:
        # If any member stands on the feature tiles, relation = "on"
        member_set = {(x, y) for _, x, y in members}
        closest: Optional[TerrainCluster] = None
        best_d = 1e9
        for f in features:
            tiles = set(f.tiles)
            if member_set & tiles:
                return f, "on"
            d = self._dist(centroid, f.centroid)
            if d < best_d:
                best_d, closest = d, f
        if closest is None:
            return None, None
        if best_d <= 3:
            return closest, "near"
        if best_d <= 6:
            return closest, "in the vicinity of"
        return closest, None

    def _faction_paragraph(self, faction: str, groups: List[UnitGroup]) -> str:
        phrases: List[str] = []
        for g in groups:
            # Detailed unit listing
            unit_details = self._describe_unit_group_detailed(g)
            loc = self._detailed_location(g.centroid, g.location_phrase)
            
            # Describe position relative to features
            near = ""
            if g.near_feature and g.relation_to_feature:
                near = f" {self._describe_position_relative_to_feature(g, g.near_feature, g.relation_to_feature)}"
            
            # Formation description
            formation = self._describe_formation(g)
            
            phrases.append(f"{unit_details} {formation} {loc}{near}")
        
        if not phrases:
            return ""
        return f"{faction} forces are deployed with " + self._oxford_join(phrases) + "."

    def _contact_paragraph(self, groups: List[UnitGroup]) -> Optional[str]:
        """Provide detailed information about opposing forces in proximity."""
        by_faction = defaultdict(list)
        for g in groups:
            by_faction[g.faction].append(g)
        factions = list(by_faction.keys())
        if len(factions) < 2:
            return None
        
        # Find specific opposing groups in proximity
        contacts = []
        for i in range(len(factions)):
            for j in range(i + 1, len(factions)):
                for a in by_faction[factions[i]]:
                    for b in by_faction[factions[j]]:
                        dist = self._dist(a.centroid, b.centroid)
                        if dist <= 5.0:  # Increased range for reporting
                            contacts.append((a, b, dist))
        
        if not contacts:
            return None
        
        # Sort by distance (closest first)
        contacts.sort(key=lambda x: x[2])
        
        # Describe the most significant contacts
        descriptions = []
        for a, b, dist in contacts[:3]:  # Report up to 3 closest contacts
            a_desc = self._group_short_name(a)
            b_desc = self._group_short_name(b)
            
            if dist <= 1.5:
                proximity = "in immediate contact with"
            elif dist <= 2.5:
                proximity = "engaged with"
            elif dist <= 3.5:
                proximity = "closing on"
            else:
                proximity = "approaching"
            
            # Add relative direction
            dx = b.centroid[0] - a.centroid[0]
            dy = b.centroid[1] - a.centroid[1]
            direction = self._relative_direction(dx, dy)
            
            descriptions.append(f"{a_desc} {proximity} {b_desc} {direction}")
        
        if len(descriptions) == 1:
            return descriptions[0].capitalize() + "."
        else:
            return "Combat is imminent: " + self._oxford_join(descriptions) + "."
    
    def _group_short_name(self, g: UnitGroup) -> str:
        """Generate a short identifier for a unit group."""
        if len(g.members) == 1:
            unit = g.members[0][0]
            name = getattr(unit, "name", "Unknown")
            return f"the {name}"
        else:
            # Use dominant type
            types = list(g.composition.keys())
            if len(types) == 1:
                count = sum(g.composition.values())
                unit_type = types[0].lower()
                if count == 2:
                    return f"two {unit_type} units"
                elif count <= 5:
                    return f"{self._count_word(count)} {unit_type} units"
                else:
                    return f"a {unit_type} formation"
            else:
                return f"a {g.faction} formation"
    
    def _relative_direction(self, dx: float, dy: float) -> str:
        """Describe relative direction between two points."""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return ""
        
        if abs(dx) > abs(dy) * 2:
            return "from the west" if dx > 0 else "from the east"
        elif abs(dy) > abs(dx) * 2:
            return "from the north" if dy > 0 else "from the south"
        else:
            if dx > 0 and dy > 0:
                return "from the northwest"
            elif dx > 0 and dy < 0:
                return "from the southwest"
            elif dx < 0 and dy > 0:
                return "from the northeast"
            else:
                return "from the southeast"

    # -------- Text helpers --------
    def _describe_shape(self, c: TerrainCluster) -> str:
        """Describe the shape of a terrain cluster in natural language."""
        dx = c.bbox[2] - c.bbox[0] + 1
        dy = c.bbox[3] - c.bbox[1] + 1
        ratio = max(dx, dy) / max(1, min(dx, dy))
        
        # Measure compactness - how many tiles vs bounding box
        bbox_area = dx * dy
        compactness = c.size / max(1, bbox_area)
        
        if ratio >= 3.0:
            if c.terrain_name == "River":
                return "winding"
            return "elongated"
        elif ratio >= 2.0:
            if c.terrain_name == "River":
                return "flowing"
            return "extended"
        elif compactness >= 0.7:
            return "compact"
        elif compactness >= 0.4:
            return "irregular"
        else:
            return "sprawling"
    
    def _describe_orientation_detailed(self, c: TerrainCluster) -> str:
        """Provide detailed orientation description."""
        if c.orientation == "east-west":
            return "running east to west"
        elif c.orientation == "north-south":
            return "running north to south"
        elif c.orientation == "diagonal":
            dx = c.bbox[2] - c.bbox[0]
            dy = c.bbox[3] - c.bbox[1]
            # Determine diagonal direction
            if dx > 0 and dy > 0:
                return "trending from northwest to southeast"
            else:
                return "trending from northeast to southwest"
        return "oriented irregularly"
    
    def _describe_extent(self, c: TerrainCluster) -> str:
        """Describe how far the feature extends."""
        dx = c.bbox[2] - c.bbox[0] + 1
        dy = c.bbox[3] - c.bbox[1] + 1
        map_span = max(self.w, self.h)
        
        max_extent = max(dx, dy)
        if max_extent >= map_span * 0.7:
            return ", spanning nearly the entire battlefield"
        elif max_extent >= map_span * 0.5:
            return ", extending across much of the region"
        elif max_extent >= map_span * 0.3:
            return ", covering a considerable distance"
        return ""
    
    def _describe_unit_group_detailed(self, g: UnitGroup) -> str:
        """List all units in a group by name and type."""
        if len(g.members) == 1:
            unit = g.members[0][0]
            name = getattr(unit, "name", "Unknown")
            unit_type = getattr(unit, "__class__", type(unit)).__name__.lower()
            return f"the {name} ({unit_type})"
        
        # Multiple units - group by type and list names
        by_type: Dict[str, List[str]] = defaultdict(list)
        for unit, _, _ in g.members:
            name = getattr(unit, "name", "Unknown")
            unit_type = getattr(unit, "__class__", type(unit)).__name__.lower()
            by_type[unit_type].append(name)
        
        parts = []
        for unit_type, names in sorted(by_type.items()):
            if len(names) == 1:
                parts.append(f"the {names[0]} ({unit_type})")
            else:
                name_list = self._oxford_join([f"the {n}" for n in names])
                parts.append(f"{name_list} ({unit_type} units)")
        
        return self._oxford_join(parts)
    
    def _describe_formation(self, g: UnitGroup) -> str:
        """Describe how units are arranged."""
        if len(g.members) == 1:
            return "positioned"
        
        # Calculate spread
        positions = [(x, y) for _, x, y in g.members]
        if len(positions) < 2:
            return "positioned"
        
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distances.append(self._dist(positions[i], positions[j]))
        
        avg_dist = sum(distances) / len(distances)
        max_dist = max(distances)
        
        if max_dist <= 1.5:
            return "massed together"
        elif max_dist <= 3.0:
            return "in close formation"
        elif avg_dist >= 4.0:
            return "dispersed"
        else:
            return "deployed"
    
    def _detailed_location(self, centroid: Tuple[float, float], base_phrase: str) -> str:
        """Provide more granular location within the region."""
        nx = 0.0 if self.w <= 1 else centroid[0] / (self.w - 1)
        ny = 0.0 if self.h <= 1 else centroid[1] / (self.h - 1)
        
        # Add refinement to base direction
        base = self._soften_center(base_phrase)
        
        # Check if near edges
        modifiers = []
        if nx < 0.15:
            modifiers.append("western edge of")
        elif nx > 0.85:
            modifiers.append("eastern edge of")
        
        if ny < 0.15:
            modifiers.append("northern edge of")
        elif ny > 0.85:
            modifiers.append("southern edge of")
        
        if modifiers:
            return f"at the {' '.join(modifiers)} {base}"
        
        return f"in {base}"
    
    def _describe_position_relative_to_feature(self, g: UnitGroup, feature: TerrainCluster, relation: str) -> str:
        """Describe detailed position relative to a terrain feature."""
        feature_name = self._feature_short(feature)
        
        if relation == "on":
            # Determine which part of the feature
            member_positions = [(x, y) for _, x, y in g.members]
            feature_center = feature.centroid
            
            # Calculate relative position within feature
            avg_x = sum(p[0] for p in member_positions) / len(member_positions)
            avg_y = sum(p[1] for p in member_positions) / len(member_positions)
            
            dx = avg_x - feature_center[0]
            dy = avg_y - feature_center[1]
            
            # Determine which part
            if abs(dx) > abs(dy) * 1.5:
                if dx > 0:
                    part = "eastern side of"
                else:
                    part = "western side of"
            elif abs(dy) > abs(dx) * 1.5:
                if dy > 0:
                    part = "southern reaches of"
                else:
                    part = "northern reaches of"
            else:
                part = "occupying"
            
            return f"{part} {feature_name}"
        
        elif relation == "near":
            # Determine direction to feature
            cx, cy = g.centroid
            fx, fy = feature.centroid
            dx = fx - cx
            dy = fy - cy
            
            if abs(dx) > abs(dy) * 1.5:
                direction = "to the east of" if dx > 0 else "to the west of"
            elif abs(dy) > abs(dx) * 1.5:
                direction = "to the south of" if dy > 0 else "to the north of"
            else:
                if dx > 0 and dy > 0:
                    direction = "to the southeast of"
                elif dx > 0 and dy < 0:
                    direction = "to the northeast of"
                elif dx < 0 and dy > 0:
                    direction = "to the southwest of"
                else:
                    direction = "to the northwest of"
            
            return f"{direction} {feature_name}"
        
        elif relation == "in the vicinity of":
            return f"near the general area of {feature_name}"
        
        return f"{relation} {feature_name}"

    def _composition_phrase(self, comp: Dict[str, int]) -> str:
        # Example: "two infantry and a cavalry", "artillery and infantry"
        names = []
        for cls, n in sorted(comp.items(), key=lambda kv: (-kv[1], kv[0])):
            word = self._count_word(n)
            names.append(f"{word} {cls.lower() if cls.isalpha() else cls}")
        return self._oxford_join(names)

    def _feature_short(self, c: TerrainCluster) -> str:
        cname = self._cluster_feature_name(c)
        if cname:
            return f"the {cname}"
        if c.terrain_name == "River":
            return "the river"
        if c.terrain_name == "Hill":
            return "the high ground"
        if c.terrain_name == "Forest":
            return "the woods"
        return "the terrain"

    # -------- Feature name harvesting --------
    def _cluster_feature_name(self, c: TerrainCluster) -> Optional[str]:
        """Return the most common named feature label across tiles in this cluster, if any.

        Requires that the map's hexes have a `features: List[str]` containing labels (e.g., from Map.label_terrain_features).
        Uses a soft threshold so that mixed clusters still pick a dominant name.
        """
        counts: Dict[str, int] = {}
        for (x, y) in c.tiles:
            try:
                feats = getattr(self.map.grid[y][x], "features", []) or []
            except Exception:
                feats = []
            for f in feats:
                counts[f] = counts.get(f, 0) + 1
        if not counts:
            return None
        # pick the most frequent; require at least 30% coverage to avoid spurious small overlaps
        total = len(c.tiles)
        name, n = max(counts.items(), key=lambda kv: kv[1])
        if n / max(1, total) >= 0.30:
            return name
        return None

    def _count_word(self, n: int) -> str:
        return ["a", "two", "three", "four", "five"][n-1] if 1 <= n <= 5 else f"{n}"

    def _scale_word(self, size: int) -> str:
        """Describe the scale/size of a terrain feature in natural language."""
        total = self.w * self.h
        frac = size / max(1, total)
        
        # Also consider absolute size
        if size >= 50:
            if frac >= 0.18:
                return "vast"
            return "extensive"
        elif size >= 30:
            if frac >= 0.18:
                return "extensive"
            return "large"
        elif size >= 15:
            if frac >= 0.10:
                return "large"
            return "substantial"
        elif size >= 8:
            if frac >= 0.05:
                return "notable"
            return "moderate"
        else:
            return "small"

    def _soften_center(self, phrase: str) -> str:
        return "the central area" if phrase == "the center" else phrase

    def _oxford_join(self, items: List[str]) -> str:
        items = [s for s in items if s]
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    # -------- Geometry helpers --------
    def _neighbors(self, x: int, y: int):
        # Match your map's offset layout (even-r or odd-r); here uses same as Map.get_neighbors
        offsets_even = [(+1, 0), (-1, 0), (0, +1), (0, -1), (-1, +1), (-1, -1)]
        offsets_odd = [(+1, 0), (-1, 0), (0, +1), (0, -1), (+1, +1), (+1, -1)]
        offsets = offsets_odd if y % 2 else offsets_even
        for dx, dy in offsets:
            yield x + dx, y + dy

    def _centroid(self, pts: List[Tuple[int, int]]) -> Tuple[float, float]:
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        n = max(1, len(pts))
        return (sx / n, sy / n)

    def _orientation(self, minx: int, miny: int, maxx: int, maxy: int) -> str:
        dx = maxx - minx + 1
        dy = maxy - miny + 1
        if dx >= dy * 1.5:
            return "east-west"
        if dy >= dx * 1.5:
            return "north-south"
        return "diagonal"

    def _direction_phrase(self, x: float, y: float) -> str:
        # Map to 3x3 grid of compass directions
        nx = 0.0 if self.w <= 1 else x / (self.w - 1)
        ny = 0.0 if self.h <= 1 else y / (self.h - 1)
        col = 0 if nx < 0.33 else (2 if nx > 0.67 else 1)
        row = 0 if ny < 0.33 else (2 if ny > 0.67 else 1)
        grid = {
            (0, 0): "the northwest", (1, 0): "the north", (2, 0): "the northeast",
            (0, 1): "the west",      (1, 1): "the center", (2, 1): "the east",
            (0, 2): "the southwest", (1, 2): "the south",  (2, 2): "the southeast",
        }
        return grid[(col, row)]

    def _dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        return math.hypot(ax - bx, ay - by)