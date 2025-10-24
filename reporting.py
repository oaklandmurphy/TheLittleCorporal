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
    def generate(self) -> str:
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

        return f"The map is a {self.w}x{self.h} hex grid. ".join(p.strip() for p in parts if p).strip()

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
        orient = f"running {c.orientation.replace('-', ' ')}" if c.orientation != "diagonal" else "cutting diagonally"
        loc = self._soften_center(c.location_phrase)
        return f"{core} {orient} across {loc}"

    def _hill_phrase(self, c: TerrainCluster) -> str:
        name = self._cluster_feature_name(c)
        scale = self._scale_word(c.size)
        loc = self._soften_center(c.location_phrase)
        orient = ""
        if c.orientation in ("east-west", "north-south"):
            orient = f" forming a {c.orientation.replace('-', ' ')} ridge"
        if name:
            # e.g., "the Monti di San Marco forming a ... in the north"
            return f"the {name}{orient} in {loc}"
        return f"{scale} hills{orient} in {loc}"

    def _forest_phrase(self, c: TerrainCluster) -> str:
        name = self._cluster_feature_name(c)
        scale = self._scale_word(c.size)
        loc = self._soften_center(c.location_phrase)
        if name:
            return f"the {name} in {loc}"
        return f"{scale} woods in {loc}"

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
            comp_str = self._composition_phrase(g.composition)
            loc = self._soften_center(g.location_phrase)
            near = ""
            if g.near_feature and g.relation_to_feature:
                near = f" {g.relation_to_feature} {self._feature_short(g.near_feature)}"
            phrases.append(f"{comp_str} {loc}{near}")
        if not phrases:
            return ""
        return f"{faction} forces are deployed with " + self._oxford_join(phrases) + "."

    def _contact_paragraph(self, groups: List[UnitGroup]) -> Optional[str]:
        # Simple proximity check: if any opposing groups are within ~3 hexes
        by_faction = defaultdict(list)
        for g in groups:
            by_faction[g.faction].append(g)
        factions = list(by_faction.keys())
        if len(factions) < 2:
            return None
        contacts = []
        for i in range(len(factions)):
            for j in range(i + 1, len(factions)):
                for a in by_faction[factions[i]]:
                    for b in by_faction[factions[j]]:
                        if self._dist(a.centroid, b.centroid) <= 3.5:
                            contacts.append((a, b))
        if not contacts:
            return None
        return "Opposing elements are in close proximity near the center and may soon engage."

    # -------- Text helpers --------
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
        total = self.w * self.h
        frac = size / max(1, total)
        if frac >= 0.18:
            return "extensive"
        if frac >= 0.10:
            return "large"
        if frac >= 0.05:
            return "notable"
        return "scattered"

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