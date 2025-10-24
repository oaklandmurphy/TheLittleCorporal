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
            
            # Mark current hex as reachable only if it's the start position or unoccupied
            current_tile = self.grid[y][x]
            if (x, y) == start or current_tile.unit is None:
                reachable[(x, y)] = current_cost

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
        def is_hill(x, y):
            return self.grid[y][x].terrain.name == "Hill"

        def is_forest(x, y):
            return self.grid[y][x].terrain.name == "Forest"

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
            "Adda", "Po", "Ticino", "Mincio", "Brenta", "Isonzo", "Adige", "Arno", "Liri",
            "Sesia", "Oglio", "Tanaro", "Taro", "Nera", "Volturno", "Piave", "Trebbia", "Chiese",
            "Ofanto", "Bormida"
        ]
        saints = [
            "San Marco", "San Pietro", "San Giorgio", "San Luca", "San Paolo", "Santa Maria",
            "Sant'Anna", "San Michele", "San Carlo", "San Giovanni"
        ]
        adjs = ["Grande", "Piccolo", "Alto", "Basso", "Lungo", "Nero", "Verde", "Rosso"]

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
                return unique(f"Colle {random.choice(roots)}")
            elif choice == 1:
                return unique(f"Monti di {random.choice(saints)}")
            else:
                return unique(f"Alture {random.choice(adjs)}")

        def name_forest() -> str:
            choice = random.randint(0, 2)
            if choice == 0:
                return unique(f"Bosco di {random.choice(saints)}")
            elif choice == 1:
                return unique(f"Selva {random.choice(roots)}")
            else:
                return unique(f"Foresta {random.choice(adjs)}")

        def name_river() -> str:
            if random.random() < 0.5:
                return unique(f"Fiume {random.choice(roots)}")
            else:
                return unique(f"Torrente {random.choice(roots)}")

        def name_valley() -> str:
            choice = random.randint(0, 2)
            if choice == 0:
                return unique(f"Valle del {random.choice(roots)}")
            elif choice == 1:
                return unique(f"Val di {random.choice(saints)}")
            else:
                return unique(f"Piana {random.choice(adjs)}")

        # Apply names to clusters by populating each hex.features
        for cluster in hill_clusters:
            label = name_hill()
            for x, y in cluster:
                self.grid[y][x].features.append(label)

        for cluster in forest_clusters:
            label = name_forest()
            for x, y in cluster:
                self.grid[y][x].features.append(label)

        for cluster in river_clusters:
            label = name_river()
            for x, y in cluster:
                self.grid[y][x].features.append(label)

        for cluster in valley_clusters:
            label = name_valley()
            for x, y in cluster:
                self.grid[y][x].features.append(label)