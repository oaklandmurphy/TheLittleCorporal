"""Terrain feature labeling and naming for map generation."""

import random
from typing import Optional, List, Tuple, Set
from terrain import HILL_ELEVATION_THRESHOLD, FOREST_TREE_COVER_THRESHOLD
import pathfinding


def label_terrain_features(grid, width: int, height: int, seed: Optional[int] = None,
                           min_sizes: Optional[dict] = None) -> None:
    """Populate each hex's `features` with names for clusters of Hills, Rivers, Forests, and Valleys.

    Valleys are approximated as contiguous Fields tiles adjacent to any River tile.

    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
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
        for y in range(height):
            for x in range(width):
                if (x, y) in seen or not predicate(x, y):
                    continue
                comp: List[Tuple[int, int]] = []
                stack = [(x, y)]
                seen.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    comp.append((cx, cy))
                    for nx, ny in pathfinding.get_neighbors(cx, cy):
                        if 0 <= nx < width and 0 <= ny < height:
                            if (nx, ny) not in seen and predicate(nx, ny):
                                seen.add((nx, ny))
                                stack.append((nx, ny))
                clusters.append(comp)
        return clusters

    def is_hill(x, y):
        # Use elevation threshold to classify hills instead of terrain name
        return grid[y][x].terrain.elevation >= HILL_ELEVATION_THRESHOLD

    def is_forest(x, y):
        # Use tree_cover threshold to classify forests instead of terrain name
        return grid[y][x].terrain.tree_cover >= FOREST_TREE_COVER_THRESHOLD

    def is_river(x, y):
        return grid[y][x].terrain.name == "River"

    # Valley candidates: Fields adjacent to any River
    river_adjacent_fields: Set[Tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            if grid[y][x].terrain.name != "Fields":
                continue
            for nx, ny in pathfinding.get_neighbors(x, y):
                if 0 <= nx < width and 0 <= ny < height:
                    if grid[ny][nx].terrain.name == "River":
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
        """Break a large cluster into smaller sub-clusters using spatial proximity."""
        if len(cluster) < min_subcluster_size:
            return []
        
        subclusters = []
        remaining = set(cluster)
        
        # Limit iterations to prevent infinite loops
        max_iterations = len(cluster) * 2
        iteration_count = 0
        
        while len(remaining) >= min_subcluster_size and iteration_count < max_iterations:
            iteration_count += 1
            
            # Pick a random seed from remaining
            seed_pos = random.choice(list(remaining))
            subcluster = [seed_pos]
            remaining.remove(seed_pos)
            
            # Create small sub-features (2-3 hexes) to maximize coverage
            target_size = random.randint(2, 3)
            
            # BFS-like growth but stay compact
            frontier = [seed_pos]
            visited = {seed_pos}
            growth_iterations = 0
            max_growth = target_size * 4
            
            while len(subcluster) < target_size and frontier and growth_iterations < max_growth:
                growth_iterations += 1
                
                if not frontier:
                    break
                    
                current = frontier.pop(0)
                neighbors = list(pathfinding.get_neighbors(current[0], current[1]))
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

    # Apply names to clusters by populating each hex.features
    for cluster in hill_clusters:
        major_label = name_hill()
        for x, y in cluster:
            grid[y][x].features.append(major_label)
        
        if len(cluster) >= 3:
            subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
            for subcluster in subclusters:
                sub_label = name_hill_sub()
                for x, y in subcluster:
                    grid[y][x].features.append(sub_label)

    for cluster in forest_clusters:
        major_label = name_forest()
        for x, y in cluster:
            grid[y][x].features.append(major_label)
        
        if len(cluster) >= 3:
            subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
            for subcluster in subclusters:
                sub_label = name_forest_sub()
                for x, y in subcluster:
                    grid[y][x].features.append(sub_label)

    for cluster in river_clusters:
        major_label = name_river()
        for x, y in cluster:
            grid[y][x].features.append(major_label)
        
        if len(cluster) >= 2:
            subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
            for subcluster in subclusters:
                sub_label = name_river_sub()
                for x, y in subcluster:
                    grid[y][x].features.append(sub_label)

    for cluster in valley_clusters:
        major_label = name_valley()
        for x, y in cluster:
            grid[y][x].features.append(major_label)
        
        if len(cluster) >= 3:
            subclusters = subdivide_cluster(cluster, min_subcluster_size=1)
            for subcluster in subclusters:
                sub_label = name_valley_sub()
                for x, y in subcluster:
                    grid[y][x].features.append(sub_label)
