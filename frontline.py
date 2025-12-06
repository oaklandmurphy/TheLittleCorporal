"""Frontline calculation and positioning for tactical deployment."""

import heapq
import math
from typing import List, Tuple, Optional, Dict, Any
import pathfinding
import combat


def hex_to_cartesian(col: int, row: int) -> Tuple[float, float]:
    """Convert even-q hex coordinates to Cartesian coordinates.
    
    Args:
        col: Column coordinate
        row: Row coordinate
        
    Returns:
        Tuple of (x, y) in Cartesian space
    """
    x = col * 1.5
    y = row * (3**0.5) + (col % 2) * (3**0.5) / 2
    return x, y


def angle_to_neighbor(x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate angle in degrees clockwise from north to neighbor.
    
    Args:
        x1, y1: Starting coordinates
        x2, y2: Neighbor coordinates
        
    Returns:
        Angle in degrees (0° = North, 90° = East, 180° = South, 270° = West)
    """
    x1c, y1c = hex_to_cartesian(x1, y1)
    x2c, y2c = hex_to_cartesian(x2, y2)
    
    dx = x2c - x1c
    dy = y2c - y1c
    
    # Calculate angle (math.atan2 returns angle from positive x-axis)
    # Convert to degrees clockwise from north
    angle_rad = math.atan2(dx, -dy)  # -dy because screen Y is inverted
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to 0-360
    return angle_deg % 360


def is_in_arc(target_angle: float, center_angle: float, half_arc_width: float) -> tuple[bool, float]:
    """Check if target angle is within the arc centered on center_angle.
    
    Args:
        target_angle: The angle to check
        center_angle: The center of the arc
        half_arc_width: Half the width of the arc
        
    Returns:
        Tuple of (is_in_arc, angular_distance_from_center)
    """
    # Normalize both angles to 0-360
    target = target_angle % 360
    center = center_angle % 360
    
    # Calculate the difference, maintaining direction
    diff = (target - center + 180) % 360 - 180
    
    # Check if within arc and return absolute distance
    return (abs(diff) <= half_arc_width, abs(diff))


def get_weighted_front_arc_advantage(grid, width: int, height: int, x: int, y: int, 
                                     angle_degrees: int, arc_width_degrees: float = 180.0) -> float:
    """Calculate weighted combat advantage for all tiles within a front arc.
    
    For a tile at (x, y) facing a given direction, this function calculates the combat
    advantage for all neighboring tiles that fall within the front arc. Each neighbor's
    contribution is weighted by its angular distance from the facing direction.
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        x: X coordinate of the tile
        y: Y coordinate of the tile
        angle_degrees: Direction angle in degrees clockwise from north (0° = North, 90° = East)
        arc_width_degrees: Width of the front arc in degrees (default 180°)

    Returns:
        Weighted sum of combat advantages from all tiles in the front arc
    """
    total_weighted_advantage = 0.0
    half_arc = arc_width_degrees / 2.0
    
    # Get all neighbors and calculate their contribution
    for nx, ny in pathfinding.get_neighbors(x, y):
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        
        # Calculate angle to this neighbor
        neighbor_angle = angle_to_neighbor(x, y, nx, ny)
        
        # Check if neighbor is within the front arc and get angular distance
        in_arc, angle_diff = is_in_arc(neighbor_angle, angle_degrees, half_arc)

        if in_arc:
            # Calculate combat advantage for this neighbor
            advantage = combat.get_combat_advantage(grid, width, height, x, y, nx, ny)
            
            # Weight based on angle: neighbors directly in front get full weight,
            # those at the arc edge get less weight
            weight = 1.0 - (angle_diff / half_arc)
            
            # Final contribution is advantage * weight
            contribution = advantage * weight
            total_weighted_advantage += contribution
    
    return total_weighted_advantage


def get_frontline_endpoints(grid, width: int, height: int, feature_coords: List[Tuple[int, int]], 
                            angle_degrees: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Find two endpoint coordinates for a frontline across a feature.

    Finds two points near the feature that maximize the perpendicular distance component 
    (in Cartesian space) relative to the given direction.

    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        feature_coords: List of (x, y) coordinates for the feature
        angle_degrees: Angle in degrees clockwise from straight north

    Returns:
        A tuple of two (x, y) coordinates: (left_point, right_point), or None if not found
    """
    if not feature_coords:
        return None

    # Convert angle in degrees clockwise from north to direction vector in Cartesian space
    angle_radians = math.radians(angle_degrees)
    dir_vec = (math.sin(angle_radians), -math.cos(angle_radians))
    
    # Get the perpendicular vector (rotate 90 degrees counterclockwise)
    perp_vec = (-dir_vec[1], dir_vec[0])

    # Get feature center in Cartesian coordinates
    feature_cart = [hex_to_cartesian(x, y) for x, y in feature_coords]
    center_x = sum(p[0] for p in feature_cart) / len(feature_cart)
    center_y = sum(p[1] for p in feature_cart) / len(feature_cart)

    # Get all candidate tiles near the feature
    candidate_tiles: List[Tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            for fx, fy in feature_coords:
                dist = pathfinding.hex_distance(x, y, fx, fy)
                if 1 <= dist <= 1:
                    candidate_tiles.append((x, y))
                    break

    candidate_tiles = list(set(candidate_tiles))

    if len(candidate_tiles) < 2:
        return None

    # Project each candidate onto the perpendicular axis
    projections = []
    cart_positions = {}
    for col, row in candidate_tiles:
        cart_x, cart_y = hex_to_cartesian(col, row)
        cart_positions[(col, row)] = (cart_x, cart_y)
        vec_x = cart_x - center_x
        vec_y = cart_y - center_y
        projection = vec_x * perp_vec[0] + vec_y * perp_vec[1]
        projections.append((projection, (col, row)))

    # Sort by projection to find extremes
    projections.sort(key=lambda p: p[0])
    
    # Find all points with minimum projection (left side)
    min_projection = projections[0][0]
    left_candidates = [p[1] for p in projections if abs(p[0] - min_projection) < 0.0001]
    
    # Find all points with maximum projection (right side)
    max_projection = projections[-1][0]
    right_candidates = [p[1] for p in projections if abs(p[0] - max_projection) < 0.0001]
    
    # If there are ties, use full Cartesian distance as tiebreaker
    best_pair = None
    max_distance = -1
    
    for left in left_candidates:
        for right in right_candidates:
            left_cart = cart_positions[left]
            right_cart = cart_positions[right]
            distance = ((right_cart[0] - left_cart[0])**2 + (right_cart[1] - left_cart[1])**2)**0.5
            
            if distance > max_distance:
                max_distance = distance
                best_pair = (left, right)
    
    return best_pair


def get_frontline(grid, width: int, height: int, start: Tuple[int, int], goal: Tuple[int, int], 
                 angle_degrees: int) -> List[Tuple[int, int]]:
    """Compute the best frontline path between two endpoints using pathfinding.

    Finds the path between start and goal that maximizes combat advantage 
    when facing the given direction. Uses Dijkstra-like pathfinding where
    advantage is the primary factor (heavily weighted) and distance is secondary.

    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        start: Starting (x, y) coordinate of the frontline
        goal: Ending (x, y) coordinate of the frontline
        angle_degrees: Angle in degrees clockwise from straight north

    Returns:
        A list of (x, y) hex coordinates forming the best frontline path.
        Returns empty list if no path exists.
    """
    def get_advantage(x: int, y: int) -> float:
        """Get weighted front arc advantage for this position."""
        return get_weighted_front_arc_advantage(grid, width, height, x, y, angle_degrees, arc_width_degrees=180.0)

    # Use Dijkstra's algorithm with cost = -advantage (so we maximize advantage)
    open_set = []
    heapq.heappush(open_set, (0.0, 0, start))
    came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    best_priority: dict[Tuple[int, int], float] = {start: 0.0}
    visited = set()

    while open_set:
        current_priority, current_dist, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        visited.add(current)

        # If we reached the goal, reconstruct path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for nx, ny in pathfinding.get_neighbors(current[0], current[1]):
            neighbor = (nx, ny)
            
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if neighbor in visited:
                continue

            # Calculate the cost to reach this neighbor
            neighbor_advantage = get_advantage(nx, ny)
            new_dist = current_dist + 1
            
            # Priority heavily weights advantage, lightly weights distance
            new_priority = -neighbor_advantage * 1000.0 + new_dist

            # If this is a better path to the neighbor, update it
            if neighbor not in best_priority or new_priority < best_priority[neighbor]:
                best_priority[neighbor] = new_priority
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_priority, new_dist, neighbor))

    # No path found
    return []


def distribute_units_along_frontline(coordinates: List[Tuple[int, int]], num_units: int) -> List[Tuple[int, int]]:
    """Take a list of coordinates and return evenly spaced points along them.

    Points are distributed with equal spacing between them and equal margins at the edges.
    For example, with num_units=2, points are placed at 1/3 and 2/3 of the path length.
    
    If num_units exceeds len(coordinates), the function returns all coordinates multiple times
    and distributes the remainder using the spacing algorithm.

    Args:
        coordinates: A list of (x, y) tuples representing a path
        num_units: The number of points to return

    Returns:
        A list of (x, y) tuples with size equal to num_units, evenly distributed along the path
    """
    if not coordinates:
        return []
    
    if num_units <= 0:
        return []
    
    path_length = len(coordinates)
    selected = []
    
    # Add all coordinates for each complete pass
    full_passes = num_units // path_length
    for _ in range(full_passes):
        selected.extend(coordinates)
    
    # Distribute remainder using spacing algorithm
    remainder = num_units % path_length
    segment_size = path_length / (remainder + 1)
    for i in range(1, remainder + 1):
        index = int(round(i * segment_size))
        index = min(max(index, 0), path_length - 1)
        selected.append(coordinates[index])
    
    return selected


def get_frontline_total_advantage(grid, width: int, height: int, frontline_coords: List[Tuple[int, int]], 
                                   angle_degrees: int, arc_width_degrees: float = 180.0) -> float:
    """Calculate the total combat advantage for a frontline facing a given direction.
    
    For each coordinate in the frontline, calculates the weighted front arc advantage
    and sums them up to get a total advantage score for the entire frontline.
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        frontline_coords: List of (x, y) coordinates that make up the frontline
        angle_degrees: Direction the frontline is facing (0° = North, 90° = East, 180° = South, 270° = West)
        arc_width_degrees: Width of the front arc in degrees (default 180°)
    
    Returns:
        Total combat advantage score for the entire frontline
    """
    if not frontline_coords:
        return 0.0
    
    total_advantage = 0.0
    
    for x, y in frontline_coords:
        if 0 <= x < width and 0 <= y < height:
            advantage = get_weighted_front_arc_advantage(
                grid, width, height, x, y, angle_degrees, arc_width_degrees
            )
            total_advantage += advantage
    
    return total_advantage


def get_best_frontline_with_advantage(grid, width: int, height: int, start: Tuple[int, int], 
                                       goal: Tuple[int, int], angle_degrees: int,
                                       arc_width_degrees: float = 180.0) -> Dict[str, Any]:
    """Calculate the best frontline from point A to point B and return it with its total advantage.
    
    Combines frontline pathfinding with advantage calculation to provide both the optimal
    frontline path and its combat effectiveness score.
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        start: Starting (x, y) coordinate of the frontline
        goal: Ending (x, y) coordinate of the frontline
        angle_degrees: Direction the frontline is facing (0° = North, 90° = East, 180° = South, 270° = West)
        arc_width_degrees: Width of the front arc in degrees (default 180°)
    
    Returns:
        Dictionary containing:
            - 'frontline': List of (x, y) coordinates forming the best frontline path
            - 'total_advantage': Total combat advantage score for this frontline
            - 'length': Number of hexes in the frontline
            - 'average_advantage': Average advantage per hex position
    """
    # Generate the best frontline path
    frontline_path = get_frontline(grid, width, height, start, goal, angle_degrees)
    
    if not frontline_path:
        return {
            'frontline': [],
            'total_advantage': 0.0,
            'length': 0,
            'average_advantage': 0.0
        }
    
    # Calculate total advantage for this frontline
    total_advantage = get_frontline_total_advantage(
        grid, width, height, frontline_path, angle_degrees, arc_width_degrees
    )
    
    # Calculate average advantage per position
    length = len(frontline_path)
    average_advantage = total_advantage / length if length > 0 else 0.0
    
    return {
        'frontline': frontline_path,
        'total_advantage': total_advantage,
        'length': length,
        'average_advantage': average_advantage
    }


def identify_defensive_features(map_obj, faction: str,
                                arc_width_degrees: float = 180.0) -> List[Dict[str, Any]]:
    """Identify terrain features with high combat advantage facing enemy forces.
    
    For each terrain feature on the map, calculates the direction toward enemy units
    and evaluates the combat advantage of defending that feature from that direction.
    Returns features sorted by their defensive value.
    
    Args:
        map_obj: The Map object containing the grid and helper methods
        faction: The faction to analyze defensive positions for
        arc_width_degrees: Width of the front arc in degrees (default 180°)
    
    Returns:
        List of dictionaries, each containing:
            - 'feature_name': Name of the terrain feature
            - 'feature_coords': List of (x, y) coordinates in the feature
            - 'feature_center': (x, y) center position of the feature
            - 'enemy_direction': Angle in degrees toward enemy forces
            - 'total_advantage': Total combat advantage score for this feature
            - 'average_advantage': Average advantage per hex in feature
            - 'size': Number of hexes in the feature
        Sorted by total_advantage (highest first)
    """
    grid = map_obj.grid
    width = map_obj.width
    height = map_obj.height
    
    # Get all terrain feature names
    feature_names = map_obj.list_feature_names()
    
    if not feature_names:
        return []
    
    # Analyze each feature
    defensive_features = []
    
    for feature_name in feature_names:
        # Use existing method to get enemy approach angle for this feature
        enemy_direction = map_obj.get_enemy_approach_angle(faction, feature_name)
        
        if enemy_direction is None:
            # No enemies found, skip this feature
            continue
        
        # Get feature coordinates
        coords = map_obj.get_feature_coordinates(feature_name)
        if not coords:
            continue
        
        # Calculate feature center
        feature_x = sum(x for x, y in coords) / len(coords)
        feature_y = sum(y for x, y in coords) / len(coords)
        feature_center = (int(feature_x), int(feature_y))
        
        # Calculate total advantage for all hexes in this feature facing enemies
        total_advantage = 0.0
        for x, y in coords:
            advantage = get_weighted_front_arc_advantage(
                grid, width, height, x, y, enemy_direction, arc_width_degrees
            )
            total_advantage += advantage
        
        # Calculate average advantage per hex
        size = len(coords)
        average_advantage = total_advantage / size if size > 0 else 0.0
        
        defensive_features.append({
            'feature_name': feature_name,
            'feature_coords': coords,
            'feature_center': feature_center,
            'enemy_direction': enemy_direction,
            'total_advantage': total_advantage,
            'average_advantage': average_advantage,
            'size': size
        })
    
    # Sort by total advantage (best defensive features first)
    defensive_features.sort(key=lambda f: f['total_advantage'], reverse=True)
    
    return defensive_features
