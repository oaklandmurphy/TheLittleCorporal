"""Combat resolution logic for hexagonal grid warfare."""

from typing import Set, Tuple, Optional
from unit import Unit
import map.pathfinding as pathfinding


def check_and_engage_combat(unit: Unit, grid, width: int, height: int) -> None:
    """Check if a unit is adjacent to enemy units and engage them in combat.
    
    Args:
        unit: The unit to check for engagements
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
    """
    for nx, ny in pathfinding.get_neighbors(unit.x, unit.y):
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        enemy = grid[ny][nx].unit
        if enemy and enemy.faction != unit.faction:
            # Both units become engaged
            unit.engaged = True
            enemy.engaged = True
            print(f"{unit.name} engages {enemy.name}!")


def check_all_engagements(grid, width: int, height: int) -> None:
    """Check all units on the map and engage adjacent enemies.
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
    """
    processed = set()
    for y in range(height):
        for x in range(width):
            unit = grid[y][x].unit
            if not unit or (x, y) in processed:
                continue
            
            # Check for adjacent enemies
            for nx, ny in pathfinding.get_neighbors(x, y):
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                enemy = grid[ny][nx].unit
                if enemy and enemy.faction != unit.faction:
                    # Both units become engaged
                    unit.engaged = True
                    enemy.engaged = True
                    if (nx, ny) not in processed:
                        print(f"{unit.name} engages {enemy.name}!")
                    processed.add((x, y))
                    processed.add((nx, ny))


def apply_engagement_damage(grid, width: int, height: int) -> None:
    """Apply combat damage to all engaged units.
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
    """
    processed = set()
    for y in range(height):
        for x in range(width):
            unit = grid[y][x].unit
            if not unit or not unit.engaged or (x, y) in processed:
                continue
            
            # Find adjacent engaged enemies
            for nx, ny in pathfinding.get_neighbors(x, y):
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                enemy = grid[ny][nx].unit
                if enemy and enemy.faction != unit.faction and enemy.engaged:
                    # Avoid processing the same pair twice
                    if (nx, ny) in processed:
                        continue
                    
                    # Calculate combat power for both units
                    unit_power = 1  # unit.combat_power(get_terrain(unit.x, unit.y))
                    enemy_power = 1  # enemy.combat_power(get_terrain(enemy.x, enemy.y))
                    
                    # Apply damage
                    damage_to_enemy = max(1, int(unit_power / 10))
                    damage_to_unit = max(1, int(enemy_power / 10))
                    
                    print(f"Combat: {unit.name} (power {unit_power:.1f}) vs {enemy.name} (power {enemy_power:.1f})")
                    
                    enemy.take_damage(damage_to_enemy)
                    unit.take_damage(damage_to_unit)
                    
                    # Remove routed units
                    if enemy.is_routed():
                        grid[ny][nx].unit = None
                    if unit.is_routed():
                        grid[y][x].unit = None
                    
                    # Mark both as processed
                    processed.add((x, y))
                    processed.add((nx, ny))
                    break  # Only process one enemy per unit per turn


def resolve_adjacent_combat(grid, width: int, height: int) -> None:
    """Make adjacent enemy units engage in combat.
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
    """
    processed = set()
    for y in range(height):
        for x in range(width):
            unit = grid[y][x].unit
            if not unit or (x, y) in processed:
                continue
            for nx, ny in pathfinding.get_neighbors(x, y):
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                enemy = grid[ny][nx].unit
                if enemy and enemy.faction != unit.faction:
                    print(f"{unit.name} engages {enemy.name} in combat!")
                    combat(unit, enemy, grid)
                    processed.add((x, y))
                    processed.add((nx, ny))


def combat(attacker: Unit, defender: Unit, grid) -> None:
    """Simple mutual combat between two units.
    
    Args:
        attacker: The attacking unit
        defender: The defending unit
        grid: The 2D grid of Hex objects
    """
    att_power = 1  # attacker.combat_power(get_terrain(attacker.x, attacker.y))
    def_power = 1  # defender.combat_power(get_terrain(defender.x, defender.y))

    damage_to_def = max(1, int(att_power / 10))
    damage_to_att = max(1, int(def_power / 10))

    defender.take_damage(damage_to_def)
    attacker.take_damage(damage_to_att)

    # Remove routed units
    if defender.is_routed():
        grid[defender.y][defender.x].unit = None
    if attacker.is_routed():
        grid[attacker.y][attacker.x].unit = None


def get_combat_advantage(grid, width: int, height: int, x1: int, y1: int, x2: int, y2: int) -> float:
    """Calculate combat advantage for unit at (x1,y1) against unit at (x2,y2).
    
    Args:
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
        x1, y1: Coordinates of the first unit
        x2, y2: Coordinates of the second unit
        
    Returns:
        Combat advantage modifier
    """
    # Get hex at (x1, y1)
    if not (0 <= x1 < width and 0 <= y1 < height):
        return 0.0
    hex1 = grid[y1][x1]
    
    # Check if tile in given direction is on the map
    if not (0 <= x2 < width and 0 <= y2 < height):
        return -10.0
    hex2 = grid[y2][x2]

    terrain1 = hex1.terrain
    terrain2 = hex2.terrain

    offense_mod = terrain1.getOffenseModifier(terrain2.elevation)
    defense_mod = terrain1.getDefenseModifier() - terrain2.getDefenseModifier()

    # Simple formula: offense modifier minus defense modifier
    advantage = offense_mod + defense_mod
    return advantage
