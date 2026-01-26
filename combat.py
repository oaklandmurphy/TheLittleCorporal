"""Combat resolution logic for hexagonal grid warfare."""

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
            # add an egaged unit
            unit.engagement += 1
            print(f"{unit.name} engages {enemy.name}!")


def check_all_engagements(faction1_units: list, faction2_units: list, grid, width: int, height: int) -> None:
    """Check units from both factions and engage adjacent enemies, with evenly distributed resolution.
    
    Args:
        faction1_units: List of units from the first faction
        faction2_units: List of units from the second faction
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
    """
    # assign engagements for faction 1
    faction1_engaged_units = []
    
    for _, unit in faction1_units:
           # engage adjacent enemies
           check_and_engage_combat(unit, grid, width, height)
           faction1_engaged_units.append(unit)

    # assign engagements for faction 2
    faction2_engaged_units = []

    for _, unit in faction2_units:
           # engage adjacent enemies
           check_and_engage_combat(unit, grid, width, height)
           faction2_engaged_units.append(unit)
    
    engaged_units = faction1_engaged_units + faction2_engaged_units
    
    # Process units in the interleaved order
    for unit in engaged_units:
        resolve_combat_for_unit(unit, grid, width, height)

def resolve_combat_for_unit(unit: Unit, grid, width: int, height: int) -> None:
    """Resolve combat for a single unit against adjacent enemies.
    
    Args:
        unit: The unit to resolve combat for
        grid: The 2D grid of Hex objects
        width: Width of the map
        height: Height of the map
    """
    for nx, ny in pathfinding.get_neighbors(unit.x, unit.y):
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        enemy = grid[ny][nx].unit
        if enemy and enemy.faction != unit.faction:
            print(f"{unit.name} engages {enemy.name} in combat!")
            combat(unit, enemy, grid)

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
