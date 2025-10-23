from map import Map
from unit import Unit

class TurnManager:
    """Manages turn order and unit actions for factions."""
    def __init__(self, game_map: Map, factions: list[str]):
        self.map = game_map
        self.factions = factions
        self.current_index = 0  # index of current faction

    def all_units(self):
        # Collect all units currently on the map
        units = []
        for row in self.map.grid:
            for hex in row:
                if hex.unit:
                    units.append(hex.unit)
        return units

    def start_turn(self):
        """Reset mobility for all units of the current faction."""
        faction = self.factions[self.current_index]
        print(f"\n--- {faction} turn ---")
        for unit in self.all_units():
            if unit.faction == faction and hasattr(unit, "set_mobility"):
                unit.set_mobility()  # reset mobility for the turn
                unit.has_moved = False

    def take_turn(self):
        """Simple AI: each unit moves to a random reachable hex and attempts to rally if morale is low."""
        import random
        faction = self.factions[self.current_index]
        for unit in [u for u in self.all_units() if u.faction == faction]:
            # Move to a random reachable hex (excluding current position)
            reachable = self.map.find_reachable_hexes(unit)
            possible = [pos for pos in reachable if pos != (unit.x, unit.y)]
            if possible:
                target_hex = random.choice(possible)
                self.map.move_unit(unit, *target_hex)
            # Attempt rally if morale low
            if hasattr(unit, "rally") and unit.morale < 5:
                unit.rally()
        # Advance to next faction
        self.current_index = (self.current_index + 1) % len(self.factions)

    def run_turns(self, num_turns: int = 1):
        for _ in range(num_turns):
            self.start_turn()
            self.take_turn()
