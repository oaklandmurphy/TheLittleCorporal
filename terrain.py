class Terrain:
    """Represents a terrain type with integer movement cost and combat modifiers."""

    def __init__(self, name: str, move_cost: int, combat_mod: float, vp_value: int = 0):
        self.name = name
        self.move_cost = move_cost
        self.combat_mod = combat_mod
        self.vp_value = vp_value


# Common terrain presets with integer move_cost
FIELDS = Terrain("Fields", move_cost=1, combat_mod=1.0, vp_value=0)
FOREST = Terrain("Forest", move_cost=2, combat_mod=0.9, vp_value=0)
RIVER = Terrain("River", move_cost=3, combat_mod=0.7, vp_value=0)
HILL = Terrain("Hill", move_cost=2, combat_mod=1.2, vp_value=0)
