class Terrain:
    """Represents a terrain type with movement and combat modifiers."""
    def __init__(self, name: str, move_cost: int, combat_modifier: float = 1.0):
        self.name = name
        self.move_cost = move_cost  # integer cost to enter this tile
        self.combat_modifier = combat_modifier  # modifies combat effectiveness

    def __repr__(self):
        return f"Terrain({self.name})"

# Common terrain presets
FIELDS = Terrain("Fields", move_cost=1, combat_modifier=1.0)
FOREST = Terrain("Forest", move_cost=2, combat_modifier=0.9)
RIVER = Terrain("River", move_cost=3, combat_modifier=0.7)
HILL = Terrain("Hill", move_cost=2, combat_modifier=1.2)
