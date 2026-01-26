import random


class Terrain:
    """Represents a terrain type with movement and combat modifiers."""
    def __init__(self, name: str, move_cost: int, elevation: float = 0.0, tree_cover: int = 0):
        self.name = name
        self.move_cost = move_cost  # integer cost to enter this tile
        
        self.elevation = elevation  # elevation value (used to classify hills via threshold)
        self.tree_cover = tree_cover  # tree cover density (0-3, used to classify forests via threshold)
        self.urbanization = 0
        self.entrenchments = 0

    def __repr__(self):
        return f"Terrain({self.name})"
    
    def getDefenseModifier(self):
        """Returns the combat modifier for defense calculations."""
        return min(self.urbanization + self.entrenchments + self.tree_cover, 5)
    
    def getOffenseModifier(self, enemy_elevation: int):
        """Returns the combat modifier for offense calculations."""
        return self.elevation - enemy_elevation

# Common terrain presets with elevation and tree_cover values
FIELDS = Terrain("Fields", move_cost=1, elevation=1, tree_cover=0)

def FOREST():
    """Create a Forest terrain with random tree cover (1-3)."""
    return Terrain("Forest", move_cost=2, elevation=1, tree_cover=random.randint(1, 3))

RIVER = Terrain("River", move_cost=3, elevation=0, tree_cover=0)
HILL = Terrain("Hill", move_cost=2, elevation=5, tree_cover=0)

# Thresholds for classifying terrain features
HILL_ELEVATION_THRESHOLD = 2
FOREST_TREE_COVER_THRESHOLD = 2  # tree_cover >= 2 is considered forest
