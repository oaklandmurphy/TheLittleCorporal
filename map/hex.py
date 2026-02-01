"""Hex tile class for the hexagonal grid map."""

from .terrain import Terrain
from map.unit import Unit
from typing import Optional


class Hex:
    """A single tile on the hex map."""
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.unit: Optional[Unit] = None
        self.victory_points: Optional[int] = 0
        self.features: list[str] = []
