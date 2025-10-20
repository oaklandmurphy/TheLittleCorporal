from __future__ import annotations
from abc import ABC, abstractmethod
import random


class Unit(ABC):
	"""Base class for all unit types."""

	def __init__(self, name: str, x: int, y: int, quality: int, size: int, morale: int, faction: str):
		self.name = name
		self.x = x
		self.y = y
		self.quality = max(1, min(5, quality))
		self.size = max(1, min(12, size))
		self.morale = max(0, min(10, morale))
		self.faction = faction 
		self.mobility = 0  # overridden by subclasses

	@abstractmethod
	def set_mobility(self):
		pass

	def move(self, new_x: int, new_y: int, game_map) -> bool:
		"""Move unit considering terrain cost from Map."""
		terrain = game_map.get_terrain(new_x, new_y)
		if not terrain:
			print(f"{self.name} cannot move out of bounds!")
			return False

		terrain_cost = terrain.move_cost
		if terrain_cost <= self.mobility:
			self.x, self.y = new_x, new_y
			self.mobility -= terrain_cost
			print(f"{self.name} moved to ({new_x}, {new_y}) over {terrain.name} (cost {terrain_cost})")
			return True
		else:
			print(f"{self.name} lacks mobility to move over {terrain.name} (needs {terrain_cost}, has {self.mobility})")
			return False

	def combat(self, enemy: Unit, game_map):
		"""Engage an adjacent enemy unit if they are from a different faction."""
		if self.faction == enemy.faction:
			print(f"{self.name} cannot attack {enemy.name} — same faction ({self.faction})")
			return  # skip combat if same faction

		terrain = game_map.get_terrain(self.x, self.y)
		terrain_mod = terrain.combat_mod if terrain else 1.0

		attack_strength = self.quality * self.size * random.uniform(0.8, 1.2)
		defense_strength = enemy.quality * enemy.size * terrain_mod * random.uniform(0.8, 1.2)

		if attack_strength > defense_strength:
			loss = int((attack_strength - defense_strength) / 20)
			enemy.size = max(0, enemy.size - loss)
			enemy.morale = max(0, enemy.morale - 1)
			print(f"{self.name} defeats {enemy.name} (enemy loses {loss} strength).")
		else:
			loss = int((defense_strength - attack_strength) / 20)
			self.size = max(0, self.size - loss)
			self.morale = max(0, self.morale - 1)
			print(f"{self.name} fails to break {enemy.name} (loses {loss} strength).")

	def rally(self):
		success_chance = 0.3 + (self.quality * 0.1)
		if random.random() < success_chance:
			self.morale = min(10, self.morale + 1)
			print(f"{self.name} rallies! Morale is now {self.morale}.")
		else:
			print(f"{self.name} fails to rally.")

	def status(self) -> str:
		"""Return a plain-English description of the unit’s condition."""

		quality_labels = {
			1: "green",
			2: "regular",
			3: "seasoned",
			4: "veteran",
			5: "elite"
		}

		morale_labels = {
			range(0, 2): "broken",
			range(2, 4): "shaken",
			range(4, 7): "steady",
			range(7, 9): "eager",
			range(9, 11): "fresh"
		}

		size_labels = {
			range(1, 4): "small",
			range(4, 7): "average-sized",
			range(7, 10): "large",
			range(10, 13): "very large"
		}

		# Helper to get label for a given value range mapping
		def label_for(value: int, table: dict) -> str:
			for key, label in table.items():
				if isinstance(key, range) and value in key:
					return label
				elif value == key:
					return label
			return "unknown"

		quality_desc = quality_labels.get(self.quality, "unknown quality")
		morale_desc = label_for(self.morale, morale_labels)
		size_desc = label_for(self.size, size_labels)

		return (
			f"{self.name} ({self.__class__.__name__}) at ({self.x}, {self.y}) — "
			f"A {size_desc}, {quality_desc} formation that is {morale_desc}."
		)

class Infantry(Unit):
	def __init__(self, name: str, x: int, y: int, quality: int, size: int, morale: int):
		super().__init__(name, x, y, quality, size, morale)
		self.set_mobility()

	def set_mobility(self):
		self.mobility = 3


class Cavalry(Unit):
	def __init__(self, name: str, x: int, y: int, quality: int, size: int, morale: int):
		super().__init__(name, x, y, quality, size, morale)
		self.set_mobility()

	def set_mobility(self):
		self.mobility = 6


class Artillery(Unit):
	def __init__(self, name: str, x: int, y: int, quality: int, size: int, morale: int):
		super().__init__(name, x, y, quality, size, morale)
		self.set_mobility()

	def set_mobility(self):
		self.mobility = 2

	def bombard(self, enemy: Unit, range_penalty: float = 1.0):
		"""Long-range attack; cannot retaliate."""
		damage = int((self.quality * self.size * random.uniform(0.6, 1.0)) / (10 * range_penalty))
		enemy.size = max(0, enemy.size - damage)
		enemy.morale = max(0, enemy.morale - 1)
		print(f"{self.name} bombards {enemy.name}! Inflicts {damage} casualties.")