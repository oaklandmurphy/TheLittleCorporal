from __future__ import annotations
from abc import ABC, abstractmethod
from map.terrain import Terrain
import random

class Unit(ABC):
	"""Represents a single combat unit on the battlefield."""
	def __init__(self, name: str, faction: str, division: str, corps: str, mobility: int, size: int, quality: int, morale: int):
		self.name = name
		self.faction = faction
		self.division = division
		self.corps = corps
		self.mobility = mobility  # integer: max distance moved in a turn
		self.remaining_mobility = mobility
		self.stance = "neutral"  # can be "aggressive", "neutral", "evasive
		self.size = size          # 1–12: number of men abstracted
		self.new_round_size = size
		self.quality = quality    # 1–5: training & experience
		self.new_round_quality = quality
		self.morale = morale      # 0–10: fighting spirit
		self.new_round_morale = morale
		self.engagement = 0	  # number of engagements
		self.has_moved = False
		self.x = None
		self.y = None

	def __repr__(self):
		return f"{self.name}({self.faction})"
	
	def move(self, target_x: int, target_y: int, distance: int) -> bool:
		self.x = target_x
		self.y = target_y
		self.remaining_mobility -= distance
		self.has_moved = True

	# --- Combat logic ---
	def offensive_power(self, terrain_mod) -> float:
		"""Calculate combat effectiveness."""
		return (self.size * 0.5 + self.quality * 2 + self.morale) + terrain_mod * 0.5
	
	def defensive_power(self, terrain_mod) -> float:
		"""Calculate combat effectiveness."""
		return (self.size * 0.5 + self.quality * 2 + self.morale) + terrain_mod * 0.5
	
	def take_damage(self, dmg: float):
		"""Simplified damage resolution."""
		self.new_round_size = max(0, self.size - int(dmg))
		if self.morale == 0:
			print(f"{self.name} ({self.faction}) loses morale and falls back!")

	def is_routed(self) -> bool:
		return self.morale <= 0
	
	def rally(self):
		success_chance = 0.3 + (self.quality * 0.1)
		if random.random() < success_chance:
			self.morale = min(10, self.morale + 1)
			print(f"{self.name} rallies! Morale is now {self.morale}.")
		else:
			print(f"{self.name} fails to rally.")

	def status_general(self) -> str:
		"""Descriptive status of the unit."""
		quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
		morale_labels = {range(0, 2): "broken", range(2, 4): "shaken", range(4, 7): "steady",
						 range(7, 9): "eager", range(9, 11): "fresh"}
		size_labels = {range(1, 4): "small", range(4, 7): "average-sized", range(7, 10): "large", range(10, 13): "very large"}

		def label_for(value, table):
			for key, label in table.items():
				if isinstance(key, range) and value in key:
					return label
				elif value == key:
					return label
			return "unknown"

		return (f"{self.name}. ({self.__class__.__name__}) " # at ({self.x}, {self.y}) — "
				f"A {label_for(self.size, size_labels)}, {label_for(self.quality, quality_labels)} "
				f"formation that is {label_for(self.morale, morale_labels)}.")
	
	def status_so(self) -> str:
		"""Descriptive status of the unit."""
		quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
		morale_labels = {range(0, 2): "broken", range(2, 4): "shaken", range(4, 7): "steady",
						range(7, 9): "eager", range(9, 11): "fresh"}
		size_labels = {range(1, 4): "small", range(4, 7): "average-sized", range(7, 10): "large", range(10, 13): "very large"}

		def label_for(value, table):
			for key, label in table.items():
				if isinstance(key, range) and value in key:
					return label
				elif value == key:
					return label
			return "unknown"

		return (f"{self.name}. ({self.__class__.__name__}) at ({self.x}, {self.y}) — "
				f"A {label_for(self.size, size_labels)}, {label_for(self.quality, quality_labels)} "
				f"formation that is {label_for(self.morale, morale_labels)}.")
	
	def change_stance(self, new_stance: str):
		if new_stance in ["aggressive", "neutral", "evasive"]:
			self.stance = new_stance
		elif new_stance == 1 or new_stance == "a":
			self.stance = "aggressive"
		elif new_stance == 0 or new_stance == "n":
			self.stance = "neutral"
		elif new_stance == -1 or new_stance == "e":
			self.stance = "evasive"
		else:
			print(f"Invalid stance: {new_stance}. No changes made.")

	def reset(self):
		self.remaining_mobility = self.mobility
		self.has_moved = False
		self.size = self.new_round_size
		self.quality = self.new_round_quality
		self.morale = self.new_round_morale


class Infantry(Unit):
	def set_mobility(self):
		self.mobility = 4