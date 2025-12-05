from map import Map
from terrain import FIELDS, HILL, RIVER, FOREST
from unit import Infantry
from turnmanager import TurnManager
from visualization import Visualization, WINDOW_W, WINDOW_H
from general import General
import os
import pygame
import json
from OpenGL.GLU import gluOrtho2D

def create_demo_map() -> Map:
	w, h = 12, 9
	game_map = Map(w, h)
	import random
	random.seed(69)

	# --- Helpers ---
	def in_bounds(x, y):
		return 0 <= x < w and 0 <= y < h

	def is_river(x, y):
		return in_bounds(x, y) and game_map.get_terrain(x, y).name == "River"

	def set_if_not_river(x, y, terrain):
		if in_bounds(x, y) and not is_river(x, y):
			game_map.set_terrain(x, y, terrain)

	# --- Carve a continuous river along the left side ---
	def carve_river():
		x = 1  # start near the left edge
		for y in range(h):
			game_map.set_terrain(x, y, RIVER)
			# occasional widening within left band
			if random.random() < 0.25:
				wx = max(0, min(2, x + random.choice([-1, 1])))
				game_map.set_terrain(wx, y, RIVER)
			# gentle meander but stay near left side
			if random.random() < 0.45:
				x = max(0, min(2, x + random.choice([-1, 0, 1])))

	# --- Add hill ridges (east-west or slight diagonal) ---
	def add_hill_ridge(start_x, start_y, length, diag=False):
		x, y = start_x, start_y
		dy = 0
		for i in range(length):
			if not in_bounds(x, y):
				break
			# ridge core
			set_if_not_river(x, y, HILL)
			# thicken ridge (one of the adjacent hexes)
			for (ax, ay) in [(x, y-1), (x, y+1)]:
				if random.random() < 0.5:
					set_if_not_river(ax, ay, HILL)
			# advance east; optional slight diagonal drift
			x += 1
			if diag:
				dy += random.choice([-1, 0, 1])
				dy = max(-1, min(1, dy))
				y += dy

	# --- Add forest clusters via BFS growth ---
	def add_forest_cluster(cx, cy, target):
		if not in_bounds(cx, cy) or is_river(cx, cy):
			return
		frontier = [(cx, cy)]
		visited = set(frontier)
		placed = 0
		while frontier and placed < target:
			nx, ny = frontier.pop(0)
			if is_river(nx, ny):
				continue
			set_if_not_river(nx, ny, FOREST())
			placed += 1
			for (qx, qy) in game_map.get_neighbors(nx, ny):
				if in_bounds(qx, qy) and (qx, qy) not in visited and random.random() < 0.7:
					visited.add((qx, qy))
					frontier.append((qx, qy))

	# Generate features
	carve_river()
	
	# Multiple hill ridges with varied orientations and positions
	add_hill_ridge(3, 1, length=4, diag=True)   # Northern ridge
	add_hill_ridge(4, 3, length=6, diag=False)  # Central ridge (horizontal)
	add_hill_ridge(5, 5, length=5, diag=True)   # South-central ridge
	add_hill_ridge(6, 7, length=4, diag=False)  # Southern ridge
	add_hill_ridge(8, 1, length=3, diag=True)   # Eastern high ground
	
	# Varied forest clusters - different sizes and locations
	add_forest_cluster(7, 2, target=6)   # Small northern woods
	add_forest_cluster(9, 3, target=12)  # Large eastern forest
	add_forest_cluster(5, 4, target=5)   # Small central copse
	add_forest_cluster(10, 6, target=9)  # Medium southeastern woods
	add_forest_cluster(4, 7, target=7)   # Southern forest patch
	add_forest_cluster(8, 8, target=4)   # Small southern woods

	# place units - expanded order of battle
	# All Blue units share the same division/corps (Napoleon's 1st Corps)
	blue_division = "1st Division"
	blue_corps = "I Corps (France)"

	# All Yellow units share the same division/corps (Austrian 3rd Corps)
	yellow_division = "2nd Division"
	yellow_corps = "III Corps (Austrian)"

	# Blue forces (French brigadier generals)
	b1 = Infantry("Brigade Friant", "French", blue_division, blue_corps, mobility=3, size=8, quality=3, morale=7)
	b2 = Infantry("Brigade Gudin", "French", blue_division, blue_corps, mobility=3, size=7, quality=3, morale=6)
	b3 = Infantry("Brigade Morand", "French", blue_division, blue_corps, mobility=3, size=5, quality=4, morale=8)
	b4 = Infantry("Brigade Petit", "French", blue_division, blue_corps, mobility=3, size=4, quality=4, morale=6)
	b5 = Infantry("Brigade Desvaux", "French", blue_division, blue_corps, mobility=3, size=4, quality=2, morale=5)

	# Yellow forces (Austrian brigadier generals)
	r1 = Infantry("Brigade Klenau", "Austrian", yellow_division, yellow_corps, mobility=3, size=8, quality=3, morale=7)
	r2 = Infantry("Brigade Hohenlohe", "Austrian", yellow_division, yellow_corps, mobility=3, size=6, quality=4, morale=8)
	r3 = Infantry("Brigade Vincent", "Austrian", yellow_division, yellow_corps, mobility=3, size=4, quality=4, morale=6)
	r4 = Infantry("Brigade Lichtenstein", "Austrian", yellow_division, yellow_corps, mobility=3, size=6, quality=4, morale=7)
	r5 = Infantry("Brigade Vukassovich", "Austrian", yellow_division, yellow_corps, mobility=3, size=5, quality=3, morale=6)

	# Place Blue units (left/center)
	game_map.place_unit(b1, 3, 3)
	game_map.place_unit(b2, 4, 3)
	game_map.place_unit(b3, 3, 2)
	game_map.place_unit(b4, 2, 4)
	game_map.place_unit(b5, 4, 4)

	# Place Yellow units (right/center)
	game_map.place_unit(r1, 5, 3)
	game_map.place_unit(r2, 9, 3)
	game_map.place_unit(r3, 8, 5)
	game_map.place_unit(r4, 9, 5)
	game_map.place_unit(r5, 7, 4)

	# label features
	game_map.label_terrain_features(seed=123)
	
	return game_map

def main():
	pygame.init()
	pygame.display.set_caption("Hex Map Renderer - Napoleonic Wargame Demo")
	screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.OPENGL | pygame.DOUBLEBUF)
	gluOrtho2D(0, WINDOW_W, WINDOW_H, 0)
	clock = pygame.time.Clock()
	game_map = create_demo_map()
	factions = ["French", "Austrian"]
	turn_manager = TurnManager(game_map, factions)
	vis = Visualization(game_map)

	# Load general presets from JSON
	with open("general_presets.json", "r", encoding="utf-8") as f:
		general_presets = json.load(f)

	blue_general_preset = general_presets["marmont_preset"]
	yellow_general_preset = general_presets["schwarzenberg_preset"]

	# set up remote host
	# use None for local ollama
	# host = "http://67.181.163.41:42069"
	host = None
	
	max_retries = 9
	num_threads = None
	num_ctx = None
	

	# specify model
	gen_model = "llama2:7b"
	gen_model = "llama3.2:3b"
	# gen_model = "gpt-oss:120b-cloud"
	# gen_model = "mistral:7b"

	# Setup generals
	generals = {
		"French": General(unit_list=game_map.get_units_by_faction("French"), faction="French", model=gen_model, identity_prompt=blue_general_preset, ollama_host=host, game_map=game_map), 
		"Austrian": General(unit_list=game_map.get_units_by_faction("Austrian"), faction="Austrian", model=gen_model, identity_prompt=yellow_general_preset, ollama_host=host, game_map=game_map)
	}

	print("0,6 weighted combat adv:", game_map.get_weighted_front_arc_advantage(0, 6, 270))

	frontline_santa_maria = game_map.get_frontline_for_feature("Santa Maria Heights", 240)
	frontline_san_simone = game_map.get_frontline_for_feature("San Simone Heights", 240)
	frontline_lodi_river = game_map.get_frontline_for_feature("Lodi Stream", 270)
	print("get frontline santa maria: ", frontline_santa_maria)
	print("get frontline san simone: ", frontline_san_simone)
	print("get frontline lodi river: ", frontline_lodi_river)

	# Run the game loop
	turn_manager.run_game_loop(
		vis=vis,
		generals=generals,
		clock=clock,
		max_retries=max_retries,
		num_threads=num_threads,
		num_ctx=num_ctx
	)

if __name__ == "__main__":
	main()