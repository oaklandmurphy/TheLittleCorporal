from map import Map
from terrain import FIELDS, HILL, RIVER, FOREST
from unit import Infantry
from turnmanager import TurnManager
from visualization import Visualization, WINDOW_W, WINDOW_H
from reporting import ReportGenerator
import threading
import queue
from general import General
from staff_officer import StaffOfficer
import os
import pygame
import sys
import json
from OpenGL.GLU import gluOrtho2D

def create_demo_map() -> Map:
	w, h = 12, 9
	game_map = Map(w, h)
	import random
	random.seed(42)

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
			set_if_not_river(nx, ny, FOREST)
			placed += 1
			for (qx, qy) in game_map.get_neighbors(nx, ny):
				if in_bounds(qx, qy) and (qx, qy) not in visited and random.random() < 0.7:
					visited.add((qx, qy))
					frontier.append((qx, qy))

	# Generate features
	carve_river()
	# a couple of ridges in center/north/south bands
	add_hill_ridge(3, 2, length=5, diag=True)
	add_hill_ridge(4, 5, length=6, diag=False)
	add_hill_ridge(6, 7, length=4, diag=True)

	# several forest clusters
	add_forest_cluster(8, 2, target=8)
	add_forest_cluster(9, 5, target=10)
	add_forest_cluster(5, 6, target=7)

	# place units - expanded order of battle
	# All Blue units share the same division/corps (Napoleon's 1st Corps)
	blue_division = "1st Division"
	blue_corps = "I Corps (France)"

	# All Yellow units share the same division/corps (Austrian 3rd Corps)
	yellow_division = "2nd Division"
	yellow_corps = "III Corps (Austrian)"

	# Blue forces (French brigadier generals)
	b1 = Infantry("Brigade Friant", "French", blue_division, blue_corps, mobility=4, size=8, quality=3, morale=7)
	b2 = Infantry("Brigade Gudin", "French", blue_division, blue_corps, mobility=4, size=7, quality=3, morale=6)
	b3 = Infantry("Brigade Morand", "French", blue_division, blue_corps, mobility=4, size=5, quality=4, morale=8)
	b4 = Infantry("Brigade Petit", "French", blue_division, blue_corps, mobility=4, size=4, quality=4, morale=6)
	b5 = Infantry("Brigade Desvaux", "French", blue_division, blue_corps, mobility=4, size=4, quality=2, morale=5)

	# Yellow forces (Austrian brigadier generals)
	r1 = Infantry("Brigade Klenau", "Austrian", yellow_division, yellow_corps, mobility=4, size=8, quality=3, morale=7)
	r2 = Infantry("Brigade Hohenlohe", "Austrian", yellow_division, yellow_corps, mobility=4, size=6, quality=4, morale=8)
	r3 = Infantry("Brigade Vincent", "Austrian", yellow_division, yellow_corps, mobility=4, size=4, quality=4, morale=6)
	r4 = Infantry("Brigade Lichtenstein", "Austrian", yellow_division, yellow_corps, mobility=4, size=6, quality=4, morale=7)
	r5 = Infantry("Brigade Vukassovich", "Austrian", yellow_division, yellow_corps, mobility=4, size=5, quality=3, morale=6)

	# Place Blue units (left/center)
	game_map.place_unit(b1, 3, 3)
	game_map.place_unit(b2, 4, 3)
	game_map.place_unit(b3, 3, 2)
	game_map.place_unit(b4, 2, 4)
	game_map.place_unit(b5, 4, 4)

	# Place Yellow units (right/center)
	game_map.place_unit(r1, 8, 3)
	game_map.place_unit(r2, 9, 3)
	game_map.place_unit(r3, 8, 5)
	game_map.place_unit(r4, 9, 5)
	game_map.place_unit(r5, 7, 4)

	# label features and print report
	game_map.label_terrain_features(seed=123)
	report = ReportGenerator(game_map).generate_general_summary()
	print(report)
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
	running = True

	# Load general presets from JSON
	with open("general_presets.json", "r", encoding="utf-8") as f:
		general_presets = json.load(f)

	blue_general_preset = general_presets["marmont_preset"]
	yellow_general_preset = general_presets["schwarzenberg_preset"]

	# set up remote host
	# use None for local ollama
	host = "25.0.75.161" # None

	# specify model
	# model = "deepseek-r1:8b"
	model = "llama3.2:3b"

	blue_general = General(unit_list=game_map.get_units_by_faction("French"), faction="French", model=model, identity_prompt=blue_general_preset, ollama_host=host)
	yellow_general = General(unit_list=game_map.get_units_by_faction("Austrian"), faction="Austrian", model=model, identity_prompt=yellow_general_preset, ollama_host=host)

	# Staff officers that translate orders into concrete moves via tools
	staff_officers = {
		"French": StaffOfficer(name=blue_general.name, unit_list=game_map.get_units_by_faction("French"), game_map=game_map, model=model, ollama_host=host),
		"Austrian": StaffOfficer(name=yellow_general.name, unit_list=game_map.get_units_by_faction("Austrian"), game_map=game_map, model=model, ollama_host=host),
	}

	prompt_queue = queue.Queue()
	# Event to gate when input is allowed to avoid prompting before model output
	input_allowed = threading.Event()
	input_allowed.set()  # allow the very first prompt

	def input_thread():
		while running:
			# Wait until the main loop signals that we're ready to accept input
			input_allowed.wait()
			if not running:
				break
			try:
				prompt = input("\nType your orders for the General (or 'quit' to exit): ")
				prompt_queue.put(prompt)
				# Prevent immediately re-prompting until the main loop finishes processing
				input_allowed.clear()
			except EOFError:
				break
	t = threading.Thread(target=input_thread, daemon=True)
	t.start()

	# Alternate turns between French and Austrian
	current_faction_index = 0
	generals = {"French": blue_general, "Austrian": yellow_general}
	faction_names = ["French", "Austrian"]

	while running:
		mouse_pos = pygame.mouse.get_pos()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				running = False

		# Only advance turn when player enters a prompt
		if not prompt_queue.empty():
			player_prompt = prompt_queue.get()
			if player_prompt.strip().lower() == "quit":
				running = False
				# Unblock input thread if it's waiting
				input_allowed.set()
				continue

			# Determine which general's turn it is
			current_faction = faction_names[current_faction_index]
			general = generals[current_faction]
			print(f"\n[{current_faction} General's Turn]")

			map_summary = ReportGenerator(game_map).generate_general_summary()
			general_response = general.get_instructions(player_instructions=player_prompt, map_summary=map_summary)
			print(f"\n[{current_faction} General's Orders]:\n" + general_response)

			# Staff officer executes the orders by calling movement tools
			try:
				so = staff_officers[current_faction]
				applied = so.process_orders(general_response, map_summary=map_summary, faction=current_faction)
				if applied.get("ok"):
					moves = applied.get("applied", [])
					validation = applied.get("validation", {})
					if moves:
						print("\n[Staff Officer Applied Moves]")
						for m in moves:
							print(f"- {m.get('tool')} {m.get('args')} -> {m.get('result')}")
					print(f"\n[Validation] All {len(moves)} units received orders.")
				else:
					# Validation failed after all retries
					print(f"\n[Staff Officer ERROR] {applied.get('error')}")
					validation = applied.get("validation", {})
					if validation:
						print(f"[Validation Details]")
						if validation.get("missing"):
							print(f"  Missing orders for: {', '.join(validation['missing'])}")
						if validation.get("duplicates"):
							print(f"  Duplicate orders for: {', '.join(validation['duplicates'])}")
						print(f"  Unit coverage: {validation.get('unit_coverage', {})}")
					print("[Turn skipped - validation failed after retries]")
					# Don't advance turn or run turn_manager when validation fails
					if running:
						input_allowed.set()
					continue
			except Exception as e:
				print(f"[Staff Officer Error] {e}")
				if running:
					input_allowed.set()
				continue

			turn_manager.run_turns(1)
			report = ReportGenerator(game_map).generate_general_summary()
			print(report)

			# Alternate to the other faction for the next prompt
			current_faction_index = (current_faction_index + 1) % 2

			# Signal the input thread we're ready for the next user prompt
			if running:
				input_allowed.set()

		hover_info = vis.get_hover_info(mouse_pos)
		vis.render(hover_info)
		pygame.display.flip()
		clock.tick(30)
	pygame.quit()
	sys.exit()

if __name__ == "__main__":
	main()