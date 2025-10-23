
from map import Map
from terrain import FIELDS, HILL, RIVER, FOREST
from unit import Infantry, Cavalry, Artillery
from turnmanager import TurnManager
from visualization import Visualization, WINDOW_W, WINDOW_H
import pygame
import sys
from OpenGL.GLU import gluOrtho2D

def create_demo_map() -> Map:
	w, h = 12, 9
	game_map = Map(w, h)
	# scatter some terrain
	game_map.set_terrain(4, 3, HILL)
	game_map.set_terrain(5, 3, HILL)
	game_map.set_terrain(6, 3, HILL)
	game_map.set_terrain(3, 4, FOREST)
	game_map.set_terrain(7, 2, FOREST)
	game_map.set_terrain(8, 5, RIVER)
	game_map.set_terrain(2, 6, RIVER)
	# place units
	u1 = Infantry("1st Line", "Blue", mobility=4, size=8, quality=3, morale=7)
	u2 = Cavalry("Horse", "Red", mobility=6, size=5, quality=4, morale=8)
	u3 = Artillery("Battery", "Blue", mobility=1, size=4, quality=4, morale=6)
	game_map.place_unit(u1, 3, 3)
	game_map.place_unit(u2, 6, 3)
	game_map.place_unit(u3, 2, 2)
	return game_map

def main():
	pygame.init()
	pygame.display.set_caption("Hex Map Renderer - Napoleonic Wargame Demo")
	screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.OPENGL | pygame.DOUBLEBUF)
	gluOrtho2D(0, WINDOW_W, WINDOW_H, 0)
	clock = pygame.time.Clock()
	game_map = create_demo_map()
	factions = ["Blue", "Red"]
	turn_manager = TurnManager(game_map, factions)
	vis = Visualization(game_map)
	running = True
	TURN_INTERVAL = 1.5  # seconds per turn
	import time
	last_turn_time = time.time()
	while running:
		mouse_pos = pygame.mouse.get_pos()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				running = False
		# Run a turn every TURN_INTERVAL seconds
		now = time.time()
		if now - last_turn_time > TURN_INTERVAL:
			turn_manager.run_turns(1)
			last_turn_time = now
		hover_info = vis.get_hover_info(mouse_pos)
		vis.render(hover_info)
		pygame.display.flip()
		clock.tick(30)
	pygame.quit()
	sys.exit()

if __name__ == "__main__":
	main()