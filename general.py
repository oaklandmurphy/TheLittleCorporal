import ollama
import json
import threading
from typing import Optional, Callable, Dict, Any, List
from pydantic import BaseModel

# =====================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# =====================================================

class UnitAssignment(BaseModel):
	"""Single unit assignment to an action."""
	unit_name: str
	action_name: str

class BattlePlanAssignments(BaseModel):
	"""Complete set of unit assignments for the battle plan."""
	assignments: List[UnitAssignment]

class General:
	def __init__(self, faction: str, identity_prompt, unit_list=None, game_map=None, model: str = "llama3.2:3b", ollama_host: str = None):
		"""
		llm_command: List[str] - The command to run the local LLM (e.g., ["ollama", "run", "mymodel"])
		ollama_host: str - The Ollama API host URL (e.g., "http://localhost:11434")
		game_map: Map - Reference to the game map for reconnaissance tools
		"""
		self.llm_command = ["ollama", "run", model]
		self.model = model
		self.ollama_host = ollama_host
		if ollama_host:
			self.client = ollama.Client(host=ollama_host)
		else:
			self.client = ollama
		self.name = identity_prompt.get("name", "General")
		self.description = identity_prompt.get("description", "")
		self.faction = faction
		self.unit_list = unit_list
		self.game_map = game_map
		self.unit_summary = self.update_unit_summary()

	def get_instructions(self, player_instructions="", map_summary="", callback: Optional[Callable[[str], None]] = None):
		"""
		Passes player instructions and map summary to the LLM and returns the LLM's response as a general.
		Uses a two-stage process:
		1. General creates high-level plan and identifies features to investigate
		2. General uses reconnaissance tools to gather detailed intelligence
		3. General writes specific orders based on gathered intelligence
		
		If callback is provided, the query runs in a background thread and callback is called with the result.
		Otherwise, blocks until result is available.
		"""
		if callback:
			# Run asynchronously in a background thread
			def run_query():
				result = self._query_general_with_tools(player_instructions, map_summary)
				callback(result)
			
			thread = threading.Thread(target=run_query, daemon=True)
			thread.start()
			return None  # callback will receive result
		else:
			# Synchronous call (for backward compatibility)
			return self._query_general_with_tools(player_instructions, map_summary)

	def _build_prompt(self, player_instructions, map_summary):
		"""
		Builds a prompt for the LLM to act as a battlefield general.
		"""
		if player_instructions.strip() == "":
			player_instructions = "You have received no orders, act according to your best judgement."

		system_prompt = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n"
			"Given the following battlefield summary and orders from the user, respond with clear, concise orders for your troops.\n"
			f"Battlefield Summary:\n{map_summary}\n"
			f"Your response must be in the form of a list of one line, direct orders to each of the following {len(self.unit_list)} units ({', '.join([unit.name for unit in self.unit_list])}) and nothing else.\n"
			"You should reference a location on the battlefield when giving orders."
		)

		prompt = f"Your orders are: {player_instructions}\n"

		return system_prompt, prompt

	def update_unit_summary(self):
		"""
		Generates a summary of the general's units.
		"""
		if not self.unit_list:
			return "No units assigned."
		summaries = []
		for unit in self.unit_list:
			summaries.append(f"{unit.status_general()}\n")
		return "\n".join(summaries)

	# =====================================================
	# RECONNAISSANCE TOOLS
	# =====================================================

	@property
	def reconnaissance_tools(self) -> List[Dict[str, Any]]:
		"""Define reconnaissance tools available to the General for intelligence gathering."""
		return [
			{
				"type": "function",
				"function": {
					"name": "reconnaissance_feature",
					"description": "Get detailed information about a specific terrain feature (hill, river, forest, valley, etc.) including units present, nearby units, terrain type, and coordinates.",
					"parameters": {
						"type": "object",
						"properties": {
							"feature_name": {
								"type": "string",
								"description": "The name of the terrain feature to investigate (e.g., 'Po River', 'San Marco Heights', 'Verde Forest')"
							}
						},
						"required": ["feature_name"]
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "assess_enemy_strength",
					"description": "Analyze enemy forces near a specific location or feature. Returns information about enemy unit positions, strengths, and capabilities.",
					"parameters": {
						"type": "object",
						"properties": {
							"location": {
								"type": "string",
								"description": "The terrain feature or area to assess enemy strength around (e.g., 'Po River', 'San Marco Heights')"
							}
						},
						"required": ["location"]
					}
				}
			},
			{
				"type": "function",
				"function": {
					"name": "survey_approaches",
					"description": "Identify possible approach routes and tactical considerations for moving toward a specific terrain feature or location.",
					"parameters": {
						"type": "object",
						"properties": {
							"target_feature": {
								"type": "string",
								"description": "The terrain feature or objective to survey approaches to (e.g., 'Po River', 'Verde Forest')"
							}
						},
						"required": ["target_feature"]
					}
				}
			}
		]

	# =====================================================
	# ACTION DEFINITION TOOLS
	# =====================================================

	@property
	def action_definition_tools(self) -> List[Dict[str, Any]]:
		"""Define tools for the General to define high-level actions."""
		return [
			{
				"type": "function",
				"function": {
					"name": "define_action",
					"description": "Define a high-level tactical action to execute. You should define 1-3 actions total. Examples: 'Attack Santa Maria Heights', 'Support Brigade Gudin', 'Defend the river crossing at Lodi Stream'.",
					"parameters": {
						"type": "object",
						"properties": {
							"action_name": {
								"type": "string",
								"description": "Short name for the action (e.g., 'Attack Heights', 'Support Center', 'Defend River')"
							},
							"description": {
								"type": "string",
								"description": "Detailed description of what this action entails and its tactical purpose"
							},
							"primary_objective": {
								"type": "string",
								"description": "The main terrain feature or unit this action focuses on (e.g., 'Santa Maria Heights', 'Brigade Gudin', 'Lodi Stream')"
							}
						},
						"required": ["action_name", "description", "primary_objective"]
					}
				}
			}
		]

	# =====================================================
	# UNIT ASSIGNMENT TOOLS
	# =====================================================

	@property
	def unit_assignment_tools(self) -> List[Dict[str, Any]]:
		"""Define tools for assigning units to actions."""
		return [
			{
				"type": "function",
				"function": {
					"name": "assign_unit_to_action",
					"description": "Assign a specific unit to one of the defined actions. Each unit must be assigned to exactly one action.",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {
								"type": "string",
								"description": "Name of the unit to assign (must exactly match one of the units under your command)"
							},
							"action_name": {
								"type": "string",
								"description": "Name of the action to assign this unit to (must match one of the actions you defined)"
							}
						},
						"required": ["unit_name", "action_name"]
					}
				}
			}
		]

	def _execute_reconnaissance_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute a reconnaissance tool and return the results."""
		if not self.game_map:
			return {"ok": False, "error": "No map available for reconnaissance"}

		try:
			if tool_name == "reconnaissance_feature":
				feature_name = args.get("feature_name", "")
				coords = self.game_map.get_feature_coordinates(feature_name)
				if not coords:
					return {"ok": False, "error": f"No feature named '{feature_name}' found."}

				# Use a battlefield scale for reporting distances in meters
				# Assumption: 1 hex ≈ 100 meters (common operational scale). Adjust if your project uses a different scale.
				METERS_PER_HEX = 100
				NEAR_RADIUS_METERS = 300  # report units within ~3 hexes as nearby
				near_radius_hex = max(1, int(round(NEAR_RADIUS_METERS / METERS_PER_HEX)))

				# Determine predominant terrain type for the feature
				terrain_count = {}
				for x, y in coords:
					tname = self.game_map.grid[y][x].terrain.name
					terrain_count[tname] = terrain_count.get(tname, 0) + 1
				terrain_type = max(terrain_count, key=terrain_count.get)

				# Compute a simple center of the feature for distance phrasing
				cx = int(sum(x for x, _ in coords) / len(coords))
				cy = int(sum(y for _, y in coords) / len(coords))

				# Units located on the feature
				units_on = []
				for x, y in coords:
					u = self.game_map.grid[y][x].unit
					if u:
						units_on.append(u)

				# Units near the feature (within NEAR_RADIUS_METERS but not on it)
				units_near: list[dict] = []
				seen_units = set()
				for yy in range(self.game_map.height):
					for xx in range(self.game_map.width):
						u = self.game_map.grid[yy][xx].unit
						if not u:
							continue
						# Skip if the unit is already counted as being on the feature
						if u in units_on:
							continue
						if id(u) in seen_units:
							continue
						# Compute minimum hex distance from the unit to any tile of the feature
						min_d = float('inf')
						for fx, fy in coords:
							d = self.game_map._hex_distance(xx, yy, fx, fy)
							if d < min_d:
								min_d = d
						# Consider as nearby if within threshold and not on the feature
						if 0 < min_d <= near_radius_hex:
							units_near.append({
								"unit": u,
								"distance_m": int(min_d * METERS_PER_HEX)
							})
							seen_units.add(id(u))

				# Compose description without mentioning hexes or grid coordinates
				desc_lines = []
				desc_lines.append(f"Feature '{feature_name}':")
				desc_lines.append(f"  Terrain: {terrain_type}")

				if units_on:
					# Group by faction for readability
					by_faction = {}
					for u in units_on:
						by_faction.setdefault(u.faction, []).append(u.name)
					parts = []
					for fac, names in sorted(by_faction.items()):
						parts.append(f"{fac}: {', '.join(sorted(names))}")
					desc_lines.append("  Units present: " + "; ".join(parts))
				else:
					desc_lines.append("  Units present: None")

				if units_near:
					# Sort by increasing distance
					units_near.sort(key=lambda e: (e["distance_m"], e["unit"].faction, e["unit"].name))
					near_parts = [
						f"{e['unit'].name} ({e['unit'].faction}) approximately {e['distance_m']} meters away"
						for e in units_near
					]
					desc_lines.append("  Nearby units: " + "; ".join(near_parts))
				else:
					desc_lines.append(f"  Nearby units: None within {NEAR_RADIUS_METERS} meters")

				description = "\n".join(desc_lines)
				return {"ok": True, "intelligence": description}

			elif tool_name == "assess_enemy_strength":
				location = args.get("location", "")
				# Get feature coordinates
				coords = self.game_map.get_feature_coordinates(location)
				if not coords:
					return {"ok": False, "error": f"Unknown location: {location}"}
				
				# Find enemy units within 2 hexes of the feature
				enemy_units = []
				for y in range(self.game_map.height):
					for x in range(self.game_map.width):
						unit = self.game_map.grid[y][x].unit
						if unit and unit.faction != self.faction:
							# Check if within 3 hexes of any feature coordinate
							for fx, fy in coords:
								dist = self.game_map._hex_distance(x, y, fx, fy)
								if dist <= 2:
									enemy_units.append({
										"name": unit.name,
										"position": (x, y),
										"size": unit.size,
										"quality": unit.quality,
										"morale": unit.morale,
										"distance_from_location": dist
									})
									break
				
				assessment = f"Enemy strength assessment near {location}:\n"
				if enemy_units:
					assessment += f"Detected {len(enemy_units)} enemy unit(s):\n"
					for eu in enemy_units:
						assessment += f"  - {eu['name']} at ({eu['position'][0]},{eu['position'][1]}): "
						assessment += f"Strength {eu['strength']}, Quality {eu['quality']}, Morale {eu['morale']}, "
						assessment += f"{eu['distance_from_location']} hexes away\n"
				else:
					assessment += "No enemy units detected within 3 hexes.\n"
				
				return {"ok": True, "intelligence": assessment}

			elif tool_name == "survey_approaches":
				target_feature = args.get("target_feature", "")
				coords = self.game_map.get_feature_coordinates(target_feature)
				if not coords:
					return {"ok": False, "error": f"Unknown feature: {target_feature}"}
				
				# Calculate average position of units under command
				if not self.unit_list:
					return {"ok": False, "error": "No units under command"}
				
				avg_x = sum(unit.x for unit in self.unit_list if unit.x is not None) / len([u for u in self.unit_list if u.x is not None])
				avg_y = sum(unit.y for unit in self.unit_list if unit.y is not None) / len([u for u in self.unit_list if u.y is not None])
				avg_pos = (int(avg_x), int(avg_y))
				
				# Calculate target center position
				target_x = sum(x for x, y in coords) / len(coords)
				target_y = sum(y for x, y in coords) / len(coords)
				target_center = (int(target_x), int(target_y))

				# How many features to highlight
				top_n_features = 3
				
				# Helper function to get cardinal direction from target perspective
				def get_cardinal_direction(from_x, from_y, to_x, to_y):
					"""Get cardinal direction of 'to' point relative to 'from' point"""
					dx = to_x - from_x
					dy = to_y - from_y
					
					# Normalize for hex grid (even-q offset coordinates)
					# In hex grids, vertical is clearer than horizontal
					abs_dx = abs(dx)
					abs_dy = abs(dy)
					
					# Determine primary and secondary directions
					primary = ""
					secondary = ""
					
					# Vertical direction (north/south)
					if abs_dy > abs_dx * 0.5:  # Primarily vertical
						if dy < 0:
							primary = "north"
						else:
							primary = "south"
						# Add east/west if significant horizontal component
						if abs_dx > abs_dy * 0.3:
							if dx > 0:
								secondary = "east"
							else:
								secondary = "west"
					else:  # Primarily horizontal
						if dx > 0:
							primary = "east"
						else:
							primary = "west"
						# Add north/south if significant vertical component
						if abs_dy > abs_dx * 0.3:
							if dy < 0:
								secondary = "north"
							else:
								secondary = "south"
					
					if secondary:
						return f"{secondary}{primary}ern"
					else:
						return primary + "ern" if primary in ["north", "south", "east", "west"] else primary

				# Helper: get enemy units near a set of coordinates (within N hexes)
				def get_enemy_near_coords(coord_list, max_distance=3):
					enemy_units = []
					for y in range(self.game_map.height):
						for x in range(self.game_map.width):
							unit = self.game_map.grid[y][x].unit
							if unit and unit.faction != self.faction:

								# compute nearest distance to the feature
								min_dist = float('inf')
								for fx, fy in coord_list:
									d = self.game_map._hex_distance(x, y, fx, fy)
									if d < min_dist:
										min_dist = d

								if min_dist <= max_distance:

									enemy_units.append({
										"name": unit.name,
										"position": (x, y),
										"strength": unit.strength,
										"quality": unit.quality,
										"morale": unit.morale,
										"distance": min_dist
									})
					return enemy_units
				
				survey = f"Approach survey for {target_feature}:\n"
				survey += f"Your forces are positioned at approximately ({avg_pos[0]}, {avg_pos[1]})\n"
				survey += f"Target feature '{target_feature}' is centered at ({target_center[0]}, {target_center[1]})\n\n"
				
				# Trace a path from average position to target
				# Use simple line drawing algorithm adapted for hex grid
				path_hexes = []
				current_x, current_y = avg_pos
				
				# Build path by moving toward target
				max_steps = 50  # Prevent infinite loops
				steps = 0
				visited = set()
				visited.add((current_x, current_y))
				
				while steps < max_steps and (current_x, current_y) not in coords:
					steps += 1
					best_neighbor = None
					best_distance = float('inf')
					
					# Find neighbor closest to target
					for nx, ny in self.game_map.get_neighbors(current_x, current_y):
						if 0 <= nx < self.game_map.width and 0 <= ny < self.game_map.height:
							if (nx, ny) not in visited:
								dist = self.game_map._hex_distance(nx, ny, target_center[0], target_center[1])
								if dist < best_distance:
									best_distance = dist
									best_neighbor = (nx, ny)
					
					if best_neighbor is None:
						break
					
					current_x, current_y = best_neighbor
					visited.add((current_x, current_y))
					path_hexes.append((current_x, current_y))
				
				# Analyze terrain along the path
				survey += f"Terrain along approach path ({len(path_hexes)} hexes):\n"
				terrain_summary = {}
				for px, py in path_hexes:
					terrain = self.game_map.grid[py][px].terrain.name
					terrain_summary[terrain] = terrain_summary.get(terrain, 0) + 1
				
				for terrain_type, count in sorted(terrain_summary.items(), key=lambda x: -x[1]):
					survey += f"  - {count} {terrain_type} hex(es)\n"

				# Build a map of all features -> list of coordinates (used for size and proximity checks)
				all_features = {}
				for y in range(self.game_map.height):
					for x in range(self.game_map.width):
						for feature in self.game_map.grid[y][x].features:
							if feature not in all_features:
								all_features[feature] = []
							all_features[feature].append((x, y))

				# Identify features along the path, then pick only the largest few and describe nearby enemies
				features_along_path = set()
				for px, py in path_hexes:
					for feature in self.game_map.grid[py][px].features:
						features_along_path.add(feature)

				if features_along_path:
					sized_path_features = [
						(feature, len(all_features.get(feature, [])))
						for feature in features_along_path
						if feature != target_feature
					]
					sized_path_features.sort(key=lambda kv: kv[1], reverse=True)
					selected_path_features = sized_path_features[:top_n_features]
					if selected_path_features:
						survey += f"\nKey features along the approach (largest first):\n"
						for feature, size in selected_path_features:
							coords_list = all_features.get(feature, [])
							enemy_units = get_enemy_near_coords(coords_list, max_distance=3)
							if enemy_units:
								# Build enemy summary
								enemy_summary = f"Detected {len(enemy_units)} enemy unit(s):\n"
								for eu in sorted(enemy_units, key=lambda e: e["distance"]):
									enemy_summary += (
										f"      * {eu['name']} at ({eu['position'][0]},{eu['position'][1]}) - "
										f"Str {eu['strength']}, Qlty {eu['quality']}, Mor {eu['morale']}, {eu['distance']} hexes away\n"
									)
							else:
								enemy_summary = "No enemy units detected within 3 hexes.\n"
							survey += f"  - {feature} ({size} hexes):\n{enemy_summary}"
				
				# Identify flanking features
				# Look for features perpendicular to the approach vector
				survey += f"\nFlanking features:\n"

				# Identify flanking features (not on the direct path but near it)
				flanking_features = {}
				for feature_name, feature_coords in all_features.items():
					if feature_name == target_feature or feature_name in features_along_path:
						continue  # Skip target and features directly on path
					
					# Calculate average distance to path
					min_dist_to_path = float('inf')
					feature_center_x = sum(x for x, y in feature_coords) / len(feature_coords)
					feature_center_y = sum(y for x, y in feature_coords) / len(feature_coords)
					
					for px, py in path_hexes:
						dist = self.game_map._hex_distance(int(feature_center_x), int(feature_center_y), px, py)
						min_dist_to_path = min(min_dist_to_path, dist)
					
					# Consider as flanking feature if within 3 hexes of path
					if min_dist_to_path <= 2:
						# Get direction relative to target
						direction = get_cardinal_direction(
							target_center[0], target_center[1],
							int(feature_center_x), int(feature_center_y)
						)
						
						# Get terrain type of the feature
						terrain_type = self.game_map.grid[int(feature_center_y)][int(feature_center_x)].terrain.name.lower()
						
						flanking_features[feature_name] = {
							"direction": direction,
							"distance": min_dist_to_path,
							"terrain": terrain_type,
							"size": len(feature_coords),
							"coords": feature_coords
						}
				
				if flanking_features:
					# Choose the largest few flanking features
					selected_flanks = sorted(flanking_features.items(), key=lambda kv: kv[1]["size"], reverse=True)[:top_n_features]
					for feature_name, info in selected_flanks:
						enemy_units = get_enemy_near_coords(info["coords"], max_distance=3)
						if enemy_units:
							enemy_summary = f"Detected {len(enemy_units)} enemy unit(s):\n"
							for eu in sorted(enemy_units, key=lambda e: e["distance"]):
								enemy_summary += (
									f"      * {eu['name']} at ({eu['position'][0]},{eu['position'][1]}) - "
									f"Str {eu['strength']}, Qlty {eu['quality']}, Mor {eu['morale']}, {eu['distance']} hexes away\n"
								)
						else:
							enemy_summary = "No enemy units detected within 3 hexes.\n"
						survey += (
							f"  - {feature_name} ({info['size']} hexes), {info['terrain']} on the {info['direction']} flank "
							f"({info['distance']} hexes from approach):\n{enemy_summary}"
						)
				else:
					survey += "  No significant flanking features detected.\n"
				
				return {"ok": True, "intelligence": survey}

			else:
				return {"ok": False, "error": f"Unknown reconnaissance tool: {tool_name}"}

		except Exception as e:
			return {"ok": False, "error": f"Reconnaissance error: {str(e)}"}

	# =====================================================
	# NEW 7-STEP WORKFLOW WITH TOOL FORCING
	# =====================================================

	def _query_general_with_tools(self, player_instructions, map_summary, num_thread=4, num_ctx=4096):
		"""
		New 7-step order generation process:
		1. System prompt with battlefield overview
		2. User gives general orders
		3. General calls reconnaissance tools for information
		4. General defines 1-3 high-level actions
		5. General calls more reconnaissance tools
		6. General receives detailed unit information
		7. General assigns units to actions and returns JSON
		"""
		if player_instructions.strip() == "":
			player_instructions = "You have received no orders, act according to your best judgement."

		# STEP 1 & 2: Battlefield overview (system prompt) + Player orders (user message)
		print(f"\n{'='*60}")
		print(f"[STEP 1-2] {self.name} receiving orders...")
		print(f"{'='*60}")
		print(f"Orders: {player_instructions}\n")

		# STEP 3: Initial reconnaissance
		print(f"{'='*60}")
		print(f"[STEP 3] {self.name} conducting initial reconnaissance...")
		print(f"{'='*60}")
		
		recon_system_1 = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"Battlefield Summary:\n{map_summary}\n\n"
			f"Your orders: {player_instructions}\n\n"
			"TASK: Gather intelligence about the battlefield.\n"
			"Call 2-4 reconnaissance tools to investigate key terrain features or enemy positions.\n"
			"Available tools:\n"
			"- reconnaissance_feature: Get detailed info about a terrain feature\n"
			"- assess_enemy_strength: Analyze enemy forces near a location\n"
			"- survey_approaches: Identify approach routes to a feature\n\n"
			"Use ONLY tool calls - no text responses."
		)
		
		messages_recon_1 = [
			{"role": "system", "content": recon_system_1},
			{"role": "user", "content": "Begin reconnaissance by calling tools."}
		]
		
		intelligence_1 = self._conduct_reconnaissance_phase(messages_recon_1, num_thread, num_ctx, min_calls=2, max_calls=4)

		# STEP 4: Define 1-3 actions
		print(f"\n{'='*60}")
		print(f"[STEP 4] {self.name} defining tactical actions...")
		print(f"{'='*60}")
		
		intelligence_summary_1 = "\n\n".join(intelligence_1) if intelligence_1 else "No intelligence gathered."
		
		actions_system = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"Your orders: {player_instructions}\n\n"
			f"INTELLIGENCE GATHERED:\n{intelligence_summary_1}\n\n"
			"TASK: Define 1-3 high-level tactical actions based on your orders and intelligence.\n"
			"Examples of actions:\n"
			"- Attack Santa Maria Heights\n"
			"- Support Brigade Gudin's advance\n"
			"- Defend the river crossing at Lodi Stream\n\n"
			"Call the 'define_action' tool 1-3 times to create your actions.\n"
			"Use ONLY tool calls - no text responses."
		)
		
		messages_actions = [
			{"role": "system", "content": actions_system},
			{"role": "user", "content": "Define your tactical actions now."}
		]
		
		defined_actions = self._define_actions_phase(messages_actions, num_thread, num_ctx)

		# STEP 5: Additional reconnaissance
		print(f"\n{'='*60}")
		print(f"[STEP 5] {self.name} conducting detailed reconnaissance...")
		print(f"{'='*60}")
		
		actions_summary = "\n".join([f"- {a['name']}: {a['description']}" for a in defined_actions])
		
		recon_system_2 = (
			f"You are {self.name}, conducting detailed reconnaissance.\n"
			f"{self.description}\n\n"
			f"Your orders: {player_instructions}\n\n"
			f"Your defined actions:\n{actions_summary}\n\n"
			f"Previous intelligence:\n{intelligence_summary_1}\n\n"
			"TASK: Gather additional intelligence to support your actions.\n"
			"Call 1-3 reconnaissance tools to investigate specific details.\n"
			"Use ONLY tool calls - no text responses."
		)
		
		messages_recon_2 = [
			{"role": "system", "content": recon_system_2},
			{"role": "user", "content": "Conduct additional reconnaissance."}
		]
		
		intelligence_2 = self._conduct_reconnaissance_phase(messages_recon_2, num_thread, num_ctx, min_calls=1, max_calls=3)

		# STEP 6: Provide detailed unit information
		print(f"\n{'='*60}")
		print(f"[STEP 6] {self.name} reviewing unit dispositions...")
		print(f"{'='*60}")
		
		unit_details = self._get_detailed_unit_info()
		print(unit_details)

		# STEP 7: Assign units to actions and return JSON
		print(f"\n{'='*60}")
		print(f"[STEP 7] {self.name} assigning units to actions...")
		print(f"{'='*60}")
		
		intelligence_summary_2 = "\n\n".join(intelligence_2) if intelligence_2 else "No additional intelligence."
		all_intelligence = f"{intelligence_summary_1}\n\n{intelligence_summary_2}"
		
		assignment_system = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"Your orders: {player_instructions}\n\n"
			f"Your defined actions:\n{actions_summary}\n\n"
			f"All intelligence gathered:\n{all_intelligence}\n\n"
			f"Your units:\n{unit_details}\n\n"
			f"TASK: Assign each of your {len(self.unit_list)} units to one of your actions.\n"
			f"Call 'assign_unit_to_action' exactly {len(self.unit_list)} times - once per unit.\n"
			"Each unit must be assigned to exactly one action.\n"
			"Use ONLY tool calls - no text responses."
		)
		
		messages_assignment = [
			{"role": "system", "content": assignment_system},
			{"role": "user", "content": "Assign your units to actions now."}
		]
		
		assignments = self._assign_units_phase(messages_assignment, defined_actions, num_thread, num_ctx)

		# Build final JSON output
		final_output = self._build_json_output(defined_actions, assignments)
		
		print(f"\n{'='*60}")
		print(f"[FINAL OUTPUT] {self.name}'s battle plan:")
		print(f"{'='*60}")
		print(json.dumps(final_output, indent=2))
		print(f"{'='*60}\n")
		
		return json.dumps(final_output, indent=2)

	def _conduct_reconnaissance_phase(self, messages, num_thread, num_ctx, min_calls=2, max_calls=5):
		"""Conduct reconnaissance and gather intelligence."""
		intelligence_gathered = []
		max_rounds = max_calls
		
		for round_num in range(max_rounds):
			response = self.client.chat(
				model=self.model,
				messages=messages,
				tools=self.reconnaissance_tools,
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			msg = response.get("message", {})
			tool_calls = msg.get("tool_calls") or []
			
			if not tool_calls:
				break
			
			for tc in tool_calls:
				fn = tc.get("function", {})
				tool_name = fn.get("name", "")
				raw_args = fn.get("arguments", {})
				
				try:
					args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
				except Exception:
					args = {}
				
				print(f"\n[Reconnaissance] {tool_name}({args})")
				
				result = self._execute_reconnaissance_tool(tool_name, args)
				
				if result.get("ok"):
					intelligence = result.get("intelligence", "")
					print(f"{intelligence}")
					intelligence_gathered.append(intelligence)
				else:
					error = result.get("error", "Unknown error")
					print(f"[Error] {error}")
				
				tool_msg = {"role": "tool", "content": json.dumps(result)}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
			
			if len(intelligence_gathered) >= min_calls:
				break
		
		return intelligence_gathered

	def _define_actions_phase(self, messages, num_thread, num_ctx):
		"""Have the general define 1-3 tactical actions."""
		defined_actions = []
		max_rounds = 5
		
		for round_num in range(max_rounds):
			response = self.client.chat(
				model=self.model,
				messages=messages,
				tools=self.action_definition_tools,
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			msg = response.get("message", {})
			tool_calls = msg.get("tool_calls") or []
			
			if not tool_calls:
				break
			
			for tc in tool_calls:
				fn = tc.get("function", {})
				tool_name = fn.get("name", "")
				raw_args = fn.get("arguments", {})
				
				try:
					args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
				except Exception:
					args = {}
				
				if tool_name == "define_action" and len(defined_actions) < 3:
					action = {
						"name": args.get("action_name", "Unnamed Action"),
						"description": args.get("description", ""),
						"primary_objective": args.get("primary_objective", "")
					}
					defined_actions.append(action)
					print(f"\n[Action Defined] {action['name']}: {action['description']}")
					
					result = {"ok": True, "message": f"Action '{action['name']}' defined successfully"}
				else:
					result = {"ok": False, "error": "Maximum 3 actions allowed"}
				
				tool_msg = {"role": "tool", "content": json.dumps(result)}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
			
			if len(defined_actions) >= 1:
				break
		
		if not defined_actions:
			# Fallback: create a default action
			defined_actions.append({
				"name": "General Advance",
				"description": "General advance against enemy positions",
				"primary_objective": "Enemy forces"
			})
		
		return defined_actions

	def _assign_units_phase(self, messages, defined_actions, num_thread, num_ctx):
		"""Have the general assign units to actions using structured output."""
		action_names = [a["name"] for a in defined_actions]
		unit_names = [u.name for u in self.unit_list]
		
		# Build comprehensive prompt for structured output
		assignment_prompt = (
			f"You must assign each of your {len(self.unit_list)} units to exactly one action.\n\n"
			f"Available actions:\n"
		)
		for i, action in enumerate(defined_actions, 1):
			assignment_prompt += f"{i}. {action['name']}: {action['description']}\n"
		
		assignment_prompt += f"\nYour units:\n"
		for i, unit in enumerate(self.unit_list, 1):
			assignment_prompt += f"{i}. {unit.name}\n"
		
		assignment_prompt += (
			f"\nAssign each unit to one action. Consider tactical factors like:\n"
			"- Unit position and mobility\n"
			"- Action objectives and unit capabilities\n"
			"- Supporting other units\n"
			"- Concentration of force\n\n"
			"Provide your assignments now."
		)
		
		# Get the last user message content to build complete context
		system_content = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
		
		try:
			# Use structured output with Pydantic schema
			response = self.client.chat(
				model=self.model,
				messages=[
					{"role": "system", "content": system_content},
					{"role": "user", "content": assignment_prompt}
				],
				format=BattlePlanAssignments.model_json_schema(),
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			# Parse the structured output
			battle_plan = BattlePlanAssignments.model_validate_json(response.message.content)
			
			# Convert to dictionary format
			assignments = {}
			for assignment in battle_plan.assignments:
				unit_name = assignment.unit_name
				action_name = assignment.action_name
				
				# Validate assignment
				if unit_name in unit_names and action_name in action_names:
					if unit_name not in assignments:  # Avoid duplicates
						assignments[unit_name] = action_name
						print(f"\n[Assignment] {unit_name} → {action_name}")
					else:
						print(f"\n[Duplicate ignored] {unit_name} was already assigned")
				else:
					if unit_name not in unit_names:
						print(f"\n[Invalid] Unknown unit '{unit_name}' - ignoring")
					if action_name not in action_names:
						print(f"\n[Invalid] Unknown action '{action_name}' - ignoring")
			
			# Assign any remaining units to the first action
			if len(assignments) < len(self.unit_list):
				default_action = defined_actions[0]["name"]
				for unit in self.unit_list:
					if unit.name not in assignments:
						assignments[unit.name] = default_action
						print(f"\n[Auto-Assignment] {unit.name} → {default_action}")
			
			return assignments
			
		except Exception as e:
			print(f"\n[Error] Structured output failed: {e}")
			print("[Fallback] Using default assignment strategy")
			
			# Fallback: assign all units to first action
			default_action = defined_actions[0]["name"]
			assignments = {}
			for unit in self.unit_list:
				assignments[unit.name] = default_action
				print(f"\n[Auto-Assignment] {unit.name} → {default_action}")
			
			return assignments

	def _get_detailed_unit_info(self):
		"""Get detailed information about units under command."""
		lines = [f"Units under your command ({len(self.unit_list)} total):"]
		for i, unit in enumerate(self.unit_list, 1):
			lines.append(f"{i}. {unit.status_general()}")
		return "\n".join(lines)

	def _build_json_output(self, defined_actions, assignments):
		"""Build the final JSON output with actions and assigned units."""
		output = {"orders": []}
		
		for action in defined_actions:
			order_entry = {
				"name": action["name"],
				"target": action["primary_objective"],
				"units": []
			}
			
			# Find all units assigned to this action
			for unit_name, action_name in assignments.items():
				if action_name == action["name"]:
					order_entry["units"].append(unit_name)
			
			output["orders"].append(order_entry)
		
		return output

	def _query_general(self, system_prompt, prompt, num_thread=4, num_ctx=4096):
		"""
		Sends the prompt to the local LLM using the ollama Python library and returns its response.
		Limits CPU and RAM usage by setting num_thread and num_ctx.
		"""
		try:
			# Compose system message with personality/system_prompt info
			system_message = {
				"role": "system",
				"content": system_prompt
			}
			user_message = {"role": "user", "content": prompt}
			response = self.client.chat(
				model=self.model,
				messages=[system_message, user_message],
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			return response["message"]["content"].strip()
		except Exception as e:
			return f"[LLM Error] {e}"