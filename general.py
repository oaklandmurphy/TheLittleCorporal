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
	"""AI-powered battlefield general that generates tactical orders using LLM reasoning."""
	
	# =====================================================
	# CLASS CONSTANTS
	# =====================================================
	METERS_PER_HEX = 100  # Battlefield scale: 1 hex ≈ 100 meters
	NEAR_RADIUS_METERS = 300  # Report units within ~3 hexes as nearby
	ENEMY_DETECTION_RANGE = 3  # Hexes for enemy detection
	MAX_ACTIONS = 3  # Maximum tactical actions to define
	
	def __init__(self, faction: str, identity_prompt, unit_list=None, game_map=None, model: str = "llama3.2:3b", ollama_host: str = None):
		"""Initialize the General with faction, identity, units, and LLM configuration.
		
		Args:
			faction: The general's faction
			identity_prompt: Dict with 'name' and 'description' keys
			unit_list: List of units under command
			game_map: Map instance for reconnaissance
			model: Ollama model name
			ollama_host: Ollama API host URL (optional)
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
		
		# Load preset action names
		self.preset_action_names = self._load_preset_action_names()

	def _load_preset_action_names(self):
		"""Load preset action names from general_presets.json.
		
		Returns:
			List of action name strings
		"""
		try:
			import os
			preset_path = os.path.join(os.path.dirname(__file__), "general_presets.json")
			with open(preset_path, 'r') as f:
				presets = json.load(f)
				return presets.get("action_names", ["Attack", "Defend", "Support", "Advance", "Withdraw", "Hold"])
		except Exception as e:
			print(f"[Warning] Could not load preset action names: {e}")
			return ["Attack", "Defend", "Support", "Advance", "Withdraw", "Hold"]

	# =====================================================
	# PUBLIC API METHODS
	# =====================================================

	def get_instructions(self, player_instructions="", map_summary="", callback: Optional[Callable[[str], None]] = None):
		"""Generate tactical orders based on player instructions and battlefield situation.
		
		Uses a 7-stage process:
		1. General receives orders and battlefield overview
		2. General conducts initial reconnaissance
		3. General defines high-level tactical actions
		4. General conducts detailed reconnaissance
		5. General reviews unit dispositions
		6. General assigns units to actions
		7. Returns structured JSON battle plan
		
		Args:
			player_instructions: Orders from the player
			map_summary: Summary of battlefield state
			callback: Optional callback for async execution
			
		Returns:
			JSON string with battle plan (if synchronous), or None (if async with callback)
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
		"""Generate a summary of the general's units."""
		if not self.unit_list:
			return "No units assigned."
		summaries = []
		for unit in self.unit_list:
			summaries.append(f"{unit.status_general()}\n")
		return "\n".join(summaries)

	# =====================================================
	# HELPER METHODS FOR RECONNAISSANCE
	# =====================================================
	
	def _get_cardinal_direction(self, from_x: int, from_y: int, to_x: int, to_y: int) -> str:
		"""Get cardinal direction of 'to' point relative to 'from' point.
		
		Args:
			from_x, from_y: Origin coordinates
			to_x, to_y: Target coordinates
			
		Returns:
			Cardinal direction string (e.g., 'northern', 'southeastern')
		"""
		dx = to_x - from_x
		dy = to_y - from_y
		
		# Normalize for hex grid (even-q offset coordinates)
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
	
	def _get_enemy_units_near_coords(self, coord_list: List[tuple], max_distance: int = 3) -> List[Dict[str, Any]]:
		"""Get enemy units near a set of coordinates.
		
		Args:
			coord_list: List of (x, y) coordinates
			max_distance: Maximum hex distance to consider
			
		Returns:
			List of enemy unit dictionaries with position, stats, and distance
		"""
		enemy_units = []
		for y in range(self.game_map.height):
			for x in range(self.game_map.width):
				unit = self.game_map.grid[y][x].unit
				if unit and unit.faction != self.faction:
					# Compute nearest distance to the feature
					min_dist = float('inf')
					for fx, fy in coord_list:
						d = self.game_map._hex_distance(x, y, fx, fy)
						if d < min_dist:
							min_dist = d
					
					if min_dist <= max_distance:
						enemy_units.append({
							"name": unit.name,
							"position": (x, y),
							"size": unit.size,
							"quality": unit.quality,
							"morale": unit.morale,
							"distance": min_dist,
							"unit": unit
						})
		return enemy_units
	
	def _get_units_on_feature(self, coords: List[tuple]) -> List[Any]:
		"""Get all units located on a terrain feature.
		
		Args:
			coords: List of (x, y) coordinates defining the feature
			
		Returns:
			List of unit objects on the feature
		"""
		units_on = []
		for x, y in coords:
			u = self.game_map.grid[y][x].unit
			if u:
				units_on.append(u)
		return units_on
	
	def _get_nearby_units(self, coords: List[tuple], near_radius_hex: int) -> List[Dict[str, Any]]:
		"""Get units near (but not on) a feature.
		
		Args:
			coords: List of (x, y) coordinates defining the feature
			near_radius_hex: Radius in hexes to search
			
		Returns:
			List of dicts with 'unit' and 'distance_m' keys
		"""
		# Compute center of feature
		cx = int(sum(x for x, _ in coords) / len(coords))
		cy = int(sum(y for _, y in coords) / len(coords))
		
		units_on = self._get_units_on_feature(coords)
		units_near = []
		seen_units = set()
		
		for yy in range(self.game_map.height):
			for xx in range(self.game_map.width):
				u = self.game_map.grid[yy][xx].unit
				if not u or u in units_on or id(u) in seen_units:
					continue
				
				# Compute minimum hex distance from unit to any tile of the feature
				min_d = float('inf')
				for fx, fy in coords:
					d = self.game_map._hex_distance(xx, yy, fx, fy)
					if d < min_d:
						min_d = d
				
				# Consider as nearby if within threshold and not on the feature
				if 0 < min_d <= near_radius_hex:
					units_near.append({
						"unit": u,
						"distance_m": int(min_d * self.METERS_PER_HEX)
					})
					seen_units.add(id(u))
		
		return units_near
	
	def _get_predominant_terrain(self, coords: List[tuple]) -> str:
		"""Get the most common terrain type for a set of coordinates.
		
		Args:
			coords: List of (x, y) coordinates
			
		Returns:
			Name of predominant terrain type
		"""
		terrain_count = {}
		for x, y in coords:
			tname = self.game_map.grid[y][x].terrain.name
			terrain_count[tname] = terrain_count.get(tname, 0) + 1
		return max(terrain_count, key=terrain_count.get)
	
	def _build_path_to_target(self, start: tuple, target_coords: List[tuple], max_steps: int = 50) -> List[tuple]:
		"""Build a path from start position to target feature.
		
		Args:
			start: Starting (x, y) position
			target_coords: List of (x, y) coordinates defining target
			max_steps: Maximum path length
			
		Returns:
			List of (x, y) coordinates forming the path
		"""
		current_x, current_y = start
		target_x = sum(x for x, _ in target_coords) / len(target_coords)
		target_y = sum(y for _, y in target_coords) / len(target_coords)
		target_center = (int(target_x), int(target_y))
		
		path_hexes = []
		steps = 0
		visited = set()
		visited.add((current_x, current_y))
		
		while steps < max_steps and (current_x, current_y) not in target_coords:
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
		
		return path_hexes

	# =====================================================
	# TOOL DEFINITIONS FOR LLM
	# =====================================================

	# =====================================================
	# RECONNAISSANCE TOOLS
	# =====================================================

	@property
	def reconnaissance_tools(self) -> List[Dict[str, Any]]:
		"""Define reconnaissance tools available to the General for intelligence gathering.
		
		Returns:
			List of tool definition dictionaries
		"""
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
		"""Define tools for the General to define high-level actions.
		
		Returns:
			List of tool definition dictionaries
		"""
		return [
			{
				"type": "function",
				"function": {
					"name": "define_action",
					"description": f"Define a tactical action to execute. Call this tool once for each action needed (often fewer action are best). Choose an action type from: {', '.join(self.preset_action_names)}.",
					"parameters": {
						"type": "object",
						"properties": {
							"action_type": {
								"type": "string",
								"enum": self.preset_action_names,
								"description": f"The type of action from this list: {', '.join(self.preset_action_names)}"
							},
							"action_name": {
								"type": "string",
								"description": "Tactical description only, do NOT include terrain feature names (e.g., 'Frontal Assault', 'Supporting Attack', 'Defensive Screen')"
							},
							"description": {
								"type": "string",
								"description": "Detailed description of what this action entails and its tactical purpose"
							},
							"primary_objective": {
								"type": "string",
								"description": "The EXACT terrain feature name to target (e.g., 'Santa Maria Heights', 'Po River', 'Verde Forest')"
							}
						},
						"required": ["action_type", "action_name", "description", "primary_objective"]
					}
				}
			}
		]

	# =====================================================
	# UNIT ASSIGNMENT TOOLS
	# =====================================================

	@property
	def unit_assignment_tools(self) -> List[Dict[str, Any]]:
		"""Define tools for assigning units to actions.
		
		Returns:
			List of tool definition dictionaries
		"""
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

	# =====================================================
	# TOOL EXECUTION METHODS
	# =====================================================

	def _execute_reconnaissance_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute a reconnaissance tool and return the results.
		
		Args:
			tool_name: Name of the reconnaissance tool
			args: Tool arguments
			
		Returns:
			Dict with 'ok' (bool) and either 'intelligence' or 'error' key
		"""
		if not self.game_map:
			return {"ok": False, "error": "No map available for reconnaissance"}

		try:
			if tool_name == "reconnaissance_feature":
				return self._recon_feature(args)
			elif tool_name == "assess_enemy_strength":
				return self._recon_assess_enemy(args)
			elif tool_name == "survey_approaches":
				return self._recon_survey_approaches(args)
			else:
				return {"ok": False, "error": f"Unknown reconnaissance tool: {tool_name}"}
		except Exception as e:
			return {"ok": False, "error": f"Reconnaissance error: {str(e)}"}
	
	def _recon_feature(self, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute reconnaissance_feature tool: detailed info about a terrain feature.
		
		Args:
			args: Dict with 'feature_name' key
			
		Returns:
			Dict with reconnaissance results
		"""
		feature_name = args.get("feature_name", "")
		coords = self.game_map.get_feature_coordinates(feature_name)
		if not coords:
			return {"ok": False, "error": f"No feature named '{feature_name}' found."}

		near_radius_hex = max(1, int(round(self.NEAR_RADIUS_METERS / self.METERS_PER_HEX)))
		
		# Get terrain type, units on feature, and nearby units
		terrain_type = self._get_predominant_terrain(coords)
		units_on = self._get_units_on_feature(coords)
		units_near = self._get_nearby_units(coords, near_radius_hex)

		# Build description
		desc_lines = [
			f"Feature '{feature_name}':",
			f"  Terrain: {terrain_type}"
		]

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
			desc_lines.append(f"  Nearby units: None within {self.NEAR_RADIUS_METERS} meters")

		return {"ok": True, "intelligence": "\n".join(desc_lines)}
	
	def _recon_assess_enemy(self, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute assess_enemy_strength tool: analyze enemy forces near a location.
		
		Args:
			args: Dict with 'location' key
			
		Returns:
			Dict with enemy strength assessment
		"""
		location = args.get("location", "")
		coords = self.game_map.get_feature_coordinates(location)
		if not coords:
			return {"ok": False, "error": f"Unknown location: {location}"}
		
		# Find enemy units within detection range
		enemy_units = self._get_enemy_units_near_coords(coords, self.ENEMY_DETECTION_RANGE)
		
		assessment = f"Enemy strength assessment near {location}:\n"
		if enemy_units:
			assessment += f"Detected {len(enemy_units)} enemy unit(s):\n"
			for eu in enemy_units:
				unit_desc = eu['unit'].status_general().replace(f"{eu['unit'].name}. ({eu['unit'].__class__.__name__}) ", "")
				assessment += (
					f"  - {eu['name']} at ({eu['position'][0]},{eu['position'][1]}): "
					f"{unit_desc} {eu['distance']} hexes away\n"
				)
		else:
			assessment += f"No enemy units detected within {self.ENEMY_DETECTION_RANGE} hexes.\n"
		
		return {"ok": True, "intelligence": assessment}
	
	def _recon_survey_approaches(self, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute survey_approaches tool: identify approach routes to a target.
		
		Args:
			args: Dict with 'target_feature' key
			
		Returns:
			Dict with approach survey results
		"""
		target_feature = args.get("target_feature", "")
		coords = self.game_map.get_feature_coordinates(target_feature)
		if not coords:
			return {"ok": False, "error": f"Unknown feature: {target_feature}"}
		
		if not self.unit_list:
			return {"ok": False, "error": "No units under command"}
		
		# Calculate average position of forces
		avg_x = sum(unit.x for unit in self.unit_list if unit.x is not None) / len([u for u in self.unit_list if u.x is not None])
		avg_y = sum(unit.y for unit in self.unit_list if unit.y is not None) / len([u for u in self.unit_list if u.y is not None])
		avg_pos = (int(avg_x), int(avg_y))
		
		# Calculate target center
		target_x = sum(x for x, y in coords) / len(coords)
		target_y = sum(y for x, y in coords) / len(coords)
		target_center = (int(target_x), int(target_y))

		survey = (
			f"Approach survey for {target_feature}:\n"
			f"Your forces are positioned at approximately ({avg_pos[0]}, {avg_pos[1]})\n"
			f"Target feature '{target_feature}' is centered at ({target_center[0]}, {target_center[1]})\n\n"
		)
		
		# Build path to target
		path_hexes = self._build_path_to_target(avg_pos, coords)
		
		# Analyze terrain along path
		survey += f"Terrain along approach path ({len(path_hexes)} hexes):\n"
		terrain_summary = {}
		for px, py in path_hexes:
			terrain = self.game_map.grid[py][px].terrain.name
			terrain_summary[terrain] = terrain_summary.get(terrain, 0) + 1
		
		for terrain_type, count in sorted(terrain_summary.items(), key=lambda x: -x[1]):
			survey += f"  - {count} {terrain_type} hex(es)\n"

		# Analyze features along path and flanking features
		survey += self._analyze_path_features(path_hexes, target_feature, target_center)
		
		return {"ok": True, "intelligence": survey}
	
	def _analyze_path_features(self, path_hexes: List[tuple], target_feature: str, target_center: tuple) -> str:
		"""Analyze features along the approach path and flanking positions.
		
		Args:
			path_hexes: List of path coordinates
			target_feature: Name of target feature
			target_center: Center coordinates of target
			
		Returns:
			Formatted string with feature analysis
		"""
		TOP_N_FEATURES = 3
		
		# Build map of all features
		all_features = {}
		for y in range(self.game_map.height):
			for x in range(self.game_map.width):
				for feature in self.game_map.grid[y][x].features:
					if feature not in all_features:
						all_features[feature] = []
					all_features[feature].append((x, y))

		# Identify features along the path
		features_along_path = set()
		for px, py in path_hexes:
			for feature in self.game_map.grid[py][px].features:
				features_along_path.add(feature)

		result = ""
		
		# Report key features along path
		if features_along_path:
			sized_path_features = [
				(feature, len(all_features.get(feature, [])))
				for feature in features_along_path
				if feature != target_feature
			]
			sized_path_features.sort(key=lambda kv: kv[1], reverse=True)
			selected_path_features = sized_path_features[:TOP_N_FEATURES]
			
			if selected_path_features:
				result += f"\nKey features along the approach (largest first):\n"
				for feature, size in selected_path_features:
					coords_list = all_features.get(feature, [])
					enemy_units = self._get_enemy_units_near_coords(coords_list, max_distance=3)
					enemy_summary = self._format_enemy_summary(enemy_units)
					result += f"  - {feature} ({size} hexes):\n{enemy_summary}"
		
		# Identify flanking features
		result += f"\nFlanking features:\n"
		flanking_features = self._identify_flanking_features(all_features, features_along_path, target_feature, path_hexes, target_center)
		
		if flanking_features:
			selected_flanks = sorted(flanking_features.items(), key=lambda kv: kv[1]["size"], reverse=True)[:TOP_N_FEATURES]
			for feature_name, info in selected_flanks:
				enemy_units = self._get_enemy_units_near_coords(info["coords"], max_distance=3)
				enemy_summary = self._format_enemy_summary(enemy_units)
				result += (
					f"  - {feature_name} ({info['size']} hexes), {info['terrain']} on the {info['direction']} flank "
					f"({info['distance']} hexes from approach):\n{enemy_summary}"
				)
		else:
			result += "  No significant flanking features detected.\n"
		
		return result
	
	def _identify_flanking_features(self, all_features: Dict, features_along_path: set, 
	                                 target_feature: str, path_hexes: List[tuple], 
	                                 target_center: tuple) -> Dict[str, Dict[str, Any]]:
		"""Identify features that could serve as flanking positions.
		
		Args:
			all_features: Dict mapping feature names to coordinate lists
			features_along_path: Set of features on the direct path
			target_feature: Name of target feature
			path_hexes: List of path coordinates
			target_center: Center coordinates of target
			
		Returns:
			Dict mapping feature names to their flanking information
		"""
		flanking_features = {}
		
		for feature_name, feature_coords in all_features.items():
			if feature_name == target_feature or feature_name in features_along_path:
				continue
			
			# Calculate distance to path
			min_dist_to_path = float('inf')
			feature_center_x = sum(x for x, y in feature_coords) / len(feature_coords)
			feature_center_y = sum(y for x, y in feature_coords) / len(feature_coords)
			
			for px, py in path_hexes:
				dist = self.game_map._hex_distance(int(feature_center_x), int(feature_center_y), px, py)
				min_dist_to_path = min(min_dist_to_path, dist)
			
			# Consider as flanking feature if within 2 hexes of path
			if min_dist_to_path <= 2:
				direction = self._get_cardinal_direction(
					target_center[0], target_center[1],
					int(feature_center_x), int(feature_center_y)
				)
				
				terrain_type = self.game_map.grid[int(feature_center_y)][int(feature_center_x)].terrain.name.lower()
				
				flanking_features[feature_name] = {
					"direction": direction,
					"distance": min_dist_to_path,
					"terrain": terrain_type,
					"size": len(feature_coords),
					"coords": feature_coords
				}
		
		return flanking_features
	
	def _format_enemy_summary(self, enemy_units: List[Dict[str, Any]]) -> str:
		"""Format enemy unit information for display.
		
		Args:
			enemy_units: List of enemy unit dicts
			
		Returns:
			Formatted string describing enemy units
		"""
		if enemy_units:
			# Summarize by quality and size rather than listing individual units
			total_units = len(enemy_units)
			closest_distance = min(eu.get('distance', 999) for eu in enemy_units)
			
			# Use descriptive labels from Unit class
			size_labels = {range(1, 4): "small", range(4, 7): "average-sized", range(7, 10): "large", range(10, 13): "very large"}
			quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
			morale_labels = {range(0, 2): "broken", range(2, 4): "shaken", range(4, 7): "steady", range(7, 9): "eager", range(9, 11): "fresh"}
			
			def label_for(value, table):
				for key, label in table.items():
					if isinstance(key, range) and value in key:
						return label
					elif value == key:
						return label
				return "unknown"
			
			# Calculate aggregate descriptions
			avg_quality = sum(eu.get('quality', 0) for eu in enemy_units) / total_units if total_units > 0 else 0
			avg_morale = sum(eu.get('morale', 0) for eu in enemy_units) / total_units if total_units > 0 else 0
			total_size = sum(eu.get('size', 0) for eu in enemy_units)
			
			size_desc = label_for(total_size // total_units if total_units > 0 else 0, size_labels)
			quality_desc = label_for(round(avg_quality), quality_labels)
			morale_desc = label_for(round(avg_morale), morale_labels)
			
			summary = (
				f"Detected {total_units} enemy unit{'s' if total_units != 1 else ''}: "
				f"Average {size_desc}, {quality_desc} formations that are {morale_desc}, "
				f"closest at {closest_distance} hexes\n"
			)
		else:
			summary = "No enemy units detected within 3 hexes.\n"
		return summary

	# =====================================================
	# MAIN WORKFLOW - 7-STEP ORDER GENERATION PROCESS
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
		print(f"\n{'='*70}")
		print(f"  STEP 1-2: {self.name.upper()} - RECEIVING ORDERS")
		print(f"{'='*70}")
		print(f"  Player Instructions: {player_instructions}")
		print(f"{'='*70}")

		# STEP 3: Initial reconnaissance
		print(f"\n{'='*70}")
		print(f"  STEP 3: {self.name.upper()} - INITIAL RECONNAISSANCE")
		print(f"{'='*70}")
		
		recon_system_1 = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"=== BATTLEFIELD INTELLIGENCE ===\n{map_summary}\n\n"
			f"Your orders from high command: {player_instructions}\n\n"
			"TASK: Gather intelligence about the battlefield.\n"
			"Review the battlefield intelligence above, then call 2-4 reconnaissance tools to investigate key terrain features or enemy positions in more detail.\n"
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
		print(f"\n{'='*70}")
		print(f"  STEP 4: {self.name.upper()} - DEFINING TACTICAL ACTIONS")
		print(f"{'='*70}")
		
		intelligence_summary_1 = "\n\n".join(intelligence_1) if intelligence_1 else "No intelligence gathered."
		
		unit_count = len(self.unit_list)
		
		actions_system = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"Your orders: {player_instructions}\n\n"
			f"You command {unit_count} unit(s).\n\n"
			f"INTELLIGENCE GATHERED:\n{intelligence_summary_1}\n\n"
			"TASK: Define tactical actions based on your orders and intelligence.\n\n"
			"DECISION GUIDANCE:\n"
			"- ONE action is often BEST for focused, coordinated operations\n"
			"- TWO actions when you need simultaneous efforts (e.g., main attack + supporting attack)\n"
			"- THREE actions only for complex operations requiring multiple objectives\n"
			f"- With {unit_count} unit(s), consider if you have enough forces to split effectively\n"
			"- You MAY target the same objective with different action types (e.g., Attack + Flank on same heights)\n\n"
			"STRUCTURE:\n"
			"- action_name: Tactical description ONLY (e.g., 'Frontal Assault', 'Eastern Flank')\n"
			"- primary_objective: EXACT terrain feature or unit name (e.g., 'Santa Maria Heights')\n\n"
			"Define your action(s) by calling 'define_action'. When finished, stop calling tools.\n"
			"Use ONLY tool calls - no text responses."
		)
		
		messages_actions = [
			{"role": "system", "content": actions_system},
			{"role": "user", "content": "Define your tactical actions now."}
		]
		
		defined_actions = self._define_actions_phase(messages_actions, num_thread, num_ctx)

		# STEP 5: Additional reconnaissance
		print(f"\n{'='*70}")
		print(f"  STEP 5: {self.name.upper()} - DETAILED RECONNAISSANCE")
		print(f"{'='*70}")
		
		actions_summary = "\n".join([f"- {a['name']} (Type: {a['type']}): {a['description']}" for a in defined_actions])
		
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
		print(f"\n{'='*70}")
		print(f"  STEP 6: {self.name.upper()} - REVIEWING UNIT DISPOSITIONS")
		print(f"{'='*70}")
		
		unit_details = self._get_detailed_unit_info()
		print(f"\n{unit_details}")

		# STEP 7: Assign units to actions and return JSON
		print(f"\n{'='*70}")
		print(f"  STEP 7: {self.name.upper()} - ASSIGNING UNITS TO ACTIONS")
		print(f"{'='*70}")
		
		intelligence_summary_2 = "\n\n".join(intelligence_2) if intelligence_2 else "No additional intelligence."
		all_intelligence = f"{intelligence_summary_1}\n\n{intelligence_summary_2}"
		
		assignment_system = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"Your orders: {player_instructions}\n\n"
			f"Your defined actions:\n{actions_summary}\n\n"
			f"All intelligence gathered:\n{all_intelligence}\n\n"
			f"Your units:\n{unit_details}\n\n"
			f"CRITICAL INSTRUCTIONS:\n"
			f"- You command EXACTLY {len(self.unit_list)} units (no more, no less)\n"
			f"- You must assign each unit to exactly ONE action\n"
			f"- Do NOT assign the same unit multiple times\n"
			f"- Do NOT create assignments for units that don't exist\n"
			f"- Each unit can only be in ONE place doing ONE thing\n\n"
			f"TASK: Assign each of your {len(self.unit_list)} units to one of your actions.\n"
			"Use ONLY tool calls - no text responses."
		)
		
		messages_assignment = [
			{"role": "system", "content": assignment_system},
			{"role": "user", "content": "Assign your units to actions now."}
		]
		
		assignments = self._assign_units_phase(messages_assignment, defined_actions, num_thread, num_ctx)

		# Build final JSON output
		final_output = self._build_json_output(defined_actions, assignments)
		
		print(f"\n{'='*70}")
		print(f"  FINAL BATTLE PLAN: {self.name.upper()}")
		print(f"{'='*70}")
		for order in final_output.get('orders', []):
			print(f"\n  Action: {order['name']} (Type: {order['type']})")
			print(f"  Target: {order['target']}")
			print(f"  Units:  {', '.join(order['units'])}")
		print(f"\n{'='*70}\n")
		
		return json.dumps(final_output, indent=2)

	# =====================================================
	# WORKFLOW PHASE METHODS
	# =====================================================

	def _conduct_reconnaissance_phase(self, messages, num_thread, num_ctx, min_calls=2, max_calls=5):
		"""Conduct reconnaissance and gather intelligence.
		
		Args:
			messages: Chat messages for LLM context
			num_thread: Number of threads for LLM
			num_ctx: Context size for LLM
			min_calls: Minimum reconnaissance calls before stopping
			max_calls: Maximum reconnaissance rounds
			
		Returns:
			List of intelligence reports gathered
		"""
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
				
				args_str = ', '.join(f"{k}='{v}'" for k, v in args.items())
				print(f"\n  → {tool_name}({args_str})")
				
				result = self._execute_reconnaissance_tool(tool_name, args)
				
				if result.get("ok"):
					intelligence = result.get("intelligence", "")
					print(f"  {intelligence}")
					intelligence_gathered.append(intelligence)
				else:
					error = result.get("error", "Unknown error")
					print(f"  ✗ Error: {error}")
				
				tool_msg = {"role": "tool", "content": json.dumps(result)}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
			
			if len(intelligence_gathered) >= min_calls:
				break
		
		return intelligence_gathered

	def _define_actions_phase(self, messages, num_thread, num_ctx):
		"""Have the general define 1-3 tactical actions.
		
		Args:
			messages: Chat messages for LLM context
			num_thread: Number of threads for LLM
			num_ctx: Context size for LLM
			
		Returns:
			List of defined action dictionaries
		"""
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
					action_type = args.get("action_type", "Attack")
					action_name = args.get("action_name", "Unnamed Action")
					primary_objective = args.get("primary_objective", "")
					
					# Validate action type is in preset list
					if action_type not in self.preset_action_names:
						print(f"\n  ⚠ Warning: Invalid action type '{action_type}'")
						print(f"  Valid types: {', '.join(self.preset_action_names)}")
						action_type = self.preset_action_names[0]  # Default to first preset
						print(f"  Auto-corrected to: '{action_type}'")
					
					action = {
						"type": action_type,
						"name": action_name,
						"description": args.get("description", ""),
						"primary_objective": primary_objective
					}
					defined_actions.append(action)
					print(f"\n  ✓ Action {len(defined_actions)}: {action['name']} (Type: {action['type']})")
					print(f"    Target: {action['primary_objective']}")
					print(f"    Details: {action['description']}")
					
					result = {"ok": True, "message": f"Action '{action['name']}' targeting '{primary_objective}' defined successfully. You may define more actions or stop."}
				else:
					result = {"ok": False, "error": "Maximum 3 actions allowed. Stop defining actions."}
				
				tool_msg = {"role": "tool", "content": json.dumps(result)}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
			
			# Continue looping to give LLM chance to define more or stop naturally
			# Don't break early - let LLM decide when to stop
		
		if not defined_actions:
			# Fallback: create a default action using preset
			print("\n  ℹ No actions defined, creating default action")
			defined_actions.append({
				"type": self.preset_action_names[0] if self.preset_action_names else "Attack",
				"name": "General Advance",
				"description": "General advance against enemy positions",
				"primary_objective": "Enemy forces"
			})
		else:
			print(f"\n  ✓ {self.name} defined {len(defined_actions)} action(s)")
		
		return defined_actions

	def _assign_units_phase(self, messages, defined_actions, num_thread, num_ctx):
		"""Have the general assign units to actions using structured output.
		
		Args:
			messages: Chat messages for LLM context
			defined_actions: List of defined actions
			num_thread: Number of threads for LLM
			num_ctx: Context size for LLM
			
		Returns:
			Dictionary mapping unit names to action names
		"""
		action_names = [a["name"] for a in defined_actions]
		unit_names = [u.name for u in self.unit_list]
		
		# Build comprehensive prompt for structured output
		assignment_prompt = (
			f"CRITICAL: You have EXACTLY {len(self.unit_list)} units. You must assign each unit to ONE action.\n"
			f"Each unit can only be assigned ONCE. Do not assign the same unit multiple times.\n\n"
			f"Available actions:\n"
		)
		for i, action in enumerate(defined_actions, 1):
			assignment_prompt += f"{i}. {action['name']}: {action['description']}\n"
		
		assignment_prompt += f"\nYour {len(self.unit_list)} units (assign each one exactly once):\n"
		for i, unit in enumerate(self.unit_list, 1):
			assignment_prompt += f"{i}. {unit.name}\n"
		
		assignment_prompt += (
			f"\nIMPORTANT:\n"
			f"- You must assign all {len(self.unit_list)} units (no more, no less)\n"
			f"- Each unit appears in the list exactly once\n"
			f"- Do not assign a unit to multiple actions\n"
			f"- Consider tactical factors: unit position, mobility, action objectives\n\n"
			"Provide your assignments now."
		)
		
		# Build system message emphasizing structure
		system_content = (
			f"You are {self.name}, assigning units to tactical actions.\n"
			"CRITICAL INSTRUCTIONS:\n"
			"1. You have EXACTLY the number of units listed below\n"
			"2. Each unit must be assigned to EXACTLY ONE action\n"
			"3. Use the action_name field to match assignments (e.g., 'Frontal Assault', not feature names)\n"
			"4. Never assign a unit more than once\n"
			"5. Match action_name EXACTLY as provided in the actions list\n"
		)
		
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
			
			print(f"\n  Received {len(battle_plan.assignments)} assignment(s) from LLM")
			if len(battle_plan.assignments) != len(self.unit_list):
				print(f"  ⚠ Warning: Expected {len(self.unit_list)} assignments, got {len(battle_plan.assignments)}")
			
			# Convert to dictionary format
			assignments = {}
			duplicates_found = []
			for assignment in battle_plan.assignments:
				unit_name = assignment.unit_name
				action_name = assignment.action_name
				
				# Validate assignment
				if unit_name in unit_names and action_name in action_names:
					if unit_name not in assignments:  # Avoid duplicates
						assignments[unit_name] = action_name
						print(f"  ✓ {unit_name} → {action_name}")
					else:
						duplicates_found.append(unit_name)
						print(f"  ✗ Duplicate ignored: {unit_name} already assigned to {assignments[unit_name]}")
				else:
					if unit_name not in unit_names:
						print(f"  ✗ Unknown unit: '{unit_name}'")
					if action_name not in action_names:
						print(f"  ✗ Unknown action: '{action_name}'")
			
			if duplicates_found:
				print(f"\n  ⚠ Warning: {len(duplicates_found)} duplicate assignment(s) detected and ignored")
			
			# Assign any remaining units to the first action
			if len(assignments) < len(self.unit_list):
				default_action = defined_actions[0]["name"]
				print(f"\n  Auto-assigning remaining units to '{default_action}':")
				for unit in self.unit_list:
					if unit.name not in assignments:
						assignments[unit.name] = default_action
						print(f"  ✓ {unit.name} → {default_action}")
			
			return assignments
			
		except Exception as e:
			print(f"\n  ✗ Error: Structured output failed - {e}")
			print(f"  → Fallback: Using default assignment strategy")
			
			# Fallback: assign all units to first action
			default_action = defined_actions[0]["name"]
			assignments = {}
			print(f"\n  Assigning all units to '{default_action}':")
			for unit in self.unit_list:
				assignments[unit.name] = default_action
				print(f"  ✓ {unit.name} → {default_action}")
			
			return assignments

	# =====================================================
	# UTILITY METHODS - OUTPUT BUILDING & PARSING
	# =====================================================

	def _get_detailed_unit_info(self):
		"""Get detailed information about units under command.
		
		Returns:
			Formatted string with unit details
		"""
		lines = [f"Units under your command ({len(self.unit_list)} total):"]
		for i, unit in enumerate(self.unit_list, 1):
			lines.append(f"{i}. {unit.status_general()}")
		return "\n".join(lines)

	def _consolidate_orders(self, orders):
		"""Consolidate orders with the same type and target into single orders.
		
		Args:
			orders: List of order dictionaries with type, name, target, units
			
		Returns:
			List of consolidated orders where orders with matching type+target are merged
		"""
		# Group orders by (type, target) key
		groups = {}
		for order in orders:
			key = (order["type"], order["target"])
			if key not in groups:
				groups[key] = {
					"type": order["type"],
					"names": [],
					"target": order["target"],
					"units": []
				}
			groups[key]["names"].append(order["name"])
			groups[key]["units"].extend(order["units"])
		
		# Build consolidated orders
		consolidated = []
		for (order_type, target), group in groups.items():
			# Combine action names if multiple were merged
			if len(group["names"]) > 1:
				combined_name = " + ".join(group["names"])
				print(f"  → Consolidated {len(group['names'])} orders into: {order_type} - {combined_name}")
				print(f"    Combined units: {', '.join(group['units'])}")
			else:
				combined_name = group["names"][0]
			
			consolidated.append({
				"type": order_type,
				"name": combined_name,
				"target": target,
				"units": group["units"]
			})
		
		return consolidated

	def _build_json_output(self, defined_actions, assignments):
		"""Build the final JSON output with actions and assigned units.
		
		Args:
			defined_actions: List of action dictionaries
			assignments: Dict mapping unit names to action names
			
		Returns:
			Dictionary with 'orders' list
		"""
		orders = []
		
		for action in defined_actions:
			order_entry = {
				"type": action["type"],
				"name": action["name"],
				"target": action["primary_objective"],
				"units": []
			}
			
			# Find all units assigned to this action
			for unit_name, action_name in assignments.items():
				if action_name == action["name"]:
					order_entry["units"].append(unit_name)
			
			orders.append(order_entry)
		
		# Consolidate orders with same type and target
		print(f"\n  Consolidating orders...")
		consolidated_orders = self._consolidate_orders(orders)
		
		return {"orders": consolidated_orders}

	def parse_orders_json(self, json_string: str) -> Dict[str, Any]:
		"""Parse the JSON text output from get_instructions into a data structure.
		
		Args:
			json_string: JSON string output from the general's get_instructions method
			
		Returns:
			Dict containing parsed orders with structure:
			{
				"orders": [
					{
						"type": str,  # Action type (Attack, Defend, Support, etc.)
						"name": str,  # Descriptive action name
						"target": str,  # Primary objective/terrain feature
						"units": List[str]  # List of unit names
					},
					...
				]
			}
			
		Raises:
			ValueError: If JSON parsing fails or structure is invalid
		"""
		try:
			orders_data = json.loads(json_string)
			
			# Validate structure
			if "orders" not in orders_data:
				raise ValueError("Missing 'orders' key in JSON structure")
			
			if not isinstance(orders_data["orders"], list):
				raise ValueError("'orders' must be a list")
			
			# Validate each order
			for i, order in enumerate(orders_data["orders"]):
				if "type" not in order or "name" not in order or "target" not in order or "units" not in order:
					raise ValueError(f"Order {i} missing required keys (type, name, target, units)")
				
				if not isinstance(order["units"], list):
					raise ValueError(f"Order {i} 'units' must be a list")
			
			print(f"\n✓ Successfully parsed {len(orders_data['orders'])} order(s)")
			return orders_data
			
		except json.JSONDecodeError as e:
			raise ValueError(f"Failed to parse JSON: {e}")
		except Exception as e:
			raise ValueError(f"Error parsing orders: {e}")

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