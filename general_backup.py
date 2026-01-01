import ollama
import json
import threading
from typing import Optional, Callable, Dict, Any, List
from pydantic import BaseModel
import map.frontline as frontline

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
			return ["Attack", "Defend", "Support", "Retreat"]

	# =====================================================
	# PUBLIC API METHODS
	# =====================================================

	def _get_valid_feature_names(self) -> List[str]:
		"""Get list of valid terrain feature names from the map.
		
		Returns:
			List of feature name strings present on the battlefield
		"""
		if not self.game_map:
			return []
		return self.game_map.list_feature_names()

	def _extract_features_from_instructions(self, player_instructions: str) -> List[str]:
		"""Extract terrain feature names mentioned in player instructions.
		
		Args:
			player_instructions: The player's orders
			
		Returns:
			List of feature names found in the instructions
		"""
		if not self.game_map:
			return []
		
		valid_features = self._get_valid_feature_names()
		if not valid_features:
			return []
		
		# Normalize instructions to lowercase for case-insensitive matching
		instructions_lower = player_instructions.lower()
		
		# Find all features mentioned in the instructions
		mentioned_features = []
		for feature in valid_features:
			# Case-insensitive match
			if feature.lower() in instructions_lower:
				mentioned_features.append(feature)
		
		return mentioned_features

	def _parse_order_specificity(self, player_instructions: str) -> Dict[str, Any]:
		"""Analyze how specific the player's orders are.
		
		Args:
			player_instructions: The player's orders
			
		Returns:
			Dict with specificity analysis
		"""
		import re
		
		instructions_lower = player_instructions.lower()
		analysis = {
			"has_numbers": bool(re.search(r'\d+', player_instructions)),
			"has_unit_refs": bool(re.search(r'\b(brigade|division|regiment|battalion|unit)s?\b', instructions_lower)),
			"allocations": [],
			"is_specific": False
		}
		
		# Look for patterns like "2 units to X", "3 brigades to Y", etc.
		allocations = re.findall(r'(\d+)\s+(?:unit|brigade|division|regiment|battalion)s?\s+(?:to|at|on)\s+(?:the\s+)?([\w\s]+?)(?:\s+(?:and|,)|$)', instructions_lower)
		
		if allocations:
			analysis["allocations"] = allocations
			analysis["is_specific"] = True
		
		return analysis

	def get_instructions(self, player_instructions="", map_summary="", callback: Optional[Callable[[str], None]] = None):
		"""Generate tactical orders based on player instructions and battlefield situation.
		
		Uses a unified 3-phase conversation pipeline where context is maintained throughout:
		
		PHASE 1 - RECONNAISSANCE:
		  The General conducts intelligence gathering using reconnaissance tools to assess
		  terrain features, enemy strength, and approach routes. Focuses on features
		  mentioned in orders or tactically significant terrain.
		
		PHASE 2 - ACTION DEFINITION:
		  Based on reconnaissance intelligence and player orders, the General defines 1-3
		  high-level tactical actions (Attack, Defend, Support, etc.) with specific terrain
		  objectives. Adapts to order specificity - follows exact allocations when given
		  (e.g., "2 units to Hill A") or uses tactical judgment for vague orders.
		
		PHASE 3 - UNIT ASSIGNMENT:
		  The General assigns each unit under command to one of the defined actions using
		  structured output with dynamic validation. Respects specified allocations from
		  orders or distributes units based on tactical considerations.
		
		All phases maintain continuous conversation context, allowing the General to make
		coherent decisions informed by earlier intelligence and reasoning.
		
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
			return f"{secondary}{primary}"
		else:
			return primary # if primary in ["north", "south", "east", "west"] else primary
	
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
					"description": "Get detailed information about a specific terrain feature including its military value, occupation status, and location.",
					"parameters": {
						"type": "object",
						"properties": {
							"feature_name": {
								"type": "string",
								"description": "The name of the terrain feature to investigate"
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
					"description": "Analyze enemy forces near a specific location or feature. Returns a high-level overview of enemy strength and a recommendation for the number of units to commit.",
					"parameters": {
						"type": "object",
						"properties": {
							"location": {
								"type": "string",
								"description": "The terrain feature or area to assess enemy strength around"
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
					"description": "Survey approach routes to a location, focusing on enemy units on the path or in overlooking positions.",
					"parameters": {
						"type": "object",
						"properties": {
							"target_feature": {
								"type": "string",
								"description": "The terrain feature or objective to survey approaches to"
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
		valid_features = self._get_valid_feature_names()
		
		# Build the primary_objective property with enum constraint
		primary_objective_property = {
			"type": "string",
			"description": f"The EXACT terrain feature name from the battlefield. Must be one of the following features: {', '.join(valid_features) if valid_features else 'No features available'}"
		}
		
		# Add enum constraint if we have valid features
		if valid_features:
			primary_objective_property["enum"] = valid_features
		
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
							"primary_objective": primary_objective_property
						},
						"required": ["action_type", "action_name", "description", "primary_objective"]
					}
				}
			}
		]

	# =====================================================
	# UNIT ASSIGNMENT TOOLS
	# =====================================================

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

		# 1. Military Value
		# Use a simplified version of frontline's identification logic
		enemy_direction = self.game_map.get_enemy_approach_angle(self.faction, feature_name)
		military_value = "Low"
		if enemy_direction is not None:
			endpoints = frontline.get_frontline_endpoints(self.game_map.grid, self.game_map.width, self.game_map.height, coords, enemy_direction)
			if endpoints:
				frontline_data = frontline.get_best_frontline_with_advantage(
					self.game_map.grid, self.game_map.width, self.game_map.height, endpoints[0], endpoints[1], enemy_direction
				)
				avg_adv = frontline_data.get('average_advantage', 0)
				if avg_adv > 1.5:
					military_value = "High"
				elif avg_adv > 0.75:
					military_value = "Medium"

		# 2. Status (Occupied, Contested, Unoccupied)
		units_on = self._get_units_on_feature(coords)
		friendly_on = any(u.faction == self.faction for u in units_on)
		enemy_on = any(u.faction != self.faction for u in units_on)
		
		status = "Unoccupied"
		if friendly_on:
			status = "Occupied by friendly forces"
		elif enemy_on:
			status = f"Occupied by {len([u for u in units_on if u.faction != self.faction])} enemy unit(s)"
		else:
			# Check for nearby enemies to determine if "Contested"
			nearby_enemies = self._get_enemy_units_near_coords(coords, max_distance=2)
			if nearby_enemies:
				status = f"Contested, with {len(nearby_enemies)} enemy unit(s) nearby"

		# 3. Location
		feature_center_x = int(sum(x for x, y in coords) / len(coords))
		feature_center_y = int(sum(y for x, y in coords) / len(coords))
		
		# Get friendly center of mass
		if self.unit_list:
			friendly_center_x = sum(u.x for u in self.unit_list) / len(self.unit_list)
			friendly_center_y = sum(u.y for u in self.unit_list) / len(self.unit_list)
			direction = self._get_cardinal_direction(friendly_center_x, friendly_center_y, feature_center_x, feature_center_y)
			distance = self.game_map._hex_distance(int(friendly_center_x), int(friendly_center_y), feature_center_x, feature_center_y)
			location_desc = f"located {distance} hexes to your {direction}"
		else:
			location_desc = f"centered at ({feature_center_x}, {feature_center_y})"

		terrain_type = self._get_predominant_terrain(coords)

		intelligence = (
			f"Intelligence on '{feature_name}' ({terrain_type}):\n"
			f"  - Military Value: {military_value}. (Based on defensive potential against likely enemy approach)\n"
			f"  - Status: {status}.\n"
			f"  - Location: {location_desc}."
		)

		return {"ok": True, "intelligence": intelligence}

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
			# If not a feature, maybe it's a coordinate string like "10,15"
			try:
				x, y = map(int, location.split(','))
				coords = [(x, y)]
			except (ValueError, IndexError):
				return {"ok": False, "error": f"Unknown location or invalid coordinate: {location}"}

		enemy_units = self._get_enemy_units_near_coords(coords, self.ENEMY_DETECTION_RANGE)
		
		if not enemy_units:
			assessment = f"No enemy units detected within {self.ENEMY_DETECTION_RANGE} hexes of {location}."
			return {"ok": True, "intelligence": assessment}

		# High-level overview
		total_strength = sum(u['size'] for u in enemy_units)
		avg_quality = sum(u['quality'] for u in enemy_units) / len(enemy_units)
		avg_morale = sum(u['morale'] for u in enemy_units) / len(enemy_units)
		
		quality_desc = self._get_unit_attribute_label(round(avg_quality), 'quality')
		morale_desc = self._get_unit_attribute_label(round(avg_morale), 'morale')

		overview = (
			f"Detected {len(enemy_units)} enemy formations near {location} with a combined strength of {total_strength}. "
			f"Average quality is '{quality_desc}' and morale is '{morale_desc}'."
		)

		# Recommendation for number of units
		# Simple combat power = size * quality
		enemy_power = sum(u['size'] * u['quality'] for u in enemy_units)
		
		friendly_avg_power = 0
		if self.unit_list:
			friendly_avg_power = sum(u.size * u.quality for u in self.unit_list) / len(self.unit_list)

		if friendly_avg_power > 0:
			# Recommend a force with 1.2x the power for a good engagement chance
			recommended_units = round((enemy_power * 1.2) / friendly_avg_power)
			recommended_units = max(1, recommended_units) # Always recommend at least one unit
			recommendation = f"To engage this force with a reasonable chance of success, it is recommended to commit approximately {recommended_units} of your own units."
		else:
			recommendation = "Cannot provide a recommendation without information on friendly units."

		assessment = f"Enemy Strength Assessment near {location}:\n- {overview}\n- {recommendation}"
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
		
		# Build path to target
		path_hexes = self._build_path_to_target(avg_pos, coords)
		if not path_hexes:
			return {"ok": True, "intelligence": f"Could not determine a clear approach path to {target_feature}."}

		# 1. Analyze threats directly ON the path
		enemies_on_path = []
		path_hex_set = set(path_hexes)
		all_enemies = self._get_enemy_units_near_coords(path_hexes, max_distance=0) # Enemies exactly on the path
		
		on_path_summary = "The approach path appears clear of immediate enemy presence."
		if all_enemies:
			enemies_on_path = [e for e in all_enemies if e['position'] in path_hex_set]
			if enemies_on_path:
				total_strength = sum(e['size'] for e in enemies_on_path)
				on_path_summary = f"WARNING: The direct path is blocked by {len(enemies_on_path)} enemy unit(s) with a total strength of {total_strength}."

		# 2. Analyze threats OVERLOOKING the path (flanking features)
		flanking_threats = []
		flanking_features = self._identify_flanking_features(self._get_all_features(), set(), target_feature, path_hexes, (0,0)) # Simplified call
		
		for feature_name, info in flanking_features.items():
			enemies_on_flank = self._get_enemy_units_near_coords(info["coords"], max_distance=0)
			if enemies_on_flank:
				flank_strength = sum(e['size'] for e in enemies_on_flank)
				flanking_threats.append(
					f"the {info['direction']} flank is overlooked by {len(enemies_on_flank)} enemy unit(s) "
					f"(strength {flank_strength}) on '{feature_name}'"
				)
		
		overlooking_summary = "No enemy units detected in flanking positions."
		if flanking_threats:
			overlooking_summary = f"CAUTION: {'; '.join(flanking_threats)}."

		intelligence = (
			f"Approach Survey to '{target_feature}':\n"
			f"- Direct Path: {on_path_summary}\n"
			f"- Flanking Threats: {overlooking_summary}"
		)
		
		return {"ok": True, "intelligence": intelligence}

	def _get_all_features(self) -> Dict[str, List[tuple]]:
		"""Helper to get all features and their coordinates from the map."""
		all_features = {}
		for y in range(self.game_map.height):
			for x in range(self.game_map.width):
				for feature in self.game_map.grid[y][x].features:
					if feature not in all_features:
						all_features[feature] = []
					all_features[feature].append((x, y))
		return all_features

	def _get_unit_attribute_label(self, value: int, table_name: str) -> str:
		"""Get descriptive label for a unit attribute."""
		quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
		morale_labels = {range(0, 2): "broken", range(2, 4): "shaken", range(4, 7): "steady",
						range(7, 9): "eager", range(9, 11): "fresh"}
		
		table = None
		if table_name == 'quality':
			table = quality_labels
		elif table_name == 'morale':
			table = morale_labels
		
		if table:
			for key, label in table.items():
				if isinstance(key, range) and value in key:
					return label
				elif value == key:
					return label
		return "unknown"

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
			if not feature_coords: continue
			feature_center_x = sum(x for x, y in feature_coords) / len(feature_coords)
			feature_center_y = sum(y for x, y in feature_coords) / len(feature_coords)
			
			for px, py in path_hexes:
				dist = self.game_map._hex_distance(int(feature_center_x), int(feature_center_y), px, py)
				min_dist_to_path = min(min_dist_to_path, dist)
			
			# Consider as flanking feature if within 2 hexes of path
			if min_dist_to_path <= 2:
				# Use average path position for directionality if target_center is 0,0
				if target_center == (0,0) and path_hexes:
					path_center_x = sum(x for x,y in path_hexes) / len(path_hexes)
					path_center_y = sum(y for x,y in path_hexes) / len(path_hexes)
					direction_origin = (path_center_x, path_center_y)
				else:
					direction_origin = target_center

				direction = self._get_cardinal_direction(
					direction_origin[0], direction_origin[1],
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
			enemy_units: List of enemy unit dictionaries
		
		Returns:
			Formatted string summarizing enemy units
		"""
		if not enemy_units:
			return "No enemy units detected."
		
		# Group by distance
		units_by_distance = {}
		for eu in enemy_units:
			distance = eu.get("distance", 0)
			if distance not in units_by_distance:
				units_by_distance[distance] = []
			units_by_distance[distance].append(eu)
		
		# Sort distances
		sorted_distances = sorted(units_by_distance.keys())
		
		# Format groups
		formatted_groups = []
		for distance in sorted_distances:
			group = units_by_distance[distance]
			total_units = len(group)
			
			# Labels for size, quality, morale
			size_labels = {0: "insignificant", 1: "very small", 2: "small", 3: "medium", 4: "large", 5: "very large"}
			quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
			morale_labels = {range(0, 2): "broken", range(2, 4): "shaken", range(4, 7): "steady",
						range(7, 9): "eager", range(9, 11): "fresh"}
			
			# Helper to get label for a value
			def label_for(value, label_dict):
				for key, label in label_dict.items():
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
				f"    - Detected {total_units} enemy unit{'s' if total_units != 1 else ''}: "
				f"Average {size_desc}, {quality_desc} formations that are {morale_desc}, "
				f"closest at {min(eu.get('distance', 999) for eu in enemy_units)} hexes\n"
			)
		else:
			summary = "    - No enemy units detected within 3 hexes.\n"
		return summary

	# =====================================================
	# MAIN WORKFLOW - 7-STEP ORDER GENERATION PROCESS
	# =====================================================

	def _query_general_with_tools(self, player_instructions, map_summary, num_thread=4, num_ctx=4096):
		"""
		Unified conversation pipeline where the General thinks through the problem step-by-step:
		1. Receive orders and conduct reconnaissance
		2. Define tactical actions based on intelligence
		3. Assign units to those actions
		
		All steps maintain conversation context for coherent decision-making.
		"""
		if player_instructions.strip() == "":
			player_instructions = "You have received no orders, act according to your best judgement."

		print(f"\n{'='*70}")
		print(f"  {self.name.upper()} - BATTLE PLANNING SESSION")
		print(f"{'='*70}")
		print(f"  Orders: {player_instructions}")
		print(f"{'='*70}\n")

		# Extract features mentioned in player instructions for focused reconnaissance
		mentioned_features = self._extract_features_from_instructions(player_instructions)
		valid_features = self._get_valid_feature_names()
		features_list = ', '.join(valid_features) if valid_features else 'No features available'
		unit_count = len(self.unit_list)
		
		# Build reconnaissance guidance
		if mentioned_features:
			recon_guidance = (
				f"Your orders reference these terrain features: {', '.join(mentioned_features)}\n"
				f"Prioritize reconnaissance of these locations.\n"
			)
			print(f"→ Key objectives identified: {', '.join(mentioned_features)}\n")
		else:
			recon_guidance = "Focus reconnaissance on tactically significant terrain near your position or enemy forces.\n"
			print(f"→ No specific objectives mentioned, conducting general reconnaissance\n")

		# Analyze order specificity
		order_analysis = self._parse_order_specificity(player_instructions)
		
		# Initialize conversation with comprehensive system prompt (defines WHO the general is and HOW to operate)
		system_prompt = (
			f"You are {self.name}, a battlefield general. {self.description}\n\n"
			f"=== BATTLEFIELD SITUATION ===\n{map_summary}\n\n"
			f"=== YOUR FORCES ===\n"
			f"You command {unit_count} units:\n{self._get_detailed_unit_info()}\n\n"
			f"=== AVAILABLE TERRAIN FEATURES ===\n{features_list}\n\n"
			f"=== COMMAND PHILOSOPHY ===\n"
			f"You serve your commander's intent. When orders are SPECIFIC (e.g., 'move 2 units to Hill A and 3 to Hill B'),\n"
			f"you MUST follow them exactly - create the specified number of actions targeting the specified locations\n"
			f"and assign the specified number of units to each. When orders are VAGUE (e.g., 'advance cautiously'),\n"
			f"use your tactical judgment to determine the best approach.\n\n"
			f"=== MISSION PLANNING PROCESS ===\n"
			f"You will receive orders from your commander. Plan your battle through these phases:\n\n"
			f"PHASE 1 - RECONNAISSANCE:\n"
			f"Use reconnaissance tools to gather intelligence about terrain and enemy forces:\n"
			f"  - reconnaissance_feature(feature_name): Get detailed info about terrain\n"
			f"  - assess_enemy_strength(location): Analyze enemy forces near a location\n"
			f"  - survey_approaches(target_feature): Identify approach routes\n"
			f"Make 2-4 reconnaissance calls, investigating each relevant feature once.\n\n"
			f"PHASE 2 - DEFINE ACTIONS:\n"
			f"Based on your intelligence and orders, define 1-3 tactical actions using define_action:\n"
			f"  - action_type: Choose from {', '.join(self.preset_action_names)}\n"
			f"  - action_name: Tactical description (e.g., 'Main Assault', 'Flanking Maneuver')\n"
			f"  - primary_objective: EXACT feature name from available list above\n"
			f"  - description: What this action accomplishes\n"
			f"If orders specify multiple objectives with unit counts, create separate actions for each.\n\n"
			f"PHASE 3 - ASSIGN UNITS:\n"
			f"Assign each of your {unit_count} units to one action using structured output.\n"
			f"If orders specified unit allocations, follow them. Otherwise use tactical judgment.\n\n"
			f"Work through each phase systematically. Use ONLY tool calls - no text responses."
		)
		
		# User message contains the actual orders (WHAT to do)
		user_orders = (
			f"Your orders: {player_instructions}\n\n"
		)
		if mentioned_features:
			user_orders += (
				f"These orders reference the following terrain features: {', '.join(mentioned_features)}\n"
				f"Prioritize reconnaissance of these locations.\n\n"
			)
		user_orders += "Begin Phase 1: Conduct reconnaissance."
		
		# Initialize conversation
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_orders}
		]
		
		# PHASE 1: Reconnaissance
		print("="*70)
		print("PHASE 1: RECONNAISSANCE")
		print("="*70)
		messages, intelligence = self._run_tool_phase(
			messages, 
			self.reconnaissance_tools,
			phase_name="reconnaissance",
			min_calls=2,
			max_calls=4,
			num_thread=num_thread,
			num_ctx=num_ctx
		)
		
		# PHASE 2: Define Actions
		print(f"\n{'='*70}")
		print("PHASE 2: DEFINE TACTICAL ACTIONS")
		print("="*70)
		
		# Provide clear intelligence summary for the LLM
		if intelligence:
			intelligence_summary = "\n\n".join([f"• {intel}" for intel in intelligence])
			transition_content = f"Reconnaissance complete. Here is what you learned:\n\n{intelligence_summary}\n\n"
		else:
			transition_content = "No reconnaissance intelligence gathered.\n\n"
		
		# Add order-specific guidance
		transition_content += "Now begin Phase 2: Define your tactical actions.\n\n"
		
		if order_analysis["is_specific"] and mentioned_features:
			# Specific orders - emphasize following them exactly
			transition_content += (
				f"CRITICAL: Your orders specify actions for {len(mentioned_features)} locations: {', '.join(mentioned_features)}\n"
				f"You MUST create {len(mentioned_features)} separate action(s), one for each location mentioned.\n"
			)
			if order_analysis["allocations"]:
				transition_content += f"Your orders also specify unit counts - remember these for Phase 3.\n"
			transition_content += f"\nCreate exactly {len(mentioned_features)} action(s) now.\n"
		else:
			# Vague orders - use tactical judgment
			transition_content += (
				"Your orders allow tactical discretion. Based on your intelligence,\n"
				"define 1-3 actions that best accomplish the mission.\n"
			)
		
		transition_content += (
			f"\nRemember: primary_objective must be one of these EXACT names: {features_list}\n"
			f"Call define_action for each action. When done, stop calling tools."
		)
		
		messages.append({"role": "user", "content": transition_content})
		
		messages, defined_actions = self._run_action_definition_phase(
			messages,
			num_thread=num_thread,
			num_ctx=num_ctx
		)
		
		# PHASE 3: Assign Units
		print(f"\n{'='*70}")
		print("PHASE 3: ASSIGN UNITS TO ACTIONS")
		print("="*70)
		
		# Build action summary for context
		action_names = [a["name"] for a in defined_actions]
		actions_summary = "\n".join([f"  - '{a['name']}' (Type: {a['type']}, Target: {a['primary_objective']})" for a in defined_actions])
		
		# Build Phase 3 transition with allocation guidance
		phase3_content = f"Actions defined. Now begin Phase 3: Assign your {unit_count} units.\n\n"
		phase3_content += f"Your defined actions:\n{actions_summary}\n\n"
		
		# Add specific allocation guidance if orders were specific
		if order_analysis["is_specific"] and order_analysis["allocations"]:
			phase3_content += "CRITICAL: Your original orders specified unit allocations.\n"
			for count, location in order_analysis["allocations"]:
				location_clean = location.strip()
				phase3_content += f"  - Assign approximately {count} units to actions targeting {location_clean}\n"
			phase3_content += "Follow these allocations as closely as possible.\n\n"
		else:
			phase3_content += "Use tactical judgment to distribute units effectively.\n\n"
		
		phase3_content += f"Assign each unit to exactly ONE action. Use these EXACT action names: {', '.join(repr(name) for name in action_names)}"
		
		messages.append({"role": "user", "content": phase3_content})
		
		assignments = self._run_unit_assignment_phase(
			messages,
			defined_actions,
			num_thread=num_thread,
			num_ctx=num_ctx
		)
		
		# Build and return final output
		final_output = self._build_json_output(defined_actions, assignments)
		
		print(f"\n{'='*70}")
		print(f"BATTLE PLAN COMPLETE")
		print("="*70)
		for order in final_output.get('orders', []):
			print(f"\n  {order['type']}: {order['name']}")
			print(f"    Target: {order['target']}")
			print(f"    Units: {', '.join(order['units'])}")
		print(f"\n{'='*70}\n")
		
		return json.dumps(final_output, indent=2)

	# =====================================================
	# WORKFLOW PHASE METHODS
	# =====================================================

	def _run_tool_phase(self, messages, tools, phase_name, min_calls, max_calls, num_thread, num_ctx):
		"""Run a phase where the LLM calls tools within a continuous conversation.
		
		Args:
			messages: Conversation history
			tools: Available tools for this phase
			phase_name: Name for logging
			min_calls: Minimum tool calls before allowing phase to end
			max_calls: Maximum rounds of tool calling
			num_thread: Thread count for LLM
			num_ctx: Context size for LLM
			
		Returns:
			Tuple of (updated_messages, gathered_data)
		"""
		gathered_data = []
		investigated = set()  # Track (tool_name, feature_name) to prevent duplicates
		
		for round_num in range(max_calls):
			response = self.client.chat(
				model=self.model,
				messages=messages,
				tools=tools,
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			msg = response.get("message", {})
			messages.append(msg)  # Add assistant response to conversation
			
			tool_calls = msg.get("tool_calls") or []
			if not tool_calls:
				if len(gathered_data) >= min_calls:
					break  # Phase complete
				else:
					# Prompt to continue
					messages.append({
						"role": "user",
						"content": f"Continue {phase_name}. You need at least {min_calls - len(gathered_data)} more call(s)."
					})
					continue
			
			# Process each tool call
			for tc in tool_calls:
				fn = tc.get("function", {})
				tool_name = fn.get("name", "")
				raw_args = fn.get("arguments", {})
				
				try:
					args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
				except Exception:
					args = {}
				
				# Extract feature name for duplicate detection
				feature_name = args.get("feature_name") or args.get("location") or args.get("target_feature") or ""
				investigation_key = (tool_name, feature_name.lower() if feature_name else "")
				
				# Check for duplicate
				if investigation_key in investigated and feature_name:
					result = {
						"ok": False,
						"error": f"Already investigated '{feature_name}' with {tool_name}. Try a different feature or tool."
					}
					args_str = ', '.join(f"{k}='{v}'" for k, v in args.items())
					print(f"  ⚠ Skipping duplicate: {tool_name}({args_str})")
				else:
					# Execute the tool
					args_str = ', '.join(f"{k}='{v}'" for k, v in args.items())
					print(f"  → {tool_name}({args_str})")
					
					result = self._execute_reconnaissance_tool(tool_name, args)
					
					if result.get("ok"):
						intelligence = result.get("intelligence", "")
						print(f"    {intelligence}")
						gathered_data.append(intelligence)
						if feature_name:
							investigated.add(investigation_key)
					else:
						error = result.get("error", "Unknown error")
						print(f"    ✗ {error}")
				
				# Add tool result to conversation in a format the LLM can easily understand
				if result.get("ok"):
					tool_content = result.get("intelligence", "No intelligence gathered")
				else:
					tool_content = f"ERROR: {result.get('error', 'Unknown error')}"
				
				tool_msg = {"role": "tool", "content": tool_content}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
			
			if len(gathered_data) >= min_calls:
				break
		
		return messages, gathered_data

	def _run_action_definition_phase(self, messages, num_thread, num_ctx):
		"""Run action definition phase within continuous conversation.
		
		Args:
			messages: Conversation history
			num_thread: Thread count for LLM
			num_ctx: Context size for LLM
			
		Returns:
			Tuple of (updated_messages, defined_actions)
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
			messages.append(msg)
			
			tool_calls = msg.get("tool_calls") or []
			if not tool_calls:
				break  # No more actions to define
			
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
					
					# Validate action type
					if action_type not in self.preset_action_names:
						print(f"  ⚠ Invalid action type '{action_type}', using '{self.preset_action_names[0]}'")
						action_type = self.preset_action_names[0]
					
					# Validate primary_objective
					valid_features = self._get_valid_feature_names()
					if valid_features and primary_objective not in valid_features:
						print(f"  ✗ Invalid feature '{primary_objective}'")
						result = {
							"ok": False,
							"error": f"Invalid feature name '{primary_objective}'. Must use: {', '.join(valid_features)}"
						}
					else:
						action = {
							"type": action_type,
							"name": action_name,
							"description": args.get("description", ""),
							"primary_objective": primary_objective
						}
						defined_actions.append(action)
						print(f"  ✓ Action {len(defined_actions)}: {action['name']} ({action['type']} → {action['primary_objective']})")
						result = {"ok": True, "message": f"Action '{action['name']}' defined successfully."}
				else:
					result = {"ok": False, "error": "Maximum 3 actions allowed."}
				
				tool_msg = {"role": "tool", "content": json.dumps(result)}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
		
		# Fallback if no actions defined
		if not defined_actions:
			print("  ℹ No actions defined, creating default")
			defined_actions.append({
				"type": self.preset_action_names[0] if self.preset_action_names else "Attack",
				"name": "General Advance",
				"description": "General advance against enemy positions",
				"primary_objective": "Enemy forces"
			})
		else:
			print(f"  ✓ {len(defined_actions)} action(s) defined")
		
		return messages, defined_actions

	def _run_unit_assignment_phase(self, messages, defined_actions, num_thread, num_ctx):
		"""Run unit assignment phase within continuous conversation using structured output.
		
		Args:
			messages: Conversation history
			defined_actions: List of defined actions
			num_thread: Thread count for LLM
			num_ctx: Context size for LLM
			
		Returns:
			Dictionary mapping unit names to action names
		"""
		action_names = [a["name"] for a in defined_actions]
		unit_names = [u.name for u in self.unit_list]
		
		try:
			# Create dynamic Pydantic schema with enum constraint
			from typing import Literal
			from pydantic import create_model
			
			if action_names:
				ActionNameEnum = Literal[tuple(action_names)]
			else:
				ActionNameEnum = str
			
			DynamicUnitAssignment = create_model(
				'DynamicUnitAssignment',
				unit_name=(str, ...),
				action_name=(ActionNameEnum, ...)
			)
			
			DynamicBattlePlanAssignments = create_model(
				'DynamicBattlePlanAssignments',
				assignments=(List[DynamicUnitAssignment], ...)
			)
			
			# Get assignment using structured output
			response = self.client.chat(
				model=self.model,
				messages=messages,
				format=DynamicBattlePlanAssignments.model_json_schema(),
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			battle_plan = DynamicBattlePlanAssignments.model_validate_json(response.message.content)
			
			# Process assignments
			assignments = {}
			for assignment in battle_plan.assignments:
				unit_name = assignment.unit_name
				action_name = assignment.action_name
				
				if unit_name in unit_names and action_name in action_names:
					if unit_name not in assignments:
						assignments[unit_name] = action_name
						print(f"  ✓ {unit_name} → {action_name}")
					else:
						print(f"  ✗ Duplicate: {unit_name} already assigned")
				else:
					if unit_name not in unit_names:
						print(f"  ✗ Unknown unit: '{unit_name}'")
					if action_name not in action_names:
						print(f"  ✗ Unknown action: '{action_name}'")
			
			# Assign remaining units to first action
			if len(assignments) < len(self.unit_list):
				default_action = defined_actions[0]["name"]
				print(f"\n  Auto-assigning remaining units to '{default_action}':")
				for unit in self.unit_list:
					if unit.name not in assignments:
						assignments[unit.name] = default_action
						print(f"  ✓ {unit.name} → {default_action}")
			
			return assignments
			
		except Exception as e:
			print(f"  ✗ Assignment error: {e}")
			print(f"  → Using fallback: all units to first action")
			
			default_action = defined_actions[0]["name"]
			assignments = {}
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

