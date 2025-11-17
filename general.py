import ollama
import json
import threading
from typing import Optional, Callable, Dict, Any, List

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
	# MULTI-STAGE ORDER GENERATION WITH TOOL FORCING
	# =====================================================

	def _query_general_with_tools(self, player_instructions, map_summary, num_thread=4, num_ctx=4096):
		"""
		Multi-stage order generation process:
		1. Generate high-level tactical plan
		2. Force reconnaissance tool calls to gather intelligence
		3. Generate specific orders using gathered intelligence
		"""
		if player_instructions.strip() == "":
			player_instructions = "You have received no orders, act according to your best judgement."

		# STAGE 1: High-Level Planning
		print(f"\n{'='*60}")
		print(f"[STAGE 1] {self.name} formulating tactical plan...")
		print(f"{'='*60}")
		
		planning_system = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			"TASK: Analyze the battlefield and create a high-level tactical plan.\n"
			f"Battlefield Summary:\n{map_summary}\n\n"
			"Your response should:\n"
			"1. Identify 1-3 key terrain features or objectives relevant to your orders\n"
			"2. State which features you want to investigate further\n"
			"3. Outline your general tactical approach\n\n"
			"Be specific about terrain feature names (e.g., 'Po River', 'San Marco Heights', 'Verde Forest').\n"
			"Keep your response brief - 3-5 sentences."
		)
		
		planning_prompt = f"Your orders are: {player_instructions}\n\nFormulate your tactical plan:"
		
		planning_response = self.client.chat(
			model=self.model,
			messages=[
				{"role": "system", "content": planning_system},
				{"role": "user", "content": planning_prompt}
			],
			options={"num_thread": num_thread, "num_ctx": num_ctx}
		)
		
		tactical_plan = planning_response["message"]["content"].strip()
		print(f"\n[Tactical Plan]:\n{tactical_plan}\n")

		# STAGE 2: Forced Reconnaissance
		print(f"{'='*60}")
		print(f"[STAGE 2] {self.name} conducting reconnaissance...")
		print(f"{'='*60}")
		
		reconnaissance_system = (
			f"You are {self.name}, conducting battlefield reconnaissance.\n"
			f"{self.description}\n\n"
			f"Your tactical plan:\n{tactical_plan}\n\n"
			"CRITICAL REQUIREMENT:\n"
			"You MUST use reconnaissance tools to gather intelligence before issuing orders.\n"
			"Call 2-4 reconnaissance tools to investigate the features mentioned in your plan.\n\n"
			"You may call the same tool with a different feature/location multiple times if needed.\n\n"
			"Available tools:\n"
			"- reconnaissance_feature: Get detailed info about a specific terrain feature\n"
			"- assess_enemy_strength: Analyze enemy forces near a location\n"
			"- survey_approaches: Identify approach routes to a feature\n\n"
			"Use ONLY tool calls - no text responses in this phase."
		)
		
		messages = [
			{"role": "system", "content": reconnaissance_system},
			{"role": "user", "content": "Begin reconnaissance by calling tools to investigate your key features."}
		]
		
		intelligence_gathered = []
		max_tool_rounds = 5
		
		for round_num in range(max_tool_rounds):
			response = self.client.chat(
				model=self.model,
				messages=messages,
				tools=self.reconnaissance_tools,
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			# Check for tool calls
			msg = response.get("message", {})
			tool_calls = msg.get("tool_calls") or []
			
			if not tool_calls:
				# LLM has finished reconnaissance
				break
			
			# Execute each tool call
			for tc in tool_calls:
				fn = tc.get("function", {})
				tool_name = fn.get("name", "")
				raw_args = fn.get("arguments", {})
				
				try:
					args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
				except Exception:
					args = {}
				
				print(f"\n[Reconnaissance] {tool_name}({args})")
				
				# Execute the tool
				result = self._execute_reconnaissance_tool(tool_name, args)
				
				if result.get("ok"):
					intelligence = result.get("intelligence", "")
					print(f"{intelligence}")
					intelligence_gathered.append(intelligence)
				else:
					error = result.get("error", "Unknown error")
					print(f"[Error] {error}")
				
				# Add tool response to conversation
				tool_msg = {"role": "tool", "content": json.dumps(result)}
				if "id" in tc:
					tool_msg["tool_call_id"] = tc["id"]
				messages.append(tool_msg)
			
			# Stop if we have enough intelligence (2-4 tool calls)
			if len(intelligence_gathered) >= 2:
				break

		# STAGE 3: Generate Specific Orders
		print(f"\n{'='*60}")
		print(f"[STAGE 3] {self.name} issuing detailed orders...")
		print(f"{'='*60}\n")
		
		intelligence_summary = "\n\n".join(intelligence_gathered) if intelligence_gathered else "No additional intelligence gathered."
		
		orders_system = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n\n"
			f"Your tactical plan:\n{tactical_plan}\n\n"
			f"INTELLIGENCE GATHERED:\n{intelligence_summary}\n\n"
			f"Your response must be in the form of a list of one line, direct orders to each of the following {len(self.unit_list)} units ({', '.join([unit.name for unit in self.unit_list])}) and nothing else.\n"
			"You should reference a specific location on the battlefield when giving orders (use coordinates or feature names from the intelligence).\n"
			"Base your orders on the intelligence you gathered - be specific and tactical."
		)
		
		orders_prompt = f"Based on your reconnaissance, issue detailed orders to your units:"
		
		final_response = self.client.chat(
			model=self.model,
			messages=[
				{"role": "system", "content": orders_system},
				{"role": "user", "content": orders_prompt}
			],
			options={"num_thread": num_thread, "num_ctx": num_ctx}
		)
		
		return final_response["message"]["content"].strip()

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