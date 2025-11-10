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
				description = self.game_map.describe_feature(feature_name)
				return {"ok": True, "intelligence": description}

			elif tool_name == "assess_enemy_strength":
				location = args.get("location", "")
				# Get feature coordinates
				coords = self.game_map.get_feature_coordinates(location)
				if not coords:
					return {"ok": False, "error": f"Unknown location: {location}"}
				
				# Find enemy units within 3 hexes of the feature
				enemy_units = []
				for y in range(self.game_map.height):
					for x in range(self.game_map.width):
						unit = self.game_map.grid[y][x].unit
						if unit and unit.faction != self.faction:
							# Check if within 3 hexes of any feature coordinate
							for fx, fy in coords:
								dist = self.game_map._hex_distance(x, y, fx, fy)
								if dist <= 3:
									enemy_units.append({
										"name": unit.name,
										"position": (x, y),
										"strength": unit.strength,
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
				
				# Analyze terrain around the feature
				survey = f"Approach survey for {target_feature}:\n"
				survey += f"Feature location: {coords}\n"
				
				# Check adjacent hexes for terrain and obstacles
				adjacent_hexes = set()
				for fx, fy in coords:
					for nx, ny in self.game_map.get_neighbors(fx, fy):
						if 0 <= nx < self.game_map.width and 0 <= ny < self.game_map.height:
							if (nx, ny) not in coords:
								adjacent_hexes.add((nx, ny))
				
				terrain_counts = {}
				blocking_units = []
				for ax, ay in adjacent_hexes:
					terrain = self.game_map.grid[ay][ax].terrain.name
					terrain_counts[terrain] = terrain_counts.get(terrain, 0) + 1
					unit = self.game_map.grid[ay][ax].unit
					if unit and unit.faction != self.faction:
						blocking_units.append(f"{unit.name} at ({ax},{ay})")
				
				survey += "Adjacent terrain:\n"
				for terrain, count in terrain_counts.items():
					survey += f"  - {count} {terrain} hex(es)\n"
				
				if blocking_units:
					survey += "Enemy units blocking approaches:\n"
					for bu in blocking_units:
						survey += f"  - {bu}\n"
				else:
					survey += "No enemy units immediately adjacent.\n"
				
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