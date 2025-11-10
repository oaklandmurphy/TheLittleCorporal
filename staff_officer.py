import json
from typing import Any, Dict, List, Optional

try:
	import ollama  # type: ignore
except Exception:  # pragma: no cover
	ollama = None  # type: ignore

from map import Map


class StaffOfficer:

	def build_context_description(self, feature_names: list, friendly_units: list, enemy_units: list, game_map: Optional[Map] = None) -> str:
		"""
		Build a prompt description for the LLM given lists of feature names, friendly units, and enemy units.
		Uses map.describe_feature for features and includes unit details.
		"""
		game_map = game_map or self.map
		desc_lines = []
		# Describe features
		for fname in feature_names:
			desc = game_map.describe_feature(fname)
			desc_lines.append(desc)
		# Describe friendly units
		if friendly_units:
			desc_lines.append("\nFRIENDLY UNITS:")
			for uname in friendly_units:
				# Find the unit object
				unit_obj = None
				for y in range(game_map.height):
					for x in range(game_map.width):
						unit = game_map.grid[y][x].unit
						if unit and unit.name == uname:
							unit_obj = unit
							break
					if unit_obj:
						break
				if unit_obj:
					desc_lines.append(f"  {unit_obj.name} ({getattr(unit_obj, 'faction', 'Unknown')}) at ({unit_obj.x},{unit_obj.y}) strength: {getattr(unit_obj, 'strength', '?')}")
				else:
					desc_lines.append(f"  {uname} (not found on map)")
		# Describe enemy units
		if enemy_units:
			desc_lines.append("\nENEMY UNITS:")
			for uname in enemy_units:
				unit_obj = None
				for y in range(game_map.height):
					for x in range(game_map.width):
						unit = game_map.grid[y][x].unit
						if unit and unit.name == uname:
							unit_obj = unit
							break
					if unit_obj:
						break
				if unit_obj:
					desc_lines.append(f"  {unit_obj.name} ({getattr(unit_obj, 'faction', 'Unknown')}) at ({unit_obj.x},{unit_obj.y}) strength: {getattr(unit_obj, 'strength', '?')}")
				else:
					desc_lines.append(f"  {uname} (not found on map)")
		return "\n".join(desc_lines)

	def extract_features_and_units(self, text: str, game_map: Optional[Map] = None, friendly_faction: Optional[str] = None) -> dict:
		"""
		Parse a string and extract any named features and units present on the map.
		Returns a dict with 'features', 'friendly_units', and 'enemy_units' keys, each a list of names found in the text and present on the map.
		"""
		import re
		game_map = game_map or self.map
		# Use new map method to get all feature names
		feature_names = set(game_map.list_feature_names())
		# Determine friendly and enemy units using get_units_by_faction
		if friendly_faction is None and self.unit_list:
			# Guess faction from first unit in unit_list
			friendly_faction = getattr(self.unit_list[0], 'faction', None)
		friendly_units = set()
		enemy_units = set()
		if friendly_faction:
			for unit in game_map.get_units_by_faction(friendly_faction):
				friendly_units.add(unit.name)
			# Assume all other units are enemy
			all_unit_names = set()
			for y in range(game_map.height):
				for x in range(game_map.width):
					unit = game_map.grid[y][x].unit
					if unit:
						all_unit_names.add(unit.name)
			enemy_units = all_unit_names - friendly_units
		else:
			# If no faction info, treat all units as friendly
			for y in range(game_map.height):
				for x in range(game_map.width):
					unit = game_map.grid[y][x].unit
					if unit:
						friendly_units.add(unit.name)
			enemy_units = set()
		# Find all feature and unit names mentioned in the text (case-insensitive, word boundaries)
		found_features = set()
		found_friendly = set()
		found_enemy = set()
		for fname in feature_names:
			if re.search(rf'\b{re.escape(fname)}\b', text, re.IGNORECASE):
				found_features.add(fname)
		for uname in friendly_units:
			if re.search(rf'\b{re.escape(uname)}\b', text, re.IGNORECASE):
				found_friendly.add(uname)
		for uname in enemy_units:
			if re.search(rf'\b{re.escape(uname)}\b', text, re.IGNORECASE):
				found_enemy.add(uname)
		return {"features": list(found_features), "friendly_units": list(found_friendly), "enemy_units": list(found_enemy)}
	"""Converts a General's written orders into concrete unit movements via LLM tools.
	
	Core workflow:
	1. Takes text orders from the general
	2. Calls LLM with tool-calling capability
	3. Validates that exactly 1 tool call is made per unit under command
	4. Executes validated tool calls on the map
	"""

	# Available movement commands
	VALID_TOOLS = {
		# "advance", 
		# "retreat", 
		# "flank_left",
		# "flank_right", 
		# "hold", 
		"march"}

	def __init__(self, name: str, game_map: Map, unit_list=None, model: str = "llama3.2:3b", ollama_host: str = None):
		"""
		Initialize the Staff Officer.
		
		Args:
			name: Name of the general this staff officer serves
			game_map: The game map instance
			unit_list: List of units under command
			model: Ollama model to use
			ollama_host: The Ollama API host URL (e.g., "http://localhost:11434")
		"""
		self.name = name
		self.map = game_map
		self.model = model
		self.ollama_host = ollama_host
		if ollama_host:
			self.client = ollama.Client(host=ollama_host)
		else:
			self.client = ollama
		self.unit_list = unit_list or []
		self.unit_summary = self._build_unit_summary()

	# =====================================================
	# TOOL DEFINITIONS
	# =====================================================

	@property
	def tools(self) -> List[Dict[str, Any]]:
		"""Define the movement tools available to the LLM."""
		return [
			# self._tool_spec("advance", "move and try to attack the enemy unit nearest the destination (x,y) using pathfinding."),
			# self._tool_spec("retreat", "Retreat the named unit toward a fallback destination (x,y), preferring distance from enemy."),
			# self._tool_spec("flank_left", "Flank left toward a destination (x,y) by biasing the path left of the direct approach."),
			# self._tool_spec("flank_right", "Flank right toward a destination (x,y) by biasing the path right of the direct approach."),
			# self._tool_spec("hold", "Have the named unit hold. Destination is required for schema consistency but ignored."),
			self._tool_spec("march", "March the unit directly toward a destination (x,y), use for fastest possible movement"),
		]

	def _tool_spec(self, name: str, description: str) -> Dict[str, Any]:
		"""Generate a standard tool specification for a movement command."""
		return {
			"type": "function",
			"function": {
				"name": name,
				"description": description,
				"parameters": {
					"type": "object",
					"properties": {
						"unit_name": {"type": "string", "description": "Name of the unit to move"},
						"x": {"type": "integer", "description": "Destination x coordinate"},
						"y": {"type": "integer", "description": "Destination y coordinate"}
					},
					"required": ["unit_name", "x", "y"],
				},
			},
		}

	# =====================================================
	# MAIN PROCESSING
	# =====================================================

	def process_orders(
		self,
		orders: str,
		faction: Optional[str] = None,
		max_rounds: int = 8,
		num_thread: int = 4,
		num_ctx: int = 4096,
		max_retries: int = 2
	) -> Dict[str, Any]:
		"""Process written orders from the general and execute tool calls on the map.
		Now parses orders for features and units, builds a context description, and uses it as map_summary.
		"""
		if ollama is None:
			return {"ok": False, "error": "ollama Python package not available"}

		# Parse orders for features and units
		parsed = self.extract_features_and_units(orders, self.map, faction)
		map_summary = self.build_context_description(
			parsed["features"], parsed["friendly_units"], parsed["enemy_units"], self.map
		)
		print(map_summary)

		messages = self._build_initial_messages(orders, faction, map_summary)

		for attempt in range(max_retries + 1):
			# Collect tool calls from LLM
			collected_calls = self._collect_tool_calls(messages, max_rounds, num_thread, num_ctx)

			# Display what was collected
			self._print_collected_orders(collected_calls, attempt, max_retries)

			# Validate that exactly one order per unit was issued
			validation = self._validate_order_coverage(collected_calls)
			self._print_validation_result(validation)

			if validation["valid"]:
				# Execute all validated tool calls
				applied = self._execute_all_tools(collected_calls)
				return {"ok": True, "applied": applied, "validation": validation}

			# Validation failed - if retries remain, add feedback and try again
			if attempt < max_retries:
				feedback = self._build_retry_feedback(validation)
				messages.append({"role": "user", "content": feedback})
			else:
				# Out of retries
				return {"ok": False, "error": validation["error"], "applied": [], "validation": validation}

		return {"ok": False, "error": "Max retries exceeded", "applied": [], "validation": validation}

	# =====================================================
	# LLM INTERACTION
	# =====================================================

	def _build_initial_messages(self, orders: str, faction: Optional[str], map_summary: str) -> List[Dict[str, Any]]:
		"""Create the initial system and user messages for the LLM."""
		num_units = len(self.unit_list)
		
		system_msg = self._build_system_prompt(faction, map_summary)
		user_msg = (
			f"GENERAL'S ORDERS:\n{orders}\n\n"
			f"Execute these orders by calling exactly {num_units} movement tools. "
			f"Issue one order per unit. Begin now with tool calls only."
		)
		
		return [
			{"role": "system", "content": system_msg},
			{"role": "user", "content": user_msg}
		]

	def _build_system_prompt(self, faction: Optional[str], map_summary: str) -> str:
		"""Build the system prompt that explains the staff officer's role."""
		num_units = len(self.unit_list)
		unit_checklist = "\n".join(f"{i}. {unit.name}" for i, unit in enumerate(self.unit_list, 1))
		
		return (
			f"You are the staff officer to General {self.name}.\n"
			f"Your job is to convert the general's written orders into concrete unit movements on a hex map.\n\n"
			f"Here is a description of relevant units and locations on the hex grid battlefield:\n"
			f"{map_summary}\n\n"
			f"When ordering a unit to a location, use one of the coordinate pairs listed below that terrain feature in the Hexes section.\n"
			f"YOUR UNITS ({num_units} total):\n"
			f"{self.unit_summary}\n"
			f"CRITICAL REQUIREMENT:\n"
			f"You MUST issue EXACTLY ONE order for EACH of these {num_units} units:\n"
			f"{unit_checklist}\n\n"
			f"INSTRUCTIONS:\n"
			f"1. Call exactly {num_units} tools - one per unit\n"
			f"2. Each tool call must specify a different unit_name from the list above\n"
			f"3. Do NOT issue multiple orders to the same unit\n"
			f"4. DO NOT order multiple units to the same hex, if two units are ordered to the same terrain feature, choose different coordinates within that feature for each unit\n"
			f"5. Do NOT skip any units\n"
			f"6. Use only tool calls - no text responses\n"
			f"7. If an order is unclear, make a reasonable tactical decision\n\n"
		)

	def _collect_tool_calls(
		self, 
		messages: List[Dict[str, Any]], 
		max_rounds: int, 
		num_thread: int, 
		num_ctx: int
	) -> List[Dict[str, Any]]:
		"""Interact with LLM to collect tool calls, tracking which units have been ordered."""
		num_units = len(self.unit_list)
		collected_calls = []
		units_ordered = set()
		
		for round_num in range(max_rounds):
			# Call LLM
			resp = self.client.chat(
				model=self.model, 
				messages=messages, 
				tools=self.tools, 
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			
			# Extract tool calls from response
			tool_calls = self._extract_tool_calls(resp)
			if not tool_calls:
				break  # LLM has no more tool calls to make
			
			# Process each tool call
			for tc in tool_calls:
				name, args = self._parse_tool_call(tc)
				unit_name = args.get("unit_name", "")
				
				# Check if we already have enough orders
				if len(collected_calls) >= num_units:
					self._add_tool_response(messages, tc, {"ok": False, "error": "All units already have orders"})
					continue
				
				# Check for duplicate orders
				if unit_name in units_ordered:
					self._add_tool_response(messages, tc, {"ok": False, "error": f"{unit_name} already has an order"})
					continue
				
				# Validate the tool call
				validation = self._validate_tool_call(name, args)
				collected_calls.append({
					"tool": name, 
					"args": args, 
					"tc_id": tc.get("id"), 
					"validation": validation
				})
				
				if validation.get("ok"):
					units_ordered.add(unit_name)
				
				# Send validation result back to LLM
				self._add_tool_response(messages, tc, validation)
			
			# Early exit if we have all orders
			if len(units_ordered) >= num_units:
				break
		
		return collected_calls

	def _extract_tool_calls(self, resp: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Extract tool calls from LLM response."""
		msg = resp.get("message", {})
		tool_calls = msg.get("tool_calls") or []
		
		# Some models use a single function_call field instead
		if not tool_calls and "function_call" in msg:
			fc = msg["function_call"]
			tool_calls = [{"id": "0", "function": fc}]
		
		return tool_calls

	def _parse_tool_call(self, tc: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
		"""Parse a tool call into name and arguments."""
		fn = tc.get("function", {})
		name = fn.get("name", "")
		raw_args = fn.get("arguments", {})
		
		try:
			args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
		except Exception:
			args = {}
		
		return name, args

	def _add_tool_response(self, messages: List[Dict[str, Any]], tc: Dict[str, Any], result: Dict[str, Any]) -> None:
		"""Add a tool response message to the conversation."""
		tool_msg = {"role": "tool", "content": json.dumps(result)}
		if "id" in tc:
			tool_msg["tool_call_id"] = tc["id"]
		messages.append(tool_msg)

	# =====================================================
	# VALIDATION
	# =====================================================

	def _validate_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate a single tool call without executing it."""
		if name not in self.VALID_TOOLS:
			return {"ok": False, "error": f"Unknown tool: {name}"}
		
		# Check for x and y coordinates
		if "x" not in args or "y" not in args:
			return {"ok": False, "error": "Missing x or y coordinate"}
		
		try:
			x = int(args["x"])
			y = int(args["y"])
		except (ValueError, TypeError):
			return {"ok": False, "error": "Invalid x or y coordinate"}
		
		unit_name = args.get("unit_name")
		if not unit_name:
			return {"ok": False, "error": "Missing unit_name"}
		
		# Check if unit exists in our unit list
		unit_names = {unit.name for unit in self.unit_list}
		if unit_name not in unit_names:
			return {"ok": False, "error": f"Unknown unit: {unit_name}"}
		
		return {"ok": True, "validated": True}

	def _validate_order_coverage(self, collected_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Validate that exactly one order was issued for each unit under command."""
		if not self.unit_list:
			return {"valid": True, "unit_coverage": {}, "missing": [], "duplicates": []}
		
		# Count orders per unit
		unit_order_count = {unit.name: 0 for unit in self.unit_list}
		for call in collected_calls:
			unit_name = call.get("args", {}).get("unit_name")
			if unit_name in unit_order_count:
				unit_order_count[unit_name] += 1
		
		# Identify problems
		missing = [name for name, count in unit_order_count.items() if count == 0]
		duplicates = [name for name, count in unit_order_count.items() if count > 1]
		valid = len(missing) == 0 and len(duplicates) == 0
		
		result = {
			"valid": valid,
			"unit_coverage": unit_order_count,
			"missing": missing,
			"duplicates": duplicates
		}
		
		if not valid:
			error_parts = []
			if missing:
				error_parts.append(f"Missing orders for: {', '.join(missing)}")
			if duplicates:
				dup_details = [f"{name} ({unit_order_count[name]} orders)" for name in duplicates]
				error_parts.append(f"Duplicate orders for: {', '.join(dup_details)}")
			result["error"] = "; ".join(error_parts)
		
		return result

	# =====================================================
	# EXECUTION
	# =====================================================

	def _execute_all_tools(self, collected_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Execute all validated tool calls on the map."""
		applied = []
		for call in collected_calls:
			result = self._execute_tool(call["tool"], call["args"])
			applied.append({"tool": call["tool"], "args": call["args"], "result": result})
		return applied

	def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute a single validated tool call on the map."""
		# Extract x and y coordinates directly from args
		try:
			x = int(args["x"])
			y = int(args["y"])
			dest = (x, y)
		except (KeyError, ValueError, TypeError):
			return {"ok": False, "error": "Invalid coordinates"}
		
		unit_name = args.get("unit_name")
		
		# Dispatch to appropriate map method
		tool_methods = {
			# "advance": self.map.advance,
			# "retreat": self.map.retreat,
			# "flank_left": self.map.flank_left,
			# "flank_right": self.map.flank_right,
			# "hold": self.map.hold,
			"march": self.map.march,
		}
		
		method = tool_methods.get(name)
		if method:
			return method(unit_name, dest)
		
		return {"ok": False, "error": f"Unknown tool: {name}"}

	# =====================================================
	# UTILITIES
	# =====================================================

	def _parse_destination(self, d) -> Optional[tuple[int, int]]:
		"""Parse destination from various formats (dict, tuple, string)."""
		if isinstance(d, dict) and "x" in d and "y" in d:
			return (int(d["x"]), int(d["y"]))
		
		if isinstance(d, (list, tuple)) and len(d) == 2:
			return (int(d[0]), int(d[1]))
		
		if isinstance(d, str):
			import re
			# Try to parse as JSON string first
			try:
				parsed = json.loads(d)
				if isinstance(parsed, dict) and "x" in parsed and "y" in parsed:
					return (int(parsed["x"]), int(parsed["y"]))
			except (json.JSONDecodeError, ValueError, KeyError):
				pass
			
			# Fall back to simple coordinate parsing
			cleaned = d.strip().strip("()").strip()
			match = re.match(r'^\s*(\d+)\s*,\s*(\d+)\s*$', cleaned)
			if match:
				return (int(match.group(1)), int(match.group(2)))
		
		return None

	def _build_unit_summary(self) -> str:
		"""Generate a summary of the general's units."""
		if not self.unit_list:
			return "No units assigned."
		return "\n".join(f"{unit.status_so()}\n" for unit in self.unit_list)

	def update_unit_summary(self) -> str:
		"""Update and return unit summary (for backwards compatibility)."""
		self.unit_summary = self._build_unit_summary()
		return self.unit_summary

	# =====================================================
	# DISPLAY / LOGGING
	# =====================================================

	def _print_collected_orders(self, collected_calls: List[Dict[str, Any]], attempt: int, max_retries: int) -> None:
		"""Print the orders that were collected from the LLM."""
		print(f"\n{'='*60}")
		print(f"STAFF OFFICER ORDERS - Attempt {attempt + 1}/{max_retries + 1}")
		print(f"{'='*60}")
		
		for i, call in enumerate(collected_calls, 1):
			unit = call["args"].get("unit_name", "Unknown")
			x = call["args"].get("x", "?")
			y = call["args"].get("y", "?")
			print(f"{i}. {call['tool']} - {unit} → ({x}, {y})")
		
		print(f"{'='*60}\n")

	def _print_validation_result(self, validation: Dict[str, Any]) -> None:
		"""Print validation results with details."""
		num_units = len(self.unit_list)
		
		print(f"VALIDATION: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
		print(f"{'-'*60}")
		
		if validation["valid"]:
			print(f"All {num_units} units received exactly one order each.")
		else:
			if validation.get("missing"):
				print(f"Missing orders: {', '.join(validation['missing'])}")
			if validation.get("duplicates"):
				dup_details = [f"{name} ({validation['unit_coverage'][name]}x)" for name in validation['duplicates']]
				print(f"Duplicate orders: {', '.join(dup_details)}")
			
			correct = [name for name, count in validation['unit_coverage'].items() if count == 1]
			if correct:
				print(f"Correctly ordered: {', '.join(correct)}")
		
		print(f"{'='*60}\n")

	def _build_retry_feedback(self, validation: Dict[str, Any]) -> str:
		"""Build feedback message for retry attempts."""
		num_units = len(self.unit_list)
		feedback_parts = [
			"\n=== VALIDATION ERROR ===",
			f"You need exactly {num_units} orders, one per unit.",
			""
		]
		
		if validation.get("missing"):
			feedback_parts.append("MISSING ORDERS FOR:")
			for unit_name in validation['missing']:
				feedback_parts.append(f"  - {unit_name}")
			feedback_parts.append("")
		
		if validation.get("duplicates"):
			feedback_parts.append("DUPLICATE ORDERS FOR:")
			for unit_name in validation['duplicates']:
				count = validation['unit_coverage'][unit_name]
				feedback_parts.append(f"  - {unit_name} (received {count} orders)")
			feedback_parts.append("")
		
		correct = [name for name, count in validation['unit_coverage'].items() if count == 1]
		if correct:
			feedback_parts.append("CORRECTLY ORDERED:")
			for unit_name in correct:
				feedback_parts.append(f"  ✓ {unit_name}")
			feedback_parts.append("")
		
		feedback_parts.extend([
			f"REQUIRED ACTION:",
			f"Issue exactly {num_units} tool calls - one for each unit listed above.",
			"Each unit must receive exactly ONE order. Start fresh and call all tools now."
		])
		
		return "\n".join(feedback_parts)

