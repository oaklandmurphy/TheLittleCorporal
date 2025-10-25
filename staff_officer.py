import json
from typing import Any, Dict, List, Optional, Tuple

try:
	import ollama  # type: ignore
except Exception:  # pragma: no cover
	ollama = None  # type: ignore

from map import Map


class StaffOfficer:
	"""Converts a General's written orders into concrete unit movements via LLM tools."""

	def __init__(self, name: str, game_map: Map, unit_list=None, model: str = "llama3.2:3b"):
		self.name = name
		self.map = game_map
		self.model = model
		self.unit_list = unit_list
		self.unit_summary = self.update_unit_summary()

	# ---- Tool specifications (function calling) ----
	@property
	def tools(self) -> List[Dict[str, Any]]:
		return [
			{
				"type": "function",
				"function": {
					"name": "advance",
					"description": "move and try to attack the enemy unit nearest the destination (x,y) using pathfinding.",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {"type": "string"},
							"destination": {
								"type": "object",
								"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
								"required": ["x", "y"]
							}
						},
						"required": ["unit_name", "destination"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "retreat",
					"description": "Retreat the named unit toward a fallback destination (x,y), preferring distance from enemy.",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {"type": "string"},
							"destination": {
								"type": "object",
								"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
								"required": ["x", "y"]
							}
						},
						"required": ["unit_name", "destination"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "flank_left",
					"description": "Flank left toward a destination (x,y) by biasing the path left of the direct approach.",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {"type": "string"},
							"destination": {
								"type": "object",
								"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
								"required": ["x", "y"]
							}
						},
						"required": ["unit_name", "destination"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "flank_right",
					"description": "Flank right toward a destination (x,y) by biasing the path right of the direct approach.",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {"type": "string"},
							"destination": {
								"type": "object",
								"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
								"required": ["x", "y"]
							}
						},
						"required": ["unit_name", "destination"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "hold",
					"description": "Have the named unit hold. Destination is required for schema consistency but ignored.",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {"type": "string"},
							"destination": {
								"type": "object",
								"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
								"required": ["x", "y"]
							}
						},
						"required": ["unit_name", "destination"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "march",
					"description": "March the unit directly toward a destination (x,y), use for fastest possible movement",
					"parameters": {
						"type": "object",
						"properties": {
							"unit_name": {"type": "string"},
							"destination": {
								"type": "object",
								"properties": {
									"x": {"type": "integer"},
									"y": {"type": "integer"},
								},
								"required": ["x", "y"],
							},
						},
						"required": ["unit_name", "destination"],
					},
				},
			},
		]

	def _parse_destination(self, d):
		"""Parse destination from various formats, ignoring extra keys except x and y."""
		if isinstance(d, dict):
			# Only use x and y, ignore all other keys
			if "x" in d and "y" in d:
				return (int(d["x"]), int(d["y"]))
		if isinstance(d, (list, tuple)) and len(d) == 2:
			return (int(d[0]), int(d[1]))
		if isinstance(d, str):
			# Handle string format like "(4, 2)" or "4, 2"
			import re
			# Remove parentheses and split by comma
			cleaned = d.strip().strip("()").strip()
			match = re.match(r'^\s*(\d+)\s*,\s*(\d+)\s*$', cleaned)
			if match:
				return (int(match.group(1)), int(match.group(2)))
		return None

	def _validate_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Validate a tool call without executing it. Returns validation result."""
		if name not in ("advance", "retreat", "flank_left", "flank_right", "hold", "march"):
			return {"ok": False, "error": f"Unknown tool: {name}"}
		
		dest = self._parse_destination(args.get("destination"))
		if dest is None:
			return {"ok": False, "error": "Invalid destination"}
		
		unit_name = args.get("unit_name")
		if not unit_name:
			return {"ok": False, "error": "Missing unit_name"}
		
		# Check if unit exists in our unit list
		if self.unit_list:
			unit_names = {unit.name for unit in self.unit_list}
			if unit_name not in unit_names:
				return {"ok": False, "error": f"Unknown unit: {unit_name}"}
		
		return {"ok": True, "validated": True}

	def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
		"""Execute a validated tool call on the map."""
		dest = self._parse_destination(args.get("destination"))
		if dest is None:
			return {"ok": False, "error": "Invalid destination"}
		
		unit_name = args.get("unit_name")
		
		if name == "advance":
			return self.map.advance(unit_name, dest)
		if name == "retreat":
			return self.map.retreat(unit_name, dest)
		if name == "flank_left":
			return self.map.flank_left(unit_name, dest)
		if name == "flank_right":
			return self.map.flank_right(unit_name, dest)
		if name == "hold":
			return self.map.hold(unit_name, dest)
		if name == "march":
			return self.map.march(unit_name, dest)
		
		return {"ok": False, "error": f"Unknown tool: {name}"}

	def _system_prompt(self, faction: Optional[str], map_summary: str = "") -> str:
		friendly = faction or "your faction"
		num_units = len(self.unit_list) if self.unit_list else 0
		
		# Create explicit checklist
		unit_checklist = []
		if self.unit_list:
			for i, unit in enumerate(self.unit_list, 1):
				unit_checklist.append(f"{i}. {unit.name}")
		checklist_str = "\n".join(unit_checklist) if unit_checklist else ""
		
		return (
			f"You are the staff officer to General {self.name}.\n"
			f"Your job is to convert the general's written orders into concrete unit movements on a hex map.\n\n"
			f"MAP COORDINATE SYSTEM:\n"
			f"- x=0 is the western edge, increasing eastward\n"
			f"- y=0 is the northern edge, increasing southward\n\n"
			f"BATTLEFIELD SITUATION:\n"
			f"{map_summary}\n\n"
			f"YOUR UNITS ({num_units} total):\n"
			f"{self.unit_summary}\n"
			f"CRITICAL REQUIREMENT:\n"
			f"You MUST issue EXACTLY ONE order for EACH of these {num_units} units:\n"
			f"{checklist_str}\n\n"
			f"INSTRUCTIONS:\n"
			f"1. Call exactly {num_units} tools - one per unit\n"
			f"2. Each tool call must specify a different unit_name from the list above\n"
			f"3. Do NOT issue multiple orders to the same unit\n"
			f"4. Do NOT skip any units\n"
			f"5. Use only tool calls - no text responses\n"
			f"6. If an order is unclear, make a reasonable tactical decision\n\n"
		)

	def process_orders(self, orders: str, faction: Optional[str] = None, max_rounds: int = 8, map_summary: str = "", num_thread=4, num_ctx=4096, max_retries: int = 2) -> Dict[str, Any]:
		"""Run the LLM with tool calling against the given orders and apply moves to the map.
		
		If validation fails (missing or duplicate orders), retry up to max_retries times with feedback.
		"""
		if ollama is None:
			return {"ok": False, "error": "ollama Python package not available"}

		# Enhanced user message with explicit instructions
		num_units = len(self.unit_list) if self.unit_list else 0
		enhanced_orders = (
			f"GENERAL'S ORDERS:\n{orders}\n\n"
			f"Execute these orders by calling exactly {num_units} movement tools. "
			f"Issue one order per unit. Begin now with tool calls only."
		)

		messages: List[Dict[str, Any]] = [
			{"role": "system", "content": self._system_prompt(faction=faction, map_summary=map_summary)},
			{"role": "user", "content": enhanced_orders},
		]

		for retry_attempt in range(max_retries + 1):
			collected_calls: List[Dict[str, Any]] = []  # Collect tool calls before execution
			units_ordered = set()  # Track which units have received orders
			
			for round_num in range(max_rounds):
				
				resp = ollama.chat(model=self.model, messages=messages, tools=self.tools, options={"num_thread": num_thread, "num_ctx": num_ctx})
				msg = resp.get("message", {})
				tool_calls = msg.get("tool_calls") or []

				# Some models might use a single tool call field
				if not tool_calls and "function_call" in msg:
					fc = msg["function_call"]
					tool_calls = [{"id": "0", "function": fc}]

				if not tool_calls:
					# No more tool calls; end
					break

				for tc in tool_calls:
					fn = tc.get("function", {})
					name = fn.get("name")
					raw_args = fn.get("arguments")
					try:
						args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
					except Exception:
						args = {}
					
					unit_name = args.get("unit_name", "")
					
					# Check if we already have enough orders or if this is a duplicate
					if len(collected_calls) >= num_units:
						# Already have enough orders, skip additional ones
						tool_msg = {
							"role": "tool",
							"content": json.dumps({"ok": False, "error": "All units already have orders"}),
						}
						if "id" in tc:
							tool_msg["tool_call_id"] = tc["id"]
						messages.append(tool_msg)
						continue
					
					if unit_name in units_ordered:
						# Duplicate order, provide feedback
						tool_msg = {
							"role": "tool",
							"content": json.dumps({"ok": False, "error": f"{unit_name} already has an order"}),
						}
						if "id" in tc:
							tool_msg["tool_call_id"] = tc["id"]
						messages.append(tool_msg)
						continue
					
					# Validate the tool call (without executing)
					validation_result = self._validate_tool_call(name, args or {})
					collected_calls.append({"tool": name, "args": args, "tc_id": tc.get("id"), "validation": validation_result})
					
					if validation_result.get("ok"):
						units_ordered.add(unit_name)
					
					# Feed validation result back to LLM
					tool_msg = {
						"role": "tool",
						"content": json.dumps(validation_result),
					}
					if "id" in tc:
						tool_msg["tool_call_id"] = tc["id"]
					messages.append(tool_msg)
				
				# Early exit if we have all orders
				if len(units_ordered) >= num_units:
					break

			# Print the orders that were issued
			print(f"\n{'='*60}")
			print(f"STAFF OFFICER ORDERS - Attempt {retry_attempt + 1}/{max_retries + 1}")
			print(f"{'='*60}")
			for i, call in enumerate(collected_calls, 1):
				unit = call["args"].get("unit_name", "Unknown")
				dest = call["args"].get("destination", {})
				# Handle destination being a dict or other format
				if isinstance(dest, dict):
					dest_str = f"({dest.get('x', '?')}, {dest.get('y', '?')})"
				else:
					dest_str = str(dest)
				print(f"{i}. {call['tool']} - {unit} → {dest_str}")
			print(f"{'='*60}\n")

			# Validate: ensure exactly one order per unit
			validation = self._validate_orders(collected_calls)
			
			# Print validation result
			print(f"VALIDATION: {'✓ PASSED' if validation['valid'] else '✗ FAILED'}")
			print(f"{'-'*60}")
			
			if validation["valid"]:
				# Success message
				print(f"All {num_units} units received exactly one order each.")
				print(f"{'='*60}\n")
				
				# NOW execute all the validated tool calls on the map
				applied: List[Dict[str, Any]] = []
				for call in collected_calls:
					result = self._execute_tool(call["tool"], call["args"])
					applied.append({"tool": call["tool"], "args": call["args"], "result": result})
				
				return {"ok": True, "applied": applied, "validation": validation}
			else:
				# Print what went wrong
				if validation.get("missing"):
					print(f"Missing orders: {', '.join(validation['missing'])}")
				if validation.get("duplicates"):
					dup_details = [f"{name} ({validation['unit_coverage'][name]}x)" for name in validation['duplicates']]
					print(f"Duplicate orders: {', '.join(dup_details)}")
				
				# Show correctly ordered units
				correct = [name for name, count in validation['unit_coverage'].items() if count == 1]
				if correct:
					print(f"Correctly ordered: {', '.join(correct)}")
				
				print(f"{'='*60}\n")
			
			# Validation failed - if we have retries left, provide feedback
			if retry_attempt < max_retries:
				feedback_parts = ["\n=== VALIDATION ERROR ==="]
				feedback_parts.append(f"You issued {len(collected_calls)} orders but need exactly {num_units}.")
				feedback_parts.append("")
				
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
				
				# Show what was correctly ordered
				correct = [name for name, count in validation['unit_coverage'].items() if count == 1]
				if correct:
					feedback_parts.append("CORRECTLY ORDERED:")
					for unit_name in correct:
						feedback_parts.append(f"  ✓ {unit_name}")
					feedback_parts.append("")
				
				feedback_parts.append(f"REQUIRED ACTION:")
				feedback_parts.append(f"Issue exactly {num_units} tool calls - one for each unit listed above.")
				feedback_parts.append("Each unit must receive exactly ONE order. Start fresh and call all tools now.")
				
				# Add feedback as user message and retry
				feedback_str = "\n".join(feedback_parts)
				messages.append({"role": "user", "content": feedback_str})
			else:
				# Out of retries - return without executing anything
				return {"ok": False, "error": validation["error"], "applied": [], "validation": validation}

		return {"ok": False, "error": "Max retries exceeded", "applied": [], "validation": validation}

	def _validate_orders(self, collected_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Validate that exactly one order was issued for each unit under command.
		
		Args:
			collected_calls: List of collected tool calls with structure {"tool": str, "args": dict, ...}
		
		Returns:
			{
				"valid": bool,
				"error": str (if invalid),
				"unit_coverage": {unit_name: order_count},
				"missing": [unit_names],
				"duplicates": [unit_names]
			}
		"""
		if not self.unit_list:
			return {"valid": True, "unit_coverage": {}, "missing": [], "duplicates": []}
		
		# Count orders per unit
		unit_order_count = {unit.name: 0 for unit in self.unit_list}
		
		for call in collected_calls:
			args = call.get("args", {})
			unit_name = args.get("unit_name")
			if unit_name in unit_order_count:
				unit_order_count[unit_name] += 1
		
		# Identify missing and duplicate orders
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

	def update_unit_summary(self):
		"""
		Generates a summary of the general's units.
		"""
		if not self.unit_list:
			return "No units assigned."
		summaries = []
		for unit in self.unit_list:
			summaries.append(f"{unit.status_so()}\n")
		return "\n".join(summaries)