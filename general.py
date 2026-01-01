"""
General - AI-powered battlefield commander.

The General class now acts as an orchestrator, delegating specialized tasks
to service classes while maintaining a clear, easy-to-follow workflow:

PHASE 1: RECONNAISSANCE - Gather battlefield intelligence
PHASE 2: ACTION DEFINITION - Define tactical actions
PHASE 3: UNIT ASSIGNMENT - Assign units to actions

This refactored design follows the Single Responsibility Principle and makes
the multi-step decision-making process transparent and maintainable.
"""

import ollama
import json
import threading
from typing import Optional, Callable, Dict, Any, List
from pydantic import BaseModel

# Import specialized services
from general_services.terrain_analyzer import TerrainAnalyzer
from general_services.reconnaissance_service import ReconnaissanceService
from general_services.action_planner import ActionPlanner
from general_services.unit_assigner import UnitAssigner
from general_services.order_formatter import OrderFormatter


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
    """
    AI-powered battlefield general that generates tactical orders using LLM reasoning.
    
    The General orchestrates a three-phase decision-making process:
    1. RECONNAISSANCE: Uses ReconnaissanceService to gather intelligence
    2. ACTION DEFINITION: Uses ActionPlanner to create tactical actions
    3. UNIT ASSIGNMENT: Uses UnitAssigner to distribute units
    
    All phases maintain conversational context for coherent decision-making.
    """
    
    # =====================================================
    # CLASS CONSTANTS
    # =====================================================
    METERS_PER_HEX = 100  # Battlefield scale: 1 hex ≈ 100 meters
    NEAR_RADIUS_METERS = 300  # Report units within ~3 hexes as nearby
    MAX_ACTIONS = 3  # Maximum tactical actions to define
    
    def __init__(self, faction: str, identity_prompt, unit_list=None, game_map=None, 
                 model: str = "llama3.2:3b", ollama_host: str = None):
        """Initialize the General with faction, identity, units, and LLM configuration.
        
        Args:
            faction: The general's faction
            identity_prompt: Dict with 'name' and 'description' keys
            unit_list: List of units under command
            game_map: Map instance for reconnaissance
            model: Ollama model name
            ollama_host: Ollama API host URL (optional)
        """
        # Core attributes
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
        
        # Initialize specialized services
        self.terrain = TerrainAnalyzer(game_map, faction)
        self.reconnaissance = ReconnaissanceService(game_map, faction, self.terrain)
        self.action_planner = ActionPlanner(game_map, self.preset_action_names, self.client, model)
        self.unit_assigner = UnitAssigner(self.client, model)
        self.formatter = OrderFormatter()

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
                return presets.get("action_names", ["Attack", "Defend", "Support", "Retreat"])
        except Exception as e:
            print(f"[Warning] Could not load preset action names: {e}")
            return ["Attack", "Defend", "Support", "Retreat"]

    # =====================================================
    # PUBLIC API - MAIN ENTRY POINT
    # =====================================================

    def get_instructions(self, player_instructions="", map_summary="", 
                        callback: Optional[Callable[[str], None]] = None):
        """
        Main entry point: Generate tactical orders based on player instructions.
        
        This method orchestrates the three-phase planning process:
        1. RECONNAISSANCE: Gather intelligence about terrain and enemy forces
        2. ACTION DEFINITION: Define tactical actions based on intelligence
        3. UNIT ASSIGNMENT: Assign units to actions
        
        Args:
            player_instructions: Orders from the player/commander
            map_summary: Summary of the battlefield situation
            callback: Optional callback for streaming responses
            
        Returns:
            JSON string containing the complete battle plan with orders
        """
        if player_instructions.strip() == "":
            player_instructions = "You have received no orders, act according to your best judgement."

        # Use threaded execution if callback provided (for UI responsiveness)
        if callback:
            result_container = {"result": None}
            
            def run_query():
                try:
                    result = self._execute_three_phase_planning(player_instructions, map_summary)
                    result_container["result"] = result
                    callback(result)
                except Exception as e:
                    error_msg = f"Error during planning: {str(e)}"
                    print(error_msg)
                    callback(json.dumps({"error": error_msg}))
            
            thread = threading.Thread(target=run_query, daemon=True)
            thread.start()
            thread.join()
            return result_container["result"]
        else:
            return self._execute_three_phase_planning(player_instructions, map_summary)

    # =====================================================
    # ORCHESTRATION - THREE-PHASE PLANNING PROCESS
    # =====================================================

    def _execute_three_phase_planning(self, player_instructions: str, map_summary: str, 
                                     num_thread: int = 4, num_ctx: int = 4096) -> str:
        """
        Execute the complete three-phase planning process.
        
        WORKFLOW OVERVIEW:
        ==================
        
        PHASE 1: RECONNAISSANCE (Intelligence Gathering)
        ------------------------------------------------
        - LLM analyzes orders and identifies key terrain features
        - Calls reconnaissance tools 2-4 times to gather intelligence:
          * reconnaissance_feature: Get details about specific terrain
          * assess_enemy_strength: Analyze enemy forces near objectives
          * survey_approaches: Identify approach routes and threats
        - Intelligence is accumulated and fed into next phase
        
        PHASE 2: ACTION DEFINITION (Tactical Planning)
        ----------------------------------------------
        - LLM receives intelligence summary from Phase 1
        - Defines 1-3 tactical actions using define_action tool:
          * Each action has: type, name, description, primary_objective
          * Actions must target valid terrain features
          * Specific orders (e.g., "2 units to Hill A") are followed exactly
          * Vague orders allow tactical discretion
        
        PHASE 3: UNIT ASSIGNMENT (Force Allocation)
        -------------------------------------------
        - LLM receives defined actions from Phase 2
        - Assigns each unit to exactly one action using structured output
        - Follows specified allocations if orders were specific
        - Uses tactical judgment for vague orders
        - All units are assigned (with fallback to first action)
        
        Args:
            player_instructions: Commander's orders
            map_summary: Battlefield situation summary
            num_thread: LLM thread count
            num_ctx: LLM context size
            
        Returns:
            JSON string with complete battle plan
        """
        print(f"\n{'='*70}")
        print(f"  {self.name.upper()} - BATTLE PLANNING SESSION")
        print(f"{'='*70}")
        print(f"  Orders: {player_instructions}")
        print(f"{'='*70}\n")

        # Analyze order specificity for intelligent planning
        mentioned_features = self.action_planner.extract_features_from_instructions(player_instructions)
        valid_features = self.action_planner.get_valid_feature_names()
        features_list = ', '.join(valid_features) if valid_features else 'No features available'
        unit_count = len(self.unit_list)
        order_analysis = self.action_planner.parse_order_specificity(player_instructions)
        
        # Build reconnaissance guidance
        if mentioned_features:
            print(f"→ Key objectives identified: {', '.join(mentioned_features)}\n")
        else:
            print(f"→ No specific objectives mentioned, conducting general reconnaissance\n")

        # Initialize conversation with comprehensive system prompt
        system_prompt = self._build_system_prompt(
            map_summary, unit_count, features_list, order_analysis
        )
        
        user_orders = self._build_initial_user_message(
            player_instructions, mentioned_features
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_orders}
        ]
        
        # PHASE 1: RECONNAISSANCE
        print("="*70)  # Visual separator for human readability
        print("PHASE 1: RECONNAISSANCE")
        print("="*70)
        messages, intelligence = self._run_reconnaissance_phase(
            messages, num_thread, num_ctx
        )
        
        # PHASE 2: DEFINE TACTICAL ACTIONS
        print(f"\n{'='*70}")
        print("PHASE 2: DEFINE TACTICAL ACTIONS")
        print("="*70)
        
        messages = self._add_phase_transition(
            messages, intelligence, mentioned_features, 
            order_analysis, features_list, phase=2
        )
        
        messages, defined_actions = self.action_planner.run_action_definition_phase(
            messages, num_thread, num_ctx
        )
        
        # PHASE 3: ASSIGN UNITS TO ACTIONS
        print(f"\n{'='*70}")
        print("PHASE 3: ASSIGN UNITS TO ACTIONS")
        print("="*70)
        
        messages = self._add_phase_transition(
            messages, None, mentioned_features,
            order_analysis, features_list, phase=3,
            defined_actions=defined_actions
        )
        
        assignments = self.unit_assigner.run_unit_assignment_phase(
            messages, defined_actions, self.unit_list, num_thread, num_ctx
        )
        
        # BUILD AND RETURN FINAL OUTPUT
        final_output = self.formatter.build_json_output(defined_actions, assignments)
        
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
    # PHASE EXECUTION METHODS
    # =====================================================

    def _run_reconnaissance_phase(self, messages: List[Dict], num_thread: int, 
                                  num_ctx: int) -> tuple:
        """
        Execute Phase 1: Reconnaissance with intelligence gathering.
        
        The LLM calls reconnaissance tools to gather battlefield intelligence.
        Each tool call is tracked to prevent duplicates.
        
        Args:
            messages: Conversation history
            num_thread: LLM thread count
            num_ctx: LLM context size
            
        Returns:
            Tuple of (updated_messages, gathered_intelligence)
        """
        gathered_data = []
        investigated = set()  # Track (tool_name, feature_name) to prevent duplicates
        min_calls = 2
        max_calls = 4
        
        for round_num in range(max_calls):
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=self.reconnaissance.get_reconnaissance_tools(),
                options={"num_thread": num_thread, "num_ctx": num_ctx}
            )
            
            msg = response.get("message", {})
            messages.append(msg)
            
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                if len(gathered_data) >= min_calls:
                    break  # Phase complete
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Continue reconnaissance. You need at least {min_calls - len(gathered_data)} more call(s)."
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
                    
                    # Pass unit_list to reconnaissance methods that need it
                    if tool_name in ["assess_enemy_strength", "survey_approaches"]:
                        result = self.reconnaissance.execute_reconnaissance_tool(tool_name, args)
                        # Inject unit_list for these specific tools
                        if tool_name == "assess_enemy_strength":
                            result = self.reconnaissance._recon_assess_enemy(args, self.unit_list)
                        elif tool_name == "survey_approaches":
                            result = self.reconnaissance._recon_survey_approaches(args, self.unit_list)
                    else:
                        result = self.reconnaissance.execute_reconnaissance_tool(tool_name, args)
                    
                    if result.get("ok"):
                        intelligence = result.get("intelligence", "")
                        print(f"    {intelligence}")
                        gathered_data.append(intelligence)
                        if feature_name:
                            investigated.add(investigation_key)
                    else:
                        error = result.get("error", "Unknown error")
                        print(f"    ✗ {error}")
                
                # Add tool result to conversation
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

    # =====================================================
    # PROMPT BUILDING HELPERS
    # =====================================================

    def _build_system_prompt(self, map_summary: str, unit_count: int, 
                            features_list: str, order_analysis: Dict) -> str:
        """Build the comprehensive system prompt that defines the General's identity and process."""
        return (
            f"You are {self.name}, a battlefield general. {self.description}\n\n"
            f"BATTLEFIELD SITUATION:\n{map_summary}\n\n"
            f"YOUR FORCES:\n"
            f"You command {unit_count} units:\n{self.formatter.get_detailed_unit_info(self.unit_list)}\n\n"
            f"AVAILABLE TERRAIN FEATURES:\n{features_list}\n\n"
            f"COMMAND PHILOSOPHY:\n"
            f"You serve your commander's intent. When orders are SPECIFIC (e.g., 'move 2 units to Hill A and 3 to Hill B'),\n"
            f"you MUST follow them exactly - create the specified number of actions targeting the specified locations\n"
            f"and assign the specified number of units to each. When orders are VAGUE (e.g., 'advance cautiously'),\n"
            f"use your tactical judgment to determine the best approach.\n\n"
            f"MISSION PLANNING PROCESS:\n"
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

    def _build_initial_user_message(self, player_instructions: str, 
                                   mentioned_features: List[str]) -> str:
        """Build the initial user message that provides orders."""
        user_orders = f"Your orders: {player_instructions}\n\n"
        if mentioned_features:
            user_orders += (
                f"These orders reference the following terrain features: {', '.join(mentioned_features)}\n"
                f"Prioritize reconnaissance of these locations.\n\n"
            )
        user_orders += "Begin Phase 1: Conduct reconnaissance."
        return user_orders

    def _add_phase_transition(self, messages: List[Dict], intelligence: List[str] = None,
                             mentioned_features: List[str] = None, order_analysis: Dict = None,
                             features_list: str = None, phase: int = 2, 
                             defined_actions: List[Dict] = None) -> List[Dict]:
        """Add phase transition message to guide the LLM through the workflow."""
        if phase == 2:
            # Transition to Action Definition
            if intelligence:
                intelligence_summary = "\n\n".join([f"• {intel}" for intel in intelligence])
                transition_content = f"Reconnaissance complete. Here is what you learned:\n\n{intelligence_summary}\n\n"
            else:
                transition_content = "No reconnaissance intelligence gathered.\n\n"
            
            transition_content += "Now begin Phase 2: Define your tactical actions.\n\n"
            
            if order_analysis and order_analysis.get("is_specific") and mentioned_features:
                transition_content += (
                    f"CRITICAL: Your orders specify actions for {len(mentioned_features)} locations: {', '.join(mentioned_features)}\n"
                    f"You MUST create {len(mentioned_features)} separate action(s), one for each location mentioned.\n"
                )
                if order_analysis.get("allocations"):
                    transition_content += f"Your orders also specify unit counts - remember these for Phase 3.\n"
                transition_content += f"\nCreate exactly {len(mentioned_features)} action(s) now.\n"
            else:
                transition_content += (
                    "Your orders allow tactical discretion. Based on your intelligence,\n"
                    "define 1-3 actions that best accomplish the mission.\n"
                )
            
            transition_content += (
                f"\nRemember: primary_objective must be one of these EXACT names: {features_list}\n"
                f"Call define_action for each action. When done, stop calling tools."
            )
            
        elif phase == 3:
            # Transition to Unit Assignment
            action_names = [a["name"] for a in defined_actions]
            actions_summary = "\n".join([
                f"  - '{a['name']}' (Type: {a['type']}, Target: {a['primary_objective']})" 
                for a in defined_actions
            ])
            
            unit_count = len(self.unit_list)
            transition_content = f"Actions defined. Now begin Phase 3: Assign your {unit_count} units.\n\n"
            transition_content += f"Your defined actions:\n{actions_summary}\n\n"
            
            if order_analysis and order_analysis.get("is_specific") and order_analysis.get("allocations"):
                transition_content += "CRITICAL: Your original orders specified unit allocations.\n"
                for count, location in order_analysis["allocations"]:
                    location_clean = location.strip()
                    transition_content += f"  - Assign approximately {count} units to actions targeting {location_clean}\n"
                transition_content += "Follow these allocations as closely as possible.\n\n"
            else:
                transition_content += "Use tactical judgment to distribute units effectively.\n\n"
            
            transition_content += f"Assign each unit to exactly ONE action. Use these EXACT action names: {', '.join(repr(name) for name in action_names)}"
        
        messages.append({"role": "user", "content": transition_content})
        return messages

    # =====================================================
    # UTILITY METHODS
    # =====================================================

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
