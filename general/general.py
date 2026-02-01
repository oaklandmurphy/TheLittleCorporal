"""
General - AI-powered battlefield commander (SIMPLIFIED VERSION).

This is a barebones single-session implementation where the general:
- Receives orders and map information
- Returns a single order that all units will execute
- No multi-phase planning, no complex services
"""

import ollama
import json
from typing import Optional, Dict, Any
from .general_tools import build_general_tools, process_tool_calls


class General:
    """
    Simplified AI battlefield general that issues one order for all units in a single LLM session.
    """
    
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
        self.model = model
        self.ollama_host = ollama_host
        if ollama_host:
            self.client = ollama.Client(host=ollama_host)
        else:
            self.client = ollama
        
        self.name = identity_prompt.get("name", "General")
        self.description = identity_prompt.get("description", "")
        self.faction = faction
        self.unit_list = unit_list if unit_list else []
        self.game_map = game_map

    def get_instructions(self, player_instructions="", map_summary="", 
                        callback: Optional[Dict] = None):
        """
        Main entry point: Generate a single tactical order for all units using LLM tool calling.
        
        Args:
            player_instructions: Orders from the player/commander
            map_summary: Summary of the battlefield situation
            callback: Unused (kept for compatibility)
            
        Returns:
            JSON string containing a single order for all units
        """
        if player_instructions.strip() == "":
            player_instructions = "You have received no orders, act according to your best judgement."

        print(f"\n{'='*70}")
        print(f"  {self.name.upper()} - ISSUING ORDERS")
        print(f"{'='*70}")
        print(f"  Commander's orders: {player_instructions}")
        print(f"{'='*70}\n")

        # Build simple prompt
        system_prompt = self._build_system_prompt(map_summary)
        user_prompt = self._build_user_prompt(player_instructions)
        
        # Define tools for the LLM to call
        tools = self._build_tools()

        # Get LLM response with tool calling
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools
            )
            
            # Check if LLM made tool calls
            if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
                # Process the tool call(s)
                orders_json = process_tool_calls(response.message.tool_calls)
                
                print(f"{'='*70}")
                print(f"ORDER ISSUED")
                print("="*70)
                order = orders_json['orders'][0]
                print(f"  {order['type']}: {order['name']}")
                print(f"  Target: {order['target']}")
                print(f"  Units: {order['units']}")
                print(f"{'='*70}\n")
                
                return json.dumps(orders_json, indent=2)
            else:
                # Fallback if no tool call was made
                print(f"\nGeneral's response (no tool call):\n{response.message.content}\n")
                print("Warning: No tool call received, using fallback order")
                return self._build_fallback_order()
            
        except Exception as e:
            print(f"Error during order generation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback order
            return self._build_fallback_order()

    def _build_system_prompt(self, map_summary: str) -> str:
        """Build the system prompt for the LLM."""
        unit_names = [u.name for u in self.unit_list] if self.unit_list else []
        unit_list_str = ", ".join(unit_names) if unit_names else "No units"
        
        return f"""You are {self.name}, a battlefield general. {self.description}

                BATTLEFIELD SITUATION:
                {map_summary}

                YOUR FORCES:
                {unit_list_str}

                TASK: Issue ONE order by calling the appropriate order function (issue_attack_order, issue_defend_order, issue_support_order, or issue_retreat_order).
                
                IMPORTANT: When specifying units, provide them as an array of strings, like ["Unit1", "Unit2"], NOT as a JSON string.
                
                You must call one of these functions to issue your order. Choose the most appropriate action based on the battlefield situation and commander's instructions.""" 

    def _build_user_prompt(self, player_instructions: str) -> str:
        """Build the user prompt for the LLM."""
        return f"""Commander's orders: {player_instructions}

Issue your order now by calling the appropriate function."""

    def _build_tools(self) -> list:
        """Build the tool definitions for LLM function calling."""
        # Get available terrain features for the enum
        available_features = []
        if self.game_map:
            available_features = self.game_map.list_feature_names()
        
        # Get all unit names
        all_unit_names = [u.name for u in self.unit_list] if self.unit_list else []
        
        return build_general_tools(available_features, all_unit_names)

    def _build_fallback_order(self) -> str:
        """Build a fallback order in case of errors."""
        all_unit_names = [u.name for u in self.unit_list] if self.unit_list else []
        fallback = {
            "orders": [{
                "type": "Defend",
                "name": "Defensive Stance",
                "target": "Current Position",
                "units": all_unit_names
            }]
        }
        return json.dumps(fallback, indent=2)

    def parse_orders_json(self, json_string: str) -> Dict[str, Any]:
        """Parse the JSON text output from get_instructions into a data structure.
        
        Args:
            json_string: JSON string output from get_instructions
            
        Returns:
            Dict containing parsed orders
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

