"""
ActionPlanner - Handles tactical action definition and planning.

Responsibilities:
- Parse player order specificity
- Extract terrain features from instructions
- Define action tools for LLM
- Execute action definition phase
"""

from typing import List, Dict, Any, Optional
import re
import json


class ActionPlanner:
    """Plans and defines tactical actions based on orders and intelligence."""
    
    def __init__(self, game_map, preset_action_names: List[str], llm_client, model: str):
        """Initialize the action planner.
        
        Args:
            game_map: The game map instance
            preset_action_names: List of valid action type names
            llm_client: Ollama client for LLM interactions
            model: LLM model name
        """
        self.game_map = game_map
        self.preset_action_names = preset_action_names
        self.client = llm_client
        self.model = model
    
    def get_valid_feature_names(self) -> List[str]:
        """Get list of valid terrain feature names from the map.
        
        Returns:
            List of feature name strings present on the battlefield
        """
        if not self.game_map:
            return []
        return self.game_map.list_feature_names()
    
    def extract_features_from_instructions(self, player_instructions: str) -> List[str]:
        """Extract terrain feature names mentioned in player instructions.
        
        Args:
            player_instructions: The player's orders
            
        Returns:
            List of feature names found in the instructions
        """
        if not self.game_map:
            return []
        
        valid_features = self.get_valid_feature_names()
        mentioned = []
        
        # Normalize instructions for matching
        instructions_lower = player_instructions.lower()
        
        for feature in valid_features:
            # Check if feature name appears in instructions (case-insensitive)
            if feature.lower() in instructions_lower:
                mentioned.append(feature)
        
        return mentioned
    
    def parse_order_specificity(self, player_instructions: str) -> Dict[str, Any]:
        """Analyze how specific the player's orders are.
        
        Determines if orders specify:
        - Exact unit counts
        - Specific terrain features
        - Multiple objectives
        
        Args:
            player_instructions: The player's orders
            
        Returns:
            Dict with:
                - is_specific (bool): Whether orders are highly specific
                - allocations (List[Tuple[int, str]]): List of (unit_count, location) pairs
                - mentioned_features (List[str]): Features referenced in orders
        """
        mentioned_features = self.extract_features_from_instructions(player_instructions)
        
        # Look for patterns like "send 2 units to X" or "move 3 units to Y"
        allocation_patterns = [
            r'(\d+)\s+units?\s+(?:to|toward|at|for)\s+([A-Za-z\s]+)',
            r'(?:send|move|dispatch)\s+(\d+)\s+(?:to|toward|at)\s+([A-Za-z\s]+)',
        ]
        
        allocations = []
        for pattern in allocation_patterns:
            matches = re.finditer(pattern, player_instructions, re.IGNORECASE)
            for match in matches:
                count = int(match.group(1))
                location = match.group(2).strip()
                allocations.append((count, location))
        
        # Orders are "specific" if they mention features AND unit counts
        is_specific = len(mentioned_features) > 0 and len(allocations) > 0
        
        return {
            "is_specific": is_specific,
            "allocations": allocations,
            "mentioned_features": mentioned_features
        }
    
    def get_action_definition_tools(self) -> List[Dict[str, Any]]:
        """Define tools for the General to define high-level actions.
        
        Returns:
            List of tool definition dictionaries for LLM
        """
        valid_features = self.get_valid_feature_names()
        
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
    
    def run_action_definition_phase(self, messages: List[Dict], num_thread: int, num_ctx: int) -> tuple:
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
                tools=self.get_action_definition_tools(),
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
                    valid_features = self.get_valid_feature_names()
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
