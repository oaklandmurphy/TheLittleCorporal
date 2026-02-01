"""
Tool definitions for the General AI to use with LLM function calling.

This module defines the available tactical orders that the General can issue
through LLM function calling.
"""

from typing import List, Dict, Any


def build_general_tools(available_features: List[str], all_unit_names: List[str]) -> List[Dict[str, Any]]:
    """Build the tool definitions for LLM function calling.
    
    Args:
        available_features: List of terrain feature names available on the map
        all_unit_names: List of all unit names under the general's command
        
    Returns:
        List of tool definitions in Ollama function calling format
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "issue_attack_order",
                "description": "Issue an attack order to advance units toward and assault a specific terrain feature or objective",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": f"The terrain feature or objective to attack. Available features: {', '.join(available_features) if available_features else 'None'}",
                        },
                        "units": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"List of unit names to participate in the attack. Available units: {', '.join(all_unit_names)}",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this attack is being ordered",
                        }
                    },
                    "required": ["target", "units"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "issue_defend_order",
                "description": "Issue a defend order to hold and fortify a specific terrain feature or position",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": f"The terrain feature or position to defend. Available features: {', '.join(available_features) if available_features else 'None'}",
                        },
                        "units": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"List of unit names to participate in the defense. Available units: {', '.join(all_unit_names)}",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why this defensive position is being ordered",
                        }
                    },
                    "required": ["target", "units"]
                }
            }
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "issue_support_order",
        #         "description": "Issue a support order to provide assistance to other units or reinforce a position",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "target": {
        #                     "type": "string",
        #                     "description": f"The unit or position to support. Can be a unit name or terrain feature. Available features: {', '.join(available_features) if available_features else 'None'}",
        #                 },
        #                 "units": {
        #                     "type": "array",
        #                     "items": {"type": "string"},
        #                     "description": f"List of unit names to provide support. Available units: {', '.join(all_unit_names)}",
        #                 },
        #                 "reasoning": {
        #                     "type": "string",
        #                     "description": "Brief explanation of what support is needed and why",
        #                 }
        #             },
        #             "required": ["target", "units"]
        #         }
        #     }
        # },
        {
            "type": "function",
            "function": {
                "name": "issue_retreat_order",
                "description": "Issue a retreat order to pull units back from engagement and move toward a fallback position. This is the only order that engaged units can execute.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": f"The fallback position or rally point to retreat toward. Available features: {', '.join(available_features) if available_features else 'None'}",
                        },
                        "units": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"List of unit names to retreat. Available units: {', '.join(all_unit_names)}",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of why retreat is necessary",
                        }
                    },
                    "required": ["target", "units"]
                }
            }
        }
    ]
    
    return tools


def process_tool_calls(tool_calls) -> Dict[str, Any]:
    """Process tool calls from the LLM and convert to orders JSON structure.
    
    Args:
        tool_calls: List of tool calls from the LLM
        
    Returns:
        Dict containing orders in the expected format
    """
    import json
    
    orders = []
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        # Arguments might already be a dict or might be a JSON string
        arguments = tool_call.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        elif not isinstance(arguments, dict):
            arguments = dict(arguments)
        
        # Map function name to action type
        action_type_map = {
            "issue_attack_order": "Attack",
            "issue_defend_order": "Defend",
            # "issue_support_order": "Support",
            "issue_retreat_order": "Retreat"
        }
        
        action_type = action_type_map.get(function_name, "Attack")
        target = arguments.get("target", "Unknown")
        units = arguments.get("units", [])
        
        # Handle case where units might be a JSON string instead of a list
        if isinstance(units, str):
            try:
                units = json.loads(units)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, treat as a single unit name
                units = [units] if units else []
        
        reasoning = arguments.get("reasoning", "")
        
        # Build order name
        order_name = f"{action_type} {target}"
        
        # Print reasoning if provided
        if reasoning:
            print(f"\nGeneral's reasoning: {reasoning}\n")
        
        orders.append({
            "type": action_type,
            "name": order_name,
            "target": target,
            "units": units
        })
    
    return {"orders": orders}
