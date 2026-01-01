"""
UnitAssigner - Handles unit assignment to tactical actions.

Responsibilities:
- Run unit assignment phase using structured LLM output
- Distribute units to actions based on orders
"""

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, create_model
import json


class UnitAssigner:
    """Assigns units to tactical actions based on commander's intent."""
    
    def __init__(self, llm_client, model: str):
        """Initialize the unit assigner.
        
        Args:
            llm_client: Ollama client for LLM interactions
            model: LLM model name
        """
        self.client = llm_client
        self.model = model
    
    def run_unit_assignment_phase(self, messages: List[Dict], defined_actions: List[Dict[str, Any]], 
                                 unit_list: List, num_thread: int, num_ctx: int) -> Dict[str, str]:
        """Run unit assignment phase within continuous conversation using structured output.
        
        Args:
            messages: Conversation history
            defined_actions: List of defined actions
            unit_list: List of unit objects to assign
            num_thread: Thread count for LLM
            num_ctx: Context size for LLM
            
        Returns:
            Dictionary mapping unit names to action names
        """
        action_names = [a["name"] for a in defined_actions]
        unit_names = [u.name for u in unit_list]
        
        try:
            # Create dynamic Pydantic schema with enum constraint
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
            if len(assignments) < len(unit_list):
                default_action = defined_actions[0]["name"]
                print(f"\n  Auto-assigning remaining units to '{default_action}':")
                for unit in unit_list:
                    if unit.name not in assignments:
                        assignments[unit.name] = default_action
                        print(f"  ✓ {unit.name} → {default_action}")
            
            return assignments
            
        except Exception as e:
            print(f"  ✗ Assignment error: {e}")
            print(f"  → Using fallback: all units to first action")
            
            default_action = defined_actions[0]["name"]
            assignments = {}
            for unit in unit_list:
                assignments[unit.name] = default_action
                print(f"  ✓ {unit.name} → {default_action}")
            
            return assignments
