"""Order execution system for commanding units on the battlefield."""

from typing import Dict, Any, List


def execute_orders(map_instance, orders_data: Dict[str, Any], faction: str) -> None:
    """Execute orders from a general by routing to specific action handlers.
    
    Args:
        map_instance: The Map instance to execute orders on
        orders_data: Parsed orders dictionary with structure:
            {
                "orders": [
                    {
                        "type": str,  # Action type (Attack, Defend, Support, etc.)
                        "name": str,  # Descriptive name for the action
                        "target": str,  # Primary objective/terrain feature
                        "units": List[str]  # List of unit names
                    },
                    ...
                ]
            }
        faction: The faction executing these orders
    """
    print(f"\n{'='*60}")
    print(f"[Executing Orders] Processing {len(orders_data.get('orders', []))} order(s) for {faction}")
    print(f"{'='*60}")
    
    for i, order in enumerate(orders_data.get("orders", []), 1):
        action_type = order.get("type", "Attack")
        action_name = order.get("name", "Unknown")
        target = order.get("target", "Unknown")
        units = order.get("units", [])
        
        print(f"\n[Order {i}] {action_type}: {action_name}")
        print(f"  → Target: {target}")
        print(f"  → Units: {', '.join(units) if units else 'None'}")
        
        # Route to appropriate handler based on action type
        action_handlers = {
            "Attack": _execute_attack,
            "Defend": _execute_defend,
            "Support": _execute_support,
            "Retreat": _execute_retreat,
        }
        
        handler = action_handlers.get(action_type, _execute_attack)
        handler(map_instance, target, units, faction)
    
    print(f"\n{'='*60}")
    print(f"[Orders Complete] All orders processed")
    print(f"{'='*60}\n")


# =====================================================
# ACTION HANDLER IMPLEMENTATIONS
# =====================================================

def _execute_attack(map_instance, target: str, units: List[str], faction: str) -> None:
    """Execute an attack order.
    
    Args:
        map_instance: The Map instance
        target: Target feature name
        units: List of unit names to attack with
        faction: The faction executing the attack
    """
    enemy_approach_angle = map_instance.get_enemy_approach_angle(faction, target)
    print(f"  [Info] Calculated enemy approach angle: {enemy_approach_angle}°")
    
    target_coords = map_instance.get_frontline_for_feature(target, enemy_approach_angle)
    if not target_coords:
        print(f"  [Error] Target feature '{target}' not found")
        return
    
    # Get frontline positions for the target
    destinations = map_instance.distribute_units_along_frontline(target_coords, len(units))
    
    if not destinations:
        print(f"  [Error] Could not determine attack positions for '{target}'")
        return
    print(f"  [Info] Assigned attack positions: {destinations}")
    
    # Assign units to destinations optimally (minimizing total distance)
    assignments = map_instance.assign_units_to_destinations_optimally(units, destinations)
    
    for unit_name, dest in assignments:
        print(f"  [Attack] Unit '{unit_name}' assigned to attack at {dest}")
        map_instance.march(unit_name, dest)


def _execute_defend(map_instance, target: str, units: List[str], faction: str) -> None:
    """Execute a defend order.
    
    Args:
        map_instance: The Map instance
        target: Target feature name to defend
        units: List of unit names to defend with
        faction: The faction executing the defense
    """
    print(f"  [TODO] Implement defend logic for {target}")
    # TODO: Implement defend logic
    pass


def _execute_support(map_instance, target: str, units: List[str], faction: str) -> None:
    """Execute a support order.
    
    Args:
        map_instance: The Map instance
        target: Target feature or unit to support
        units: List of unit names to provide support
        faction: The faction executing support
    """
    print(f"  [TODO] Implement support logic for {target}")
    # TODO: Implement support logic
    pass

def _execute_retreat(map_instance, target: str, units: List[str], faction: str) -> None:
    """Execute a retreat order.
    
    Retreat is the only order that engaged units can execute, allowing them to 
    disengage from combat. Units will move away from enemies toward the target position.
    
    Args:
        map_instance: The Map instance
        target: Fallback position or rally point (can be a feature name or coordinates)
        units: List of unit names to retreat
        faction: The faction retreating
    """
    # Try to interpret target as a feature name first
    target_coords = map_instance.get_feature_coordinates(target)
    
    if target_coords:
        # If it's a feature, use the center of the feature as the destination
        center_x = sum(x for x, y in target_coords) // len(target_coords)
        center_y = sum(y for x, y in target_coords) // len(target_coords)
        destination = (center_x, center_y)
        print(f"  [Retreat] Retreating toward feature '{target}' at {destination}")
    else:
        # Otherwise assume it's coordinates (would need parsing in real implementation)
        print(f"  [Retreat] Target '{target}' not found as feature, using as-is")
        # For now, just pick a reasonable fallback
        destination = (map_instance.width // 2, map_instance.height // 2)
    
    for unit_name in units:
        print(f"  [Retreat] Unit '{unit_name}' ordered to retreat to {destination}")
        result = map_instance.retreat(unit_name, destination)
        
        if not result.get("ok"):
            reason = result.get("reason", "Unknown")
            print(f"  [Warning] {unit_name} failed to retreat: {reason}")
