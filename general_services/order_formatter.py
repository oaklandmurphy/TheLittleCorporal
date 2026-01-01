"""
OrderFormatter - Handles formatting and consolidation of tactical orders.

Responsibilities:
- Format unit attributes and enemy summaries
- Build and consolidate orders
- Generate JSON output
"""

from typing import List, Dict, Any


class OrderFormatter:
    """Formats tactical orders and generates output structures."""
    
    @staticmethod
    def get_unit_attribute_label(value: int, table_name: str) -> str:
        """Get descriptive label for a unit attribute.
        
        Args:
            value: Numeric value of the attribute
            table_name: Type of attribute ('quality' or 'morale')
            
        Returns:
            Descriptive label string
        """
        quality_labels = {1: "green", 2: "regular", 3: "seasoned", 4: "veteran", 5: "elite"}
        morale_labels = {
            range(0, 2): "broken", 
            range(2, 4): "shaken", 
            range(4, 7): "steady",
            range(7, 9): "eager", 
            range(9, 11): "fresh"
        }
        
        table = None
        if table_name == 'quality':
            table = quality_labels
        elif table_name == 'morale':
            table = morale_labels
        
        if table:
            for key, label in table.items():
                if isinstance(key, range) and value in key:
                    return label
                elif value == key:
                    return label
        return "unknown"
    
    @staticmethod
    def format_enemy_summary(enemy_units: List[Dict[str, Any]]) -> str:
        """Format enemy unit information for display.
        
        Args:
            enemy_units: List of enemy unit dictionaries
        
        Returns:
            Formatted string summarizing enemy units
        """
        if not enemy_units:
            return "No enemy units detected."
        
        # Group by distance
        units_by_distance = {}
        for eu in enemy_units:
            distance = eu.get("distance", 0)
            if distance not in units_by_distance:
                units_by_distance[distance] = []
            units_by_distance[distance].append(eu)
        
        # Sort distances
        sorted_distances = sorted(units_by_distance.keys())
        
        # Format groups
        size_labels = {0: "insignificant", 1: "very small", 2: "small", 3: "medium", 4: "large", 5: "very large"}
        
        # Helper to get label for a value
        def label_for(value, label_dict):
            for key, label in label_dict.items():
                if isinstance(key, range) and value in key:
                    return label
                elif value == key:
                    return label
            return "unknown"
        
        # Calculate aggregate descriptions
        total_units = len(enemy_units)
        avg_quality = sum(eu.get('quality', 0) for eu in enemy_units) / total_units if total_units > 0 else 0
        avg_morale = sum(eu.get('morale', 0) for eu in enemy_units) / total_units if total_units > 0 else 0
        total_size = sum(eu.get('size', 0) for eu in enemy_units)
        
        size_desc = label_for(total_size // total_units if total_units > 0 else 0, size_labels)
        quality_desc = OrderFormatter.get_unit_attribute_label(round(avg_quality), 'quality')
        morale_desc = OrderFormatter.get_unit_attribute_label(round(avg_morale), 'morale')
        
        summary = (
            f"    - Detected {total_units} enemy unit{'s' if total_units != 1 else ''}: "
            f"Average {size_desc}, {quality_desc} formations that are {morale_desc}, "
            f"closest at {min(eu.get('distance', 999) for eu in enemy_units)} hexes\n"
        )
        
        return summary
    
    @staticmethod
    def get_detailed_unit_info(unit_list: List) -> str:
        """Get detailed information about units under command.
        
        Args:
            unit_list: List of unit objects
        
        Returns:
            Formatted string with unit details
        """
        lines = [f"Units under your command ({len(unit_list)} total):"]
        for i, unit in enumerate(unit_list, 1):
            lines.append(f"{i}. {unit.status_general()}")
        return "\n".join(lines)
    
    @staticmethod
    def consolidate_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate orders with the same type and target into single orders.
        
        Args:
            orders: List of order dictionaries with type, name, target, units
            
        Returns:
            List of consolidated orders where orders with matching type+target are merged
        """
        # Group orders by (type, target) key
        groups = {}
        for order in orders:
            key = (order["type"], order["target"])
            if key not in groups:
                groups[key] = {
                    "type": order["type"],
                    "names": [],
                    "target": order["target"],
                    "units": []
                }
            groups[key]["names"].append(order["name"])
            groups[key]["units"].extend(order["units"])
        
        # Build consolidated orders
        consolidated = []
        for (order_type, target), group in groups.items():
            # Combine action names if multiple were merged
            if len(group["names"]) > 1:
                combined_name = " + ".join(group["names"])
                print(f"  → Consolidated {len(group['names'])} orders into: {order_type} - {combined_name}")
                print(f"    Combined units: {', '.join(group['units'])}")
            else:
                combined_name = group["names"][0]
            
            consolidated.append({
                "type": order_type,
                "name": combined_name,
                "target": target,
                "units": group["units"]
            })
        
        return consolidated
    
    @staticmethod
    def build_json_output(defined_actions: List[Dict[str, Any]], 
                         assignments: Dict[str, str]) -> Dict[str, Any]:
        """Build the final JSON output with actions and assigned units.
        
        Args:
            defined_actions: List of action dictionaries
            assignments: Dict mapping unit names to action names
            
        Returns:
            Dictionary with 'orders' list
        """
        orders = []
        
        for action in defined_actions:
            order_entry = {
                "type": action["type"],
                "name": action["name"],
                "target": action["primary_objective"],
                "units": []
            }
            
            # Find all units assigned to this action
            for unit_name, action_name in assignments.items():
                if action_name == action["name"]:
                    order_entry["units"].append(unit_name)
            
            orders.append(order_entry)
        
        # Consolidate orders with same type and target
        print(f"\n  Consolidating orders...")
        consolidated_orders = OrderFormatter.consolidate_orders(orders)
        
        return {"orders": consolidated_orders}
