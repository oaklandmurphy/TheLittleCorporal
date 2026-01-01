"""
ReconnaissanceService - Handles battlefield intelligence gathering.

Responsibilities:
- Define reconnaissance tools for LLM
- Execute reconnaissance operations
- Generate intelligence reports
"""

from typing import List, Dict, Any, Optional
import map.frontline as frontline
from .terrain_analyzer import TerrainAnalyzer


class ReconnaissanceService:
    """Provides intelligence gathering capabilities for battlefield operations."""
    
    # Constants
    ENEMY_DETECTION_RANGE = 3  # Hexes for enemy detection
    
    def __init__(self, game_map, faction: str, terrain_analyzer: TerrainAnalyzer):
        """Initialize the reconnaissance service.
        
        Args:
            game_map: The game map instance
            faction: The faction this service serves
            terrain_analyzer: Terrain analysis service
        """
        self.game_map = game_map
        self.faction = faction
        self.terrain = terrain_analyzer
    
    def get_reconnaissance_tools(self) -> List[Dict[str, Any]]:
        """Define reconnaissance tools available to the General for intelligence gathering.
        
        Returns:
            List of tool definition dictionaries for LLM
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "reconnaissance_feature",
                    "description": "Get detailed information about a specific terrain feature including its military value, occupation status, and location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feature_name": {
                                "type": "string",
                                "description": "The name of the terrain feature to investigate"
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
                    "description": "Analyze enemy forces near a specific location or feature. Returns a high-level overview of enemy strength and a recommendation for the number of units to commit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The terrain feature or area to assess enemy strength around"
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
                    "description": "Survey approach routes to a location, focusing on enemy units on the path or in overlooking positions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_feature": {
                                "type": "string",
                                "description": "The terrain feature or objective to survey approaches to"
                            }
                        },
                        "required": ["target_feature"]
                    }
                }
            }
        ]
    
    def execute_reconnaissance_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a reconnaissance tool and return the results.
        
        Args:
            tool_name: Name of the reconnaissance tool
            args: Tool arguments
            
        Returns:
            Dict with 'ok' (bool) and either 'intelligence' or 'error' key
        """
        if not self.game_map:
            return {"ok": False, "error": "No map available for reconnaissance"}

        try:
            if tool_name == "reconnaissance_feature":
                return self._recon_feature(args)
            elif tool_name == "assess_enemy_strength":
                return self._recon_assess_enemy(args)
            elif tool_name == "survey_approaches":
                return self._recon_survey_approaches(args)
            else:
                return {"ok": False, "error": f"Unknown reconnaissance tool: {tool_name}"}
        except Exception as e:
            return {"ok": False, "error": f"Reconnaissance error: {str(e)}"}
    
    def _recon_feature(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reconnaissance_feature tool: detailed info about a terrain feature.
        
        Args:
            args: Dict with 'feature_name' key
            
        Returns:
            Dict with reconnaissance results
        """
        feature_name = args.get("feature_name", "")
        coords = self.game_map.get_feature_coordinates(feature_name)
        if not coords:
            return {"ok": False, "error": f"No feature named '{feature_name}' found."}

        # 1. Military Value
        # Use a simplified version of frontline's identification logic
        enemy_direction = self.game_map.get_enemy_approach_angle(self.faction, feature_name)
        military_value = "Low"
        if enemy_direction is not None:
            endpoints = frontline.get_frontline_endpoints(self.game_map.grid, self.game_map.width, self.game_map.height, coords, enemy_direction)
            if endpoints:
                frontline_data = frontline.get_best_frontline_with_advantage(
                    self.game_map.grid, self.game_map.width, self.game_map.height, endpoints[0], endpoints[1], enemy_direction
                )
                avg_adv = frontline_data.get('average_advantage', 0)
                if avg_adv > 1.5:
                    military_value = "High"
                elif avg_adv > 0.75:
                    military_value = "Medium"

        # 2. Status (Occupied, Contested, Unoccupied)
        units_on = self.terrain.get_units_on_feature(coords)
        friendly_on = any(u.faction == self.faction for u in units_on)
        enemy_on = any(u.faction != self.faction for u in units_on)
        
        status = "Unoccupied"
        if friendly_on:
            status = "Occupied by friendly forces"
        elif enemy_on:
            status = f"Occupied by {len([u for u in units_on if u.faction != self.faction])} enemy unit(s)"
        else:
            # Check for nearby enemies to determine if "Contested"
            nearby_enemies = self.terrain.get_enemy_units_near_coords(coords, max_distance=2)
            if nearby_enemies:
                status = f"Contested, with {len(nearby_enemies)} enemy unit(s) nearby"

        # 3. Location description
        feature_center_x = int(sum(x for x, y in coords) / len(coords))
        feature_center_y = int(sum(y for x, y in coords) / len(coords))
        location_desc = f"centered at ({feature_center_x}, {feature_center_y})"

        terrain_type = self.terrain.get_predominant_terrain(coords)

        # Generate tactical recommendation based on intelligence
        recommendations = []
        
        # Military value assessment
        if military_value == "High":
            recommendations.append("This position offers excellent defensive advantages and should be prioritized")
        elif military_value == "Medium":
            recommendations.append("This position provides moderate tactical benefit")
        else:  # Low
            recommendations.append("This position offers minimal defensive value - committing forces here may not be tactically sound")
        
        # Occupation status recommendations
        if enemy_on:
            enemy_count = len([u for u in units_on if u.faction != self.faction])
            if enemy_count > 1:
                recommendations.append(f"Strongly defended by {enemy_count} enemy units - will require significant force to dislodge")
            else:
                recommendations.append("Enemy occupied - assault or containment required")
        elif friendly_on:
            recommendations.append("Already secured by friendly forces - consider using as staging point or reinforcing")
        elif nearby_enemies:
            recommendations.append(f"Contested area with {len(nearby_enemies)} nearby enemy units - rapid movement recommended to secure before enemy arrival")
        else:
            recommendations.append("Undefended - can be secured easily if deemed valuable")

        intelligence = (
            f"Intelligence on '{feature_name}' ({terrain_type}):\n"
            f"  - Military Value: {military_value}. (Based on defensive potential against likely enemy approach)\n"
            f"  - Status: {status}.\n"
            f"  - Location: {location_desc}.\n"
            f"  - Recommendation: {' '.join(recommendations)}"
        )

        return {"ok": True, "intelligence": intelligence}
    
    def _recon_assess_enemy(self, args: Dict[str, Any], unit_list: List = None) -> Dict[str, Any]:
        """Execute assess_enemy_strength tool: analyze enemy forces near a location.
        
        Args:
            args: Dict with 'location' key
            unit_list: List of friendly units for recommendation calculations
            
        Returns:
            Dict with enemy strength assessment
        """
        location = args.get("location", "")
        coords = self.game_map.get_feature_coordinates(location)
        if not coords:
            # If not a feature, maybe it's a coordinate string like "10,15"
            try:
                x, y = map(int, location.split(','))
                coords = [(x, y)]
            except (ValueError, IndexError):
                return {"ok": False, "error": f"Unknown location or invalid coordinate: {location}"}

        enemy_units = self.terrain.get_enemy_units_near_coords(coords, self.ENEMY_DETECTION_RANGE)
        
        if not enemy_units:
            assessment = f"No enemy units detected within {self.ENEMY_DETECTION_RANGE} hexes of {location}."
            return {"ok": True, "intelligence": assessment}

        # High-level overview
        total_strength = sum(u['size'] for u in enemy_units)
        avg_quality = sum(u['quality'] for u in enemy_units) / len(enemy_units)
        avg_morale = sum(u['morale'] for u in enemy_units) / len(enemy_units)
        
        from .order_formatter import OrderFormatter
        quality_desc = OrderFormatter.get_unit_attribute_label(round(avg_quality), 'quality')
        morale_desc = OrderFormatter.get_unit_attribute_label(round(avg_morale), 'morale')

        overview = (
            f"Detected {len(enemy_units)} enemy formations near {location} with a combined strength of {total_strength}. "
            f"Average quality is '{quality_desc}' and morale is '{morale_desc}'."
        )

        # Recommendation for number of units
        # Simple combat power = size * quality
        enemy_power = sum(u['size'] * u['quality'] for u in enemy_units)
        
        friendly_avg_power = 0
        if unit_list:
            friendly_avg_power = sum(u.size * u.quality for u in unit_list) / len(unit_list)

        if friendly_avg_power > 0:
            # Recommend a force with 1.2x the power for a good engagement chance
            recommended_units = round((enemy_power * 1.2) / friendly_avg_power)
            recommended_units = max(1, recommended_units)  # Always recommend at least one unit
            
            # Calculate force ratio for tactical guidance
            total_friendly_power = sum(u.size * u.quality for u in unit_list)
            force_ratio = total_friendly_power / enemy_power if enemy_power > 0 else float('inf')
            
            # Generate tactical recommendation based on force ratio
            if force_ratio >= 2.0:
                tactical_advice = "You have overwhelming superiority - consider aggressive action with confidence."
            elif force_ratio >= 1.5:
                tactical_advice = "You have favorable odds - a well-coordinated attack should succeed."
            elif force_ratio >= 1.0:
                tactical_advice = "Forces are roughly balanced - success depends on tactical execution and terrain advantage."
            elif force_ratio >= 0.7:
                tactical_advice = "Enemy has slight advantage - consider defensive posture, flanking maneuvers, or reinforcement before engagement."
            else:
                tactical_advice = "Enemy has significant advantage - direct engagement not recommended. Consider containment, delay tactics, or repositioning."
            
            recommendation = (
                f"To engage this force with a reasonable chance of success, commit approximately {recommended_units} of your {len(unit_list)} available units. "
                f"{tactical_advice}"
            )
        else:
            recommendation = "Cannot provide a recommendation without information on friendly units."

        assessment = f"Enemy Strength Assessment near {location}:\n- {overview}\n- {recommendation}"
        return {"ok": True, "intelligence": assessment}
    
    def _recon_survey_approaches(self, args: Dict[str, Any], unit_list: List = None) -> Dict[str, Any]:
        """Execute survey_approaches tool: identify approach routes to a target.
        
        Args:
            args: Dict with 'target_feature' key
            unit_list: List of friendly units to calculate average position
            
        Returns:
            Dict with approach survey results
        """
        target_feature = args.get("target_feature", "")
        coords = self.game_map.get_feature_coordinates(target_feature)
        if not coords:
            return {"ok": False, "error": f"Unknown feature: {target_feature}"}
        
        if not unit_list:
            return {"ok": False, "error": "No units under command"}
        
        # Calculate average position of forces
        valid_units = [u for u in unit_list if u.x is not None and u.y is not None]
        if not valid_units:
            return {"ok": False, "error": "No units with valid positions"}
        
        avg_x = sum(unit.x for unit in valid_units) / len(valid_units)
        avg_y = sum(unit.y for unit in valid_units) / len(valid_units)
        avg_pos = (int(avg_x), int(avg_y))
        
        # Build path to target
        path_hexes = self.terrain.build_path_to_target(avg_pos, coords)
        if not path_hexes:
            return {"ok": True, "intelligence": f"Could not determine a clear approach path to {target_feature}."}

        # 1. Analyze threats directly ON the path
        enemies_on_path = []
        path_hex_set = set(path_hexes)
        all_enemies = self.terrain.get_enemy_units_near_coords(path_hexes, max_distance=0)  # Enemies exactly on the path
        
        on_path_summary = "The approach path appears clear of immediate enemy presence."
        if all_enemies:
            enemies_on_path = [e for e in all_enemies if e['position'] in path_hex_set]
            if enemies_on_path:
                total_strength = sum(e['size'] for e in enemies_on_path)
                on_path_summary = f"WARNING: The direct path is blocked by {len(enemies_on_path)} enemy unit(s) with a total strength of {total_strength}."

        # 2. Analyze threats OVERLOOKING the path (flanking features)
        flanking_threats = []
        flanking_features = self.terrain.identify_flanking_features(
            self.terrain.get_all_features(), set(), target_feature, path_hexes, (0, 0)
        )
        
        for feature_name, info in flanking_features.items():
            enemies_on_flank = self.terrain.get_enemy_units_near_coords(info["coords"], max_distance=0)
            if enemies_on_flank:
                flank_strength = sum(e['size'] for e in enemies_on_flank)
                flanking_threats.append(
                    f"the {info['direction']} flank is overlooked by {len(enemies_on_flank)} enemy unit(s) "
                    f"(strength {flank_strength}) on '{feature_name}'"
                )
        
        overlooking_summary = "No enemy units detected in flanking positions."
        if flanking_threats:
            overlooking_summary = f"CAUTION: {'; '.join(flanking_threats)}."
        
        # Generate tactical recommendation based on approach analysis
        approach_recommendation = []
        
        if enemies_on_path:
            approach_recommendation.append("Direct approach is blocked - consider alternative routes, preliminary bombardment, or clearing operations before main advance.")
        elif flanking_threats:
            approach_recommendation.append("Direct path clear but flanks exposed - recommend securing flanking positions first or advancing with flank guards to prevent envelopment.")
        else:
            approach_recommendation.append("Approach is clear and secure - favorable conditions for rapid advance or maneuver.")
        
        # Add path length consideration
        if len(path_hexes) > 10:
            approach_recommendation.append(f"Long approach distance ({len(path_hexes)} hexes) - consider logistics and potential enemy repositioning during advance.")
        elif len(path_hexes) > 5:
            approach_recommendation.append(f"Moderate approach distance ({len(path_hexes)} hexes) - can be reached in reasonable time.")
        else:
            approach_recommendation.append(f"Short approach distance ({len(path_hexes)} hexes) - objective can be reached quickly.")

        intelligence = (
            f"Approach Survey to '{target_feature}':\n"
            f"- Direct Path: {on_path_summary}\n"
            f"- Flanking Threats: {overlooking_summary}\n"
            f"- Tactical Assessment: {' '.join(approach_recommendation)}"
        )
        
        return {"ok": True, "intelligence": intelligence}
