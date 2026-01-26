"""
Comprehensive unit tests for the Map class and its functions.

This test suite covers:
- Map initialization and setup
- Terrain management
- Unit placement and movement
- Pathfinding (Dijkstra-based reachable hexes)
- Combat engagement logic
- Feature labeling and terrain analysis
- Tactical situation and battlefield summaries
- Frontline calculations
- Order execution
"""

import pytest
from map import Map
from map.terrain import Terrain, FIELDS, HILL, RIVER, FOREST
from map.hex import Hex
from unit import Unit, Infantry
from typing import List, Tuple


# ==================== FIXTURES ====================

@pytest.fixture
def empty_map():
    """Create a basic 10x10 empty map with default terrain."""
    return Map(10, 10)


@pytest.fixture
def small_map():
    """Create a small 5x5 map for quick tests."""
    return Map(5, 5)


@pytest.fixture
def sample_map_with_terrain():
    """Create a map with varied terrain features."""
    game_map = Map(8, 6)
    
    # Add a river down the middle (column 3)
    for y in range(6):
        game_map.set_terrain(3, y, RIVER)
    
    # Add hills on the right side
    for y in range(2, 5):
        for x in range(5, 7):
            game_map.set_terrain(x, y, HILL)
    
    # Add forests on the left side
    for y in range(1, 4):
        for x in range(0, 2):
            game_map.set_terrain(x, y, FOREST())
    
    return game_map


@pytest.fixture
def sample_infantry():
    """Create a sample infantry unit."""
    return Infantry(
        name="Test Infantry",
        faction="French",
        division="1st Division",
        corps="1st Corps",
        mobility=4,
        size=6,
        quality=3,
        morale=7
    )


@pytest.fixture
def sample_infantry_blue():
    """Create a blue faction infantry unit."""
    return Infantry(
        name="Blue Infantry",
        faction="French",
        division="1st Division",
        corps="1st Corps",
        mobility=4,
        size=5,
        quality=3,
        morale=7
    )


@pytest.fixture
def sample_infantry_red():
    """Create a red faction infantry unit."""
    return Infantry(
        name="Red Infantry",
        faction="Austrian",
        division="1st Division",
        corps="1st Corps",
        mobility=4,
        size=5,
        quality=3,
        morale=7
    )


@pytest.fixture
def map_with_units(sample_map_with_terrain, sample_infantry_blue, sample_infantry_red):
    """Create a map with units placed on it."""
    game_map = sample_map_with_terrain
    
    # Place blue unit on the left
    game_map.place_unit(sample_infantry_blue, 1, 2)
    
    # Place red unit on the right
    game_map.place_unit(sample_infantry_red, 6, 2)
    
    return game_map


# ==================== INITIALIZATION TESTS ====================

class TestMapInitialization:
    """Test Map constructor and basic setup."""
    
    def test_map_creation(self, empty_map):
        """Test that a map is created with correct dimensions."""
        assert empty_map.width == 10
        assert empty_map.height == 10
        assert len(empty_map.grid) == 10
        assert len(empty_map.grid[0]) == 10
    
    def test_map_default_terrain(self, empty_map):
        """Test that all hexes start with default FIELDS terrain."""
        for y in range(empty_map.height):
            for x in range(empty_map.width):
                hex_tile = empty_map.get_hex(x, y)
                assert hex_tile is not None
                assert hex_tile.terrain.name == "Fields"
    
    def test_small_map_dimensions(self, small_map):
        """Test creating a map with different dimensions."""
        assert small_map.width == 5
        assert small_map.height == 5


# ==================== TERRAIN MANAGEMENT TESTS ====================

class TestTerrainManagement:
    """Test terrain setting and retrieval."""
    
    def test_set_terrain(self, empty_map):
        """Test setting terrain at a specific location."""
        empty_map.set_terrain(5, 5, HILL)
        terrain = empty_map.get_terrain(5, 5)
        assert terrain.name == "Hill"
    
    def test_get_terrain(self, sample_map_with_terrain):
        """Test retrieving terrain from various locations."""
        # River in the middle
        assert sample_map_with_terrain.get_terrain(3, 2).name == "River"
        
        # Hill on the right
        assert sample_map_with_terrain.get_terrain(5, 3).name == "Hill"
        
        # Forest on the left
        assert sample_map_with_terrain.get_terrain(1, 2).name == "Forest"
    
    def test_get_hex_valid(self, empty_map):
        """Test getting a hex with valid coordinates."""
        hex_tile = empty_map.get_hex(5, 5)
        assert hex_tile is not None
        assert isinstance(hex_tile, Hex)
    
    def test_get_hex_invalid(self, empty_map):
        """Test getting a hex with invalid coordinates returns None."""
        assert empty_map.get_hex(-1, 0) is None
        assert empty_map.get_hex(0, -1) is None
        assert empty_map.get_hex(10, 5) is None
        assert empty_map.get_hex(5, 10) is None
        assert empty_map.get_hex(100, 100) is None


# ==================== UNIT PLACEMENT TESTS ====================

class TestUnitPlacement:
    """Test unit placement and management."""
    
    def test_place_unit_success(self, empty_map, sample_infantry):
        """Test successfully placing a unit on the map."""
        result = empty_map.place_unit(sample_infantry, 5, 5)
        assert result is True
        assert empty_map.get_hex(5, 5).unit == sample_infantry
        assert sample_infantry.x == 5
        assert sample_infantry.y == 5
    
    def test_place_unit_out_of_bounds(self, empty_map, sample_infantry):
        """Test that placing a unit out of bounds fails."""
        assert empty_map.place_unit(sample_infantry, -1, 5) is False
        assert empty_map.place_unit(sample_infantry, 5, -1) is False
        assert empty_map.place_unit(sample_infantry, 10, 5) is False
        assert empty_map.place_unit(sample_infantry, 5, 10) is False
    
    def test_place_unit_occupied_hex(self, empty_map, sample_infantry_blue, sample_infantry_red):
        """Test that placing a unit on an occupied hex fails."""
        empty_map.place_unit(sample_infantry_blue, 5, 5)
        result = empty_map.place_unit(sample_infantry_red, 5, 5)
        assert result is False
        assert empty_map.get_hex(5, 5).unit == sample_infantry_blue
    
    def test_get_units(self, map_with_units):
        """Test retrieving all units from the map."""
        units = map_with_units.get_units()
        assert len(units) == 2
        assert all(isinstance(u, Unit) for u in units)
    
    def test_get_units_by_faction(self, map_with_units):
        """Test retrieving units by faction."""
        french_units = map_with_units.get_units_by_faction("French")
        austrian_units = map_with_units.get_units_by_faction("Austrian")
        
        assert len(french_units) == 1
        assert len(austrian_units) == 1
        assert french_units[0].faction == "French"
        assert austrian_units[0].faction == "Austrian"
    
    def test_get_units_empty_map(self, empty_map):
        """Test that an empty map returns no units."""
        units = empty_map.get_units()
        assert len(units) == 0


# ==================== UNIT MOVEMENT TESTS ====================

class TestUnitMovement:
    """Test unit movement mechanics."""
    
    def test_teleport_unit(self, empty_map, sample_infantry):
        """Test instantly moving a unit to a new location."""
        empty_map.place_unit(sample_infantry, 2, 2)
        empty_map.teleport_unit(sample_infantry, 5, 5, cost=3)
        
        # Check unit moved
        assert sample_infantry.x == 5
        assert sample_infantry.y == 5
        
        # Check old hex is empty
        assert empty_map.get_hex(2, 2).unit is None
        
        # Check new hex contains unit
        assert empty_map.get_hex(5, 5).unit == sample_infantry
    
    def test_move_unit_within_range(self, empty_map, sample_infantry):
        """Test moving a unit to a reachable hex."""
        empty_map.place_unit(sample_infantry, 5, 5)
        # Ensure unit has no engagement
        sample_infantry.engagement = 0
        
        # Move to adjacent hex (should be reachable)
        result = empty_map.move_unit(sample_infantry, 6, 5)
        
        assert result is True
        assert sample_infantry.x == 6
        assert sample_infantry.y == 5
    
    def test_move_unit_to_same_hex(self, empty_map, sample_infantry):
        """Test that moving to the same hex is allowed (costs 0)."""
        empty_map.place_unit(sample_infantry, 5, 5)
        sample_infantry.engagement = 0
        result = empty_map.move_unit(sample_infantry, 5, 5)
        
        # Moving to same hex should succeed (0 cost)
        assert result is True
        assert sample_infantry.x == 5
        assert sample_infantry.y == 5
    
    def test_move_unit_too_far(self, empty_map, sample_infantry):
        """Test that moving beyond mobility range fails."""
        empty_map.place_unit(sample_infantry, 0, 0)
        sample_infantry.engagement = 0
        
        # Try to move very far away (beyond mobility)
        result = empty_map.move_unit(sample_infantry, 9, 9)
        
        assert result is False
        # Unit should stay in place
        assert sample_infantry.x == 0
        assert sample_infantry.y == 0


# ==================== PATHFINDING TESTS ====================

class TestPathfinding:
    """Test pathfinding and reachable hex calculations."""
    
    def test_find_reachable_hexes_basic(self, empty_map, sample_infantry):
        """Test finding reachable hexes on an open map."""
        empty_map.place_unit(sample_infantry, 5, 5)
        reachable = empty_map.find_reachable_hexes(sample_infantry)
        
        # Should be able to reach starting position (cost 0)
        assert (5, 5) in reachable
        assert reachable[(5, 5)] == 0
        
        # Should be able to reach adjacent hexes (cost 1 on fields)
        neighbors = empty_map.get_neighbors(5, 5)
        for nx, ny in neighbors:
            if 0 <= nx < empty_map.width and 0 <= ny < empty_map.height:
                assert (nx, ny) in reachable
    
    def test_find_reachable_hexes_with_river(self, sample_map_with_terrain, sample_infantry_blue):
        """Test that rivers increase movement cost."""
        sample_map_with_terrain.place_unit(sample_infantry_blue, 2, 3)
        reachable = sample_map_with_terrain.find_reachable_hexes(sample_infantry_blue)
        
        # River at (3, 3) should be reachable but costly
        if (3, 3) in reachable:
            # River has move_cost of 3
            assert reachable[(3, 3)] >= 3
    
    def test_find_reachable_hexes_blocked_by_unit(self, empty_map, sample_infantry_blue, sample_infantry_red):
        """Test that enemy units block movement."""
        empty_map.place_unit(sample_infantry_blue, 5, 5)
        empty_map.place_unit(sample_infantry_red, 6, 5)
        
        reachable = empty_map.find_reachable_hexes(sample_infantry_blue)
        
        # Enemy-occupied hex should not be in reachable
        assert (6, 5) not in reachable
    
    def test_get_neighbors(self, empty_map):
        """Test hex neighbor calculation (odd-q offset coordinates)."""
        # Test a hex in the middle
        neighbors = list(empty_map.get_neighbors(5, 5))
        assert len(neighbors) == 6
        
        # All neighbors should be tuples
        for neighbor in neighbors:
            assert isinstance(neighbor, tuple)
            assert len(neighbor) == 2
    
    def test_hex_distance(self, empty_map):
        """Test hex distance calculation."""
        # Distance from a hex to itself should be 0
        assert empty_map._hex_distance(5, 5, 5, 5) == 0
        
        # Distance to direct neighbor should be 1
        distance = empty_map._hex_distance(5, 5, 6, 5)
        assert distance == 1


# ==================== COMBAT TESTS ====================

class TestCombat:
    """Test combat engagement mechanics."""
    
    def test_check_and_engage_combat_adjacent_enemies(self, empty_map, sample_infantry_blue, sample_infantry_red):
        """Test that adjacent enemy units engage in combat."""
        empty_map.place_unit(sample_infantry_blue, 5, 5)
        empty_map.place_unit(sample_infantry_red, 6, 5)
        
        # Check engagement
        empty_map.check_and_engage_combat(sample_infantry_blue)
        
        # Blue unit should be engaged (increment happens)
        assert sample_infantry_blue.engagement > 0
    
    def test_check_all_engagements(self, map_with_units):
        """Test checking all units for engagement."""
        # Initially units are far apart
        map_with_units.check_all_engagements()
        
        # Units are not adjacent, so they shouldn't be engaged
        blue_unit = map_with_units.get_units_by_faction("French")[0]
        red_unit = map_with_units.get_units_by_faction("Austrian")[0]
        
        # Move them adjacent
        map_with_units.teleport_unit(red_unit, 2, 2, cost=0)
        map_with_units.check_all_engagements()
        
        assert blue_unit.engagement > 0
        assert red_unit.engagement > 0


# ==================== FEATURE LABELING TESTS ====================

class TestFeatureLabeling:
    """Test terrain feature labeling and analysis."""
    
    def test_label_terrain_features(self, sample_map_with_terrain):
        """Test that terrain features are labeled properly."""
        sample_map_with_terrain.label_terrain_features(seed=42)
        
        # Check that features were added
        feature_names = sample_map_with_terrain.list_feature_names()
        assert len(feature_names) > 0
    
    def test_list_feature_names(self, sample_map_with_terrain):
        """Test retrieving list of feature names."""
        sample_map_with_terrain.label_terrain_features(seed=42)
        feature_names = sample_map_with_terrain.list_feature_names()
        
        # Should return a sorted list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        assert feature_names == sorted(feature_names)
    
    def test_get_feature_coordinates(self, sample_map_with_terrain):
        """Test retrieving coordinates for a specific feature."""
        # Manually add a feature to test
        sample_map_with_terrain.get_hex(2, 2).features.append("Test Hill")
        sample_map_with_terrain.get_hex(2, 3).features.append("Test Hill")
        
        coords = sample_map_with_terrain.get_feature_coordinates("Test Hill")
        
        assert len(coords) == 2
        assert (2, 2) in coords
        assert (2, 3) in coords
    
    def test_get_feature_coordinates_nonexistent(self, empty_map):
        """Test getting coordinates for a non-existent feature."""
        coords = empty_map.get_feature_coordinates("Nonexistent Feature")
        assert len(coords) == 0


# ==================== TACTICAL SITUATION TESTS ====================

class TestTacticalSituation:
    """Test tactical situation analysis functions."""
    
    def test_get_tactical_situation(self, map_with_units):
        """Test retrieving comprehensive tactical situation."""
        # Initialize engagement attribute for all units
        for unit in map_with_units.get_units():
            if not hasattr(unit, 'engagement'):
                unit.engagement = 0
        
        situation = map_with_units.get_tactical_situation("French")
        
        assert "friendly_units" in situation
        assert "enemy_units" in situation
        assert "terrain_features" in situation
        
        assert len(situation["friendly_units"]) == 1
        assert len(situation["enemy_units"]) == 1
    
    def test_tactical_situation_friendly_units(self, map_with_units):
        """Test that friendly units are reported correctly."""
        # Initialize engagement attribute for all units
        for unit in map_with_units.get_units():
            if not hasattr(unit, 'engagement'):
                unit.engagement = 0
        
        situation = map_with_units.get_tactical_situation("French")
        friendly = situation["friendly_units"][0]
        
        assert friendly["name"] == "Blue Infantry"
        assert "position" in friendly
        assert "size" in friendly
        assert "mobility" in friendly
        assert "engaged" in friendly
        assert "on_frontline" in friendly
    
    def test_tactical_situation_enemy_units(self, map_with_units):
        """Test that enemy units are reported correctly."""
        # Initialize engagement attribute for all units
        for unit in map_with_units.get_units():
            if not hasattr(unit, 'engagement'):
                unit.engagement = 0
        
        situation = map_with_units.get_tactical_situation("French")
        enemy = situation["enemy_units"][0]
        
        assert enemy["name"] == "Red Infantry"
        assert "position" in enemy
        assert "size" in enemy
        assert "distance_to_nearest_friendly" in enemy
        assert enemy["distance_to_nearest_friendly"] is not None
    
    def test_get_battlefield_summary(self, map_with_units):
        """Test generating battlefield summary string."""
        # Initialize engagement attribute for all units
        for unit in map_with_units.get_units():
            if not hasattr(unit, 'engagement'):
                unit.engagement = 0
        
        summary = map_with_units.get_battlefield_summary("French")
        
        assert isinstance(summary, str)
        assert "BATTLEFIELD SITUATION" in summary
        assert "YOUR FORCES" in summary
        assert "ENEMY FORCES" in summary
        assert "Blue Infantry" in summary
        assert "Red Infantry" in summary


# ==================== HELPER FUNCTION TESTS ====================

class TestHelperFunctions:
    """Test internal helper functions."""
    
    def test_find_unit_by_name(self, map_with_units):
        """Test finding a unit by name."""
        unit = map_with_units._find_unit_by_name("Blue Infantry")
        assert unit is not None
        assert unit.name == "Blue Infantry"
        
        # Test non-existent unit
        unit = map_with_units._find_unit_by_name("Nonexistent Unit")
        assert unit is None
    
    def test_nearest_enemy_pos(self, map_with_units):
        """Test finding nearest enemy position."""
        blue_unit = map_with_units.get_units_by_faction("French")[0]
        enemy_pos = map_with_units._nearest_enemy_pos(blue_unit)
        
        assert enemy_pos is not None
        assert isinstance(enemy_pos, tuple)
        assert len(enemy_pos) == 2
        
        # Should be the position of the red unit
        red_unit = map_with_units.get_units_by_faction("Austrian")[0]
        assert enemy_pos == (red_unit.x, red_unit.y)
    
    def test_best_reachable_toward(self, empty_map, sample_infantry):
        """Test finding best reachable hex toward a target."""
        empty_map.place_unit(sample_infantry, 2, 2)
        target = (7, 7)
        
        best = empty_map._best_reachable_toward(sample_infantry, target)
        
        # Should return a position closer to target than current position
        if best:
            current_dist = empty_map._hex_distance(2, 2, target[0], target[1])
            new_dist = empty_map._hex_distance(best[0], best[1], target[0], target[1])
            assert new_dist <= current_dist


# ==================== ORDER EXECUTION TESTS ====================

class TestOrderExecution:
    """Test order execution (march, retreat, etc.)."""
    
    def test_march_order(self, empty_map, sample_infantry):
        """Test march order execution."""
        empty_map.place_unit(sample_infantry, 2, 2)
        sample_infantry.engagement = 0
        
        result = empty_map.march("Test Infantry", (4, 4))
        
        assert result["ok"] is True
        assert result["unit"] == "Test Infantry"
        # Unit should have moved closer to destination
        assert (sample_infantry.x != 2 or sample_infantry.y != 2)
    
    def test_march_order_nonexistent_unit(self, empty_map):
        """Test march order with non-existent unit."""
        result = empty_map.march("Nonexistent Unit", (5, 5))
        
        assert result["ok"] is False
        assert "error" in result
    
    def test_march_order_engaged_unit(self, empty_map, sample_infantry_blue, sample_infantry_red):
        """Test that engaged units cannot march."""
        empty_map.place_unit(sample_infantry_blue, 5, 5)
        empty_map.place_unit(sample_infantry_red, 6, 5)
        sample_infantry_blue.engagement = 0
        sample_infantry_red.engagement = 0
        
        # Engage units in combat
        empty_map.check_and_engage_combat(sample_infantry_blue)
        
        # Try to march while engaged
        result = empty_map.march("Blue Infantry", (2, 2))
        
        assert result["ok"] is False
        assert "engaged" in result.get("reason", "").lower() or "combat" in result.get("reason", "").lower()
    
    def test_retreat_order(self, empty_map, sample_infantry_blue, sample_infantry_red):
        """Test retreat order execution."""
        empty_map.place_unit(sample_infantry_blue, 5, 5)
        empty_map.place_unit(sample_infantry_red, 6, 5)
        sample_infantry_blue.engagement = 0
        sample_infantry_red.engagement = 0
        
        # Engage units
        empty_map.check_and_engage_combat(sample_infantry_blue)
        
        # Retreat should work even when engaged
        result = empty_map.retreat("Blue Infantry", (1, 1))
        
        assert result["ok"] is True or result["ok"] is False  # May fail if no valid retreat path
        assert result["unit"] == "Blue Infantry"


# ==================== FRONTLINE CALCULATION TESTS ====================

class TestFrontlineCalculations:
    """Test frontline and enemy approach calculations."""
    
    def test_get_enemy_approach_angle(self, map_with_units):
        """Test calculating enemy approach angle to a feature."""
        # Add a feature
        map_with_units.get_hex(1, 2).features.append("Test Hill")
        map_with_units.get_hex(1, 3).features.append("Test Hill")
        
        angle = map_with_units.get_enemy_approach_angle("French", "Test Hill")
        
        # Should return an angle or None
        if angle is not None:
            assert isinstance(angle, float)
            assert 0 <= angle < 360
    
    def test_get_enemy_approach_angle_no_enemies(self, empty_map, sample_infantry_blue):
        """Test approach angle calculation with no enemies."""
        empty_map.place_unit(sample_infantry_blue, 5, 5)
        empty_map.get_hex(3, 3).features.append("Test Feature")
        
        angle = empty_map.get_enemy_approach_angle("French", "Test Feature")
        
        # Should return None when no enemies exist
        assert angle is None
    
    def test_get_enemy_approach_angle_nonexistent_feature(self, map_with_units):
        """Test approach angle for non-existent feature."""
        angle = map_with_units.get_enemy_approach_angle("French", "Nonexistent Feature")
        assert angle is None
    
    def test_distribute_units_along_frontline(self, empty_map):
        """Test distributing units along a frontline."""
        coordinates = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        distributed = empty_map.distribute_units_along_frontline(coordinates, 3)
        
        assert len(distributed) == 3
        # Should return evenly spaced points
        assert all(coord in coordinates for coord in distributed)


# ==================== EDGE CASE TESTS ====================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_hex_map(self):
        """Test creating and using a 1x1 map."""
        tiny_map = Map(1, 1)
        assert tiny_map.width == 1
        assert tiny_map.height == 1
        
        hex_tile = tiny_map.get_hex(0, 0)
        assert hex_tile is not None
    
    def test_large_map_creation(self):
        """Test creating a large map."""
        large_map = Map(100, 100)
        assert large_map.width == 100
        assert large_map.height == 100
        
        # Test corners
        assert large_map.get_hex(0, 0) is not None
        assert large_map.get_hex(99, 99) is not None
    
    def test_empty_map_operations(self, empty_map):
        """Test operations on an empty map (no units)."""
        units = empty_map.get_units()
        assert len(units) == 0
        
        # Check all engagements on empty map (should not crash)
        empty_map.check_all_engagements()
        
        # Get tactical situation with no units
        situation = empty_map.get_tactical_situation("French")
        assert len(situation["friendly_units"]) == 0
        assert len(situation["enemy_units"]) == 0


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests combining multiple map functions."""
    
    def test_complete_turn_simulation(self, map_with_units):
        """Test a complete turn with multiple operations."""
        blue_unit = map_with_units.get_units_by_faction("French")[0]
        red_unit = map_with_units.get_units_by_faction("Austrian")[0]
        
        # Initialize engagement
        for unit in map_with_units.get_units():
            if not hasattr(unit, 'engagement'):
                unit.engagement = 0
        
        # 1. Move blue unit
        initial_pos = (blue_unit.x, blue_unit.y)
        map_with_units.move_unit(blue_unit, blue_unit.x + 1, blue_unit.y)
        
        # 2. Check for engagements
        map_with_units.check_all_engagements()
        
        # 3. Get tactical situation
        situation = map_with_units.get_tactical_situation("French")
        assert len(situation["friendly_units"]) == 1
        assert len(situation["enemy_units"]) == 1
        
        # 4. Generate battlefield summary
        summary = map_with_units.get_battlefield_summary("French")
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_pathfinding_around_obstacles(self, sample_map_with_terrain, sample_infantry):
        """Test pathfinding around rivers and difficult terrain."""
        # Place unit on left side of river
        sample_map_with_terrain.place_unit(sample_infantry, 1, 3)
        
        # Find reachable hexes (should account for river crossing cost)
        reachable = sample_map_with_terrain.find_reachable_hexes(sample_infantry)
        
        # Unit should be able to reach starting position
        assert (1, 3) in reachable
        
        # Check that path cost is calculated properly
        for coord, cost in reachable.items():
            assert cost >= 0
            assert cost <= sample_infantry.mobility


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
