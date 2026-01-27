# Map Testing Documentation

## Overview

This test suite provides comprehensive unit tests for the `map.py` file in TheLittleCorporal game. The tests verify all core map functionality including terrain management, unit placement, pathfinding, combat engagement, and tactical situation analysis.

## Test Framework

- **Framework**: pytest 9.0.2
- **Coverage Tool**: pytest-cov 7.0.0
- **Python Version**: 3.14.0
- **Current Coverage**: 63% of map.py

## Running Tests

### Run all tests:
```bash
python -m pytest test_map.py -v
```

### Run specific test class:
```bash
python -m pytest test_map.py::TestMapInitialization -v
```

### Run with coverage report:
```bash
python -m pytest test_map.py --cov=map.map --cov-report=term-missing
```

### Run specific test:
```bash
python -m pytest test_map.py::TestUnitMovement::test_move_unit_within_range -v
```

## Test Structure

### 1. TestMapInitialization (3 tests)
Tests basic map creation and initialization:
- Map dimensions are correctly set
- Default terrain (Fields) is applied to all hexes
- Different map sizes work correctly

### 2. TestTerrainManagement (4 tests)
Tests terrain setting and retrieval:
- Setting terrain at specific coordinates
- Getting terrain from hexes
- Valid and invalid hex coordinate handling

### 3. TestUnitPlacement (6 tests)
Tests unit placement on the map:
- Successful unit placement
- Out-of-bounds placement rejection
- Occupied hex rejection
- Getting all units from map
- Filtering units by faction

### 4. TestUnitMovement (4 tests)
Tests unit movement mechanics:
- Teleporting units (instant movement)
- Moving units within mobility range
- Moving to the same hex
- Rejecting movement beyond range

### 5. TestPathfinding (5 tests)
Tests Dijkstra-based pathfinding:
- Finding reachable hexes within mobility
- Terrain cost impact (rivers, hills, forests)
- Enemy unit blocking
- Hex neighbor calculation
- Hex distance calculation

### 6. TestCombat (2 tests)
Tests combat engagement mechanics:
- Adjacent enemy engagement
- Mass engagement checking

### 7. TestFeatureLabeling (4 tests)
Tests terrain feature labeling system:
- Labeling terrain features (hills, forests, rivers)
- Listing all feature names
- Getting feature coordinates
- Handling non-existent features

### 8. TestTacticalSituation (4 tests)
Tests tactical analysis functions:
- Comprehensive tactical situation retrieval
- Friendly unit reporting
- Enemy unit reporting with distances
- Natural language battlefield summaries

### 9. TestHelperFunctions (3 tests)
Tests internal helper functions:
- Finding units by name
- Finding nearest enemy position
- Finding best reachable hex toward target

### 10. TestOrderExecution (4 tests)
Tests order execution (march, retreat):
- March orders to destinations
- Handling non-existent units
- Preventing engaged units from marching
- Retreat orders (even when engaged)

### 11. TestFrontlineCalculations (4 tests)
Tests frontline and approach angle calculations:
- Enemy approach angle to features
- Handling maps with no enemies
- Handling non-existent features
- Unit distribution along frontlines

### 12. TestEdgeCases (3 tests)
Tests boundary conditions:
- Single-hex maps (1x1)
- Large maps (100x100)
- Operations on empty maps

### 13. TestIntegration (2 tests)
Integration tests combining multiple operations:
- Complete turn simulation
- Pathfinding around obstacles

## Test Fixtures

The test suite uses pytest fixtures for common test setup:

### Map Fixtures
- `empty_map`: 10x10 map with default terrain
- `small_map`: 5x5 map for quick tests
- `sample_map_with_terrain`: 8x6 map with rivers, hills, and forests

### Unit Fixtures
- `sample_infantry`: Generic test infantry unit
- `sample_infantry_blue`: French faction infantry
- `sample_infantry_red`: Austrian faction infantry
- `map_with_units`: Pre-populated map with units

## Key Test Coverage Areas

### ✅ Fully Covered
- Map initialization and setup
- Terrain management (get/set)
- Basic unit placement
- Hex coordinate validation
- Feature labeling and queries
- Pathfinding (reachable hexes)
- Tactical situation reporting

### ⚠️ Partially Covered (63% coverage)
- Some edge cases in movement logic
- Advanced combat scenarios
- Complex order execution flows
- Detailed frontline calculations
- Battlefield summary formatting

### ❌ Not Yet Covered
- Order execution module integration
- Some advanced tactical analysis functions
- Edge cases in retreat pathfinding
- Stance-based movement modifiers

## Common Test Patterns

### Testing Map Operations
```python
def test_operation(self, empty_map):
    # Arrange: Set up test data
    game_map = empty_map
    
    # Act: Perform operation
    result = game_map.some_function()
    
    # Assert: Verify results
    assert result == expected_value
```

### Testing with Units
```python
def test_unit_operation(self, empty_map, sample_infantry):
    # Place unit
    empty_map.place_unit(sample_infantry, 5, 5)
    
    # Ensure engagement is initialized
    sample_infantry.engagement = 0
    
    # Perform operation
    result = empty_map.move_unit(sample_infantry, 6, 5)
    
    # Verify
    assert result is True
```

## Known Issues and Considerations

1. **Engagement Attribute**: The `Unit` class uses `engagement` (counter), not `engaged` (boolean). Tests must initialize `unit.engagement = 0`.

2. **Generator Returns**: Some functions like `get_neighbors()` return generators. Convert to list when testing: `list(map.get_neighbors(x, y))`.

3. **Random Terrain**: The `FOREST()` function returns terrain with random tree cover. Use seeds or test ranges rather than exact values.

4. **Movement Costs**: Unit mobility is depleted by movement. Reset with `unit.remaining_mobility = unit.mobility` between tests if needed.

## Future Improvements

1. **Increase Coverage**: Add tests for uncovered edge cases (currently 37% uncovered)
2. **Performance Tests**: Add benchmarks for pathfinding on large maps
3. **Property-Based Tests**: Use hypothesis for fuzz testing
4. **Mocking**: Mock external dependencies (combat, orders modules)
5. **Parameterized Tests**: Use `@pytest.mark.parametrize` for testing multiple scenarios

## Test Execution Summary

**Total Tests**: 48  
**Passing**: 48 ✅  
**Failing**: 0 ❌  
**Coverage**: 63%  
**Execution Time**: ~0.08 seconds  

## Contributing

When adding new functionality to `map.py`:

1. Write tests first (TDD approach)
2. Ensure existing tests still pass
3. Aim for >80% coverage on new code
4. Update this documentation with new test descriptions
5. Use descriptive test names following pattern: `test_<function>_<scenario>`

## Troubleshooting

### Import Errors
Ensure you're running from the project root:
```bash
cd C:\Users\aidan\Documents\TheLittleCorporal
python -m pytest test_map.py
```

### AttributeError on 'engagement'
Make sure to initialize the engagement attribute:
```python
unit.engagement = 0
```

### Generator Length Errors
Convert generators to lists before checking length:
```python
neighbors = list(map.get_neighbors(x, y))
assert len(neighbors) == 6
```
