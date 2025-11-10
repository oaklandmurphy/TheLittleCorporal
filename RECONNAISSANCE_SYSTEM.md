# Reconnaissance-Driven Order Generation System

## Overview

The General LLM now uses a **three-stage intelligence-gathering workflow** to generate more informed, tactical orders. Instead of generating orders directly from a battlefield summary, the General is forced to use reconnaissance tools to gather detailed intelligence first.

## How It Works

### Stage 1: High-Level Planning
The General receives:
- Player instructions (e.g., "Attack across the river")
- Basic battlefield summary (from tactical report)

The General responds with:
- A brief 3-5 sentence tactical plan
- Identification of 1-3 key terrain features to investigate
- General tactical approach

**Example Output:**
```
"I intend to advance across the Po River and secure the high ground beyond. 
I will investigate the Po River crossing points and assess enemy strength 
near San Marco Heights. My approach will be to concentrate forces for a 
decisive thrust while maintaining flanking security."
```

### Stage 2: Forced Reconnaissance
The General **must** call 2-4 reconnaissance tools before proceeding. Available tools:

#### 1. `reconnaissance_feature`
Get detailed information about a specific terrain feature:
- Terrain type
- Exact hex coordinates
- Units present on the feature
- Units nearby

**Example Call:**
```python
reconnaissance_feature(feature_name="Po River")
```

**Example Response:**
```
Feature 'Po River':
  Terrain type: River
  Hexes: [(1, 0), (1, 1), (1, 2), (2, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8)]
  Units present: None
  Units nearby: Brigade Friant (French) at (3,3), Brigade Klenau (Austrian) at (5,3)
```

#### 2. `assess_enemy_strength`
Analyze enemy forces within 3 hexes of a location:
- Enemy unit names and positions
- Strength, quality, and morale statistics
- Distance from the target location

**Example Call:**
```python
assess_enemy_strength(location="San Marco Heights")
```

**Example Response:**
```
Enemy strength assessment near San Marco Heights:
Detected 2 enemy unit(s):
  - Brigade Hohenlohe at (9,3): Strength 6, Quality 4, Morale 8, 2 hexes away
  - Brigade Lichtenstein at (9,5): Strength 6, Quality 4, Morale 7, 3 hexes away
```

#### 3. `survey_approaches`
Identify approach routes and tactical considerations:
- Adjacent terrain types
- Enemy units blocking approaches
- Tactical assessment of accessibility

**Example Call:**
```python
survey_approaches(target_feature="Verde Forest")
```

**Example Response:**
```
Approach survey for Verde Forest:
Feature location: [(7, 2), (8, 2), (9, 2), (9, 3), (10, 3)]
Adjacent terrain:
  - 8 Fields hex(es)
  - 2 Hill hex(es)
Enemy units blocking approaches:
  - Brigade Hohenlohe at (9,3)
```

### Stage 3: Generate Specific Orders
Using the intelligence gathered, the General writes detailed orders:
- One order per unit under command
- References specific coordinates or feature names from intelligence
- Tactical reasoning based on reconnaissance data

**Example Output:**
```
Brigade Friant: Advance to Po River at (1,3) and prepare to cross
Brigade Gudin: Support Friant's advance, move to (2,4)
Brigade Morand: Flank left through Verde Forest to (7,2)
Brigade Petit: Hold position at (2,4) as reserve
Brigade Desvaux: Screen right flank, advance to (4,5)
```

## Benefits

### 1. More Informed Decision-Making
The General now has access to:
- Specific unit positions and capabilities
- Detailed terrain information
- Enemy strength assessments

### 2. Better Order Quality
Orders are now:
- **Grounded in specific intelligence** rather than vague summaries
- **Reference actual coordinates** from reconnaissance
- **Tactically sound** based on enemy dispositions

### 3. Consistent Reconnaissance Process
The system **forces** the General to:
- Identify objectives before acting
- Gather intelligence systematically
- Base orders on concrete data

### 4. Transparency
You can see the General's thought process:
- Stage 1 shows the tactical plan
- Stage 2 shows what intelligence is gathered
- Stage 3 shows how intelligence influences orders

## Implementation Details

### Modified Files

#### `general.py`
- Added `game_map` parameter to `__init__`
- Added `reconnaissance_tools` property with 3 tool definitions
- Added `_execute_reconnaissance_tool()` method
- Replaced `_query_general()` with `_query_general_with_tools()` for multi-stage workflow
- Modified `get_instructions()` to use new workflow

#### `main.py`
- Updated General initialization to pass `game_map` parameter

### Workflow Control

The reconnaissance stage will:
- Execute up to 5 rounds of tool calls
- Stop after 2-4 successful intelligence reports
- Handle tool errors gracefully
- Fall back to basic orders if no map is available

## Configuration

### Adjusting Intelligence Requirements

You can modify the minimum intelligence threshold in `_query_general_with_tools()`:

```python
# Stop if we have enough intelligence (2-4 tool calls)
if len(intelligence_gathered) >= 2:  # Change this threshold
    break
```

### Adjusting Tool Behavior

Modify reconnaissance ranges in `_execute_reconnaissance_tool()`:

```python
# Find enemy units within 3 hexes of the feature
if dist <= 3:  # Change search radius
```

## Example Full Turn

```
============================================================
[STAGE 1] General Auguste Marmont formulating tactical plan...
============================================================

[Tactical Plan]:
I intend to advance methodically across the Po River to secure key terrain 
on the eastern bank. I will investigate the Po River crossing points and 
assess enemy strength near the Adda Hill and Grande Ridge positions.

============================================================
[STAGE 2] General Auguste Marmont conducting reconnaissance...
============================================================

[Reconnaissance] reconnaissance_feature({'feature_name': 'Po River'})
Feature 'Po River':
  Terrain type: River
  Hexes: [(1, 0), (1, 1), (1, 2), (2, 3), (1, 4), (1, 5)]
  Units present: None
  Units nearby: Brigade Friant (French) at (3,3)

[Reconnaissance] assess_enemy_strength({'location': 'Adda Hill'})
Enemy strength assessment near Adda Hill:
Detected 3 enemy unit(s):
  - Brigade Klenau at (5,3): Strength 8, Quality 3, Morale 7, 1 hexes away
  - Brigade Vincent at (8,5): Strength 4, Quality 4, Morale 6, 3 hexes away

============================================================
[STAGE 3] General Auguste Marmont issuing detailed orders...
============================================================

[Austrian General's Orders]:
Brigade Friant: Advance to Po River crossing at (2,3) and prepare to engage
Brigade Gudin: Support crossing, move to (3,2)
Brigade Morand: Flank through Adda Hill approach at (4,3)
Brigade Petit: Hold reserve position at (2,4)
Brigade Desvaux: Screen southern approach at (4,5)
```

## Future Enhancements

Potential improvements:
1. **Adaptive reconnaissance** - Vary tool calls based on player instructions
2. **Historical intelligence** - Remember reconnaissance from previous turns
3. **Collaborative intelligence** - Share reconnaissance between friendly generals
4. **Priority targeting** - Automatically focus on high-value objectives
5. **Counter-intelligence** - Detect and respond to enemy reconnaissance

## Troubleshooting

### General not calling tools
- Ensure model supports tool calling (llama3.2+ recommended)
- Check that `game_map` is passed to General initialization
- Verify ollama host is accessible

### Too many/too few tool calls
- Adjust `max_tool_rounds` in `_query_general_with_tools()`
- Modify intelligence threshold check

### Tool execution errors
- Check that terrain features are properly labeled on map
- Verify feature names match those in map labels
- Ensure map has correct dimensions

## Testing

To test the system:
1. Start a game with `python main.py`
2. Observe the three-stage output in the console
3. Verify orders reference specific coordinates from reconnaissance
4. Compare order quality to previous implementation

The new system should produce more specific, contextually-aware orders that demonstrate tactical reasoning.
