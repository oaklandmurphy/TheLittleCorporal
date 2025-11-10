# Quick Start: Reconnaissance System

## What Changed?

Your General LLM now uses a **three-stage process** to generate orders:

```
Stage 1: Plan → Stage 2: Reconnoiter → Stage 3: Order
```

Instead of immediately generating orders, the General:
1. Creates a tactical plan
2. **Uses tools to gather intelligence** 
3. Issues orders based on that intelligence

## Running the System

No changes needed to run your game:

```bash
python main.py
```

## What You'll See

### Old Behavior (Before)
```
[French General's Orders]:
Brigade Friant: Move forward
Brigade Gudin: Support the attack
...
```

### New Behavior (After)
```
============================================================
[STAGE 1] General Auguste Marmont formulating tactical plan...
============================================================

[Tactical Plan]:
I will attack across the Po River to secure the eastern heights...

============================================================
[STAGE 2] General Auguste Marmont conducting reconnaissance...
============================================================

[Reconnaissance] reconnaissance_feature({'feature_name': 'Po River'})
Feature 'Po River':
  Terrain type: River
  Hexes: [(1, 0), (1, 1), (1, 2)]
  Units nearby: Brigade Friant (French) at (3,3)

[Reconnaissance] assess_enemy_strength({'location': 'Po River'})
Enemy strength assessment near Po River:
Detected 2 enemy unit(s):
  - Brigade Klenau at (5,3): Strength 8, Quality 3, Morale 7

============================================================
[STAGE 3] General Auguste Marmont issuing detailed orders...
============================================================

[Austrian General's Orders]:
Brigade Friant: Advance to Po River at (1,3) and engage enemy
Brigade Gudin: Support at (2,3)
...
```

## Key Features

### 1. Specific Coordinates
Orders now reference **actual hex coordinates** from reconnaissance:
- ✅ "Advance to Po River at (1,3)"
- ❌ "Move toward the river"

### 2. Intelligence-Based Reasoning
Orders show awareness of:
- Enemy positions and strength
- Terrain details
- Approach routes

### 3. Transparent Process
You can see:
- What the General plans to do
- What intelligence is gathered
- How orders are informed by recon

## Available Reconnaissance Tools

The General can use three tools:

| Tool | Purpose | Example |
|------|---------|---------|
| `reconnaissance_feature` | Get details about terrain features | "What's at Po River?" |
| `assess_enemy_strength` | Find enemy units near a location | "Who's defending the heights?" |
| `survey_approaches` | Analyze routes to a feature | "How do I get there?" |

The system **automatically forces** the General to use 2-4 tools before issuing orders.

## Tips for Better Results

### 1. Give Feature-Specific Orders
✅ "Attack the Po River crossing"
✅ "Secure San Marco Heights"
❌ "Advance"
❌ "Move forward"

### 2. Reference Terrain
The General will investigate features mentioned in your orders.

### 3. Trust the Process
The three-stage workflow takes longer but produces better orders.

## Troubleshooting

**Problem:** General not using tools
- **Solution:** Ensure using llama3.2+ model (tool-calling support)

**Problem:** Orders still too vague
- **Solution:** Give more specific player instructions referencing terrain features

**Problem:** Reconnaissance takes too long
- **Solution:** Reduce `max_tool_rounds` in `general.py` line 261

## Configuration

### Change minimum intelligence requirements
In `general.py`, line 283:
```python
if len(intelligence_gathered) >= 2:  # Require at least 2 recon reports
    break
```

### Change enemy detection range
In `general.py`, line 115:
```python
if dist <= 3:  # Detect enemies within 3 hexes
```

## Performance Impact

- **Before:** 1 LLM call per turn
- **After:** 3 LLM calls + tool executions per turn
- **Time increase:** ~2-3x longer per turn
- **Quality increase:** Significantly more tactical orders

## Example Order Comparison

### Before Reconnaissance System
```
Brigade Friant: Advance toward enemy
Brigade Gudin: Support the attack
Brigade Morand: Move to high ground
```

### After Reconnaissance System
```
Brigade Friant: Advance to Po River at (1,3) and prepare to cross
Brigade Gudin: Support crossing operation at (2,3) covering north approach
Brigade Morand: Secure Adda Hill at (4,2) to provide covering fire
```

Notice:
- ✅ Specific coordinates
- ✅ Clear tactical purpose
- ✅ Coordinated actions
- ✅ Based on actual terrain/enemy intel

## Next Steps

1. Run the game and observe the three-stage process
2. Compare order quality to previous runs
3. Experiment with different player instructions
4. Adjust configuration parameters if needed

See `RECONNAISSANCE_SYSTEM.md` for detailed technical documentation.
