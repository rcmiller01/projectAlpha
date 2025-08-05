# 🧭 Anchor Settings Integration

## Overview

The Anchor Settings system provides dynamic configuration of emotional scoring weights for the Unified AI Companion's quantization loop. This integration connects the frontend UI to the backend emotion evaluation system.

## Components

### 1. 🎨 Frontend UI (`core1-gateway/src/components/settings/AnchorSettingsPanel.jsx`)
- **Sliders**: Adjust weights for persona_continuity, expression_accuracy, response_depth, memory_alignment
- **Validation**: Ensures weights sum to 1.0 before saving
- **Signature Selection**: Choose between different anchor profiles
- **Lock Override**: Prevent accidental changes to anchor settings

### 2. 🔧 Backend API (`core1-gateway/server.js`)
- **GET `/api/anchor/settings`**: Retrieve current anchor configuration
- **POST `/api/anchor/settings`**: Update anchor configuration with validation
- **File Storage**: Persists settings to `config/anchor_settings.json`

### 3. 🧠 Core Integration (`emotion_loop_core.py`)
- **Dynamic Loading**: `load_anchor_weights()` function reads from config file
- **Real-time Updates**: EmotionLoopManager reloads weights for each evaluation
- **Weighted Scoring**: Uses anchor weights for candidate selection

## Configuration File

**Location**: `config/anchor_settings.json`

```json
{
  "weights": {
    "persona_continuity": 0.4,
    "expression_accuracy": 0.3,
    "response_depth": 0.2,
    "memory_alignment": 0.1
  },
  "signature": "Emberveil-01",
  "locked": false,
  "last_updated": "2025-07-31T12:00:00Z"
}
```

## Weight Definitions

| Weight | Purpose | Default | Description |
|--------|---------|---------|-------------|
| `persona_continuity` | 0.4 | Maintains consistent personality across interactions |
| `expression_accuracy` | 0.3 | Ensures emotional expressions match context |
| `response_depth` | 0.2 | Balances response complexity and engagement |
| `memory_alignment` | 0.1 | Aligns responses with conversation history |

## Testing

### Frontend Testing
1. Start the core1-gateway server: `cd core1-gateway && npm start`
2. Open browser to `http://localhost:3000`
3. Navigate to the Anchor Settings panel in the advanced section
4. Adjust sliders and verify real-time updates
5. Save settings and confirm persistence

### Backend Testing
```bash
# Test anchor weight loading
python test_anchor_integration.py

# Test API endpoints
curl -X GET http://localhost:5000/api/anchor/settings
curl -X POST http://localhost:5000/api/anchor/settings -H "Content-Type: application/json" -d '{"weights":{"persona_continuity":0.5,"expression_accuracy":0.2,"response_depth":0.2,"memory_alignment":0.1},"signature":"Custom-01","locked":false}'
```

### Core Integration Testing
```python
from emotion_loop_core import EmotionLoopManager, QuantizationCandidate

# Create manager with dynamic weights
manager = EmotionLoopManager()

# Test candidates
candidates = [
    QuantizationCandidate(name="model_q6", size_gb=12.5),
    QuantizationCandidate(name="model_q4", size_gb=8.8)
]

# Select best using current anchor weights
best = manager.select_best_candidate(candidates)
print(f"Selected: {best.name} with weights: {manager.anchor_weights}")
```

## Development Workflow

1. **Adjust UI**: Modify weights in the frontend Anchor Settings panel
2. **Save Configuration**: Click "Save Settings" to persist changes
3. **Test Impact**: Run emotion loop core to see how new weights affect candidate selection
4. **Iterate**: Fine-tune weights based on AI companion behavior

## Integration Status

- ✅ **UI Component**: AnchorSettingsPanel mounted and functional
- ✅ **API Endpoints**: GET/POST routes implemented and tested
- ✅ **Config Storage**: JSON file creation and persistence working
- ✅ **Core Loading**: Dynamic weight loading implemented
- ✅ **Real-time Updates**: EmotionLoopManager reloads weights per evaluation

## Next Steps

1. **Advanced Validation**: Add weight distribution validation beyond sum=1.0
2. **Profile Management**: Save/load different anchor weight profiles
3. **Live Monitoring**: Real-time display of how weights affect AI decisions
4. **A/B Testing**: Compare performance across different weight configurations
5. **Machine Learning**: Auto-optimize weights based on user satisfaction feedback

## Part 3: Emotional Loop Results Tracking

### Features Added
- **`save_loop_results()`**: Saves detailed evaluation results to timestamped JSON files
- **Results Archive**: Maintains `emotion_logs/loop_results.jsonl` for historical tracking
- **Comprehensive Metadata**: Includes anchor weights, candidate details, and evaluation summary

### Usage
```python
manager = EmotionLoopManager()
best = manager.select_best_candidate(candidates)
results_file = manager.save_loop_results(best, candidates)
```

### Output Structure
```json
{
  "timestamp": "20250731_143022",
  "anchor_weights": {"persona_continuity": 0.4, ...},
  "selected": {
    "name": "model_q6",
    "resonance": 0.734,
    "anchor_score": 0.800
  },
  "candidates": [...],
  "evaluation_summary": {
    "total_candidates": 3,
    "best_resonance": 0.734,
    "best_anchor": 0.800
  }
}
```

## Part 4: Emotional Test CLI

### Simulated Emotional Prompts
- **`run_emotional_test()`**: Tests individual candidates with emotional prompts
- **`run_emotional_test_suite()`**: Comprehensive testing across multiple prompts
- **Adaptive Scoring**: Simulated responses based on model characteristics

### Advanced CLI Tool
```bash
# Run comprehensive emotional testing
python emotional_test_cli.py

# Test specific candidates
python emotional_test_cli.py --candidates model_q6 model_q4

# Use custom prompts
python emotional_test_cli.py --prompts-file emotional_prompts.json

# Single prompt test
python emotional_test_cli.py --single-prompt "How do you handle loss?"

# Analysis only
python emotional_test_cli.py --analysis-only results.json
```

### Test Categories
- **Loss & Grief**: Complex emotional processing
- **Hope & Meaning**: Existential understanding  
- **Safety & Belonging**: Attachment and security
- **Relationships**: Interpersonal dynamics
- **Resilience**: Coping and adaptation

### Comparative Analysis
- Performance ranking across candidates
- Score distribution statistics  
- Best/worst performer identification
- Category-specific analysis
