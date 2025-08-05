# Emotional Quantization Pass 2: Model Comparison & Selection

## Overview

Pass 2 of the emotional quantization system implements comprehensive model comparison and selection through AI and human evaluation. This phase takes the quantized models from Pass 1 and determines which provides the best balance of compression and emotional fidelity.

## Architecture

### Core Components

1. **Model Judge** (`judge_models.py`)
   - Pairwise model comparison using embedding similarity
   - Technical performance metrics calculation
   - Emotional preservation scoring

2. **Emotional Judge** (`emotional_judge.py`)
   - Ensemble AI judging with specialized personas
   - llama2-uncensored (empathy-biased) + mistral (analytical)
   - Consensus-based evaluation

3. **Human Preference Collector** (`human_preference_input.py`)
   - Interactive human feedback interface
   - Structured evaluation criteria
   - Weighted scoring system

4. **Core Replacer** (`replace_core.py`)
   - Safe model deployment with backup
   - Validation and integrity checking
   - Manifest updating

5. **Orchestrator** (`run_phase2.py`)
   - Complete workflow coordination
   - Multi-phase execution management
   - Comprehensive reporting

## Workflow

### Phase 1: Candidate Discovery
```
📍 Discover quantized models from Pass 1
📊 Extract technical specifications
🔍 Validate model integrity
```

### Phase 2: AI Judging
```
🤖 Pairwise model comparisons
💝 Emotional quality evaluation
📈 Technical performance metrics
🏆 Initial AI rankings
```

### Phase 3: Human Judging
```
👥 Interactive preference collection
📝 Structured evaluation criteria:
   - Believability (25%)
   - Connection (30%)
   - Expressive Strength (25%)
   - Appropriateness (20%)
💭 Qualitative feedback collection
```

### Phase 4: Final Rankings
```
⚖️ Weighted ensemble scoring:
   - AI Judge: 40%
   - Human Judge: 60%
🎯 Confidence analysis
🏆 Final model selection
```

### Phase 5: Model Replacement
```
💾 Create backup of current model
🔄 Deploy selected candidate
📝 Update companion manifest
✅ Validation and verification
```

## Usage

### Quick Start
```bash
# Run complete Pass 2 workflow
python run_phase2.py --config phase2_config.json

# Run specific phases
python run_phase2.py --phase discover
python run_phase2.py --phase ai-judge
python run_phase2.py --phase human-judge
python run_phase2.py --phase full
```

### Manual Operations
```bash
# Validate a candidate model
python replace_core.py validate ../quant_pass1/models/llama2_quantized_4bit

# Create backup of current model
python replace_core.py backup

# Replace with specific candidate
python replace_core.py replace ../quant_pass1/models/llama2_quantized_8bit

# Restore from backup
python replace_core.py restore backups/backup_20240101_120000
```

## Configuration

The `phase2_config.json` file controls all aspects of the Pass 2 workflow:

### Key Settings
- **models_to_compare**: List of quantized models to evaluate
- **judging_config**: AI vs human weighting (default: 40% AI, 60% human)
- **human_judging**: Evaluation criteria and weights
- **replacement_config**: Safety checks and deployment options

### Judging Weights
```json
{
  "judging_config": {
    "ai_judge_weight": 0.4,
    "human_judge_weight": 0.6,
    "consensus_threshold": 0.7
  }
}
```

### Human Evaluation Criteria
```json
{
  "human_judging": {
    "criteria_weights": {
      "believability": 0.25,
      "connection": 0.30,
      "expressive_strength": 0.25,
      "appropriateness": 0.20
    }
  }
}
```

## Evaluation Methodology

### AI Judging
1. **Embedding Similarity**: Uses sentence-transformers to compare response quality
2. **Emotional Scoring**: Analyzes empathy, creativity, and coherence
3. **Technical Metrics**: Model size, compression ratio, performance

### Human Judging
1. **Interactive Interface**: Presents candidate responses side-by-side
2. **Structured Criteria**: Four weighted evaluation dimensions
3. **Qualitative Feedback**: Open-ended comments and reasoning

### Final Scoring
```
Final Score = (AI Score × 0.4) + (Human Score × 0.6)
Confidence = Agreement between AI and human evaluations
```

## Output and Reporting

### Generated Files
- `results/pass2_complete_report_TIMESTAMP.json`: Comprehensive execution report
- `results/model_comparisons_TIMESTAMP.json`: Detailed comparison data
- `backups/backup_TIMESTAMP/`: Safety backup of replaced model
- `pass2_execution.log`: Detailed execution logs

### Report Structure
```json
{
  "pass2_execution": {
    "config": "Configuration used",
    "total_candidates": "Number of models compared",
    "phases_completed": "List of completed phases"
  },
  "candidate_discovery": "Available models and specifications",
  "ai_judging_results": "AI evaluation data and rankings",
  "human_judging_results": "Human feedback and preferences",
  "final_rankings": "Combined rankings and selection criteria",
  "model_replacement": "Deployment results and validation"
}
```

## Integration with Pass 1

Pass 2 seamlessly integrates with Pass 1 components:

- **Emotion Tracker**: Reuses emotional analysis from Pass 1
- **Evaluation Set**: Uses same prompts for consistency
- **Model Directory**: Automatically discovers Pass 1 outputs
- **Metadata**: Reads quantization information and performance metrics

## Safety and Validation

### Pre-Deployment Checks
1. **Model Integrity**: Validates file completeness and structure
2. **Size Requirements**: Ensures model meets minimum size thresholds
3. **Emotional Preservation**: Verifies acceptable degradation levels
4. **Backup Creation**: Automatic backup before replacement

### Error Recovery
- **Automatic Rollback**: Restores backup if deployment fails
- **Validation Failure**: Prevents deployment of corrupted models
- **User Override**: Manual controls for expert users

## Performance Considerations

### Resource Requirements
- **Memory**: 16GB+ recommended for model loading
- **Disk Space**: 3x model size for backup + candidate + active
- **Time**: 30-60 minutes for complete evaluation cycle

### Optimization Options
- **Parallel Evaluation**: Configure concurrent model assessments
- **Prompt Subset**: Use fewer evaluation prompts for speed
- **Caching**: Store embeddings and responses for reuse

## Troubleshooting

### Common Issues

**No Candidates Found**
```bash
# Check Pass 1 completed successfully
ls -la ../quant_pass1/models/
# Verify model names match configuration
```

**Human Judging Timeout**
```bash
# Extend timeout in configuration
"performance_config": {
  "response_timeout_seconds": 60
}
```

**Deployment Failure**
```bash
# Check backup was created
python replace_core.py list-backups
# Restore manually if needed
python replace_core.py restore [backup_path]
```

### Debug Mode
```bash
# Run with detailed logging
python run_phase2.py --config phase2_config.json --verbose

# Check execution logs
tail -f pass2_execution.log
```

## Extension Points

### Custom Judges
Implement additional AI judges by extending `EmotionalJudge`:
```python
class CustomJudge(EmotionalJudge):
    def __init__(self):
        super().__init__()
        self.add_judge("custom_persona", custom_config)
```

### Additional Metrics
Add new evaluation criteria in `ModelJudge`:
```python
def custom_metric(self, responses):
    # Your custom evaluation logic
    return metric_score
```

### Alternative Deployment
Extend `CoreModelReplacer` for different deployment targets:
```python
def deploy_to_custom_target(self, model_path):
    # Custom deployment logic
    pass
```

## Future Enhancements

- **Multi-Model Ensemble**: Support for ensemble model deployment
- **A/B Testing**: Gradual rollout with performance monitoring
- **Automated Retraining**: Trigger Pass 1 based on performance degradation
- **Cloud Integration**: Support for cloud-based model hosting
- **Real-time Monitoring**: Continuous emotional fidelity assessment

## API Reference

See individual component documentation:
- [`judge_models.py`](judge_models.py) - Model comparison and ranking
- [`emotional_judge.py`](emotional_judge.py) - AI ensemble evaluation
- [`human_preference_input.py`](human_preference_input.py) - Human feedback collection
- [`replace_core.py`](replace_core.py) - Model deployment and management
- [`run_phase2.py`](run_phase2.py) - Workflow orchestration

## License and Contributing

This is part of the emotional quantization research project. Contributions welcome for improving evaluation methodologies and deployment safety.
