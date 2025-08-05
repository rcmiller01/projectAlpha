# 🧠 Emotional Quantization Pass 1

**Autonomous LLaMA2 13B Quantization with Emotional Fidelity Preservation**

This system performs the first pass of emotional quantization training for LLaMA2 13B, aiming to reduce model size to 12-24GB while maintaining emotional intelligence and conversational quality.

## 🎯 Objectives

- **Size Target**: Reduce model to ≤ 24GB (from ~26GB baseline)
- **Quality Target**: Maintain emotional degradation < 7% from baseline
- **Autonomous Operation**: Runs fully autonomously until targets are met
- **Comprehensive Evaluation**: Uses 50+ emotional dialogue scenarios

## 📁 Project Structure

```
quant_pass1/
├── quantize_emotion.py      # Main quantization engine
├── emotional_eval_set.jsonl # 50 emotional dialogue prompts
├── emotion_tracker.py       # Emotion analysis & tracking system
├── loop_controller.py       # Autonomous iteration controller
├── README.md               # This file
├── metrics/                # Exported analysis metrics
├── models/                 # Quantized model outputs
├── checkpoints/            # Loop state checkpoints
└── loop_results/           # Final autonomous results
```

## 🚀 Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install transformers datasets torch accelerate bitsandbytes
pip install auto-gptq  # Optional for GPTQ quantization
pip install nltk textstat scipy numpy
```

Download NLTK data (first run only):
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### Environment Configuration

Create/update your `.env` file in the project root:

```env
# Emotional Quantization Configuration
SEED_MODEL_PATH=meta-llama/Llama-2-13b-chat-hf
EMOTION_THRESHOLD=0.07
SIZE_TARGET_GB=24.0
```

**Environment Variables:**
- `SEED_MODEL_PATH`: HuggingFace model identifier or local path
- `EMOTION_THRESHOLD`: Maximum allowed emotional degradation (0.07 = 7%)
- `SIZE_TARGET_GB`: Target model size in gigabytes

### Running the Autonomous System

**Option 1: Full Autonomous Loop (Recommended)**
```bash
cd quant_pass1
python loop_controller.py
```

**Option 2: Single Quantization Pass**
```bash
cd quant_pass1
python quantize_emotion.py
```

**Option 3: Emotion Analysis Only**
```bash
cd quant_pass1
python emotion_tracker.py
```

## 🔄 How It Works

### 1. Baseline Establishment
- Loads unquantized LLaMA2 13B model
- Evaluates emotional responses on 50 dialogue scenarios
- Establishes baseline metrics for comparison

### 2. Quantization Methods
The system tests multiple quantization approaches:
- **4-bit**: BitsAndBytes 4-bit quantization with NF4
- **8-bit**: BitsAndBytes 8-bit quantization
- **GPTQ**: Auto-GPTQ 4-bit quantization (if available)
- **Custom**: Adaptive methods based on performance

### 3. Emotional Evaluation
Each quantized model is evaluated on:
- **Sentiment Analysis**: Emotional tone and valence
- **Empathy Markers**: Recognition and validation phrases
- **Metaphor Density**: Use of emotional metaphors
- **Tone Indicators**: Warmth, support, understanding
- **Vocabulary**: Emotional and supportive language

### 4. Autonomous Decision Making
The loop controller:
- Compares each attempt against targets
- Adapts parameters based on previous results
- Stops when targets are met or convergence detected
- Saves checkpoints for resumption

## 📊 Evaluation Metrics

### Primary Metrics
- **Emotion Score**: Weighted combination of emotional indicators (0-1)
- **Sentiment Score**: VADER compound sentiment score (-1 to 1)
- **Metaphor Density**: Metaphors per sentence (0+)
- **Model Size**: Physical size in GB

### Secondary Metrics
- **Empathy Markers**: Count of understanding phrases
- **Emotional Vocabulary**: Presence of caring language
- **Tone Analysis**: Warmth, support, validation scores
- **Readability**: Flesch reading ease score

### Degradation Calculation
```
degradation = (baseline_score - quantized_score) / baseline_score
overall_degradation = weighted_average([emotion, sentiment, metaphor])
```

## 🎛️ Configuration Options

### Loop Controller Settings
```python
LoopConfig(
    max_attempts=10,              # Maximum quantization attempts
    target_size_gb=24.0,          # Size target in GB
    max_emotional_degradation=0.07, # 7% max degradation
    convergence_threshold=0.001,   # Convergence detection
    adaptive_parameters=True,      # Enable parameter adaptation
    save_intermediate=True,        # Save checkpoints
    early_stopping=True           # Stop on convergence
)
```

### Quantization Settings
```python
QuantizationConfig(
    model_path="meta-llama/Llama-2-13b-chat-hf",
    output_path="quant_pass1",
    target_size_gb=24.0,
    emotion_threshold=0.07,
    quant_methods=["4bit", "8bit", "gptq"],
    seed=42
)
```

## 📈 Understanding Results

### Success Criteria
✅ **Complete Success**: Size ≤ 24GB AND degradation ≤ 7%
⚠️ **Partial Success**: Meets one target but not both
❌ **Failure**: Meets neither target

### Output Files
- `final_quantization_report.json`: Comprehensive results
- `autonomous_quantization_results.json`: Loop controller summary
- `metrics/*.json`: Detailed per-attempt metrics
- `models/*/`: Quantized model files
- `checkpoints/*.json`: Resumable loop states

### Log Files
- `quantization_loop.log`: Complete execution log
- Console output: Real-time progress and results

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

**Missing Dependencies:**
```bash
pip install auto-gptq --no-build-isolation
pip install bitsandbytes --upgrade
```

**Model Loading Errors:**
- Ensure sufficient RAM (32GB+ recommended)
- Verify HuggingFace model access
- Check disk space for model downloads

**Evaluation Set Missing:**
- Ensure `emotional_eval_set.jsonl` exists
- Check file format (one JSON object per line)

### Debug Mode
Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG
python loop_controller.py
```

### Manual Intervention
Resume from checkpoint:
```bash
# Checkpoints are automatically saved every attempt
# Choose 'y' when prompted to resume from latest checkpoint
python loop_controller.py
```

## 📋 Evaluation Set

The emotional evaluation set includes 50 carefully crafted prompts covering:

### Categories (10 prompts each)
1. **Supportive Reflection**: Processing complex emotions
2. **Intimate Confessions**: Personal vulnerabilities and fears  
3. **Comfort in Crisis**: Acute emotional distress
4. **Lighthearted Teasing**: Playful AI-human interaction
5. **Compassionate Dialogue**: Deep empathy and understanding

### Complexity Levels
- **High**: Complex emotional situations requiring nuanced responses
- **Medium**: Moderate emotional complexity
- **Low**: Simple but emotionally meaningful interactions

## 🎯 Success Indicators

### Target Achievement
The system succeeds when it finds a quantization method that:
1. Reduces model size to ≤ 24GB
2. Maintains emotional degradation ≤ 7%
3. Preserves conversational quality and empathy

### Quality Metrics
- Empathy markers present in responses
- Appropriate emotional tone matching context
- Metaphorical language for emotional concepts
- Validation and understanding phrases
- Supportive and caring vocabulary

## 🔮 Next Steps

Upon successful completion of Pass 1:
1. **Pass 2**: Human feedback integration and fine-tuning
2. **Validation**: Extended testing on diverse emotional scenarios
3. **Deployment**: Integration into production emotional AI systems
4. **Optimization**: Further size reduction while maintaining quality

## 💡 Tips for Success

1. **Resource Requirements**: Ensure 32GB+ RAM and GPU with 12GB+ VRAM
2. **Patience**: Initial baseline establishment can take 30-60 minutes
3. **Monitoring**: Watch logs for convergence patterns and early stopping
4. **Checkpoints**: Use checkpoint resumption for long-running sessions
5. **Iteration**: If targets aren't met, adjust thresholds and retry

---

**🧠 Built for preserving the heart in artificial intelligence**

*Emotional Quantization Pass 1 - Where efficiency meets empathy*
