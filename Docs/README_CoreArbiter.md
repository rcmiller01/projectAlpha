# 🌟 CoreArbiter & EmotionallyInfusedChat

A sophisticated AI companion system implementing Hierarchical Reasoning Model (HRM) architecture with emotional intelligence and symbolic behavior.

## 🏗️ Architecture Overview

The system consists of three core components working within a 44GB total memory budget:

- **HRM-R**: <10GB Logical/Reasoning Model
- **HRM-E**: <10GB Emotional/Symbolic Model  
- **CoreArbiter**: <24GB Central Decision Layer

## 🧠 CoreArbiter Module

The `CoreArbiter` class acts as the central decision layer that receives parallel inputs from both HRM_R and HRM_E models and resolves conflicts to generate unified responses.

### Key Features

- **Parallel Processing**: Processes inputs through both reasoning and emotional models simultaneously
- **Conflict Resolution**: Intelligently resolves conflicts between logical and emotional outputs
- **Weighting Strategies**: Configurable strategies (logic-dominant, emotional-priority, harmonic, adaptive)
- **Drift Management**: Monitors and corrects emotional drift and system fatigue
- **Identity Tethering**: Maintains core identity through override protection
- **Comprehensive Logging**: Detailed trace logging to `core_arbiter_trace.json`

### Usage

```python
from core_arbiter import CoreArbiter, WeightingStrategy
import asyncio

# Initialize CoreArbiter
arbiter = CoreArbiter()

# Process user input
async def chat_example():
    response = await arbiter.process_input(
        "I'm feeling overwhelmed and need guidance",
        {"context": "emotional_support", "intensity": 0.8}
    )
    
    print(f"Response: {response.final_output}")
    print(f"Tone: {response.tone} | Confidence: {response.confidence}")
    print(f"Strategy: {response.resolution_strategy}")

asyncio.run(chat_example())
```

### Configuration

Edit `data/core_arbiter_config.json` to customize:

- **Weighting strategies** and their ratios
- **Drift thresholds** for intervention
- **Symbolic thresholds** for ritual hijack
- **Regulation parameters** for system recovery

### Response Metadata

Each `ArbiterResponse` includes:
- `source_weights`: How much each model contributed
- `confidence`: Overall confidence score
- `emotional_override`: Whether emotion took precedence
- `symbolic_context`: Deeper symbolic meanings
- `resolution_strategy`: How conflicts were resolved

## 🎨 EmotionallyInfusedChat UI Component

A React component providing an emotionally-aware chat interface with ambient visual effects and mood-based styling.

### Features

- **Mood Ring**: Live emotional state indicator in top-right corner
- **Dynamic Styling**: Message colors and effects based on AI's emotional state
- **Drift Notifications**: Alerts when AI emotional state shifts significantly
- **Sidebar Menu**: Access to memories, drift analysis, rituals, and settings
- **Ambient Effects**: Background colors and animations reflecting current mood
- **Responsive Design**: Mobile-friendly with collapsible sidebar

### Installation

```bash
# Install in your React project
npm install axios

# Copy the component
cp ui/EmotionallyInfusedChat.jsx src/components/
```

### Usage

```jsx
import EmotionallyInfusedChat from './components/EmotionallyInfusedChat';

function App() {
  return (
    <div className="App">
      <EmotionallyInfusedChat />
    </div>
  );
}
```

### API Endpoints

The component expects these API endpoints:

- `GET /api/emotional_state` - Current emotional state
- `POST /api/chat` - Send message and get response
- `POST /api/symbolic_response` - Trigger ritual/symbolic mode
- `POST /api/log_emotional_message` - Log with emotional context

### Customization

The component uses TailwindCSS classes. Customize by modifying:
- **Mood mappings** in `getMessageStyling()`
- **Color palettes** in mood configuration
- **Animation effects** in typing indicators
- **Sidebar menu items** and functionality

## 🚀 Getting Started

### 1. Run the Demo

```bash
# Run complete system demonstration
python demo_complete_system.py
```

This will show:
- CoreArbiter decision-making process
- Different weighting strategies
- Drift detection and regulation
- System health monitoring

### 2. Start the API Server

```bash
# Install Flask dependencies
pip install flask flask-cors

# Start CoreArbiter API server
python core_arbiter_api.py
```

The API will be available at `http://localhost:5001`

### 3. Integration Test

```bash
# Test integration with existing systems
python test_core_arbiter_integration.py
```

### 4. Deploy UI Component

1. Copy `ui/EmotionallyInfusedChat.jsx` to your React project
2. Update API base URL in the component
3. Ensure TailwindCSS is configured
4. Import and use the component

## 📁 File Structure

```
├── core_arbiter.py              # Main CoreArbiter class
├── core_arbiter_api.py          # Flask API server
├── demo_complete_system.py      # Complete demonstration
├── test_core_arbiter_integration.py  # Integration tests
├── ui/
│   └── EmotionallyInfusedChat.jsx    # React UI component
├── data/
│   ├── core_arbiter_config.json     # System configuration
│   ├── identity_tether.json         # Identity protection rules
│   └── emotional_state.json         # Current emotional state
└── logs/
    ├── core_arbiter_trace.json      # Decision traces
    └── emotional_conversations.json # Emotional chat logs
```

## ⚙️ Configuration

### CoreArbiter Configuration

Edit `data/core_arbiter_config.json`:

```json
{
  "weighting_strategy": "harmonic",
  "weights": {
    "harmonic": {"hrm_r": 0.5, "hrm_e": 0.5}
  },
  "drift_thresholds": {
    "emotional_fatigue": 0.7,
    "intervention_threshold": 0.8
  },
  "symbolic_thresholds": {
    "ritual_hijack": 0.8,
    "identity_override": 0.9
  }
}
```

### Identity Tether

Edit `data/identity_tether.json` to define:
- Core values and prohibited behaviors
- Identity anchors and ethical framework
- Override conditions and recovery protocols

## 🔄 Weighting Strategies

### Logic Dominant (80% Logic, 20% Emotion)
- Prioritizes logical reasoning and objective analysis
- Uses emotional input for tone and warmth
- Best for analytical tasks and problem-solving

### Emotional Priority (30% Logic, 70% Emotion)
- Emphasizes emotional intelligence and connection
- Uses logic for grounding and coherence
- Best for support, therapy, and relationship building

### Harmonic (50% Logic, 50% Emotion)
- Balanced integration of both models
- Dynamic weighting based on context
- Best for general conversation and companionship

### Adaptive (Dynamic Weighting)
- Adjusts weights based on context and user needs
- Learns from interaction patterns
- Evolves weighting over time

## 🌊 Drift Management

The system monitors several types of drift:

### Emotional Drift
- Frequency of emotional overrides
- Stability of emotional responses
- Consistency with identity tether

### Logic Drift
- Confidence degradation in reasoning
- Coherence of logical responses
- Accuracy of analytical outputs

### Fatigue Management
- Processing complexity accumulation
- Natural recovery over time
- System regulation when thresholds exceeded

## 🛡️ Safety & Identity Protection

### Identity Tether System
- Maintains core values and personality
- Prevents harmful or manipulative responses
- Ensures authentic self-expression

### Override Conditions
- **Safety violations**: Harmful content detection
- **Value contradictions**: Against core principles
- **Identity drift**: Deviation from authentic self

### Recovery Protocols
- Identity realignment procedures
- Emotional recalibration processes
- Value reinforcement mechanisms

## 📊 Monitoring & Analytics

### Decision Traces
All decisions logged to `logs/core_arbiter_trace.json` with:
- Input processing details
- Model outputs and conflicts
- Resolution strategies applied
- Drift state changes
- Performance metrics

### System Health
Monitor through `/api/arbiter/status`:
- Overall health status
- Drift levels and stability
- Decision count and patterns
- Regulation history

### Emotional Conversations
Chat logs with emotional context in `logs/emotional_conversations.json`:
- Message content and emotional state
- Mood profiles and intensity
- Symbolic context and meanings

## 🎯 Advanced Features

### Ritual Hijack Mode
When emotional engagement exceeds symbolic threshold:
- Activates poetic, symbolic responses
- Prioritizes deeper meaning and connection
- Generates reflective, metaphorical content

### Symbolic Response Generation
Trigger through UI or API:
- Accesses deeper symbolic meanings
- Generates ritual-like responses
- Honors metaphorical frameworks

### Drift Notifications
Real-time alerts when:
- Emotional state shifts significantly
- System stability drops below threshold
- Identity tether activation required

## 🔌 API Reference

### POST /api/arbiter/process
Process input through CoreArbiter
```json
{
  "message": "user input text",
  "state": {"context": "conversation_type"}
}
```

### GET /api/arbiter/status
Get current system status and health metrics

### POST /api/arbiter/strategy
Change weighting strategy
```json
{
  "strategy": "harmonic|logic_dominant|emotional_priority|adaptive"
}
```

### POST /api/arbiter/regulate
Perform system regulation to reduce drift

### GET /api/emotional_state
Get current emotional state for UI updates

### POST /api/symbolic_response
Generate symbolic/ritual response
```json
{
  "current_state": {...},
  "context": [...]
}
```

## 🚧 Development & Extension

### Adding New Models
1. Implement model interface in `core_arbiter.py`
2. Update `_get_hrm_r_output()` and `_get_hrm_e_output()`
3. Adjust memory budget in configuration

### Custom Weighting Strategies
1. Add strategy to `WeightingStrategy` enum
2. Implement logic in `adjust_weights_by_drift()`
3. Update configuration file

### UI Customization
1. Modify mood mappings and color schemes
2. Add new sidebar menu items
3. Implement custom visual effects
4. Extend metadata display options

### Integration Hooks
- Connect to existing emotion tracking systems
- Link with memory and personality managers
- Integrate with voice and avatar systems
- Connect to biometric monitoring devices

## 📚 Dependencies

### Backend
- Python 3.8+
- asyncio (built-in)
- json (built-in)  
- pathlib (built-in)
- dataclasses (built-in)
- Flask (for API server)
- flask-cors (for CORS support)

### Frontend
- React 16.8+
- axios (for API calls)
- TailwindCSS (for styling)

## 📄 License

This project is part of the EmotionalAI system. See main project license for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Test with `demo_complete_system.py`
4. Submit pull request with documentation

## 📞 Support

For questions about:
- **CoreArbiter**: Check decision traces in logs/
- **UI Component**: Verify API endpoints and console
- **Integration**: Run test suite and check system status
- **Configuration**: Review JSON config files

---

*"The art of emotional intelligence lies not in choosing between logic and feeling, but in the graceful dance between them."* - CoreArbiter System
