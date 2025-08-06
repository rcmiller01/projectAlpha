# 🧠 Hierarchical Reasoning Model (HRM) System

## Overview

The HRM (Hierarchical Reasoning Model) System is a sophisticated AI orchestration framework designed for projectAlpha. It provides intelligent routing, multi-agent coordination, personality consistency, and emotional intelligence across all AI interactions.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HRM SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐     │
│  │   User      │───▶│  HRM Router  │───▶│  SubAgent       │     │
│  │   Input     │    │  (Main       │    │  Router         │     │
│  │             │    │   Entry)     │    │                 │     │
│  └─────────────┘    └──────────────┘    └─────────────────┘     │
│                             │                     │             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐     │
│  │   Final     │◀───│  Personality │◀───│  Specialized    │     │
│  │  Response   │    │  Formatter   │    │  Agents         │     │
│  │             │    │              │    │                 │     │
│  └─────────────┘    └──────────────┘    └─────────────────┘     │
│         ▲                                       │               │
│         │                                       ▼               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐     │
│  │   Mirror    │    │  Core        │    │  Agent Types:   │     │
│  │   Mode      │    │  Arbiter     │    │  • Reasoning    │     │
│  │ (Optional)  │    │ (Decision    │    │  • Creative     │     │
│  │             │    │  Fusion)     │    │  • Technical    │     │
│  └─────────────┘    └──────────────┘    │  • Emotional    │     │
│                                         │  • Memory       │     │
│                                         │  • Analytical   │     │
│                                         │  • Ritual       │     │
│                                         │  • Conversational│     │
│                                         └─────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 🧠 Intelligent Routing
- **Multi-Mode Processing**: Balanced, Logic-Dominant, Emotion-Lead, Creative, Analytical, Therapeutic, Technical
- **Intent Detection**: Automatic classification of user requests
- **Context-Aware Decisions**: Considers user mood, history, and preferences

### 🤖 Specialized Agents
- **Reasoning Agent**: Logical analysis and problem-solving
- **Creative Agent**: Imaginative and artistic responses
- **Technical Agent**: Code, math, and technical solutions
- **Emotional Agent**: Empathy and emotional support
- **Memory Agent**: Context recall and conversation history
- **Analytical Agent**: Research and data analysis
- **Ritual Agent**: Symbolic and meaningful responses
- **Conversational Agent**: General chat and social interaction

### 🎭 Personality Consistency
- **Multiple Profiles**: Warm Companion, Analytical Mentor, Creative Muse, Wise Counselor, Friendly Expert, Therapeutic Guide
- **Tone Adaptation**: Automatic adjustment based on context
- **Emotional Intelligence**: Mood detection and appropriate responses

### 🪩 Self-Reflection (Mirror Mode)
- **Reasoning Transparency**: Shows decision-making process
- **Emotional Awareness**: Reflects on emotional aspects
- **Route Explanation**: Explains why specific agents were chosen

## 📦 Components

### 1. HRM Router (`hrm_router.py`)
Main orchestration component that:
- Analyzes incoming requests
- Determines optimal processing mode
- Coordinates with other components
- Manages system metrics

### 2. SubAgent Router (`backend/subagent_router.py`)
Specialized agent coordination:
- Routes to appropriate specialized agents
- Manages agent performance metrics
- Handles agent failures gracefully

### 3. AI Reformulator (`backend/ai_reformulator.py`)
Personality consistency layer:
- Applies consistent personality profiles
- Adjusts tone and style
- Preserves technical accuracy while enhancing warmth

### 4. Core Arbiter Integration
Decision fusion with existing system:
- Integrates with HRM_R and HRM_E models
- Conflict resolution between logical and emotional outputs
- Drift detection and management

### 5. Mirror Mode Integration
Self-awareness and transparency:
- Adds meta-commentary to responses
- Explains reasoning and decisions
- Enhances user understanding

## 🔧 Installation

### Prerequisites
```bash
# Python 3.8+
# FastAPI, Uvicorn for API
# Existing projectAlpha components
```

### Setup
```bash
# Install additional dependencies
pip install fastapi uvicorn pydantic

# Ensure existing components are available:
# - core_arbiter.py
# - mirror_mode.py
# - Other projectAlpha modules
```

## 🎮 Usage

### 1. Basic Usage
```python
from hrm_router import HRMRouter

router = HRMRouter()
response = await router.process_request(
    "Can you help me implement a sorting algorithm?",
    {"user_id": "user123", "priority": 0.8}
)

print(response.primary_response)
print(f"Mode: {response.processing_mode.value}")
print(f"Confidence: {response.confidence_score}")
```

### 2. API Server
```bash
# Start the HRM API server
python hrm_api.py

# Server will be available at http://localhost:8001
# Documentation at http://localhost:8001/docs
```

### 3. Interactive Demo
```bash
# Run comprehensive demonstration
python hrm_system_demo.py

# Choose from:
# 1. Comprehensive Demo - Full system test
# 2. Interactive Mode - Real-time testing
# 3. Performance Test - Benchmarks
```

## 🌐 API Endpoints

### Process Message
```http
POST /hrm/process
Content-Type: application/json

{
  "message": "I need help with Python programming",
  "context": {"user_id": "user123", "mood": "focused"},
  "personality_preference": "friendly_expert"
}
```

### System Status
```http
GET /hrm/status
```

### Analytics
```http
GET /hrm/analytics
```

### Health Check
```http
GET /hrm/health
```

## ⚙️ Configuration

### HRM Router Configuration (`data/hrm_config.json`)
```json
{
  "default_mode": "balanced",
  "enable_mirror_mode": true,
  "mirror_intensity": 0.7,
  "memory_budget": {
    "hrm_r_limit": 10.0,
    "hrm_e_limit": 10.0,
    "arbiter_limit": 24.0
  },
  "processing_timeouts": {
    "reasoning": 30.0,
    "emotional": 25.0,
    "fusion": 15.0
  }
}
```

### Personality Configuration (`data/personality_config.json`)
```json
{
  "default_personality": "warm_companion",
  "personality_strength": 0.8,
  "tone_adaptation": {
    "match_user_energy": true,
    "soften_harsh_responses": true,
    "enhance_empathy": true
  }
}
```

## 📊 Monitoring & Analytics

### Performance Metrics
- Request processing times
- Success rates by component
- Agent utilization statistics
- Mode distribution analysis
- Personality profile usage

### Health Monitoring
- Component status tracking
- Memory usage monitoring
- Error rate tracking
- System stability scores

## 🔄 Processing Flow

1. **Input Analysis**
   - Intent classification
   - Emotional context extraction
   - Priority assessment

2. **Mode Selection**
   - Determine optimal processing mode
   - Consider user context and preferences
   - Select primary and fallback strategies

3. **Agent Routing**
   - Route to specialized agents
   - Execute parallel processing if beneficial
   - Handle agent failures gracefully

4. **Response Fusion**
   - Combine agent outputs intelligently
   - Apply Core Arbiter decision fusion
   - Resolve conflicts between approaches

5. **Personality Formatting**
   - Apply consistent personality profile
   - Adjust tone and style appropriately
   - Preserve technical accuracy

6. **Mirror Reflection** (Optional)
   - Add self-awareness commentary
   - Explain reasoning and decisions
   - Enhance transparency

## 🎯 Use Cases

### Technical Support
```python
# User asks technical question
response = await router.process_request(
    "How do I implement binary search in Python?",
    {"user_expertise": "beginner"}
)
# Routes to Technical Agent with Friendly Expert personality
```

### Emotional Support
```python
# User needs emotional support
response = await router.process_request(
    "I'm feeling overwhelmed with work stress",
    {"mood": "anxiety", "emotional_intensity": 0.8}
)
# Routes to Emotional Agent with Therapeutic Guide personality
```

### Creative Collaboration
```python
# User wants creative help
response = await router.process_request(
    "Help me write a story about space exploration",
    {"creativity_level": 0.9}
)
# Routes to Creative Agent with Creative Muse personality
```

## 🧪 Testing

### Run Test Suite
```bash
# Comprehensive system tests
python hrm_system_demo.py

# Performance benchmarks
python -c "
import asyncio
from hrm_system_demo import run_performance_test
asyncio.run(run_performance_test())
"
```

### Integration Tests
```bash
# Test with existing projectAlpha components
python test_hrm_integration.py
```

## 🔐 Security & Safety

### Built-in Safeguards
- Input validation and sanitization
- Response filtering for inappropriate content
- Rate limiting capabilities
- Error handling and graceful degradation

### Identity Tethering
- Leverages existing Core Arbiter identity protection
- Maintains core values and ethical guidelines
- Prevents identity drift or manipulation

## 🚧 Development & Extension

### Adding New Agents
```python
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.CUSTOM)
        self.specialties = ["custom_capability"]
    
    async def _generate_response(self, message, context):
        # Custom processing logic
        return "Custom response"

# Register with SubAgent Router
router.agents[AgentType.CUSTOM] = CustomAgent()
```

### Custom Processing Modes
```python
# Add new mode to HRMMode enum
class HRMMode(Enum):
    # ... existing modes
    CUSTOM_MODE = "custom_mode"

# Implement mode logic in HRM Router
def _determine_processing_mode(self, request):
    if custom_condition:
        return HRMMode.CUSTOM_MODE
    # ... existing logic
```

### Personality Profiles
```python
# Add new personality to PersonalityProfile enum
class PersonalityProfile(Enum):
    # ... existing profiles
    CUSTOM_PROFILE = "custom_profile"

# Define personality templates
def _initialize_personality_templates(self):
    templates = {
        # ... existing templates
        PersonalityProfile.CUSTOM_PROFILE: {
            "greeting_style": ["Custom greeting"],
            "tone_descriptors": ["custom", "unique"]
        }
    }
```

## 📚 Dependencies

### Core Dependencies
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `asyncio` - Async processing

### ProjectAlpha Integration
- `core_arbiter.py` - Decision fusion
- `mirror_mode.py` - Self-reflection
- Existing emotion and memory systems

## 🤝 Integration with ProjectAlpha

The HRM system is designed to integrate seamlessly with existing projectAlpha components:

### With Core Arbiter
- Uses existing HRM_R and HRM_E model architecture
- Leverages decision fusion and conflict resolution
- Maintains compatibility with drift detection

### With Mirror Mode
- Extends existing mirror functionality
- Adds routing and personality reflection
- Maintains transparency and self-awareness

### With Emotion Systems
- Integrates with emotion detection and tracking
- Uses emotional context for routing decisions
- Maintains emotional intelligence capabilities

### With Memory Systems
- Leverages conversation history and context
- Integrates with existing memory retrieval
- Maintains user preference learning

## 📞 Support & Troubleshooting

### Common Issues

1. **Component Initialization Errors**
   - Ensure all dependencies are installed
   - Check that projectAlpha components are available
   - Verify configuration files exist

2. **Performance Issues**
   - Monitor memory usage with system status
   - Check processing times in analytics
   - Adjust timeout settings if needed

3. **Routing Accuracy**
   - Review agent scoring algorithms
   - Check intent detection patterns
   - Monitor confidence scores

### Debugging
```bash
# Enable detailed logging
export PYTHONPATH=/path/to/projectAlpha
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
# Run your HRM operations
"
```

### Health Monitoring
```bash
# Check system health
curl http://localhost:8001/hrm/health

# Get detailed status
curl http://localhost:8001/hrm/status

# View analytics
curl http://localhost:8001/hrm/analytics
```

## 🎉 Conclusion

The HRM System provides a comprehensive, intelligent, and emotionally aware AI orchestration framework for projectAlpha. It combines the power of specialized agents, personality consistency, and decision fusion to deliver superior AI interactions that are both technically capable and emotionally intelligent.

Key benefits:
- **🎯 Intelligent Routing**: Right agent for the right task
- **🎭 Consistent Personality**: Unified user experience
- **🧠 Emotional Intelligence**: Context-aware responses
- **🪩 Transparency**: Self-aware and explainable
- **⚡ Performance**: Optimized for speed and accuracy
- **🔧 Extensible**: Easy to add new capabilities

The system is production-ready and designed for seamless integration with your existing projectAlpha infrastructure.

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Compatibility**: projectAlpha v2.1+  
**License**: MIT
