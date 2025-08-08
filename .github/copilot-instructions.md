# ProjectAlpha AI Agent Instructions

## Project Overview
ProjectAlpha is a sophisticated AI orchestration platform featuring hierarchical reasoning models (HRM), specialized SLiM agents, voice cadence modulation, and emotional AI. The system intelligently routes between local and cloud AI models with a React webapp frontend and Python backend.

## Core Architecture

### 🧠 Multi-Agent System Stack
```
React Frontend (webapp/frontend) → Node.js Proxy → Python Backend (src/)
├── HRM Router (src/core/hrm_router.py) - Main orchestration hub
├── CoreConductor (src/core/core_conductor.py) - Model management
├── CoreArbiter (src/core/core_arbiter.py) - Decision fusion layer
└── SLiM Agents (src/agents/) - Specialized cognitive modules
```

### 🔧 Key Integration Points
- **HRM System**: Routes requests through 7 processing modes (Balanced, Logic-Dominant, Emotion-Lead, etc.)
- **SLiM Models**: 8 specialized agents (logic_high, emotion_creative, etc.) with hemispheric processing
- **MoE System**: Mixture-of-Experts loader for dynamic RAM-constrained model management
- **Voice Cadence**: 603-line modulation system integrated with React hooks
- **GraphRAG Memory**: Semantic entity linking for conversation context
- **Emotional State**: Real-time mood tracking with drift detection

## Development Patterns

### 🏗️ Model Loading Convention
```python
# Auto-detection pattern used throughout
def load_models():
    slim_vars = ["LOGIC_HIGH_MODEL", "EMOTION_VALENCE_MODEL", ...]
    has_slim = any(os.getenv(var) for var in slim_vars)
    return load_all_models() if has_slim else load_conductor_models()
```

### 🎛️ HRM Router Integration
```python
# Standard integration pattern for all agents
class Agent(SLiMAgent):
    def process(self, input_text):
        # Pre-processing: GraphRAG memory context
        context = self.hrm_router.process_agent_input(input_text, self.agent_type)

        # Core processing with CoreConductor
        response = self.conductor.generate(self.role, enhanced_prompt)

        # Post-processing: Memory storage + tool suggestions
        self.hrm_router.process_agent_output(response, context, self.agent_type)
        return response
```

### 🎨 React Voice Integration
```javascript
// Voice hook pattern used in all agent interactions
const { generateVoiceParams, voiceSettings } = useVoiceCadence();

// Generate voice parameters for agent responses
const voiceParams = generateVoiceParams({
  agentType: 'deduction',
  emotion: 'contemplative',
  urgency: 'medium'
}, responseText);
```

## Essential Commands

### 🚀 Development Workflow
```bash
# Full system startup
cd webapp && npm run dev    # React frontend + Node proxy
python src/core/core_conductor.py  # Python backend

# Model verification
python src/core/core_conductor.py --list-models

# System demonstrations
python demo_moe_system.py          # MoE expert routing
python demos/hrm_system_demo.py    # HRM pipeline
python demo_complete_system.py     # Full integration
```

### 🔍 Testing & Debugging
```bash
# Component testing
python -m pytest tests/            # Backend tests
cd webapp/frontend && npm test     # React tests

# System health checks
python quick_verification.py       # Model connectivity
python src/core/init_models.py     # Model loading test
```

## Project-Specific Conventions

### 📁 File Organization
- `src/core/` - Core orchestration (HRM, Conductor, Arbiter)
- `src/agents/` - Specialized SLiM agent implementations
- `src/api/` - FastAPI endpoints and routing
- `webapp/frontend/src/` - React app (pages, components, hooks, store)
- `memory/` - GraphRAG semantic memory system
- `Docs/` - Architecture documentation and integration guides

### 🔧 Configuration Strategy
Environment variables drive model selection:
- **Standard Config**: 4 models (conductor, logic, emotion, creative)
- **SLiM Config**: 12 models (conductor + 8 SLiMs + legacy compatibility)
- **MoE Config**: 12 expert domains with RAM constraints

### 🎯 Agent Specialization Mapping
```python
# Left Brain (Logic Hemisphere)
logic_high → phi4-mini-reasoning:3.8b    # Advanced reasoning
logic_code → qwen2.5-coder:3b           # Code generation
logic_proof → deepseek-r1:1.5b          # Mathematical proofs

# Right Brain (Emotion/Creativity Hemisphere)
emotion_valence → gemma3:1b              # Emotional analysis
creative_metaphor → Custom model        # Metaphorical expression
```

## Critical Integration Notes

### ⚡ HRM Router as Central Hub
All agent interactions must flow through `HRMRouter` for memory hooks and tool routing. Direct model calls bypass the memory system.

### 🧠 Memory System Threading
GraphRAG operations are thread-safe but expensive. Use `process_agent_input()` for context retrieval and `process_agent_output()` for fact storage.

### 🎵 Voice Cadence Integration
VoiceCadenceModulator requires initialization in React components. Use `useVoiceCadence` hook for all agent voice parameter generation.

### 📱 Webapp Architecture
Two-column layout with sidebar navigation. All agent interactions use Material-UI + Framer Motion. Zustand for state management with localStorage persistence.

## Common Pitfalls

- **Circular imports**: Import models through `init_models.py`, not directly
- **Missing HRM integration**: Always use HRMRouter for agent orchestration
- **Voice initialization**: Check `isInitialized` before generating voice parameters
- **SLiM detection**: Use environment variable detection pattern for model selection
- **Memory context**: Include GraphRAG context in all agent processing

## Key Documentation References
- `Docs/README_HRM_SYSTEM.md` - HRM architecture deep dive
- `MoE_INTEGRATION_GUIDE.md` - Mixture-of-Experts implementation
- `VOICE_INTEGRATION_COMPLETE.md` - Voice system integration
- `SLiM_AGENT_IMPLEMENTATION_SUMMARY.md` - Specialized agent details
