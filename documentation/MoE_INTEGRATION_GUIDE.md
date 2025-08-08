# MoE Integration Guide for ProjectAlpha

## Overview
This guide shows how to integrate the Mixture-of-Experts (MoE) system with existing ProjectAlpha components like CoreConductor and HRM Router.

## Quick Start

### 1. Initialize MoE System
```python
from src.core.init_models import load_conductor_models

# Load conductor suite with MoE integration
models = load_conductor_models(use_moe=True)
moe_adapter = models.get("moe_loader")
```

### 2. Basic Usage
```python
# Direct MoE usage
response = moe_adapter.generate("Debug this Python code")
print(response)  # Routes to logic_code expert

# Emotional query
response = moe_adapter.generate("I feel overwhelmed")
print(response)  # Routes to emote_valence expert
```

### 3. Integration with CoreConductor
```python
# In core_conductor.py - add MoE routing
class CoreConductor:
    def __init__(self):
        self.models = load_conductor_models(use_moe=True)
        self.moe_system = self.models.get("moe_loader")

    def process_query(self, query: str, context: dict = None) -> str:
        # Use MoE for intelligent routing
        if self.moe_system:
            return self.moe_system.generate(query, context)
        else:
            # Fallback to standard models
            return self.models["conductor"].generate(query)
```

## Advanced Integration

### 1. HRM Router Enhancement
```python
# In hrm_router.py - replace direct SLiM calls
class HRMRouter:
    def __init__(self):
        self.models = load_conductor_models(use_moe=True)
        self.moe_system = self.models.get("moe_loader")

    def route_to_slim(self, agent_type: str, query: str) -> str:
        # Instead of direct SLiM model loading:
        # Old: slim_model = load_slim_model(agent_type)

        # New: Use MoE routing based on query content
        if self.moe_system:
            context = {"agent_type": agent_type}
            return self.moe_system.generate(query, context)
        else:
            # Fallback to legacy routing
            return self.legacy_route(agent_type, query)
```

### 2. Memory System Integration
```python
# In memory_system.py - use MoE for memory recall
class MemorySystem:
    def __init__(self):
        self.models = load_conductor_models(use_moe=True)
        self.moe_system = self.models.get("moe_loader")

    def recall_memory(self, query: str) -> str:
        # Route memory queries to specialized expert
        memory_query = f"Recall: {query}"
        if self.moe_system:
            return self.moe_system.generate(memory_query)
        else:
            return self.default_memory_search(query)
```

### 3. Ritual System Integration
```python
# In ritual system - use symbolic expert
class RitualSystem:
    def __init__(self):
        self.models = load_conductor_models(use_moe=True)
        self.moe_system = self.models.get("moe_loader")

    def process_ritual(self, ritual_description: str) -> str:
        ritual_query = f"Design ritual: {ritual_description}"
        if self.moe_system:
            return self.moe_system.generate(ritual_query)
        else:
            return self.default_ritual_processing(ritual_description)
```

## Configuration Management

### 1. Environment Setup
```bash
# Copy MoE configuration
cp .env.moe .env

# Adjust RAM limits for your system
echo "MOE_MAX_RAM_GB=4.0" >> .env  # For 8GB systems
echo "MOE_MAX_RAM_GB=16.0" >> .env # For 32GB+ systems
```

### 2. Mobile Deployment
```python
# For mobile/constrained environments
import os
os.environ["MOE_MOBILE_MODE"] = "true"
os.environ["MOE_MOBILE_MAX_RAM_GB"] = "2.0"

models = load_conductor_models(use_moe=True)
```

### 3. Development Testing
```python
# Use mock models for development
import os
os.environ["MOE_USE_MOCK_MODELS"] = "true"
os.environ["MOE_DEBUG"] = "true"

models = load_conductor_models(use_moe=True)
```

## Performance Monitoring

### 1. Real-time Stats
```python
moe_system = models["moe_loader"].moe_loader
stats = moe_system.get_performance_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"RAM usage: {stats['ram_usage']}")
```

### 2. Expert Usage Tracking
```python
# See which experts are currently loaded
loaded = moe_system.get_loaded_experts()
print(f"Loaded experts: {loaded}")

# Memory usage breakdown
memory = moe_system.get_memory_usage()
for expert, ram_gb in memory.items():
    print(f"{expert}: {ram_gb}GB")
```

## Migration Strategy

### Phase 1: Parallel Operation
- Keep existing model loading alongside MoE
- Route specific query types to MoE
- Compare responses for quality

### Phase 2: Gradual Replacement
- Replace SLiM direct calls with MoE routing
- Update HRM Router to use MoE
- Maintain fallback to legacy models

### Phase 3: Full Integration
- Remove legacy model loading
- Use MoE as primary routing system
- Optimize expert registry for production

## Troubleshooting

### Common Issues
1. **RAM Limits**: Adjust MOE_MAX_RAM_GB based on system
2. **Model Paths**: Ensure expert model files exist
3. **Backend Availability**: Falls back to mock models if no backends available

### Debug Commands
```python
# Test MoE system
python demo_moe_system.py

# Check model loading
from src.core.init_models import initialize_moe_system
moe = initialize_moe_system(max_ram_gb=4.0)
print(f"MoE ready: {moe is not None}")
```

## Next Steps
1. Integrate with CoreConductor
2. Update HRM Router
3. Add real SLiM model support
4. Create web interface for expert management
5. Implement performance benchmarking
