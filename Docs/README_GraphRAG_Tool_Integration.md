# GraphRAG Memory + Tool Router Integration

## Overview

This system implements a modular GraphRAG memory and tool router architecture for multi-agent AI systems. It enhances reasoning capabilities by integrating semantic entity linking through a memory graph and enables autonomous tool usage via a lightweight router.

## Key Features

- **GraphRAG Memory System**: Semantic entity linking using NetworkX-based directed graphs
- **Tool Request Router**: Modular tool registration and execution with thread-safe operations
- **HRM Integration**: Seamless compatibility with existing High-Resolution Memory stack
- **Enhanced Conductor**: Strategic reasoning with memory and tool integration
- **Thread-Safe**: Concurrent operations with proper synchronization
- **Extensible**: Ready for future SLiM agent integration

## Architecture

```
projectAlpha/
├── memory/
│   └── graphrag_memory.py          # GraphRAG memory system
├── src/
│   ├── tools/
│   │   ├── __init__.py
│   │   └── tool_request_router.py  # Tool routing system
│   └── core/
│       ├── hrm_router.py           # HRM integration layer
│       └── core_conductor.py       # Enhanced conductor
├── examples/
│   └── graphrag_tool_integration_demo.py  # Complete demo
├── data/                           # Memory persistence
├── logs/                           # Tool request logs
└── requirements_graphrag.txt       # Additional dependencies
```

## Components

### 1. GraphRAG Memory System (`memory/graphrag_memory.py`)

Thread-safe memory graph with semantic entity linking:

```python
from memory.graphrag_memory import GraphRAGMemory

# Initialize memory
memory = GraphRAGMemory("data/my_memory.json")

# Add semantic facts
memory.add_fact("user", "prefers", "chocolate", confidence=0.9, source="conversation")

# Query related concepts
result = memory.query_related("user", depth=2)
print(f"Found {len(result.related_concepts)} related concepts")

# Save/load persistence
memory.save_memory()
```

**Key Methods:**
- `add_fact(subject, relation, object, confidence, source)` - Add semantic relationships
- `query_related(node, depth, min_confidence)` - Query connected concepts
- `save_memory()` / `load_memory()` - Persistence operations
- `get_memory_stats()` - Memory graph statistics

### 2. Tool Request Router (`src/tools/tool_request_router.py`)

Thread-safe tool registration and execution:

```python
from src.tools.tool_request_router import ToolRequestRouter

# Initialize router
router = ToolRequestRouter("logs/tool_requests.jsonl")

# Register a tool
def my_tool(query: str, **kwargs) -> dict:
    return {"result": f"Processed: {query}", "trace": kwargs.get('trace_id')}

router.register_tool("my_tool", my_tool)

# Execute tool
response = router.route_request("my_tool", {"query": "test"})
print(f"Success: {response.success}, Result: {response.result}")
```

**Key Methods:**
- `register_tool(name, handler)` - Register callable tools
- `route_request(intent, parameters)` - Execute tools by name
- `list_tools()` - Get available tools and metadata
- `get_stats()` - Router statistics

### 3. HRM Router Integration (`src/core/hrm_router.py`)

Integration layer between GraphRAG memory, tool router, and HRM stack:

```python
from src.core.hrm_router import HRMRouter

# Initialize with both memory and tools
hrm = HRMRouter(
    memory_file="data/hrm_memory.json",
    tool_log_file="logs/hrm_tools.jsonl"
)

# Process agent input with memory context
result = hrm.process_agent_input(
    "User wants personalized recommendations", 
    agent_type="conductor"
)

# Execute tools
tool_response = hrm.execute_tool("query_memory", {"concept": "user", "depth": 2})
```

**Key Methods:**
- `process_agent_input(text, agent_type, context)` - Input processing with memory hooks
- `process_agent_output(text, input_context, agent_type)` - Output processing with fact extraction
- `execute_tool(tool_name, parameters, agent_type)` - Tool execution for agents
- `get_integration_stats()` - System statistics

### 4. Enhanced Core Conductor (`src/core/core_conductor.py`)

Strategic reasoning enhanced with memory and tool capabilities:

```python
from src.core.core_conductor import CoreConductor

# Initialize conductor
conductor = CoreConductor(
    memory_file="data/conductor_memory.json",
    tool_log_file="logs/conductor_tools.jsonl"
)

# Set objectives
conductor.set_objectives([
    "Improve user engagement",
    "Enhance personalization"
])

# Make strategic decision with full context
decision = conductor.make_strategic_decision(
    situation="User asking complex questions",
    objectives=["Provide thoughtful responses"],
    constraints=["Maintain accuracy"]
)

print(f"Decision confidence: {decision.confidence}")
print(f"Action plan: {decision.action_plan}")
```

**Key Methods:**
- `make_strategic_decision(situation, objectives, constraints)` - Strategic reasoning
- `set_objectives(objectives)` - Set strategic goals
- `get_status()` - Conductor status and statistics
- `save_state()` - Persist conductor state

## Usage Patterns

### Basic Memory Operations

```python
# Initialize memory system
memory = GraphRAGMemory("data/user_memory.json")

# Build user model
memory.add_fact("user", "likes", "science_fiction", confidence=0.8, source="conversation")
memory.add_fact("user", "works_in", "technology", confidence=0.9, source="profile")
memory.add_fact("science_fiction", "relates_to", "technology", confidence=0.6, source="association")

# Query for recommendations
related = memory.query_related("user", depth=3)
for concept in related.related_concepts:
    if concept['relation_type'] == 'likes':
        print(f"User likes: {concept['concept']}")
```

### Tool Development

```python
from src.tools.tool_request_router import tool_handler, requires_params

@tool_handler("search_web", "Search the web for information")
@requires_params("query")
def web_search_tool(query: str, max_results: int = 5, **kwargs) -> dict:
    # Implement web search logic
    return {
        "results": [{"title": "Result", "url": "https://example.com"}],
        "trace": kwargs.get('trace_id')
    }

# Register with router
router.register_tool("search_web", web_search_tool)
```

### Agent Integration

```python
# Initialize HRM Router
hrm = HRMRouter()

# Process input from any agent type
context = hrm.process_agent_input(
    "User needs help with recommendations",
    agent_type="supervisor",
    context={"user_id": "12345", "session": "active"}
)

# Get memory-enhanced context
memory_concepts = context['memory_related']
suggested_tools = context['suggested_tools']

# Execute suggested tools
for tool_name in suggested_tools:
    response = hrm.execute_tool(tool_name, {"concept": "user_preferences"})
    if response.success:
        print(f"Tool {tool_name} result: {response.result}")
```

### Conductor Strategic Planning

```python
# Enhanced conductor with memory and tools
conductor = CoreConductor()

# Set strategic context
conductor.set_objectives([
    "Improve response quality",
    "Increase user satisfaction",
    "Optimize resource usage"
])

# Strategic decision with full context
decision = conductor.make_strategic_decision(
    situation="Users reporting inconsistent response quality",
    constraints=["Limited computational resources", "Must maintain speed"]
)

# Execute recommended actions
for action in decision.action_plan:
    print(f"Recommended action: {action}")

# Use recommended tools
for tool in decision.tool_recommendations:
    result = conductor.hrm_router.execute_tool(tool, {})
    print(f"Tool {tool} executed: {result.success}")
```

## Threading and Concurrency

All components are thread-safe:

```python
import threading
from concurrent.futures import ThreadPoolExecutor

def agent_task(agent_id, hrm_router):
    result = hrm_router.process_agent_input(
        f"Request from agent {agent_id}",
        agent_type="worker"
    )
    return result

# Concurrent agent operations
hrm = HRMRouter()
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(agent_task, i, hrm) for i in range(10)]
    results = [f.result() for f in futures]
```

## Integration with Existing HRM Stack

The system is designed for seamless integration:

```python
# Your existing HRM components
from your_existing_hrm import ExistingHRMCore

# New GraphRAG integration
from src.core.hrm_router import HRMRouter

class EnhancedHRMCore(ExistingHRMCore):
    def __init__(self):
        super().__init__()
        self.graphrag_router = HRMRouter()
    
    def process_input(self, input_text, context):
        # Use existing HRM processing
        base_result = super().process_input(input_text, context)
        
        # Enhance with GraphRAG memory
        enhanced_context = self.graphrag_router.process_agent_input(
            input_text, 
            agent_type="hrm_core",
            context=context
        )
        
        # Combine results
        base_result['memory_enhanced'] = enhanced_context
        return base_result
```

## Installation and Setup

1. **Install dependencies:**
```bash
pip install -r requirements_graphrag.txt
```

2. **Run the demo:**
```bash
python examples/graphrag_tool_integration_demo.py
```

3. **Initialize in your code:**
```python
from src.core.hrm_router import HRMRouter

# Basic setup
hrm = HRMRouter()

# Custom configuration
hrm = HRMRouter(
    memory_file="data/custom_memory.json",
    tool_log_file="logs/custom_tools.jsonl",
    enable_memory_hooks=True,
    enable_tool_routing=True
)
```

## Future Extensions

### SLiM Agent Integration

The system is designed for future SLiM agent integration:

```python
# Future SLiM agent integration pattern
class SLiMAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.hrm_router = HRMRouter()
    
    def autonomous_reasoning(self, input_text):
        # Get memory context
        context = self.hrm_router.process_agent_input(
            input_text, 
            agent_type="slim_agent"
        )
        
        # Use tools autonomously
        for tool in context['suggested_tools']:
            result = self.hrm_router.execute_tool(tool, {})
            # Use tool results for reasoning
        
        return self.generate_response(context)
```

### Advanced Memory Features

```python
# Planned enhancements
memory.add_temporal_fact(subject, relation, object, timestamp, duration)
memory.query_temporal(node, time_range)
memory.add_confidence_decay(fact_id, decay_rate)
memory.merge_similar_concepts(similarity_threshold)
```

## Performance Considerations

- **Memory Graph Size**: Optimize for graphs with 1K-10K nodes
- **Query Depth**: Limit depth to 2-3 for real-time operations
- **Tool Concurrency**: Thread-safe for concurrent tool execution
- **Persistence**: JSON serialization suitable for moderate graph sizes

## Monitoring and Debugging

```python
# Get system statistics
hrm_stats = hrm.get_integration_stats()
print(f"Memory nodes: {hrm_stats['memory_stats']['total_nodes']}")
print(f"Active tools: {hrm_stats['tool_stats']['total_tools']}")

# Monitor tool requests
with open("logs/tool_requests.jsonl") as f:
    for line in f:
        request = json.loads(line)
        print(f"Tool: {request['request']['tool_name']}, "
              f"Success: {request['response']['success']}")

# Memory graph analysis
memory_stats = memory.get_memory_stats()
for node, degree in memory_stats['most_connected_nodes'][:5]:
    print(f"Hub concept: {node} (degree: {degree})")
```

## Error Handling

All components include comprehensive error handling:

```python
try:
    result = memory.query_related("concept", depth=2)
    if not result.related_concepts:
        print("No related concepts found")
except Exception as e:
    logger.error(f"Memory query failed: {e}")

# Tool execution with error handling
response = router.route_request("tool_name", {})
if not response.success:
    print(f"Tool failed: {response.error_message}")
    # Fallback logic
```

## Best Practices

1. **Memory Management**: Regularly save memory state and monitor graph size
2. **Tool Design**: Keep tools stateless and include trace IDs for debugging
3. **Error Recovery**: Implement graceful degradation when components fail
4. **Performance**: Limit query depth and memory graph complexity for real-time use
5. **Logging**: Enable comprehensive logging for debugging and monitoring
6. **Threading**: Use proper locks when extending the system with custom components

## Contributing

When extending the system:

1. Follow the established patterns for thread safety
2. Include comprehensive error handling
3. Add logging for debugging
4. Write unit tests for new components
5. Update documentation with examples

The system is designed to be modular and extensible while maintaining compatibility with the existing HRM stack and preparing for future SLiM agent integration.
