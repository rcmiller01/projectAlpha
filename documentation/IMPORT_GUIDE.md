# Import Guide After Reorganization

## New Import Paths

### Backend Components
```python
# HRM Router
from backend.hrm_router import HRMRouter, HRMMode, RequestType

# HRM API
from backend.hrm_api import hrm_api_function

# Subagent Router
from backend.subagent_router import SubAgentRouter, AgentType
```

### Core Components
```python
# Core Arbiter
from core.core_arbiter import CoreArbiter, ArbiterResponse

# Mirror Mode
from core.mirror_mode import MirrorModeManager, MirrorType

# Symbolic Drift
from core.symbolic_drift import SymbolicDriftTracker

# Emotion Loop
from core.emotion_loop_core import EmotionState, process_emotional_loop
```

### Source Components
```python
# Agents
from src.agents.deduction_agent import DeductionAgent
from src.agents.metaphor_agent import MetaphorAgent

# Core
from src.core.core_conductor import CoreConductor
from src.core.hrm_router import HRMRouter  # If not moved to backend

# API
from src.api.hrm_api import HRMApi
```

### Testing
```python
# Test utilities (now in testing/)
from testing.test_hrm_integration import test_function
from testing.comprehensive_verification_suite import verify_system
```

## Common Import Fixes

| Old Import | New Import |
|------------|------------|
| `from hrm_router import` | `from backend.hrm_router import` |
| `from core_arbiter import` | `from core.core_arbiter import` |
| `from mirror_mode import` | `from core.mirror_mode import` |
| `from symbolic_drift import` | `from core.symbolic_drift import` |

## Path Resolution

Make sure your Python path includes the project root:

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root))
```
