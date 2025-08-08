#!/usr/bin/env python3
"""
Import Path Fixer - Update import paths after file reorganization

This script automatically fixes import paths in Python files after
the project reorganization moved files to appropriate directories.
"""

import os
import re
import sys
from pathlib import Path

def fix_import_paths():
    """Fix import paths throughout the project"""

    # Define path mappings for common import fixes
    import_mappings = {
        r'from backend.hrm_router import': r'from backend.hrm_router import',
        r'import backend.hrm_router as hrm_router': r'import backend.hrm_router as hrm_router',
        r'from backend.hrm_api import': r'from backend.hrm_api import',
        r'import backend.hrm_api as hrm_api': r'import backend.hrm_api as hrm_api',
        r'from core.core_arbiter import': r'from core.core_arbiter import',
        r'import core.core_arbiter as core_arbiter': r'import core.core_arbiter as core_arbiter',
        r'from core.mirror_mode import': r'from core.mirror_mode import',
        r'import core.mirror_mode as mirror_mode': r'import core.mirror_mode as mirror_mode',
        r'from core.symbolic_drift import': r'from core.symbolic_drift import',
        r'import core.symbolic_drift as symbolic_drift': r'import core.symbolic_drift as symbolic_drift',
    }

    # Files and directories to scan
    scan_directories = [
        'src/',
        'backend/',
        'tests/',
        'testing/',
        'webapp/',
        'examples/',
        'demos/',
        'utils/',
        'scripts/'
    ]

    # Get project root
    project_root = Path(__file__).parent.parent
    files_updated = 0

    print("🔧 Starting import path fixes...")

    for directory in scan_directories:
        dir_path = project_root / directory
        if not dir_path.exists():
            continue

        print(f"📁 Scanning {directory}")

        # Find all Python files
        for py_file in dir_path.rglob("*.py"):
            if fix_file_imports(py_file, import_mappings):
                files_updated += 1
                print(f"   ✅ Fixed: {py_file.relative_to(project_root)}")

    print(f"\n🎉 Import fixing complete! Updated {files_updated} files.")

    return files_updated

def fix_file_imports(file_path: Path, mappings: dict) -> bool:
    """
    Fix imports in a single file

    Args:
        file_path (Path): Path to the Python file
        mappings (dict): Dictionary of import pattern mappings

    Returns:
        bool: True if file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply each mapping
        for old_pattern, new_pattern in mappings.items():
            content = re.sub(old_pattern, new_pattern, content)

        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

    except Exception as e:
        print(f"   ❌ Error fixing {file_path}: {e}")

    return False

def validate_imports():
    """Validate that critical imports work after fixes"""

    print("\n🔍 Validating critical imports...")

    test_imports = [
        ('backend.hrm_router', 'HRMRouter'),
        ('backend.hrm_api', 'hrm_api'),
        ('core.core_arbiter', 'CoreArbiter'),
        ('core.mirror_mode', 'MirrorModeManager'),
        ('core.symbolic_drift', 'symbolic_drift'),
    ]

    validation_passed = 0

    for module_name, class_name in test_imports:
        try:
            # Add project root to path
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))

            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"   ✅ {module_name}.{class_name}")
                validation_passed += 1
            else:
                print(f"   ⚠️  {module_name} imported but {class_name} not found")

        except ImportError as e:
            print(f"   ❌ Failed to import {module_name}: {e}")
        except Exception as e:
            print(f"   ⚠️  Error validating {module_name}: {e}")

    print(f"\n📊 Validation complete: {validation_passed}/{len(test_imports)} imports successful")
    return validation_passed

def create_import_guide():
    """Create a guide for the new import structure"""

    guide_content = """# Import Guide After Reorganization

## New Import Paths

### Backend Components
```python
# HRM Router
from backend.hrm_router import HRMRouter, HRMMode, RequestType

# HRM API
from backend.hrm_api import backend.hrm_api as hrm_api_function

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
| `from backend.hrm_router import` | `from backend.hrm_router import` |
| `from core.core_arbiter import` | `from core.core_arbiter import` |
| `from core.mirror_mode import` | `from core.mirror_mode import` |
| `from core.symbolic_drift import` | `from core.symbolic_drift import` |

## Path Resolution

Make sure your Python path includes the project root:

```python
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # Adjust as needed
sys.path.insert(0, str(project_root))
```
"""

    guide_path = Path(__file__).parent.parent / "documentation" / "IMPORT_GUIDE.md"
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)

    print(f"📖 Created import guide: {guide_path}")

if __name__ == "__main__":
    print("🚀 Project Import Path Fixer")
    print("=" * 50)

    # Fix import paths
    files_updated = fix_import_paths()

    # Validate critical imports
    validation_passed = validate_imports()

    # Create import guide
    create_import_guide()

    print(f"\n✨ Summary:")
    print(f"   📝 Files updated: {files_updated}")
    print(f"   ✅ Imports validated: {validation_passed}")
    print(f"   📖 Import guide created")

    if files_updated > 0:
        print(f"\n⚠️  Please test your application to ensure all imports work correctly!")
    else:
        print(f"\n🎯 No import fixes needed - all paths appear to be correct!")
