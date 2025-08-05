# Scripts Directory

This directory contains organized scripts for different aspects of the Unified AI Companion project.

## 📁 Directory Structure

```
scripts/
├── llm_engines/          # Core LLM engine startup scripts
│   ├── run_mythomax.sh   # MythoMax LLM engine (unified companion)
│   ├── run_openchat.sh   # OpenChat LLM engine (backup/alternative)
│   ├── run_qwen2_chat.sh # Qwen2 Chat LLM engine (creative tasks)
│   └── run_kimik2.sh     # KimiK2 LLM engine (technical tasks)
├── deployment/           # Deployment and infrastructure scripts
├── testing/              # Test scripts and test runners
│   ├── test_phase1.py    # Core system tests
│   ├── test_phase2.py    # Advanced feature tests
│   └── test_phase3.py    # Integration tests
└── development/          # Development and setup scripts
    └── start_phase1.py   # System startup script
```

## 🚀 Usage

### LLM Engines
Start the unified companion system with different LLM backends:

```bash
# Start MythoMax (primary unified companion)
cd scripts/llm_engines
./run_mythomax.sh

# Start OpenChat (alternative companion)
./run_openchat.sh

# Start Qwen2 Chat (creative specialization)
./run_qwen2_chat.sh

# Start KimiK2 (technical specialization)
./run_kimik2.sh
```

### Testing
Run tests for different system components:

```bash
# Test core unified system
cd scripts/testing
python test_phase1.py

# Test advanced system features
python test_phase2.py

# Test full system integration
python test_phase3.py
```

### Development
Start development environment:

```bash
# Start unified companion development
cd scripts/development
python start_phase1.py
```

## 🔧 Script Details

### LLM Engine Scripts
- **run_mythomax.sh**: Starts MythoMax model for unified companion responses
- **run_openchat.sh**: Starts OpenChat model for alternative companion personality
- **run_qwen2_chat.sh**: Starts Qwen2 Chat for creative and poetic responses
- **run_kimik2.sh**: Starts KimiK2 for technical and analytical responses

### Test Scripts
- **test_phase1.py**: Tests core unified companion features and conversation flow
- **test_phase2.py**: Tests advanced features, creative discovery, and memory system
- **test_phase3.py**: Tests full system integration and comprehensive functionality

### Development Scripts
- **start_phase1.py**: Initializes unified companion development environment and dependencies

## 📋 Adding New Scripts

When adding new scripts, place them in the appropriate directory:

1. **LLM Engines**: New model startup scripts for creative discovery
2. **Deployment**: Infrastructure, Docker, and deployment scripts
3. **Testing**: Test runners and validation scripts for unified system
4. **Development**: Setup, configuration, and development tools

## 🎯 Next Steps

### Planned Scripts
- **deployment/docker_build.sh**: Automated Docker image building
- **deployment/deploy_production.sh**: Production deployment automation
- **testing/integration_tests.py**: Full unified system integration tests
- **development/setup_environment.sh**: Complete environment setup

### Script Improvements
- Add error handling and logging for unified system
- Implement configuration management for companion settings
- Add script documentation for new unified architecture
- Create comprehensive testing framework for unified companion 