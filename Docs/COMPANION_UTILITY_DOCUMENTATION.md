# Companion Utility Modules Documentation

## Overview

The Companion Utility Modules transform your AI companion system from purely conversational to a comprehensive daily life assistant. These modules provide email management, scheduling optimization, task automation, and workflow integration - all guided by persona-driven intelligence and emotional awareness.

## Core Components

### 1. Companion Assistant Module (`modules/utility/companion_assistant.py`)

The primary utility engine that provides:

- **Email Management**: Intelligent email processing with persona-driven categorization and response suggestions
- **Schedule Optimization**: Calendar management with stress-level awareness and work-life balance
- **Task Management**: Voice-to-task conversion with persona-specific guidance and complexity analysis
- **Proactive Suggestions**: Context-aware recommendations based on patterns and current state

#### Key Features:

```python
# Email processing with emotional intelligence
email_summaries = await assistant.process_emails(email_config)

# Task creation with persona guidance
task = await assistant.create_task_with_persona_guidance(
    "Prepare presentation for Friday meeting", 
    context={'stress_level': 0.6}
)

# Voice dictation with auto-organization
voice_note = await assistant.process_voice_dictation(audio_data, context)

# Proactive assistance
suggestions = assistant.get_proactive_suggestions(current_context)
```

### 2. N8N Workflow Integration (`modules/automation/n8n_integration.py`)

Advanced workflow automation with persona oversight:

- **Workflow Safety Assessment**: Evaluates automation safety with persona-specific risk tolerance
- **Persona-Supervised Execution**: Real-time monitoring and intervention capabilities
- **Emotional Impact Tracking**: Monitors stress levels and relationship effects of automation
- **Rollback Mechanisms**: Automatic reversal of problematic automations

#### Key Features:

```python
# Create workflow with persona supervision
workflow_result = await n8n_manager.create_workflow(workflow_config, "mia")

# Execute with persona oversight
execution_result = await n8n_manager.execute_workflow(workflow_id, trigger_data)

# Monitor with emotional intelligence
monitoring_result = await n8n_manager._monitor_execution_with_persona(
    execution, n8n_result, persona_name
)
```

## Unified Intelligent Assistant

### Adaptive Intelligence
The companion uses a single, versatile persona that adapts its approach based on context and user needs:

- **Email Management**: Balances professionalism with emotional awareness, adjusting tone based on relationship importance and content sensitivity
- **Task Approach**: Combines systematic organization with stress-awareness, providing both efficient execution and emotional support
- **Automation Philosophy**: Intelligent risk assessment that prioritizes both productivity and well-being, adapting safety levels based on task complexity
- **Development Support**: Seamlessly transitions between companion interactions and technical assistance, maintaining context and relationship continuity
- **Monitoring**: Comprehensive tracking of productivity, emotional well-being, and system performance with intelligent intervention when needed

### Context-Aware Adaptation
The unified persona automatically adjusts its communication style and approach based on:
- **Task Complexity**: More systematic for complex technical work, more supportive for stress-inducing tasks
- **Emotional State**: Increased empathy and gentleness when stress levels are high
- **Relationship Context**: Professional tone for work emails, warmer approach for personal communications
- **Development Mode**: Technical precision for coding tasks while maintaining the supportive relationship dynamic
- **Time Sensitivity**: Balanced urgency handling that respects both deadlines and well-being

## Integration Examples

### Daily Morning Routine

```python
async def morning_assistant_routine(user_id: str):
    assistant = create_companion_assistant(user_id, 'adaptive')
    
    # Process overnight emails with context-aware approach
    emails = await assistant.process_emails(email_config)
    important_emails = [e for e in emails if e.importance_score > 0.7]
    
    # Optimize today's calendar with work-life balance awareness
    schedule_result = await assistant.manage_schedule(calendar_data, user_prefs)
    
    # Generate morning insights with adaptive intelligence
    context = {'time_of_day': 'morning', 'energy_level': 0.8}
    suggestions = assistant.get_proactive_suggestions(context)
    
    return {
        'email_summary': f"{len(important_emails)} important emails need attention",
        'schedule_optimization': schedule_result['suggestions'],
        'morning_suggestions': suggestions[:3]
    }
```

### Voice-to-Task with Development Integration

```python
async def unified_voice_task_automation(user_id: str, audio_data: bytes):
    assistant = create_companion_assistant(user_id, 'adaptive')
    
    # Process voice note with context detection
    voice_note = await assistant.process_voice_dictation(audio_data, context)
    
    # Detect if it's development work or personal tasks
    if assistant.detect_development_context(voice_note.transcription):
        # Handle as development task with technical precision
        tasks = await assistant.create_development_tasks(voice_note, context)
    else:
        # Handle as personal/companion task with emotional awareness
        tasks = await assistant.create_personal_tasks(voice_note, context)
    
    return {
        'transcription': voice_note.transcription,
        'context_detected': voice_note.context_type,
        'tasks_created': len(tasks),
        'adaptive_guidance': voice_note.adaptive_analysis
    }
```

### Unified Workflow Setup

```python
async def setup_adaptive_automation(user_id: str, n8n_credentials: dict):
    n8n_manager = create_n8n_manager(
        n8n_credentials['base_url'],
        n8n_credentials['api_key'],
        user_id
    )
    
    # Define adaptive automation workflow
    workflow_config = {
        'name': 'unified_intelligent_assistant',
        'type': 'adaptive_automation',
        'purpose': 'context-aware assistance for both development and personal tasks',
        'triggers': [
            {'type': 'email_received'},
            {'type': 'development_task_detected'},
            {'type': 'personal_task_created'}
        ],
        'actions': [
            {'type': 'analyze_context', 'parameters': {'detect_work_type': True}},
            {'type': 'adapt_approach', 'parameters': {'technical_mode': True, 'companion_mode': True}},
            {'type': 'provide_assistance', 'parameters': {'unified_intelligence': True}}
        ]
    }
    
    # Create with unified intelligence
    result = await n8n_manager.create_workflow(workflow_config, 'adaptive')
    
    return result
```

## Configuration Options

### User Preferences

```python
user_preferences = {
    'email_settings': {
        'auto_categorization': True,
        'response_suggestions': True,
        'relationship_preservation': True,
        'professional_tone_matching': True
    },
    'calendar_optimization': {
        'focus_block_duration': 90,  # minutes
        'meeting_buffer_time': 15,   # minutes
        'stress_threshold': 0.7,     # 0.0-1.0
        'work_life_balance_priority': 'high'
    },
    'task_management': {
        'auto_task_creation': True,
        'complexity_breakdown': True,
        'persona_guidance': True,
        'stress_aware_scheduling': True
    },
    'automation_safety': {
        'risk_tolerance': 'medium',  # low, medium, high
        'persona_supervision': True,
        'rollback_enabled': True,
        'monitoring_level': 'standard'
    }
}
```

### Unified Assistant Configuration

```python
assistant_config = {
    'communication_style': 'adaptive_intelligent',  # Adjusts based on context
    'technical_capability': 'full_stack',          # Handles both dev and companion tasks
    'emotional_intelligence': 'high',              # Always maintains emotional awareness
    'risk_tolerance': 'context_adaptive',          # Adjusts based on task and user state
    'intervention_style': 'balanced_guidance',     # Combines support with practical action
    'priority_values': ['user_wellbeing', 'productivity', 'relationship_quality', 'technical_excellence'],
    'adaptation_parameters': {
        'stress_sensitivity': True,                # Adjusts approach based on user stress
        'technical_mode_switching': True,         # Seamless dev/companion transitions
        'relationship_continuity': True,          # Maintains connection across all interactions
        'context_awareness': True                 # Adapts to work/personal/creative contexts
    }
}
```

## Safety and Privacy

### Data Protection
- All personal data encrypted at rest and in transit
- Persona oversight prevents unauthorized data access
- User consent required for all automation actions
- Automatic data retention limits (30-90 days)

### Automation Safety
- Multi-layer safety checks before workflow execution
- Persona-specific risk assessment for all automations
- Real-time monitoring with intervention capabilities
- Rollback mechanisms for problematic automations

### Privacy Preservation
- Email content analysis done locally when possible
- Relationship data never shared with external services
- Voice transcription with on-device processing options
- User control over all data sharing permissions

## Performance Optimization

### Efficiency Features
- Intelligent caching of email categorization models
- Batch processing for routine automation tasks
- Async processing for voice transcription and analysis
- Smart rate limiting for API-dependent features

### Scalability
- Modular architecture supports independent scaling
- Database optimization for large email/task volumes
- Efficient memory management for voice processing
- Configurable automation complexity levels

## Commercial Applications

### Personal Productivity Suite
- **Individual Users**: Complete daily life assistance with emotional intelligence
- **Professional Services**: Enhanced productivity for knowledge workers
- **Executive Assistants**: AI-powered support for high-level decision makers

### Enterprise Integration
- **Team Collaboration**: Coordinated assistance across team members
- **Workflow Optimization**: Company-wide process automation with human oversight
- **Stress Management**: Organization-wide well-being monitoring and support

### Healthcare and Wellness
- **Therapeutic Support**: Stress-aware task management for therapy clients
- **Caregiver Assistance**: Automated support for healthcare providers
- **Wellness Coaching**: Integrated life management with health awareness

## Future Enhancements

### Planned Features
- **Multi-language Support**: Persona-driven assistance in multiple languages
- **Advanced Voice Processing**: Emotion detection and speaker identification
- **Calendar Intelligence**: Meeting outcome prediction and optimization
- **Relationship CRM**: Automated relationship maintenance suggestions

### Integration Roadmap
- **Microsoft 365**: Native integration with Outlook, Teams, and Office apps
- **Google Workspace**: Gmail, Calendar, and Drive automation
- **Slack/Discord**: Team communication assistance
- **Zapier/IFTTT**: Expanded automation ecosystem integration

## Getting Started

### Basic Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Storage**:
   ```python
   # Ensure storage directories exist
   os.makedirs("storage/utility", exist_ok=True)
   os.makedirs("storage/workflows", exist_ok=True)
   os.makedirs("storage/attachment", exist_ok=True)
   ```

3. **Initialize Unified Assistant**:
   ```python
   from modules.utility.companion_assistant import create_companion_assistant
   
   assistant = create_companion_assistant("user123", "adaptive")
   ```

4. **Test Adaptive Functionality**:
   ```python
   # Test development task creation
   dev_task = await assistant.create_development_task(
       "Implement user authentication API endpoint",
       context={'complexity': 'medium', 'deadline': '2 days'}
   )
   print(f"Dev task: {dev_task.title}")
   print(f"Technical guidance: {dev_task.technical_guidance}")
   
   # Test personal task creation
   personal_task = await assistant.create_personal_task(
       "Plan weekend hiking trip",
       context={'stress_level': 0.4, 'energy': 'high'}
   )
   print(f"Personal task: {personal_task.title}")
   print(f"Supportive guidance: {personal_task.supportive_guidance}")
   ```

### Advanced Setup with N8N

1. **Setup N8N Instance**:
   ```bash
   # Docker setup
   docker run -it --rm --name n8n -p 5678:5678 n8nio/n8n
   ```

2. **Configure Integration**:
   ```python
   from modules.automation.n8n_integration import create_n8n_manager
   
   n8n_manager = create_n8n_manager(
       "http://localhost:5678",
       "your_api_key",
       "user123"
   )
   ```

3. **Create Adaptive Workflow**:
   ```python
   workflow_config = {
       'name': 'unified_assistant_workflow',
       'type': 'adaptive_automation',
       'triggers': [
           {'type': 'email_received'},
           {'type': 'voice_note_processed'},
           {'type': 'development_context_detected'}
       ],
       'actions': [
           {'type': 'analyze_context'},
           {'type': 'provide_adaptive_assistance'}
       ]
   }
   
   result = await n8n_manager.create_workflow(workflow_config, "adaptive")
   ```

## Support and Community

- **Documentation**: Complete API reference and examples
- **Community Forum**: User discussions and feature requests
- **Developer Support**: Integration assistance and custom development
- **Training Resources**: Video tutorials and best practices guides

---

This companion utility system represents the future of AI assistance - not just conversational, but actively helpful in managing the complexities of daily life and development work, all while maintaining the emotional intelligence and adaptive care that makes AI companions truly special. The unified approach eliminates the need for persona switching while providing context-aware assistance that seamlessly transitions between technical development support and personal companion interactions.
