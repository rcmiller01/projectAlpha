# Emotion Quantization Autopilot System

**Fully Autonomous Emotional Quantization for Local LLMs**

## Overview

The Emotion Quantization Autopilot is a comprehensive system that enables fully autonomous execution of emotional quantization refinement loops for local Language Models. The system monitors system idle time, manages quantization job queues, and executes quantization processes while preserving emotional intelligence in the models.

## 🌟 Key Features

### 🤖 **Autonomous Operation**
- **Idle-Triggered Execution**: Automatically starts quantization when system is idle
- **Queue-Based Management**: Prioritized job queue with automatic execution
- **Safety Limits**: Daily execution limits, disk space monitoring, emergency stops
- **Background Processing**: Runs quantization jobs without user intervention

### 📊 **Intelligent Monitoring**
- **System Resource Tracking**: CPU, memory, and disk usage monitoring
- **User Activity Detection**: Mouse/keyboard activity detection (optional)
- **Configurable Thresholds**: Customizable idle detection parameters
- **Real-time Status**: Live monitoring of system state and job progress

### 🔄 **Integration with Existing Pipeline**
- **Pass 1 Quantization Loop**: Direct integration with existing quantization system
- **Emotional Dataset Builder**: Automatic evaluation using predefined scenarios
- **Training Tracker**: Comprehensive logging of all quantization attempts
- **Database Persistence**: SQLite backend with automatic backups

### 🛡️ **Production-Ready Safety**
- **Emergency Stop Mechanism**: Immediate halt capability
- **Resource Limits**: Configurable CPU, memory, and disk constraints
- **Error Recovery**: Graceful handling of failures and timeouts
- **Comprehensive Logging**: Detailed audit trails and notifications

## 📂 System Architecture

```
emotion_quant_autopilot/
├── quant_autopilot.py          # Main autopilot controller
├── idle_monitor.py             # System idle state monitoring
├── autopilot_config.json       # Configuration settings
├── setup_autopilot.py          # Setup and installation script
├── logs/                       # Log files and notifications
├── temp/                       # Temporary processing files
└── EMERGENCY_STOP              # Emergency stop trigger file
```

## 🚀 Quick Start

### 1. Installation & Setup

```bash
# Navigate to the autopilot directory
cd emotion_quant_autopilot

# Run setup script
python setup_autopilot.py

# Install optional dependencies for enhanced monitoring
pip install pynput requests
```

### 2. Configuration

Edit `autopilot_config.json` to customize settings:

```json
{
  "min_idle_minutes": 30,
  "max_active_loops_per_day": 3,
  "target_model_size_range_gb": [12, 24],
  "preferred_base_models": [
    "llama2-chat-hf",
    "mistral-instruct"
  ],
  "quantization_methods": [
    "q8_0", "q6_K", "q5_K_M", "q4_K_M", "q3_K_L"
  ]
}
```

### 3. Basic Usage

```bash
# Check system status
python quant_autopilot.py status

# Start autopilot (runs continuously)
python quant_autopilot.py start

# Add specific quantization job
python quant_autopilot.py add-job --model llama2-chat-hf --quant q4_K_M --priority 8

# View job queue
python quant_autopilot.py queue

# Populate default jobs
python quant_autopilot.py populate
```

### 4. Emergency Stop

```bash
# Immediate stop (creates emergency stop file)
touch emotion_quant_autopilot/EMERGENCY_STOP

# Resume operations (remove emergency stop file)
rm emotion_quant_autopilot/EMERGENCY_STOP
```

## 📋 Configuration Reference

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `min_idle_minutes` | `30` | Minimum idle time before starting quantization |
| `max_active_loops_per_day` | `3` | Daily limit for quantization runs |
| `target_model_size_range_gb` | `[12, 24]` | Target size range for quantized models |
| `auto_start` | `true` | Start monitoring immediately |

### System Monitoring

| Setting | Default | Description |
|---------|---------|-------------|
| `check_interval_seconds` | `300` | How often to check system state |
| `cpu_threshold_percent` | `15` | Maximum CPU usage for idle state |
| `memory_threshold_percent` | `80` | Maximum memory usage threshold |
| `disk_space_threshold_gb` | `50` | Minimum free disk space required |

### Safety Limits

| Setting | Default | Description |
|---------|---------|-------------|
| `max_concurrent_processes` | `1` | Maximum simultaneous quantization jobs |
| `max_disk_usage_gb` | `100` | Maximum disk usage for model storage |
| `emergency_stop_file` | `EMERGENCY_STOP` | Emergency stop trigger file path |

### Evaluation Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `min_judgment_score` | `0.7` | Minimum acceptable quality score |
| `evaluation_prompt_count` | `25` | Number of prompts for evaluation |
| `timeout_minutes` | `120` | Maximum time per quantization job |

## 🔧 System Components

### 1. Idle Monitor (`idle_monitor.py`)

**Purpose**: Monitors system resources and user activity to determine optimal execution windows.

**Features**:
- CPU and memory usage tracking
- User input detection (mouse/keyboard)
- Configurable thresholds and timeouts
- Thread-safe operation with callbacks

**Usage**:
```python
from idle_monitor import IdleMonitor, IdleConfig

config = IdleConfig(min_idle_minutes=30)
monitor = IdleMonitor(config)
monitor.start_monitoring()
```

### 2. Autopilot Controller (`quant_autopilot.py`)

**Purpose**: Master control loop managing the entire autopilot system.

**Features**:
- Queue-based job management
- Integration with quantization pipeline
- Database persistence and tracking
- Safety limit enforcement

**Key Classes**:
- `QuantizationAutopilot`: Main controller
- `AutopilotDatabase`: Database management
- `QuantizationJob`: Job queue representation
- `AutopilotRun`: Execution run tracking

### 3. Configuration (`autopilot_config.json`)

**Purpose**: Centralized configuration for all autopilot components.

**Sections**:
- Core autopilot settings
- System monitoring thresholds
- Safety limits and constraints
- Output paths and directories
- Notification settings (email/Slack)

## 🗄️ Database Schema

The autopilot extends the existing `emotion_training.db` with additional tables:

### `autopilot_runs` Table

```sql
CREATE TABLE autopilot_runs (
    id INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    trigger_type TEXT NOT NULL,        -- "idle", "manual", "scheduled"
    timestamp TEXT NOT NULL,
    model_path TEXT NOT NULL,
    base_model TEXT NOT NULL,
    quantization_method TEXT NOT NULL,
    target_size_gb REAL NOT NULL,
    result_summary TEXT NOT NULL,
    judgment_score REAL NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT DEFAULT '',
    execution_time_minutes REAL DEFAULT 0.0
);
```

### `quantization_queue` Table

```sql
CREATE TABLE quantization_queue (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    base_model TEXT NOT NULL,
    quantization_method TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',     -- "pending", "running", "completed", "failed"
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    started_at TEXT NULL,
    completed_at TEXT NULL,
    run_id TEXT NULL
);
```

## 🔍 Monitoring & Logging

### Log Files

- **Autopilot Logs**: `logs/autopilot_YYYYMMDD.log`
- **Idle Monitor Logs**: `logs/idle_monitor_YYYYMMDD.log`
- **Notifications**: `logs/notifications.log`

### Status Monitoring

```bash
# Real-time status
python quant_autopilot.py status

# Log monitoring
tail -f emotion_quant_autopilot/logs/autopilot_*.log

# System metrics
python idle_monitor.py --test --duration 60
```

### Status Response Example

```json
{
  "is_running": true,
  "idle_status": {
    "current_state": "idle",
    "state_duration_minutes": 45.2,
    "user_idle_minutes": 52.1,
    "is_idle": true,
    "metrics": {
      "cpu_percent": 8.5,
      "memory_percent": 65.2,
      "disk_free_gb": 120.5
    }
  },
  "current_job": {
    "job_id": "abc-123",
    "base_model": "llama2-chat-hf",
    "quantization_method": "q4_K_M",
    "priority": 5
  },
  "pending_jobs_count": 12,
  "daily_runs": 2,
  "max_daily_runs": 3
}
```

## 🛠️ Advanced Usage

### Custom Job Scheduling

```python
from quant_autopilot import QuantizationAutopilot

autopilot = QuantizationAutopilot("custom_config.json")

# Add high-priority job
job_id = autopilot.add_quantization_job(
    base_model="custom-model-path",
    quantization_method="q3_K_L",
    priority=9
)

# Start monitoring
autopilot.start()
```

### Programmatic Control

```python
# Get detailed status
status = autopilot.get_status()

# Force idle state (for testing)
autopilot.idle_monitor.force_idle_state()

# Check safety limits
is_safe = autopilot._check_safety_limits()
```

### Integration with External Systems

```python
# Custom notification callback
def custom_notification(message: str):
    # Send to external monitoring system
    send_to_slack(message)
    log_to_database(message)

# Replace default notification
autopilot._send_notification = custom_notification
```

## 🔧 Service Deployment (Linux)

### Systemd Service

1. **Edit service template**:
```bash
nano emotion_quant_autopilot/quantization-autopilot.service
```

2. **Update paths and user**:
```ini
[Service]
User=quantization_user
WorkingDirectory=/path/to/emotion_quant_autopilot
ExecStart=/usr/bin/python3 /path/to/quant_autopilot.py start
```

3. **Install and enable**:
```bash
sudo cp quantization-autopilot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantization-autopilot
sudo systemctl start quantization-autopilot
```

4. **Monitor service**:
```bash
sudo systemctl status quantization-autopilot
sudo journalctl -u quantization-autopilot -f
```

## 🚨 Troubleshooting

### Common Issues

**Autopilot won't start**:
```bash
# Check configuration
python -c "import json; json.load(open('autopilot_config.json'))"

# Check dependencies
python setup_autopilot.py

# Check permissions
ls -la emotion_quant_autopilot/
```

**Idle detection not working**:
```bash
# Test idle monitor
python idle_monitor.py --test --duration 60

# Check pynput installation
pip install pynput

# Check system thresholds
python quant_autopilot.py status
```

**Jobs not executing**:
```bash
# Check daily limits
python quant_autopilot.py status

# Check emergency stop
ls emotion_quant_autopilot/EMERGENCY_STOP

# Check queue
python quant_autopilot.py queue

# Check disk space
df -h
```

**Database errors**:
```bash
# Check database integrity
sqlite3 emotion_training.db ".schema"

# Reset database (WARNING: loses data)
rm emotion_training.db
python quant_autopilot.py status  # Recreates database
```

### Log Analysis

```bash
# Recent errors
grep -i error emotion_quant_autopilot/logs/autopilot_*.log | tail -20

# Quantization failures
grep -i "failed\|exception" emotion_quant_autopilot/logs/autopilot_*.log

# System idle patterns
grep -i "idle\|active" emotion_quant_autopilot/logs/idle_monitor_*.log

# Job execution timeline
grep -i "starting\|completed" emotion_quant_autopilot/logs/autopilot_*.log
```

## 📈 Performance Optimization

### Resource Management

1. **CPU Throttling**: Adjust `cpu_threshold_percent` for system responsiveness
2. **Memory Limits**: Monitor memory usage during quantization
3. **Disk Management**: Implement automatic cleanup of old models
4. **Concurrent Jobs**: Increase `max_concurrent_processes` for powerful systems

### Optimization Tips

```json
{
  "system_monitoring": {
    "check_interval_seconds": 60,     // Reduce for faster detection
    "cpu_threshold_percent": 25,      // Increase for busy systems
    "memory_threshold_percent": 70    // Lower for memory-constrained systems
  },
  "evaluation_settings": {
    "evaluation_prompt_count": 50,    // Increase for better evaluation
    "timeout_minutes": 180            // Increase for large models
  }
}
```

## 🔮 Future Enhancements

### Planned Features

1. **Multi-Model Support**: Simultaneous quantization of different base models
2. **Cloud Integration**: Remote model storage and distributed processing
3. **Advanced Scheduling**: Time-based and condition-based job scheduling
4. **Web Dashboard**: Real-time monitoring and control interface
5. **Model Registry**: Automatic model versioning and deployment
6. **Custom Evaluation**: User-defined evaluation metrics and datasets

### Integration Opportunities

1. **CI/CD Pipelines**: Automated quantization in model deployment workflows
2. **Kubernetes**: Container orchestration for scalable deployment
3. **Monitoring Systems**: Integration with Prometheus, Grafana, etc.
4. **Cloud Platforms**: AWS, GCP, Azure integration for hybrid deployments

---

**Built for the Dolphin AI Orchestrator v2.1 Emotional Quantization System**

*Autonomous emotional intelligence preservation in quantized language models*
