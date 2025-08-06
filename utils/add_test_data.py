#!/usr/bin/env python3
"""Add multiple test results for the dashboard demo"""

import sys
import os
sys.path.append('.')

from quant_tracking import QuantTracker, QuantLoopResult
import uuid
from datetime import datetime, timedelta
import time

# Create tracker
tracker = QuantTracker()

# Test results with varying quality to show different scenarios
test_results = [
    {
        'model_name': 'llama-3.2-1b-instruct',
        'quant_format': 'Q6_K',
        'size_mb': 1024.5,
        'emotional_score': 0.923,
        'token_quality': 0.856,
        'passed': True,
        'duration': 38.7,
        'memory_peak': 680.2,
        'cpu_avg': 74.3
    },
    {
        'model_name': 'llama-3.2-1b-instruct',
        'quant_format': 'Q2_K',
        'size_mb': 412.8,
        'emotional_score': 0.634,
        'token_quality': 0.589,
        'passed': False,
        'duration': 52.1,
        'memory_peak': 345.6,
        'cpu_avg': 82.1
    },
    {
        'model_name': 'llama-3.2-3b-instruct',
        'quant_format': 'Q4_K_M',
        'size_mb': 1812.4,
        'emotional_score': 0.889,
        'token_quality': 0.824,
        'passed': True,
        'duration': 67.2,
        'memory_peak': 923.7,
        'cpu_avg': 71.8
    },
    {
        'model_name': 'llama-3.2-3b-instruct',
        'quant_format': 'Q5_K_M',
        'size_mb': 2156.8,
        'emotional_score': 0.931,
        'token_quality': 0.867,
        'passed': True,
        'duration': 71.5,
        'memory_peak': 1124.3,
        'cpu_avg': 68.9
    },
    {
        'model_name': 'llama-3.2-1b-instruct',
        'quant_format': 'Q8_0',
        'size_mb': 1435.2,
        'emotional_score': 0.964,
        'token_quality': 0.913,
        'passed': True,
        'duration': 28.4,
        'memory_peak': 789.1,
        'cpu_avg': 59.6
    }
]

print(f"Adding {len(test_results)} test results...")

for i, data in enumerate(test_results):
    # Create result with slight time offsets
    timestamp = datetime.now() - timedelta(minutes=(len(test_results) - i) * 10)
    
    result = QuantLoopResult(
        loop_id=str(uuid.uuid4()),
        model_name=data['model_name'],
        quant_format=data['quant_format'],
        size_mb=data['size_mb'],
        emotional_score=data['emotional_score'],
        token_quality=data['token_quality'],
        passed_threshold=data['passed'],
        timestamp=timestamp,
        duration_seconds=data['duration'],
        error_count=0 if data['passed'] else 2,
        memory_peak_mb=data['memory_peak'],
        cpu_avg_percent=data['cpu_avg'],
        sentiment_variance=0.08 + (i * 0.02),
        coherence_score=data['emotional_score'] * 0.9,
        creativity_index=0.65 + (i * 0.03)
    )
    
    tracker.save_loop_result(result)
    print(f"✅ Added {data['model_name']} ({data['quant_format']}) - Emotion: {data['emotional_score']:.3f}")
    time.sleep(0.1)  # Small delay between saves

# Load and display summary
results = tracker.load_results()
print(f"\n📊 Total results in tracking system: {len(results)}")

# Show performance summary
summary = tracker.get_performance_summary()
print(f"📈 Performance Summary:")
print(f"   - Total loops: {summary['total_loops']}")
print(f"   - Success rate: {summary['success_rate']:.1%}")
print(f"   - Avg emotional score: {summary['avg_emotional_score']:.3f}")
print(f"   - Avg token quality: {summary['avg_token_quality']:.3f}")

print("\n🎯 Test data ready for dashboard!")
