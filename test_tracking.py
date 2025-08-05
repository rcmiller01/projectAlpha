#!/usr/bin/env python3
"""Simple test of the quantization tracking system"""

import sys
import os
sys.path.append('.')

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

try:
    from quant_tracking import QuantTracker, QuantLoopResult
    import uuid
    from datetime import datetime
    
    print("✅ Imports successful")
    
    # Create tracker
    tracker = QuantTracker()
    print("✅ Tracker initialized")
    
    # Create test result
    result = QuantLoopResult(
        loop_id=str(uuid.uuid4()),
        model_name='llama-3.2-1b-instruct',
        quant_format='Q4_K_M',
        size_mb=756.2,
        emotional_score=0.847,
        token_quality=0.792,
        passed_threshold=True,
        duration_seconds=45.3,
        error_count=0,
        memory_peak_mb=512.4,
        cpu_avg_percent=67.8,
        sentiment_variance=0.12,
        coherence_score=0.84,
        creativity_index=0.73
    )
    print("✅ Test result created")
    
    # Save result
    tracker.save_loop_result(result)
    print("✅ Result saved")
    
    # Check file
    if os.path.exists('data/quant_results.jsonl'):
        size = os.path.getsize('data/quant_results.jsonl')
        print(f"✅ File created: {size} bytes")
    else:
        print("❌ File not found")
    
    # Load results
    results = tracker.load_results()
    print(f"✅ Loaded {len(results)} results")
    
    if results:
        r = results[0]
        print(f"   - Model: {r.model_name}")
        print(f"   - Format: {r.quant_format}")
        print(f"   - Emotion: {r.emotional_score}")
        print(f"   - Quality: {r.token_quality}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
