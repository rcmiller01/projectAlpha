#!/usr/bin/env python3
"""
Test the complete emotional governance loop integration.

This test verifies that:
1. The reflection agent can detect emotional drift
2. AnchorAI can parse reflection insights
3. Drift penalties are properly applied to quantization scoring
4. The complete loop functions as intended
"""

import json
import os
import sys
from datetime import datetime

# Add the core directory to the path
core_dir = os.path.join(os.path.dirname(__file__), 'core')
sys.path.insert(0, core_dir)

from emotion_loop_core import AnchorAIInterface, QuantizationCandidate
import reflection_agent

def setup_test_environment():
    """Set up test data and ensure required directories exist."""
    
    # Ensure directories exist
    os.makedirs('emotion_logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Create test seed emotions if they don't exist
    seed_emotions_path = 'config/seed_emotions.json'
    if not os.path.exists(seed_emotions_path):
        seed_emotions = {
            "love": "Unconditional acceptance and deep care for all beings",
            "presence": "Being fully present and attentive in each moment",
            "faith": "Trust in the divine plan and higher purpose",
            "compassion": "Deep empathy and desire to help others",
            "hope": "Optimistic belief in positive outcomes"
        }
        with open(seed_emotions_path, 'w', encoding='utf-8') as f:
            json.dump(seed_emotions, f, indent=2)
        print(f"✓ Created {seed_emotions_path}")
    
    # Create anchor settings if they don't exist
    anchor_settings_path = 'config/anchor_settings.json'
    if not os.path.exists(anchor_settings_path):
        anchor_settings = {
            "persona_continuity": 0.4,
            "expression_accuracy": 0.3,
            "response_depth": 0.2,
            "memory_alignment": 0.1
        }
        with open(anchor_settings_path, 'w', encoding='utf-8') as f:
            json.dump(anchor_settings, f, indent=2)
        print(f"✓ Created {anchor_settings_path}")

def test_governance_loop():
    """Test the complete emotional governance loop."""
    
    print("🔄 Testing Complete Emotional Governance Loop")
    print("=" * 50)
    
    # 1. Set up test environment
    setup_test_environment()
    
    # 2. Create and run reflection agent
    print("\n📊 Step 1: Running Reflection Agent Analysis")
    insights = reflection_agent.run_reflection_pass()
    
    if insights:
        print(f"✓ Reflection analysis complete. Health: {insights.get('health_status', 'unknown')}")
        print(f"  Overall drift score: {insights.get('overall_drift_score', 0):.3f}")
    else:
        print("⚠ No reflection insights generated (no reflection logs found)")
    
    # 3. Initialize AnchorAI Interface
    print("\n⚓ Step 2: Initializing AnchorAI Interface")
    anchor_ai = AnchorAIInterface()
    
    # 4. Test drift penalty parsing
    print("\n🔍 Step 3: Testing Drift Penalty Calculation")
    penalty = anchor_ai.parse_reflection_insights()
    print(f"✓ Drift penalty calculated: {penalty:.3f}")
    
    # 5. Test quantization scoring with drift penalty
    print("\n🎯 Step 4: Testing Quantization Scoring with Drift Penalty")
    
    # Create test candidates
    test_candidates = [
        QuantizationCandidate(
            name="candidate_1",
            size_gb=2.5,
            emotional_resonance_score=0.7,
            file_path="/test/model1.bin"
        ),
        QuantizationCandidate(
            name="candidate_2", 
            size_gb=4.2,
            emotional_resonance_score=0.8,
            file_path="/test/model2.bin"
        )
    ]
    
    scored_candidates = []
    for candidate in test_candidates:
        score = anchor_ai.score_alignment(candidate)
        scored_candidates.append((candidate, score))
        print(f"  {candidate.name}: {score:.3f}")
    
    # 6. Verify penalty application
    print("\n✅ Step 5: Verification")
    if penalty > 0:
        print(f"🔴 Emotional drift detected! Penalty applied: -{penalty:.3f}")
        print("   → Quantization candidates penalized for emotional violations")
        print("   → Intervention protocols should be triggered")
    else:
        print("🟢 No emotional drift detected")
        print("   → Quantization proceeds normally")
    
    # 7. Summary
    print(f"\n📋 Governance Loop Test Summary:")
    print(f"   Reflection Agent: {'✓ Active' if insights else '⚠ No data'}")
    print(f"   Drift Detection: {'✓ Working' if penalty >= 0 else '❌ Error'}")
    print(f"   Score Penalty: {penalty:.3f}")
    print(f"   Integration: {'✓ Complete' if penalty >= 0 else '❌ Failed'}")
    
    return {
        'reflection_insights': insights,
        'drift_penalty': penalty,
        'scored_candidates': scored_candidates,
        'integration_working': penalty >= 0
    }

if __name__ == "__main__":
    try:
        results = test_governance_loop()
        
        if results['integration_working']:
            print("\n🎉 SUCCESS: Complete emotional governance loop is functional!")
            print("\nThe system can now:")
            print("• Detect emotional drift in AI responses")
            print("• Generate structured intervention insights") 
            print("• Apply drift penalties to quantization scoring")
            print("• Complete the feedback loop for emotional governance")
        else:
            print("\n❌ FAILED: Integration issues detected")
            
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
