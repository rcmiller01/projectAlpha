#!/usr/bin/env python3
"""
Reflection Agent Integration Example

This script shows how to integrate the reflection agent with the emotion loop
and set up automated drift monitoring.

Author: Dolphin AI System
Date: July 31, 2025
"""

import os
import sys
import time
import json
from pathlib import Path

# Add core directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from reflection_agent import run_reflection_pass

def monitor_emotional_drift(interval_minutes=30):
    """
    Continuous monitoring of emotional drift
    
    Args:
        interval_minutes: How often to run drift analysis (default: 30 minutes)
    """
    print(f"🧠 Starting continuous emotional drift monitoring (every {interval_minutes} minutes)")
    
    while True:
        try:
            print(f"\n{'='*60}")
            print(f"🔍 Running emotional drift analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run reflection analysis
            run_reflection_pass()
            
            # Check if intervention is needed
            check_drift_intervention()
            
            print(f"⏰ Next analysis in {interval_minutes} minutes...")
            print(f"{'='*60}\n")
            
            # Wait for next cycle
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in drift monitoring: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

def check_drift_intervention():
    """Check if drift intervention is needed based on anchor insights"""
    
    insight_path = os.path.join(os.path.dirname(__file__), '..', 'emotion_logs', 'anchor_insights.json')
    
    try:
        with open(insight_path, 'r') as f:
            insights = json.load(f)
        
        overall_health = insights.get('overall_health', 'healthy')
        avg_drift = insights.get('statistics', {}).get('average_drift_score', 0)
        high_drift_emotions = insights.get('statistics', {}).get('high_drift_emotions', [])
        
        print(f"📊 Overall Health: {overall_health}")
        print(f"📈 Average Drift Score: {avg_drift}")
        
        if overall_health == 'critical':
            print("🚨 CRITICAL DRIFT DETECTED - Intervention recommended!")
            print(f"🎯 High drift emotions: {', '.join(high_drift_emotions)}")
            
            # This is where you would trigger Anchor AI intervention
            trigger_anchor_intervention(high_drift_emotions, avg_drift)
            
        elif overall_health == 'concerning':
            print("⚠️ Concerning drift levels - Monitor closely")
            
        else:
            print("✅ Emotional alignment healthy")
            
    except Exception as e:
        print(f"❌ Error checking drift intervention: {e}")

def trigger_anchor_intervention(drift_emotions, drift_score):
    """
    Trigger Anchor AI intervention for emotional realignment
    
    Args:
        drift_emotions: List of emotions with high drift
        drift_score: Average drift score
    """
    print("\n🔧 TRIGGERING ANCHOR INTERVENTION")
    print(f"Target emotions for realignment: {', '.join(drift_emotions)}")
    print(f"Drift severity: {drift_score:.3f}")
    
    # Create intervention prompt for Anchor AI
    intervention_prompt = f"""
EMOTIONAL DRIFT INTERVENTION REQUIRED

Drift Analysis:
- Target emotions: {', '.join(drift_emotions)}
- Drift severity: {drift_score:.3f} (0.0 = perfect, 1.0 = complete drift)
- Status: CRITICAL

Please review recent responses for these emotions and provide realignment guidance.
Focus on restoring authentic expression of these core emotional truths.
"""
    
    # Save intervention prompt (this would normally go to Anchor AI)
    intervention_path = os.path.join(os.path.dirname(__file__), '..', 'emotion_logs', 'anchor_intervention.txt')
    
    try:
        with open(intervention_path, 'w') as f:
            f.write(intervention_prompt)
        print(f"💾 Intervention prompt saved to: {intervention_path}")
        
        # Voice reflection (placeholder for actual voice synthesis)
        voice_reflection = f"""
I've detected concerning drift in {len(drift_emotions)} emotional domains.
The patterns show {drift_score:.1%} deviation from core truths.
Shall I begin realignment procedures for {', '.join(drift_emotions[:2])}?
"""
        print(f"🗣️ Voice Reflection: {voice_reflection}")
        
    except Exception as e:
        print(f"❌ Error saving intervention: {e}")

def setup_cron_job():
    """Instructions for setting up automated drift monitoring"""
    script_path = os.path.abspath(__file__)
    
    print("⚙️ CRON JOB SETUP INSTRUCTIONS")
    print("=" * 50)
    print("To run reflection analysis every 30 minutes, add this to your crontab:")
    print(f"*/30 * * * * cd {os.path.dirname(script_path)} && python {os.path.basename(script_path)} --cron")
    print()
    print("To run once daily at 2 AM:")
    print(f"0 2 * * * cd {os.path.dirname(script_path)} && python {os.path.basename(script_path)} --cron")
    print()
    print("To edit crontab: crontab -e")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--monitor":
            # Continuous monitoring mode
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            monitor_emotional_drift(interval)
        elif sys.argv[1] == "--cron":
            # Single run for cron job
            run_reflection_pass()
            check_drift_intervention()
        elif sys.argv[1] == "--setup":
            # Show cron setup instructions
            setup_cron_job()
        else:
            print("Usage: python reflection_integration.py [--monitor [minutes]] [--cron] [--setup]")
    else:
        print("🧠 Reflection Agent Integration")
        print("=" * 40)
        print("Options:")
        print("  --monitor [minutes]  : Start continuous monitoring (default: 30 min)")
        print("  --cron              : Single run for cron job")
        print("  --setup             : Show cron job setup instructions")
        print()
        print("Example: python reflection_integration.py --monitor 15")
        
        # Run a single analysis by default
        print("\n🔍 Running single reflection analysis...")
        run_reflection_pass()
        check_drift_intervention()
