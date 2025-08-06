#!/usr/bin/env python3
"""
CoreArbiter and EmotionallyInfusedChat Demo

This script demonstrates both the CoreArbiter backend system and how it would
integrate with the EmotionallyInfusedChat React component.
"""

import asyncio
import json
import time
from pathlib import Path
from core_arbiter import CoreArbiter, WeightingStrategy

async def demo_complete_system():
    """Demonstrate the complete CoreArbiter system"""
    print("🌟 === CoreArbiter & EmotionallyInfusedChat Demo ===\n")
    
    # Initialize CoreArbiter
    print("1. Initializing CoreArbiter...")
    arbiter = CoreArbiter()
    print(f"   ✅ Initialized with strategy: {arbiter.weighting_strategy.value}")
    print(f"   📊 Health status: {arbiter._calculate_health_status()}")
    
    # Simulate chat conversation scenarios
    conversation_scenarios = [
        {
            "user_input": "Hello! I'm feeling a bit lost today. Can you help me?",
            "ui_context": {
                "current_emotion": "uncertainty",
                "user_state": "seeking_connection",
                "session_context": "new_conversation"
            }
        },
        {
            "user_input": "I'm struggling with a difficult decision about my career.",
            "ui_context": {
                "current_emotion": "anxiety",
                "user_state": "decision_support_needed",
                "session_context": "ongoing_conversation"
            }
        },
        {
            "user_input": "Sometimes I feel like nobody really understands me.",
            "ui_context": {
                "current_emotion": "loneliness",
                "user_state": "emotional_vulnerability",
                "session_context": "deepening_connection"
            }
        },
        {
            "user_input": "Can you analyze this data and give me a logical breakdown?",
            "ui_context": {
                "current_emotion": "analytical",
                "user_state": "task_focused",
                "session_context": "utilitarian_request"
            }
        }
    ]
    
    print("\n2. Simulating Conversation Flow:\n")
    
    for i, scenario in enumerate(conversation_scenarios, 1):
        print(f"   💬 Conversation Turn {i}")
        print(f"   User: {scenario['user_input']}")
        print(f"   Context: {scenario['ui_context']['current_emotion']} | {scenario['ui_context']['user_state']}")
        
        # Process through CoreArbiter
        response = await arbiter.process_input(
            scenario['user_input'], 
            scenario['ui_context']
        )
        
        # Simulate UI mood ring update
        mood_mapping = {
            "emotional": {"color": "#EC4899", "icon": "💖", "intensity": "high"},
            "balanced": {"color": "#8B5CF6", "icon": "💭", "intensity": "medium"},
            "objective": {"color": "#06B6D4", "icon": "🤔", "intensity": "low"}
        }
        
        ui_mood = mood_mapping.get(response.tone, mood_mapping["balanced"])
        
        print(f"   🎭 UI Mood Ring: {ui_mood['icon']} {ui_mood['color']} ({ui_mood['intensity']} intensity)")
        print(f"   🤖 AI Response: {response.final_output}")
        print(f"   📊 Metadata: {response.tone} tone, {response.confidence:.0%} confidence")
        
        if response.reflection:
            print(f"   ✨ Reflection: {response.reflection}")
        
        if response.emotional_override:
            print(f"   🌊 Emotional Override Active")
        
        # Simulate drift notification
        if arbiter.drift_state.stability_score < 0.6:
            print(f"   🌀 Drift Notification: 'I feel myself changing... (stability: {arbiter.drift_state.stability_score:.1%})'")
        
        print(f"   ⚖️  Resolution: {response.resolution_strategy}")
        print()
        
        # Simulate processing delay
        await asyncio.sleep(0.5)
    
    # Demonstrate weighting strategy changes
    print("3. Demonstrating Strategy Changes:\n")
    
    test_input = "How do you balance logic and emotion?"
    test_state = {"context": "philosophical_inquiry"}
    
    strategies = [
        WeightingStrategy.LOGIC_DOMINANT,
        WeightingStrategy.EMOTIONAL_PRIORITY, 
        WeightingStrategy.HARMONIC
    ]
    
    for strategy in strategies:
        print(f"   🎯 Strategy: {strategy.value}")
        arbiter.set_weighting_strategy(strategy)
        
        response = await arbiter.process_input(test_input, test_state)
        print(f"   Response: {response.final_output[:80]}...")
        print(f"   Weights: Logic={response.source_weights['hrm_r']:.1%}, Emotion={response.source_weights['hrm_e']:.1%}")
        print()
    
    # Demonstrate drift and regulation
    print("4. Drift Detection and System Regulation:\n")
    
    # Induce some drift by simulating intensive emotional processing
    for _ in range(5):
        emotional_input = "I need deep emotional support right now."
        emotional_state = {"context": "emotional_crisis", "intensity": 0.9}
        await arbiter.process_input(emotional_input, emotional_state)
    
    print(f"   📉 After intensive processing:")
    print(f"      Emotional drift: {arbiter.drift_state.emotional_drift:.1%}")
    print(f"      Fatigue level: {arbiter.drift_state.fatigue_level:.1%}")
    print(f"      Stability: {arbiter.drift_state.stability_score:.1%}")
    
    # Perform system regulation
    print(f"\n   🔧 Performing system regulation...")
    await arbiter.regulate_system()
    
    print(f"   📈 After regulation:")
    print(f"      Stability: {arbiter.drift_state.stability_score:.1%}")
    print(f"      Fatigue level: {arbiter.drift_state.fatigue_level:.1%}")
    
    # Show system status
    print("\n5. Final System Status:\n")
    status = arbiter.get_system_status()
    
    print(f"   🏥 Health Status: {status['health_status'].upper()}")
    print(f"   🎛️  Current Strategy: {status['weighting_strategy']}")
    print(f"   📊 Decisions Made: {status['decision_count']}")
    print(f"   📈 Stability Score: {status['drift_state']['stability_score']:.1%}")
    
    # Show trace file info
    trace_path = Path("logs/core_arbiter_trace.json")
    if trace_path.exists():
        with open(trace_path, 'r') as f:
            traces = json.load(f)
        print(f"   📝 Trace Entries: {len(traces)}")
        print(f"   💾 Trace File: {trace_path}")

def demo_ui_integration():
    """Demonstrate UI integration concepts"""
    print("\n🎨 === UI Integration Demo ===\n")
    
    print("The EmotionallyInfusedChat React component provides:")
    print("   💭 Real-time mood ring that reflects AI emotional state")
    print("   🌈 Dynamic message styling based on emotional tone")
    print("   🔄 Ambient visual effects matching current mood")
    print("   🌀 Drift notifications when emotional state shifts")
    print("   📱 Responsive mobile-friendly design")
    print("   ⚙️  Sidebar with emotional controls and settings")
    
    print("\nKey Features:")
    print("   🎭 fetchMoodProfile() - Updates mood ring from API")
    print("   📝 logMessageWithEmotion() - Logs with emotional context")
    print("   ✨ triggerSymbolicResponse() - Activates ritual/symbolic mode")
    print("   🌀 openDriftPanel() - Shows drift analysis interface")
    
    print("\nVisual Design:")
    print("   🎨 TailwindCSS with emotional color palettes")
    print("   🌊 Gradient backgrounds based on mood")
    print("   ✨ Smooth animations and transitions")
    print("   📊 Metadata tooltips for transparency")
    
    print("\nAPI Integration:")
    print("   🔗 RESTful API calls to CoreArbiter")
    print("   📡 WebSocket support for real-time updates")
    print("   💾 Local state management with React hooks")
    print("   🔄 Automatic retry and error handling")

def demo_technical_architecture():
    """Show technical architecture overview"""
    print("\n🏗️  === Technical Architecture ===\n")
    
    print("CoreArbiter (Backend):")
    print("   🧠 HRM_R: <10GB Reasoning Model")
    print("   💖 HRM_E: <10GB Emotional Model") 
    print("   ⚖️  Core Arbiter: <24GB Decision Layer")
    print("   📊 Total Budget: ~44GB VRAM")
    
    print("\nDecision Flow:")
    print("   1️⃣  Parallel processing of user input")
    print("   2️⃣  Conflict evaluation and resolution")
    print("   3️⃣  Weight adjustment based on drift")
    print("   4️⃣  Identity tether and safety checks")
    print("   5️⃣  Response fusion and generation")
    print("   6️⃣  Metadata and trace logging")
    
    print("\nUI Components:")
    print("   💬 EmotionallyInfusedChat (Main Interface)")
    print("   🎭 MoodRing (Emotional State Indicator)")
    print("   🌀 DriftNotification (System Health)")
    print("   📱 Sidebar (Controls and Settings)")
    print("   📊 MessageMetadata (Transparency)")
    
    print("\nData Flow:")
    print("   User Input → CoreArbiter → Dual Model Processing")
    print("   → Conflict Resolution → Response Generation")
    print("   → UI Update → Mood Ring → Visual Effects")

async def main():
    """Main demo function"""
    await demo_complete_system()
    demo_ui_integration()
    demo_technical_architecture()
    
    print("\n🎯 === Next Steps ===\n")
    print("1. Integrate actual HRM_R and HRM_E models")
    print("2. Deploy EmotionallyInfusedChat React component")
    print("3. Set up WebSocket connections for real-time updates")
    print("4. Implement drift monitoring dashboard")
    print("5. Add biometric integration for enhanced emotional detection")
    print("6. Create ritual/symbolic response generation system")
    print("7. Build memory and anchor management interfaces")
    
    print(f"\n✨ Demo completed! Check 'logs/core_arbiter_trace.json' for decision traces.")

if __name__ == "__main__":
    asyncio.run(main())
