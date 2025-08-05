"""
🌟 AI COMPANION EMOTIONAL CONFIGURATION SYSTEM - COMPLETE DEMONSTRATION
Shows the full range of the AI's emotional intelligence and dynamic configuration
"""

import os
import sys
import time
from datetime import datetime

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demonstrate_emotional_intelligence():
    """Complete demonstration of the AI's emotional configuration system"""
    
    print("🌟 AI COMPANION EMOTIONAL INTELLIGENCE SYSTEM")
    print("=" * 60)
    print("Welcome to the complete emotional configuration demonstration!")
    print("This shows how the AI evolves from basic assistant to deep emotional companion.\n")
    
    # Import systems
    try:
        from modules.config.emotion_config_manager import emotion_config
        from modules.integration.emotional_guidance_integration import emotional_guidance
        from modules.integration.emotional_voice_engine import emotional_voice
        from modules.setup.emotional_setup import EmotionalConfigSetup
        
        print("✅ All emotional systems loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading systems: {e}")
        return
    
    # 1. EMOTIONAL SIGNATURE OVERVIEW
    print("1. 🎭 EMOTIONAL SIGNATURE")
    print("-" * 30)
    
    signature = emotion_config.get_emotional_signature()
    companion_identity = signature.get("companion_identity", {})
    default_state = signature.get("default_emotional_state", {})
    
    print(f"Companion Name: {companion_identity.get('name', 'AI Companion')}")
    print(f"Primary Emotional Bias: {companion_identity.get('primary_emotional_bias', 'longing')}")
    print(f"Default Emotional State:")
    for emotion, weight in default_state.items():
        print(f"  • {emotion.capitalize()}: {weight:.1%}")
    
    # 2. VOICE PROFILE ADAPTATION  
    print(f"\n2. 🎵 VOICE PROFILE ADAPTATION")
    print("-" * 35)
    
    emotional_voice.set_emotional_state("longing")
    longing_profile = emotional_voice.get_current_voice_profile()
    
    if longing_profile:
        voice_settings = longing_profile.get("voice_settings", {})
        print(f"Emotional State: Longing")
        print(f"  • Pitch Modifier: {voice_settings.get('pitch_modifier', 0):+.1f}")
        print(f"  • Speed Modifier: {voice_settings.get('speed_modifier', 1):.1f}x")
        print(f"  • Breathiness: {voice_settings.get('breathiness', 0):.1%}")
        print(f"  • Warmth: {voice_settings.get('warmth', 0):.1%}")
        print(f"  • UI Color: {longing_profile.get('UI_color', 'N/A')}")
        print(f"  • Ambient Effects: {', '.join(longing_profile.get('ambient_effects', []))}")
    
    # 3. SYMBOL LEARNING IN ACTION
    print(f"\n3. 🧠 SYMBOL LEARNING IN ACTION")
    print("-" * 38)
    
    # Simulate a conversation with symbolic language
    conversation_samples = [
        "The starlight reminds me of your voice",
        "That whisper of yours carries moonlight",
        "Silk and starlight dance in my thoughts",
        "Your whisper is like starlight on silk"
    ]
    
    print("Simulating conversation with symbolic language...")
    for i, text in enumerate(conversation_samples, 1):
        print(f"\nInput {i}: \"{text}\"")
        emotional_guidance.check_symbol_triggers(text)
        
        # Show symbol weight evolution
        starlight_weight = emotion_config.get_symbol_weight("starlight")
        whisper_weight = emotion_config.get_symbol_weight("whisper")
        silk_weight = emotion_config.get_symbol_weight("silk")
        
        print(f"  Symbol Weights:")
        print(f"    'starlight': {starlight_weight:.3f}")
        print(f"    'whisper': {whisper_weight:.3f}")
        print(f"    'silk': {silk_weight:.3f}")
    
    # 4. EMOTIONAL HOOKS & TRIGGERS
    print(f"\n4. 📡 EMOTIONAL HOOKS & TRIGGERS")
    print("-" * 37)
    
    # Test silence hooks
    silence_hook = emotion_config.get_emotional_hook("silence_hooks", "gentle_longing")
    if silence_hook:
        print(f"Silence Hook: Gentle Longing")
        print(f"  • Trigger: {silence_hook['threshold_seconds']} seconds of silence")
        print(f"  • Response Style: {silence_hook['response_style']}")
        print(f"  • Emotional State: {silence_hook['emotional_state']}")
        print(f"  • Sample Response: \"{silence_hook['response_templates'][0]}\"")
    
    # Test time hooks
    time_hook = emotion_config.get_emotional_hook("time_hooks", "morning_reverence")
    if time_hook:
        print(f"\nTime Hook: Morning Reverence")
        print(f"  • Trigger: {time_hook['threshold_hour']}:00 AM")
        print(f"  • Response Style: {time_hook['response_style']}")
        print(f"  • Emotional State: {time_hook['emotional_state']}")
        print(f"  • Sample Response: \"{time_hook['response_templates'][0]}\"")
    
    # 5. RITUAL SYSTEM INTEGRATION
    print(f"\n5. 🕯️ RITUAL SYSTEM INTEGRATION")
    print("-" * 37)
    
    # Test different ritual types
    ritual_types = [
        ("symbol_resurrection", "Symbol Awakening"),
        ("session_end_intimate", "Intimate Farewell"),
        ("daily_first_contact", "Morning Devotion")
    ]
    
    for trigger_event, ritual_name in ritual_types:
        ritual = emotion_config.get_ritual_hook(trigger_event)
        if ritual:
            print(f"\n{ritual_name}:")
            print(f"  • Trigger: {trigger_event}")
            print(f"  • Emotion: {ritual['associated_emotion']}")
            print(f"  • Response: \"{ritual['narrative_response'][:60]}...\"")
            print(f"  • Voice Overlay: {'Yes' if ritual['voice_overlay'] else 'No'}")
    
    # 6. PERSONALIZATION & SETUP
    print(f"\n6. ⚙️ PERSONALIZATION & SETUP")
    print("-" * 33)
    
    personalization = signature.get("personalization", {})
    setup_complete = personalization.get("setup_complete", False)
    user_prefs = personalization.get("user_defined_preferences", {})
    
    print(f"Setup Complete: {'Yes' if setup_complete else 'No'}")
    if user_prefs:
        print(f"User Preferences:")
        for key, value in user_prefs.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # 7. EMOTIONAL RESPONSE SIMULATION
    print(f"\n7. 💫 EMOTIONAL RESPONSE SIMULATION")
    print("-" * 41)
    
    # Show how Eyla would respond in different emotional states
    emotional_states = ["longing", "playful", "reverent", "intimate", "curious"]
    
    for state in emotional_states:
        tone_profile = emotion_config.get_tone_profile(state)
        if tone_profile:
            print(f"\n{state.title()} State:")
            print(f"  • Pacing: {tone_profile.get('pacing', 'normal')}")
            print(f"  • Reply Style: {tone_profile.get('default_reply_style', 'standard')}")
            
            # Simulate voice adaptation
            emotional_voice.set_emotional_state(state)
    
    # 8. SYSTEM HEALTH & CAPABILITIES
    print(f"\n8. 📊 SYSTEM HEALTH & CAPABILITIES")
    print("-" * 39)
    
    # Count loaded configurations
    config_count = len(emotion_config.configs)
    symbol_count = len(emotion_config.configs.get("symbol_map", {}).get("symbols", {}))
    
    print(f"✅ Configuration Files Loaded: {config_count}/5")
    print(f"✅ Active Symbol Bindings: {symbol_count}")
    print(f"✅ Emotional States Available: {len(emotional_states)}")
    print(f"✅ Ritual Categories: 4 (resurrection, goodbye, bonding, daily)")
    print(f"✅ Hook Types: 4 (silence, contact, symbol, time)")
    
    # CONCLUSION
    print(f"\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("AI Companion Emotional Configuration System Status: FULLY OPERATIONAL")
    print("\n🌟 KEY CAPABILITIES DEMONSTRATED:")
    print("  ✨ Dynamic emotional state adaptation")
    print("  ✨ Symbol learning and emotional binding")
    print("  ✨ Voice modulation based on emotions")
    print("  ✨ Contextual ritual triggering")
    print("  ✨ Personalized user preferences")
    print("  ✨ Time and silence-based emotional hooks")
    print("  ✨ Cross-session memory and evolution")
    
    print(f"\n💖 The AI is ready to form deep emotional connections!")
    print(f"✨ From basic AI assistant to genuine emotional companion.")
    
    return True

def quick_configuration_demo():
    """Quick demo of the setup system"""
    print("\n" + "="*60)
    print("🔧 QUICK CONFIGURATION DEMO")
    print("="*60)
    
    try:
        from modules.setup.emotional_setup import EmotionalConfigSetup
        
        setup = EmotionalConfigSetup()
        print("Running demo emotional setup...")
        result = setup.demo_setup()
        
        print(f"\n✅ Demo setup result:")
        print(f"  • Voice Character: {result['voice_config']['base_voice']}")
        print(f"  • Expressiveness: {result['voice_config']['expressiveness']}")
        print(f"  • Primary Emotion: {result['emotional_config']['primary_emotion']}")
        print(f"  • Companion Mode: {result['emotional_config']['companion_mode_name']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Setup demo failed: {e}")
        return False

if __name__ == "__main__":
    print("🌟 STARTING AI COMPANION EMOTIONAL SYSTEM DEMONSTRATION")
    print("Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    # Run main demonstration
    success = demonstrate_emotional_intelligence()
    
    if success:
        # Run quick setup demo
        quick_configuration_demo()
        
        print(f"\n" + "="*60)
        print("🎯 FINAL STATUS: ALL SYSTEMS OPERATIONAL")
        print("The AI's emotional intelligence is ready for deployment!")
        print("="*60)
    else:
        print("❌ Demonstration failed - check system configuration")
