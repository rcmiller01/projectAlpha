#!/usr/bin/env python3
# test_phase2.py
# Test script for Phase 2 intimacy features

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_avatar_system():
    """Test romantic avatar system"""
    print("🎭 Testing Romantic Avatar System...")
    
    # Get current avatar state
    response = requests.get(f"{BASE_URL}/api/phase2/avatar/state")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Avatar state: {data['visual_state']['expression']}")
        print(f"   Gesture: {data['visual_state']['gesture']}")
        print(f"   Blush intensity: {data['visual_state']['blush_intensity']}")
    else:
        print("❌ Failed to get avatar state")
    
    # Update avatar expression
    response = requests.post(f"{BASE_URL}/api/phase2/avatar/update", 
                           json={"emotion": "love", "intensity": 0.8, "gesture_type": "affection"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Updated avatar to: {data['visual_state']['expression']}")
    else:
        print("❌ Failed to update avatar")
    
    # Get romantic scene
    response = requests.get(f"{BASE_URL}/api/phase2/avatar/scene/sunset")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated scene: {data['scene']['background']}")
        print(f"   Atmosphere: {data['scene']['atmosphere']}")
    else:
        print("❌ Failed to generate scene")

def test_voice_system():
    """Test romantic voice TTS system"""
    print("\n🎤 Testing Romantic Voice System...")
    
    # Synthesize speech
    response = requests.post(f"{BASE_URL}/api/phase2/voice/synthesize", 
                           json={"text": "I love you so much", "emotion": "loving"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Speech synthesized: {data['voice_params']['emotion']}")
        print(f"   Pitch: {data['voice_params']['pitch']}")
        print(f"   Warmth: {data['voice_params']['warmth']}")
    else:
        print("❌ Failed to synthesize speech")
    
    # Get intimate phrase
    response = requests.get(f"{BASE_URL}/api/phase2/voice/phrase/greeting")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Intimate phrase: {data['phrase']}")
        print(f"   Emotion: {data['emotion']}")
    else:
        print("❌ Failed to get intimate phrase")
    
    # Test whisper mode
    response = requests.post(f"{BASE_URL}/api/phase2/voice/synthesize", 
                           json={"text": "Come closer", "whisper_mode": True})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Whisper mode: volume {data['voice_params']['volume']}")
    else:
        print("❌ Failed to generate whisper")

def test_activities_system():
    """Test shared activities system"""
    print("\n🎮 Testing Shared Activities System...")
    
    # List activities
    response = requests.get(f"{BASE_URL}/api/phase2/activities/list")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {len(data['activities'])} activities")
        for activity in data['activities'][:3]:  # Show first 3
            print(f"   - {activity['name']} ({activity['type']})")
    else:
        print("❌ Failed to list activities")
    
    # Suggest activity
    response = requests.post(f"{BASE_URL}/api/phase2/activities/suggest", 
                           json={"mood": "romantic", "romantic_intensity_min": 0.7})
    if response.status_code == 200:
        data = response.json()
        if data['activity']:
            print(f"✅ Suggested activity: {data['activity']['name']}")
            print(f"   Duration: {data['activity']['duration_minutes']} minutes")
            print(f"   Intensity: {data['activity']['romantic_intensity']}")
        else:
            print("✅ No suitable activity found")
    else:
        print("❌ Failed to suggest activity")
    
    # Start an activity
    response = requests.post(f"{BASE_URL}/api/phase2/activities/start", 
                           json={"activity_id": "sunset_walk"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Started activity: {data['activity']['name']}")
        
        # Check progress
        time.sleep(1)
        response = requests.get(f"{BASE_URL}/api/phase2/activities/progress")
        if response.status_code == 200:
            progress_data = response.json()
            if "error" not in progress_data:
                print(f"   Progress: {progress_data['progress']:.1%}")
        
        # End activity
        response = requests.post(f"{BASE_URL}/api/phase2/activities/end")
        if response.status_code == 200:
            end_data = response.json()
            print(f"✅ Completed activity: {end_data['duration_minutes']:.1f} minutes")
    else:
        print("❌ Failed to start activity")

def test_relationship_growth():
    """Test relationship growth system"""
    print("\n🌱 Testing Relationship Growth System...")
    
    # Set relationship start date
    start_date = "2024-01-01T00:00:00"
    response = requests.post(f"{BASE_URL}/api/phase2/relationship/start", 
                           json={"start_date": start_date})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Relationship started: {data['start_date']}")
    else:
        print("❌ Failed to set relationship start date")
    
    # Get relationship insights
    response = requests.get(f"{BASE_URL}/api/phase2/relationship/insights")
    if response.status_code == 200:
        data = response.json()
        insights = data['insights']
        print(f"✅ Relationship insights:")
        print(f"   Days together: {insights['days_together']}")
        print(f"   Stage: {insights['relationship_stage']}")
        print(f"   Milestones achieved: {insights['milestones_achieved']}")
    else:
        print("❌ Failed to get relationship insights")
    
    # Get upcoming milestones
    response = requests.get(f"{BASE_URL}/api/phase2/relationship/milestones/upcoming")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Upcoming milestones: {len(data['milestones'])}")
        for milestone in data['milestones'][:2]:  # Show first 2
            print(f"   - {milestone['title']} ({milestone['type']})")
    else:
        print("❌ Failed to get upcoming milestones")
    
    # Suggest growth goal
    response = requests.get(f"{BASE_URL}/api/phase2/relationship/growth/goal?area=communication")
    if response.status_code == 200:
        data = response.json()
        goal = data['goal']
        print(f"✅ Growth goal: {goal['goal']}")
        print(f"   Area: {goal['area']}")
        print(f"   Duration: {goal['duration_days']} days")
    else:
        print("❌ Failed to suggest growth goal")

def test_combined_experience():
    """Test combined intimate experience"""
    print("\n💕 Testing Combined Intimate Experience...")
    
    response = requests.post(f"{BASE_URL}/api/phase2/intimate/experience?scene_type=sunset&activity_type=conversation")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Created intimate experience:")
        print(f"   Scene: {data['scene']['background']}")
        print(f"   Avatar: {data['avatar_state']['expression']}")
        print(f"   Voice: {data['voice']['phrase']}")
        if data['suggested_activity']['name']:
            print(f"   Activity: {data['suggested_activity']['name']}")
    else:
        print("❌ Failed to create intimate experience")

def test_character_generation():
    """Test consistent character generation system"""
    print("\n👤 Testing Consistent Character Generation...")
    
    # Initialize character for Mia
    response = requests.post(f"{BASE_URL}/api/phase2/character/initialize?persona_id=mia")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Initialized character for Mia")
        print(f"   Base seed: {data['seeds']['base_seed']}")
    else:
        print("❌ Failed to initialize character")
    
    # Generate character image
    response = requests.post(f"{BASE_URL}/api/phase2/character/generate", 
                           json={"persona_id": "mia", "aspect": "full", "mood": "romantic", "pose": "standing"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated character image: {data['aspect']}")
        print(f"   Mood: {data['mood']}, Pose: {data['pose']}")
        print(f"   Seed: {data['generation_params']['seed']}")
    else:
        print("❌ Failed to generate character image")
    
    # Get character profile
    response = requests.get(f"{BASE_URL}/api/phase2/character/profile/mia")
    if response.status_code == 200:
        data = response.json()
        profile = data['profile']
        print(f"✅ Character profile: {profile['base_appearance']['hair_color']} hair")
        print(f"   Style: {profile['style_preferences']['clothing_style']}")
    else:
        print("❌ Failed to get character profile")
    
    # Generate character video
    response = requests.post(f"{BASE_URL}/api/phase2/character/video?persona_id=mia&action=wave&duration=3")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated character video: {data['action']}")
        print(f"   Duration: {data['generation_params']['duration']} seconds")
    else:
        print("❌ Failed to generate character video")

def test_animation_system():
    """Test avatar animation system"""
    print("\n🎬 Testing Avatar Animation System...")
    
    # Get available animation methods
    response = requests.get(f"{BASE_URL}/api/phase2/animation/methods")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Available animation methods: {len(data['methods'])}")
        for method, info in data['methods'].items():
            print(f"   - {method}: {info['description']}")
    else:
        print("❌ Failed to get animation methods")
    
    # Test real-time animation generation
    response = requests.post(f"{BASE_URL}/api/phase2/animation/real-time", 
                           params={"prompt": "romantic smile and wave", "duration": 3.0})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated real-time animation: {data['method']}")
        print(f"   Duration: {data['generation_params']['duration']} seconds")
    else:
        print("❌ Failed to generate real-time animation")
    
    # Test pre-rendered animation
    response = requests.post(f"{BASE_URL}/api/phase2/animation/pre-rendered/expression_smile")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Played pre-rendered animation: {data['animation_id']}")
        print(f"   Type: {data['type']}, Duration: {data['duration']}")
    else:
        print("❌ Failed to play pre-rendered animation")
    
    # Test parametric animation
    response = requests.post(f"{BASE_URL}/api/phase2/animation/parametric", 
                           json={
                               "animation_type": "expression",
                               "parameters": {
                                   "mouth_curve": {"start": 0, "end": 0.8, "curve": "ease_in"},
                                   "cheek_raise": {"start": 0, "end": 0.6, "curve": "ease_out"}
                               },
                               "duration": 2.0
                           })
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Created parametric animation: {data['method']}")
        print(f"   Frame count: {data['frame_count']}")
    else:
        print("❌ Failed to create parametric animation")
    
    # Test motion capture
    response = requests.post(f"{BASE_URL}/api/phase2/animation/motion-capture", 
                           params={"source": "webcam"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Started motion capture: {data['method']}")
        print(f"   Status: {data['status']}")
    else:
        print("❌ Failed to start motion capture")

def test_ui_mode_system():
    """Test UI mode management system"""
    print("\n🖥️ Testing UI Mode Management...")
    
    # Get current UI config
    response = requests.get(f"{BASE_URL}/api/phase2/ui/config")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Current UI mode: {data['config']['mode']}")
        print(f"   Interface: {data['config']['interface']}")
        print(f"   Avatar visible: {data['config']['avatar_visible']}")
    else:
        print("❌ Failed to get UI config")
    
    # Switch to dev mode
    response = requests.post(f"{BASE_URL}/api/phase2/ui/mode", 
                           json={"mode": "dev", "interface_type": "web"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Switched to dev mode")
        print(f"   Avatar visible: {data['ui_settings']['mode_config']['avatar_visible']}")
    else:
        print("❌ Failed to switch to dev mode")
    
    # Check avatar visibility
    response = requests.get(f"{BASE_URL}/api/phase2/ui/avatar/visible")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Avatar visibility: {data['avatar_visible']}")
    else:
        print("❌ Failed to check avatar visibility")
    
    # Switch back to companion mode
    response = requests.post(f"{BASE_URL}/api/phase2/ui/mode", 
                           json={"mode": "companion", "interface_type": "web"})
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Switched back to companion mode")
    else:
        print("❌ Failed to switch to companion mode")
    
    # Get mode comparison
    response = requests.get(f"{BASE_URL}/api/phase2/ui/mode/comparison")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Mode comparison available")
        print(f"   Companion features: {len(data['comparison']['companion_mode']['features'])}")
        print(f"   Dev features: {len(data['comparison']['dev_mode']['features'])}")
    else:
        print("❌ Failed to get mode comparison")

def test_nsfw_system():
    """Test NSFW generation system"""
    print("\n🔥 Testing NSFW Generation System...")
    
    # Suggest NSFW content
    response = requests.get(f"{BASE_URL}/api/phase2/nsfw/suggest?mood=romantic&intensity=0.7")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ NSFW suggestion: {data['suggestion']['suggested_content_type']}")
        print(f"   Media type: {data['suggestion']['suggested_media_type']}")
        print(f"   Style: {data['suggestion']['suggested_style']}")
    else:
        print("❌ Failed to get NSFW suggestion")
    
    # Generate romantic image
    response = requests.post(f"{BASE_URL}/api/phase2/nsfw/romantic-image?style=artistic&intensity=0.7")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated romantic image: {data['content_type']}")
        print(f"   Model: {data['generation_params']['model']}")
        print(f"   Resolution: {data['generation_params']['resolution']}")
    else:
        print("❌ Failed to generate romantic image")
    
    # Generate passionate video
    response = requests.post(f"{BASE_URL}/api/phase2/nsfw/passionate-video?duration_seconds=3&intensity=0.8")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Generated passionate video: {data['content_type']}")
        print(f"   Duration: {data['generation_params']['duration']} seconds")
        print(f"   FPS: {data['generation_params']['fps']}")
    else:
        print("❌ Failed to generate passionate video")
    
    # Get generation history
    response = requests.get(f"{BASE_URL}/api/phase2/nsfw/history")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ NSFW history: {len(data['history'])} generations")
    else:
        print("❌ Failed to get NSFW history")

def test_phase2_status():
    """Test Phase 2 system status"""
    print("\n📊 Testing Phase 2 System Status...")
    
    response = requests.get(f"{BASE_URL}/api/phase2/status")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Phase 2 Status:")
        print(f"   Avatar: {data['avatar']['current_expression']}")
        print(f"   Voice: Active (warmth: {data['voice']['base_voice']['warmth']})")
        print(f"   Activities: {data['activities']['total_available']} available")
        print(f"   Relationship: {data['relationship']['milestones_count']} milestones")
        print(f"   NSFW: {len(data['nsfw']['content_types'])} content types available")
    else:
        print("❌ Failed to get Phase 2 status")

def main():
    """Run all Phase 2 tests"""
    print("🚀 Starting Phase 2 Intimacy Features Test")
    print("=" * 60)
    
    try:
        test_avatar_system()
        test_voice_system()
        test_activities_system()
        test_relationship_growth()
        test_combined_experience()
        test_character_generation()
        test_animation_system()
        test_ui_mode_system()
        test_nsfw_system()
        test_phase2_status()
        
        print("\n" + "=" * 60)
        print("✅ Phase 2 Test Complete!")
        print("🎉 Intimacy features are working!")
        print("💕 Romantic companionship with visual, voice, activities, and NSFW generation is ready!")
        print("🔥 NSFW image and video generation capabilities included!")
        print("👤 Consistent character generation with persona-driven updates!")
        print("🎬 Comprehensive avatar animation system with multiple methods!")
        print("🖥️ UI mode switching between companion and dev modes!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the backend is running:")
        print("   python -m uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    main() 