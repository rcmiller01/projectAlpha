#!/usr/bin/env python3
# test_phase1.py
# Test script for Phase 1 romantic features

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_emotion_detection():
    """Test romantic emotion detection"""
    print("🧪 Testing Romantic Emotion Detection...")
    
    test_messages = [
        "I love you so much",
        "I miss you terribly",
        "You make me feel so passionate",
        "I feel so safe with you",
        "You're so tender and caring"
    ]
    
    for message in test_messages:
        response = requests.post(f"{BASE_URL}/emotion/from_text", 
                               json={"text": message})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ '{message}' -> {data['primary_mood']}")
            if 'romantic_context' in data:
                print(f"   Romantic context: {data['romantic_context']}")
        else:
            print(f"❌ Failed to process: {message}")

def test_romantic_interaction():
    """Test romantic interaction API"""
    print("\n💕 Testing Romantic Interactions...")
    
    test_interactions = [
        {
            "message": "I love you with all my heart",
            "interaction_type": "conversation",
            "intensity": 0.9
        },
        {
            "message": "I miss you so much",
            "interaction_type": "conversation", 
            "intensity": 0.8
        },
        {
            "message": "Your touch is magical",
            "interaction_type": "touch",
            "intensity": 0.9
        }
    ]
    
    for interaction in test_interactions:
        response = requests.post(f"{BASE_URL}/api/romantic/interact", 
                               json=interaction)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {interaction['message']}")
            print(f"   Mia's response: {data['mia_response']}")
            print(f"   Relationship stage: {data['relationship_stage']}")
        else:
            print(f"❌ Failed interaction: {interaction['message']}")

def test_mia_thoughts():
    """Test Mia's self-talk generation"""
    print("\n🤔 Testing Mia's Thoughts...")
    
    response = requests.get(f"{BASE_URL}/api/romantic/mia/thoughts")
    if response.status_code == 200:
        data = response.json()
        if data['thought']:
            print(f"✅ Mia's thought: {data['thought']['thought']}")
            print(f"   Emotion: {data['thought']['emotion']}")
            print(f"   Delivery mode: {data['thought']['delivery_mode']}")
        if data['memory']:
            print(f"   Memory: {data['memory']}")
    else:
        print("❌ Failed to get Mia's thoughts")

def test_relationship_status():
    """Test relationship status API"""
    print("\n📊 Testing Relationship Status...")
    
    response = requests.get(f"{BASE_URL}/api/romantic/status")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Relationship stage: {data['relationship_stage']}")
        print(f"   Interaction count: {data['interaction_count']}")
        print(f"   Dominant emotion: {data['dominant_romantic_emotion']}")
        print(f"   Romantic intensity: {data['romantic_intensity']}")
        print(f"   Milestones: {data['milestones_count']}")
        print(f"   Shared memories: {data['shared_memories_count']}")
    else:
        print("❌ Failed to get relationship status")

def test_memory_system():
    """Test memory system"""
    print("\n🧠 Testing Memory System...")
    
    # Add a milestone
    milestone = {
        "milestone": "First romantic conversation",
        "emotion": "love"
    }
    response = requests.post(f"{BASE_URL}/api/romantic/milestone", json=milestone)
    if response.status_code == 200:
        print("✅ Milestone added successfully")
    
    # Get memories
    response = requests.get(f"{BASE_URL}/api/romantic/memories")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Retrieved {len(data['memories'])} memories")
        for memory in data['memories'][:3]:  # Show first 3
            print(f"   - {memory['description']}")
    else:
        print("❌ Failed to get memories")

def main():
    """Run all Phase 1 tests"""
    print("🚀 Starting Phase 1 Romantic Features Test")
    print("=" * 50)
    
    try:
        test_emotion_detection()
        test_romantic_interaction()
        test_mia_thoughts()
        test_relationship_status()
        test_memory_system()
        
        print("\n" + "=" * 50)
        print("✅ Phase 1 Test Complete!")
        print("🎉 Romantic companionship features are working!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the backend is running:")
        print("   python -m uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    main() 