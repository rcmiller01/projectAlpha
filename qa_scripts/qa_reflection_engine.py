#!/usr/bin/env python3
"""
🔄 Reflection Engine Targeted QA Script
Tests reflection engine context building, insight generation, and tone influence
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class ReflectionEngineQA:
    """Targeted testing for the Reflection Engine"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"reflection_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_initial_state(self):
        """Test initial reflection engine state"""
        print("🔍 Testing initial reflection engine state...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/reflection/summary") as response:
                if response.status == 200:
                    data = await response.json()
                    initial_count = data.get('total_reflections', 0)
                    print(f"✅ Initial reflection count: {initial_count}")
                    return initial_count
                else:
                    print(f"❌ Could not get initial state: {response.status}")
                    return 0
        except Exception as e:
            print(f"❌ Error getting initial state: {e}")
            return 0
    
    async def enable_reflection_engine(self):
        """Enable the reflection engine for testing"""
        print("🔄 Enabling reflection engine...")
        
        try:
            async with self.session.post(f"{self.base_url}/api/reflection/enable") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Reflection engine enabled: {data.get('status')}")
                    return True
                else:
                    print(f"❌ Failed to enable reflection engine: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error enabling reflection engine: {e}")
            return False
    
    async def build_conversation_context(self):
        """Build rich conversation context for reflection analysis"""
        print("💬 Building conversation context for reflection...")
        
        # Simulate a meaningful conversation that should trigger reflection
        conversation_flow = [
            {
                "message": "I've been feeling really stressed about my work lately. I'm working on a big project but I'm not sure if I'm making the right decisions.",
                "expected_sentiment": "negative"
            },
            {
                "message": "The project is about implementing an AI system for our company. It's exciting but also overwhelming.",
                "expected_sentiment": "mixed"
            },
            {
                "message": "I think what I really need is a structured approach. Maybe I should break it down into smaller tasks?",
                "expected_sentiment": "neutral"
            },
            {
                "message": "Actually, talking about this is helping me think more clearly. I feel like I have a better direction now.",
                "expected_sentiment": "positive"
            },
            {
                "message": "You know what, I'm actually getting excited about this project again. Thanks for listening!",
                "expected_sentiment": "positive"
            }
        ]
        
        responses = []
        for i, turn in enumerate(conversation_flow):
            try:
                chat_data = {
                    "message": turn["message"],
                    "session_id": self.session_id,
                    "persona": "companion"
                }
                
                async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '')
                        print(f"  Turn {i+1}: ✅ Response received ({len(response_text)} chars)")
                        responses.append({
                            "user_message": turn["message"],
                            "ai_response": response_text,
                            "expected_sentiment": turn["expected_sentiment"]
                        })
                        
                        # Small delay to allow processing
                        await asyncio.sleep(1)
                    else:
                        print(f"  Turn {i+1}: ❌ Chat failed with status {response.status}")
                        
            except Exception as e:
                print(f"  Turn {i+1}: ❌ Error: {e}")
        
        print(f"✅ Built conversation context with {len(responses)} turns")
        return responses
    
    async def trigger_manual_reflection(self):
        """Manually trigger reflection analysis"""
        print("🎯 Triggering manual reflection analysis...")
        
        try:
            trigger_data = {
                "session_id": self.session_id,
                "force_analysis": True
            }
            
            async with self.session.post(f"{self.base_url}/api/reflection/manual-trigger", 
                                       json=trigger_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Manual reflection triggered: {data.get('message')}")
                    return data
                else:
                    print(f"❌ Failed to trigger reflection: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error triggering reflection: {e}")
            return None
    
    async def validate_reflection_insights(self):
        """Validate that reflection insights were generated and stored"""
        print("🧠 Validating reflection insights...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/reflection/insights") as response:
                if response.status == 200:
                    insights = await response.json()
                    
                    if insights:
                        print(f"✅ Generated {len(insights)} reflection insights:")
                        for i, insight in enumerate(insights[:3]):  # Show first 3
                            insight_type = insight.get('type', 'unknown')
                            content = insight.get('content', '')[:100] + "..."
                            confidence = insight.get('confidence', 0)
                            print(f"  {i+1}. {insight_type}: {content} (confidence: {confidence:.2f})")
                        return insights
                    else:
                        print("⚠️ No reflection insights generated yet")
                        return []
                else:
                    print(f"❌ Failed to get insights: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error getting insights: {e}")
            return []
    
    async def test_tone_influence(self):
        """Test if reflection insights influence AI tone in subsequent conversations"""
        print("🎭 Testing reflection influence on AI tone...")
        
        # Send a follow-up message that should be influenced by reflection
        follow_up_data = {
            "message": "I'm starting a new project now. Any advice?",
            "session_id": self.session_id,
            "persona": "companion"
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=follow_up_data) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get('response', '')
                    
                    # Check for reflection influence indicators
                    influence_indicators = [
                        "based on our previous conversation",
                        "considering what you mentioned earlier",
                        "given your recent experience",
                        "building on what you shared",
                        "reflecting on our discussion"
                    ]
                    
                    found_indicators = [indicator for indicator in influence_indicators 
                                      if indicator.lower() in response_text.lower()]
                    
                    if found_indicators:
                        print(f"✅ Reflection influence detected: {found_indicators}")
                        print(f"   Response preview: {response_text[:200]}...")
                        return True
                    else:
                        print("⚠️ No clear reflection influence detected in response")
                        print(f"   Response preview: {response_text[:200]}...")
                        return False
                else:
                    print(f"❌ Follow-up chat failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error testing tone influence: {e}")
            return False
    
    async def validate_reflection_summary(self):
        """Validate the final reflection summary"""
        print("📊 Validating reflection summary...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/reflection/summary") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    total_reflections = data.get('total_reflections', 0)
                    active_sessions = data.get('active_sessions', 0)
                    reflection_types = data.get('reflection_types', {})
                    
                    print(f"✅ Reflection Summary:")
                    print(f"   Total Reflections: {total_reflections}")
                    print(f"   Active Sessions: {active_sessions}")
                    print(f"   Reflection Types: {reflection_types}")
                    
                    return data
                else:
                    print(f"❌ Failed to get reflection summary: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting reflection summary: {e}")
            return None
    
    async def run_reflection_qa_suite(self):
        """Run the complete reflection engine QA suite"""
        print("🔄 REFLECTION ENGINE QA SUITE")
        print("=" * 50)
        
        # Test sequence
        initial_count = await self.test_initial_state()
        
        if not await self.enable_reflection_engine():
            print("❌ Cannot continue without reflection engine")
            return False
        
        conversation_data = await self.build_conversation_context()
        if not conversation_data:
            print("❌ Failed to build conversation context")
            return False
        
        # Wait a moment for background processing
        print("⏳ Waiting for background reflection processing...")
        await asyncio.sleep(3)
        
        # Trigger manual reflection to ensure analysis
        await self.trigger_manual_reflection()
        
        # Wait for processing
        await asyncio.sleep(2)
        
        insights = await self.validate_reflection_insights()
        tone_influenced = await self.test_tone_influence()
        summary = await self.validate_reflection_summary()
        
        # Final assessment
        print("\n" + "=" * 50)
        print("🎯 REFLECTION ENGINE QA RESULTS:")
        print(f"✅ Context Built: {len(conversation_data)} conversation turns")
        print(f"✅ Insights Generated: {len(insights)} reflection insights")
        print(f"{'✅' if tone_influenced else '⚠️'} Tone Influence: {'Detected' if tone_influenced else 'Not clearly detected'}")
        print(f"✅ Summary Available: {summary is not None}")
        
        if insights and summary:
            print("\n🎉 Reflection Engine QA: PASSED")
            return True
        else:
            print("\n❌ Reflection Engine QA: PARTIAL - Some features may need attention")
            return False

async def main():
    """Main QA execution"""
    print("Starting Reflection Engine QA...")
    print("Make sure Dolphin backend is running on http://localhost:8000\n")
    
    async with ReflectionEngineQA() as qa:
        success = await qa.run_reflection_qa_suite()
        return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
