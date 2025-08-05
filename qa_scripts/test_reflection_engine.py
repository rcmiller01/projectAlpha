#!/usr/bin/env python3
"""
🔄 Reflection Engine - Targeted QA Script
Tests conversation pattern analysis and insight generation
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class ReflectionEngineQA:
    """Targeted tests for the Reflection Engine"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_session_id = f"reflection_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def simulate_conversation_session(self):
        """Build enough conversation context to trigger reflection"""
        print("🎭 Simulating conversation session to build context...")
        
        # Conversation that shifts from technical to emotional
        conversation_flow = [
            {
                "message": "Can you help me debug this Python code? I'm getting a syntax error.",
                "expected_tone": "technical"
            },
            {
                "message": "Thanks, that fixed it. But I'm really struggling with this project overall.",
                "expected_tone": "supportive"
            },
            {
                "message": "I've been working on this for weeks and I feel like I'm not making progress.",
                "expected_tone": "emotional"
            },
            {
                "message": "Maybe I should just give up and find a different career path.",
                "expected_tone": "concerning"
            },
            {
                "message": "Actually, you know what? Let me try a different approach to this problem.",
                "expected_tone": "determined"
            },
            {
                "message": "This is actually working now! I think I understand the pattern.",
                "expected_tone": "positive"
            },
            {
                "message": "Can you explain more about object-oriented programming principles?",
                "expected_tone": "learning"
            }
        ]
        
        responses = []
        for i, chat_data in enumerate(conversation_flow):
            print(f"   💬 Message {i+1}: {chat_data['message'][:50]}...")
            
            payload = {
                "message": chat_data["message"],
                "session_id": self.test_session_id,
                "persona": "companion"
            }
            
            try:
                async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        responses.append({
                            "message": chat_data["message"],
                            "response": data.get("response", ""),
                            "expected_tone": chat_data["expected_tone"],
                            "timestamp": datetime.now().isoformat()
                        })
                        print(f"   ✅ Response received ({len(data.get('response', ''))} chars)")
                    else:
                        print(f"   ❌ Chat failed: {response.status}")
                        
                # Small delay between messages
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"   ❌ Chat error: {e}")
        
        print(f"✅ Conversation simulation complete: {len(responses)} exchanges")
        return responses
    
    async def enable_reflection_engine(self):
        """Enable the reflection engine"""
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
            print(f"❌ Reflection engine enable error: {e}")
            return False
    
    async def trigger_manual_reflection(self):
        """Manually trigger reflection analysis"""
        print("🧠 Triggering manual reflection analysis...")
        
        try:
            async with self.session.post(f"{self.base_url}/api/reflection/manual-trigger") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Manual reflection triggered: {data.get('message')}")
                    return data
                else:
                    print(f"❌ Manual reflection failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Manual reflection error: {e}")
            return None
    
    async def wait_for_background_reflection(self):
        """Wait for background reflection to process"""
        print("⏳ Waiting for background reflection processing...")
        
        max_wait = 30  # 30 seconds max wait
        check_interval = 2  # Check every 2 seconds
        
        for i in range(max_wait // check_interval):
            try:
                async with self.session.get(f"{self.base_url}/api/reflection/summary") as response:
                    if response.status == 200:
                        data = await response.json()
                        total_reflections = data.get('total_reflections', 0)
                        if total_reflections > 0:
                            print(f"✅ Background reflection complete: {total_reflections} reflections generated")
                            return data
                        else:
                            print(f"   ⏳ Waiting... ({i*check_interval}s) - No reflections yet")
                            await asyncio.sleep(check_interval)
                    else:
                        print(f"   ⚠️ Reflection status check failed: {response.status}")
                        await asyncio.sleep(check_interval)
            except Exception as e:
                print(f"   ⚠️ Reflection check error: {e}")
                await asyncio.sleep(check_interval)
        
        print("❌ Background reflection timeout")
        return None
    
    async def validate_reflection_insights(self):
        """Validate that meaningful insights were generated"""
        print("🔍 Validating reflection insights...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/reflection/insights") as response:
                if response.status == 200:
                    insights = await response.json()
                    
                    if not insights:
                        print("❌ No insights generated")
                        return False
                    
                    print(f"✅ Found {len(insights)} insights:")
                    
                    # Check for different types of insights
                    insight_types = set()
                    mood_shifts_detected = False
                    engagement_tracked = False
                    
                    for insight in insights:
                        insight_type = insight.get('type', 'unknown')
                        insight_types.add(insight_type)
                        content = insight.get('content', '')
                        
                        print(f"   📊 {insight_type}: {content[:100]}...")
                        
                        # Check for mood shift detection
                        if 'mood' in content.lower() or 'emotional' in content.lower():
                            mood_shifts_detected = True
                        
                        # Check for engagement tracking
                        if 'engagement' in content.lower() or 'interest' in content.lower():
                            engagement_tracked = True
                    
                    print(f"   🎯 Insight types detected: {', '.join(insight_types)}")
                    print(f"   😊 Mood shift detection: {'✅' if mood_shifts_detected else '❌'}")
                    print(f"   📈 Engagement tracking: {'✅' if engagement_tracked else '❌'}")
                    
                    # Validate quality metrics
                    quality_score = 0
                    if len(insight_types) >= 2:  # Multiple insight types
                        quality_score += 1
                    if mood_shifts_detected:  # Emotional awareness
                        quality_score += 1
                    if engagement_tracked:  # Engagement tracking
                        quality_score += 1
                    if len(insights) >= 3:  # Sufficient insights
                        quality_score += 1
                    
                    print(f"   🏆 Quality score: {quality_score}/4")
                    
                    return quality_score >= 2
                    
                else:
                    print(f"❌ Failed to get insights: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Insight validation error: {e}")
            return False
    
    async def test_reflection_influence_on_tone(self):
        """Test if reflection insights influence future responses"""
        print("🎭 Testing reflection influence on tone...")
        
        # Send a similar message to earlier ones and see if tone adapts
        follow_up_message = "I'm working on another coding project and feeling a bit stuck again."
        
        payload = {
            "message": follow_up_message,
            "session_id": self.test_session_id,
            "persona": "companion"
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data.get("response", "").lower()
                    
                    # Check for adaptive tone indicators
                    adaptive_indicators = [
                        "remember", "like before", "similar to", "progress",
                        "breakthrough", "pattern", "approach", "working through"
                    ]
                    
                    found_indicators = [ind for ind in adaptive_indicators if ind in response_text]
                    
                    if found_indicators:
                        print(f"✅ Adaptive tone detected: {', '.join(found_indicators)}")
                        return True
                    else:
                        print(f"❌ No adaptive tone indicators found")
                        print(f"   Response sample: {response_text[:200]}...")
                        return False
                        
                else:
                    print(f"❌ Follow-up chat failed: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Tone influence test error: {e}")
            return False
    
    async def run_reflection_qa_suite(self):
        """Run the complete reflection engine QA suite"""
        print("🔄 REFLECTION ENGINE QA SUITE")
        print("=" * 50)
        
        test_results = {
            "conversation_simulation": False,
            "reflection_enablement": False,
            "insight_generation": False,
            "insight_quality": False,
            "tone_influence": False
        }
        
        # Step 1: Enable reflection engine
        test_results["reflection_enablement"] = await self.enable_reflection_engine()
        
        if not test_results["reflection_enablement"]:
            print("❌ Cannot proceed without reflection engine")
            return test_results
        
        # Step 2: Simulate conversation to build context
        responses = await self.simulate_conversation_session()
        test_results["conversation_simulation"] = len(responses) >= 5
        
        # Step 3: Trigger manual reflection (immediate)
        reflection_data = await self.trigger_manual_reflection()
        test_results["insight_generation"] = reflection_data is not None
        
        # Step 4: Wait for background processing
        background_data = await self.wait_for_background_reflection()
        if background_data:
            test_results["insight_generation"] = True
        
        # Step 5: Validate insight quality
        test_results["insight_quality"] = await self.validate_reflection_insights()
        
        # Step 6: Test reflection influence on future responses
        test_results["tone_influence"] = await self.test_reflection_influence_on_tone()
        
        # Results summary
        print("\n" + "=" * 50)
        print("🏆 REFLECTION ENGINE QA RESULTS")
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print(f"\n📊 Overall Score: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 ALL REFLECTION ENGINE TESTS PASSED!")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Reflection engine mostly functional")
        else:
            print("⚠️ Reflection engine needs attention")
        
        return test_results

async def main():
    """Run the reflection engine QA suite"""
    print("🔄 Starting Reflection Engine QA Script...")
    print("Ensure Dolphin backend is running on http://localhost:8000\n")
    
    async with ReflectionEngineQA() as qa:
        results = await qa.run_reflection_qa_suite()
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
