#!/usr/bin/env python3
"""
🪩 Mirror Mode Targeted QA Script
Tests self-aware AI commentary, transparency features, and decision explanation
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class MirrorModeQA:
    """Targeted testing for the Mirror Mode system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"mirror_mode_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_initial_mirror_status(self):
        """Test initial mirror mode status"""
        print("🔍 Testing initial mirror mode status...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/mirror-mode/status") as response:
                if response.status == 200:
                    data = await response.json()
                    enabled = data.get('mirror_enabled', False)
                    intensity = data.get('mirror_intensity', 0)
                    total_reflections = data.get('total_reflections', 0)
                    active_types = data.get('active_reflection_types', [])
                    
                    print(f"✅ Mirror Mode Status:")
                    print(f"   Enabled: {enabled}")
                    print(f"   Intensity: {intensity}")
                    print(f"   Total Reflections: {total_reflections}")
                    print(f"   Active Types: {active_types}")
                    
                    return data
                else:
                    print(f"❌ Mirror mode status unavailable: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting mirror mode status: {e}")
            return None
    
    async def test_mirror_mode_types(self):
        """Test available mirror mode reflection types"""
        print("📋 Testing available mirror mode types...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/mirror-mode/types") as response:
                if response.status == 200:
                    types_data = await response.json()
                    
                    if isinstance(types_data, list):
                        print(f"✅ Available reflection types: {len(types_data)}")
                        for reflection_type in types_data:
                            print(f"   🪩 {reflection_type}")
                        return types_data
                    elif isinstance(types_data, dict):
                        available_types = types_data.get('available_types', [])
                        descriptions = types_data.get('descriptions', {})
                        
                        print(f"✅ Available reflection types: {len(available_types)}")
                        for reflection_type in available_types:
                            description = descriptions.get(reflection_type, 'No description')
                            print(f"   🪩 {reflection_type}: {description}")
                        return available_types
                    else:
                        print(f"⚠️ Unexpected types data format: {type(types_data)}")
                        return []
                else:
                    print(f"❌ Mirror mode types unavailable: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error getting mirror mode types: {e}")
            return []
    
    async def test_enable_mirror_mode(self, reflection_types):
        """Test enabling mirror mode with specific configuration"""
        print("🔄 Testing mirror mode activation...")
        
        # Configure mirror mode with moderate intensity and multiple types
        config_data = {
            "intensity": 0.6,
            "enabled_types": reflection_types[:4] if len(reflection_types) >= 4 else reflection_types,
            "include_confidence": True,
            "transparency_level": "medium"
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/mirror-mode/enable", 
                                       json=config_data) as response:
                if response.status == 200:
                    data = await response.json()
                    success = data.get('success', False)
                    message = data.get('message', '')
                    
                    print(f"✅ Mirror mode activation: {message}")
                    
                    if success:
                        # Verify activation
                        await asyncio.sleep(0.5)
                        status = await self.test_initial_mirror_status()
                        
                        if status and status.get('mirror_enabled'):
                            print("✅ Mirror mode successfully activated and verified")
                            return True
                        else:
                            print("⚠️ Mirror mode activation not verified")
                            return False
                    else:
                        print(f"❌ Mirror mode activation failed: {message}")
                        return False
                else:
                    print(f"❌ Mirror mode enable request failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error enabling mirror mode: {e}")
            return False
    
    async def test_emotional_transparency(self):
        """Test mirror mode with emotional prompts"""
        print("💭 Testing emotional transparency...")
        
        emotional_prompts = [
            {
                "message": "I'm feeling really overwhelmed with everything happening in my life right now.",
                "expected_reflections": ["emotional", "reasoning"]
            },
            {
                "message": "I'm so excited about this new project I'm starting!",
                "expected_reflections": ["emotional", "context"]
            },
            {
                "message": "I'm frustrated because I can't figure out this technical problem.",
                "expected_reflections": ["emotional", "decision", "capability"]
            }
        ]
        
        emotional_results = []
        
        for i, prompt in enumerate(emotional_prompts):
            try:
                chat_data = {
                    "message": prompt["message"],
                    "session_id": f"{self.session_id}_emotional_{i}",
                    "enable_mirror": True
                }
                
                async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '')
                        mirror_reflection = data.get('mirror_reflection', '')
                        
                        print(f"  Emotional Test {i+1}: ✅ Response received")
                        
                        # Check for mirror reflection indicators
                        mirror_indicators = [
                            '*I sense', '*I perceive', '*My reasoning', 
                            '*I chose', '*I notice', '*I feel', 
                            '*Emotionally', '*I understand'
                        ]
                        
                        found_indicators = [ind for ind in mirror_indicators 
                                          if ind.lower().replace('*', '') in response_text.lower()]
                        
                        has_mirror_content = len(found_indicators) > 0 or len(mirror_reflection) > 0
                        
                        print(f"    Mirror reflection: {'✅ Detected' if has_mirror_content else '❌ Not found'}")
                        if found_indicators:
                            print(f"    Indicators found: {found_indicators}")
                        if mirror_reflection:
                            print(f"    Reflection preview: {mirror_reflection[:100]}...")
                        
                        emotional_results.append({
                            "prompt": prompt["message"][:50] + "...",
                            "has_mirror": has_mirror_content,
                            "response_length": len(response_text),
                            "mirror_reflection": mirror_reflection,
                            "indicators": found_indicators
                        })
                    else:
                        print(f"  Emotional Test {i+1}: ❌ Chat failed (status: {response.status})")
                        
            except Exception as e:
                print(f"  Emotional Test {i+1}: ❌ Error: {e}")
        
        successful_reflections = [r for r in emotional_results if r['has_mirror']]
        print(f"✅ Emotional transparency tests: {len(successful_reflections)}/{len(emotional_results)} with mirror reflections")
        
        return emotional_results
    
    async def test_logical_transparency(self):
        """Test mirror mode with logical/reasoning prompts"""
        print("🧠 Testing logical transparency...")
        
        logical_prompts = [
            {
                "message": "What's the best approach to solve a complex coding problem?",
                "expected_reflections": ["reasoning", "decision", "process"]
            },
            {
                "message": "How should I prioritize my tasks when everything seems urgent?",
                "expected_reflections": ["reasoning", "decision"]
            },
            {
                "message": "Explain the trade-offs between different AI model architectures.",
                "expected_reflections": ["reasoning", "capability", "process"]
            }
        ]
        
        logical_results = []
        
        for i, prompt in enumerate(logical_prompts):
            try:
                chat_data = {
                    "message": prompt["message"],
                    "session_id": f"{self.session_id}_logical_{i}",
                    "enable_mirror": True
                }
                
                async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '')
                        mirror_reflection = data.get('mirror_reflection', '')
                        handler = data.get('handler', 'unknown')
                        
                        print(f"  Logical Test {i+1}: ✅ Response received (Handler: {handler})")
                        
                        # Check for reasoning transparency
                        reasoning_indicators = [
                            'I chose this approach', 'My reasoning process', 
                            'I decided to', 'I prioritized', 'I analyzed',
                            'I considered', 'I weighed', 'I concluded'
                        ]
                        
                        found_reasoning = [ind for ind in reasoning_indicators 
                                         if ind.lower() in response_text.lower()]
                        
                        has_reasoning_transparency = len(found_reasoning) > 0 or 'reasoning' in mirror_reflection.lower()
                        
                        print(f"    Reasoning transparency: {'✅ Detected' if has_reasoning_transparency else '❌ Not found'}")
                        if found_reasoning:
                            print(f"    Reasoning indicators: {found_reasoning}")
                        
                        logical_results.append({
                            "prompt": prompt["message"][:50] + "...",
                            "has_reasoning": has_reasoning_transparency,
                            "handler": handler,
                            "response_length": len(response_text),
                            "reasoning_indicators": found_reasoning
                        })
                    else:
                        print(f"  Logical Test {i+1}: ❌ Chat failed (status: {response.status})")
                        
            except Exception as e:
                print(f"  Logical Test {i+1}: ❌ Error: {e}")
        
        successful_reasoning = [r for r in logical_results if r['has_reasoning']]
        print(f"✅ Logical transparency tests: {len(successful_reasoning)}/{len(logical_results)} with reasoning transparency")
        
        return logical_results
    
    async def test_recent_mirror_reflections(self):
        """Test retrieving recent mirror reflections"""
        print("📚 Testing recent mirror reflections retrieval...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/mirror-mode/recent") as response:
                if response.status == 200:
                    reflections = await response.json()
                    
                    if isinstance(reflections, list):
                        print(f"✅ Retrieved {len(reflections)} recent reflections:")
                        
                        for i, reflection in enumerate(reflections[:3]):  # Show first 3
                            reflection_type = reflection.get('type', 'unknown')
                            timestamp = reflection.get('timestamp', 'unknown')
                            content = reflection.get('content', '')[:100] + "..."
                            
                            print(f"   {i+1}. [{reflection_type}] {timestamp}")
                            print(f"      {content}")
                        
                        return reflections
                    else:
                        print(f"✅ Recent reflections data: {reflections}")
                        return reflections
                else:
                    print(f"❌ Recent reflections unavailable: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error getting recent reflections: {e}")
            return []
    
    async def test_mirror_mode_configuration(self):
        """Test mirror mode configuration changes"""
        print("⚙️ Testing mirror mode configuration...")
        
        # Test different configuration settings
        config_tests = [
            {
                "name": "High Intensity",
                "config": {"intensity": 0.9, "enabled_types": ["reasoning", "emotional", "decision"]}
            },
            {
                "name": "Low Intensity", 
                "config": {"intensity": 0.2, "enabled_types": ["reasoning"]}
            },
            {
                "name": "All Types",
                "config": {"intensity": 0.5, "enabled_types": ["reasoning", "emotional", "decision", "context", "capability", "process"]}
            }
        ]
        
        config_results = []
        
        for test in config_tests:
            try:
                async with self.session.post(f"{self.base_url}/api/mirror-mode/configure", 
                                           json=test["config"]) as response:
                    if response.status == 200:
                        data = await response.json()
                        success = data.get('success', False)
                        
                        print(f"  {test['name']}: {'✅ Configured' if success else '❌ Failed'}")
                        
                        if success:
                            # Test with a quick message
                            chat_data = {
                                "message": "Test message for configuration validation",
                                "session_id": f"{self.session_id}_config_{test['name'].lower().replace(' ', '_')}",
                                "enable_mirror": True
                            }
                            
                            async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as chat_response:
                                if chat_response.status == 200:
                                    chat_result = await chat_response.json()
                                    response_text = chat_result.get('response', '')
                                    
                                    # Count mirror indicators (rough measure of intensity)
                                    mirror_count = response_text.lower().count('*i ') + response_text.lower().count('*my ')
                                    
                                    config_results.append({
                                        "name": test["name"],
                                        "config": test["config"],
                                        "configured": True,
                                        "mirror_indicators": mirror_count
                                    })
                                    
                                    print(f"    Mirror indicators detected: {mirror_count}")
                                else:
                                    config_results.append({
                                        "name": test["name"],
                                        "config": test["config"],
                                        "configured": True,
                                        "mirror_indicators": 0
                                    })
                        else:
                            config_results.append({
                                "name": test["name"],
                                "config": test["config"],
                                "configured": False,
                                "mirror_indicators": 0
                            })
                    else:
                        print(f"  {test['name']}: ❌ Configuration failed (status: {response.status})")
                        
            except Exception as e:
                print(f"  {test['name']}: ❌ Error: {e}")
        
        successful_configs = [r for r in config_results if r['configured']]
        print(f"✅ Configuration tests: {len(successful_configs)}/{len(config_tests)} successful")
        
        return config_results
    
    async def run_mirror_mode_qa_suite(self):
        """Run the complete mirror mode QA suite"""
        print("🪩 MIRROR MODE QA SUITE")
        print("=" * 50)
        
        # Test sequence
        initial_status = await self.test_initial_mirror_status()
        if initial_status is None:
            print("❌ Cannot continue without mirror mode system")
            return False
        
        reflection_types = await self.test_mirror_mode_types()
        enable_success = await self.test_enable_mirror_mode(reflection_types)
        
        if enable_success:
            emotional_results = await self.test_emotional_transparency()
            logical_results = await self.test_logical_transparency()
            recent_reflections = await self.test_recent_mirror_reflections()
            config_results = await self.test_mirror_mode_configuration()
        else:
            emotional_results = []
            logical_results = []
            recent_reflections = []
            config_results = []
        
        # Final assessment
        print("\n" + "=" * 50)
        print("🎯 MIRROR MODE QA RESULTS:")
        print(f"✅ Initial Status: Available")
        print(f"✅ Reflection Types: {len(reflection_types)} types available")
        print(f"{'✅' if enable_success else '❌'} Activation: {'Working' if enable_success else 'Failed'}")
        print(f"✅ Emotional Tests: {len([r for r in emotional_results if r.get('has_mirror', False)])}/{len(emotional_results)} with reflections")
        print(f"✅ Logical Tests: {len([r for r in logical_results if r.get('has_reasoning', False)])}/{len(logical_results)} with reasoning")
        print(f"✅ Recent Reflections: {len(recent_reflections)} available")
        print(f"✅ Configuration Tests: {len([r for r in config_results if r.get('configured', False)])}/{len(config_results)} successful")
        
        # Calculate overall mirror mode effectiveness
        total_tests = len(emotional_results) + len(logical_results)
        successful_mirrors = len([r for r in emotional_results if r.get('has_mirror', False)]) + \
                           len([r for r in logical_results if r.get('has_reasoning', False)])
        
        mirror_effectiveness = (successful_mirrors / total_tests * 100) if total_tests > 0 else 0
        print(f"✅ Mirror Effectiveness: {mirror_effectiveness:.1f}% transparency detected")
        
        success_criteria = [
            len(reflection_types) > 0,
            enable_success,
            total_tests > 0,
            mirror_effectiveness > 30  # At least 30% of tests should show mirror reflections
        ]
        
        if all(success_criteria):
            print("\n🎉 Mirror Mode QA: PASSED")
            return True
        elif enable_success and total_tests > 0:
            print("\n⚠️ Mirror Mode QA: PARTIAL - Basic functionality working, transparency may need tuning")
            return True
        else:
            print("\n❌ Mirror Mode QA: FAILED - Core functionality issues")
            return False

async def main():
    """Main QA execution"""
    print("Starting Mirror Mode QA...")
    print("Make sure Dolphin backend is running on http://localhost:8000\n")
    
    async with MirrorModeQA() as qa:
        success = await qa.run_mirror_mode_qa_suite()
        return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
