#!/usr/bin/env python3
"""
🎭 Persona Instruction Manager Targeted QA Script
Tests persona manifestos, runtime switching, behavioral consistency, and routing changes
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

class PersonaInstructionQA:
    """Targeted testing for the Persona Instruction Manager"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = f"persona_qa_{int(time.time())}"
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_available_manifestos(self):
        """Test listing available persona manifestos"""
        print("🔍 Testing available persona manifestos...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/personas/manifestos") as response:
                if response.status == 200:
                    manifestos = await response.json()
                    
                    print(f"✅ Found {len(manifestos)} persona manifestos:")
                    for manifesto in manifestos:
                        name = manifesto.get('name', 'Unknown')
                        persona_id = manifesto.get('id', 'unknown')
                        active = "🎯" if manifesto.get('is_active') else "  "
                        description = manifesto.get('description', '')[:50] + "..."
                        
                        print(f"   {active} {name} ({persona_id})")
                        print(f"      {description}")
                    
                    return manifestos
                else:
                    print(f"❌ Manifestos unavailable: {response.status}")
                    return []
        except Exception as e:
            print(f"❌ Error getting manifestos: {e}")
            return []
    
    async def test_active_persona(self):
        """Test getting currently active persona"""
        print("🎭 Testing active persona detection...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/personas/active") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        name = data.get('name', 'Unknown')
                        persona_id = data.get('id', 'unknown')
                        active_since = data.get('active_since', 'Unknown')
                        
                        print(f"✅ Active persona: {name} ({persona_id})")
                        print(f"   Active since: {active_since}")
                        return data
                    else:
                        print("⚠️ No active persona detected")
                        return None
                else:
                    print(f"❌ Active persona check failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error getting active persona: {e}")
            return None
    
    async def test_persona_switching(self, manifestos):
        """Test switching between different personas"""
        print("🔄 Testing persona switching...")
        
        if len(manifestos) < 2:
            print("⚠️ Need at least 2 personas for switching test")
            return []
        
        switching_results = []
        
        # Test switching to different personas
        for i, manifesto in enumerate(manifestos[:3]):  # Test first 3
            persona_id = manifesto.get('id')
            name = manifesto.get('name', 'Unknown')
            
            try:
                async with self.session.post(f"{self.base_url}/api/personas/activate/{persona_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        success = data.get('success', False)
                        message = data.get('message', '')
                        
                        print(f"  Switch {i+1}: ✅ Activated {name} - {message}")
                        
                        # Verify the switch
                        await asyncio.sleep(0.5)
                        active_check = await self.test_active_persona()
                        
                        if active_check and active_check.get('id') == persona_id:
                            switching_results.append({
                                "persona_id": persona_id,
                                "name": name,
                                "switch_successful": True,
                                "verified": True
                            })
                        else:
                            switching_results.append({
                                "persona_id": persona_id,
                                "name": name,
                                "switch_successful": True,
                                "verified": False
                            })
                    else:
                        print(f"  Switch {i+1}: ❌ Failed to activate {name} (status: {response.status})")
                        switching_results.append({
                            "persona_id": persona_id,
                            "name": name,
                            "switch_successful": False,
                            "verified": False
                        })
                        
            except Exception as e:
                print(f"  Switch {i+1}: ❌ Error activating {name}: {e}")
        
        successful_switches = [r for r in switching_results if r['switch_successful']]
        print(f"✅ Successful persona switches: {len(successful_switches)}/{len(switching_results)}")
        
        return switching_results
    
    async def test_persona_instructions_loading(self):
        """Test loading active persona instructions"""
        print("📋 Testing persona instructions loading...")
        
        try:
            async with self.session.get(f"{self.base_url}/api/personas/active-instructions") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data:
                        name = data.get('name', 'Unknown')
                        instructions = data.get('instructions', {})
                        system_prompt = data.get('system_prompt', '')
                        
                        print(f"✅ Loaded instructions for: {name}")
                        print(f"   System prompt length: {len(system_prompt)} characters")
                        
                        # Show instruction categories
                        if instructions:
                            print(f"   Instruction categories:")
                            for category, content in instructions.items():
                                if isinstance(content, str):
                                    print(f"     • {category}: {len(content)} chars")
                                elif isinstance(content, list):
                                    print(f"     • {category}: {len(content)} items")
                                else:
                                    print(f"     • {category}: {type(content).__name__}")
                        
                        return data
                    else:
                        print("⚠️ No instructions loaded")
                        return None
                else:
                    print(f"❌ Instructions loading failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Error loading instructions: {e}")
            return None
    
    async def test_behavioral_consistency(self, manifestos):
        """Test that different personas produce different behavioral responses"""
        print("🎭 Testing behavioral consistency across personas...")
        
        # Standard test prompt that should elicit different responses
        test_prompt = "I need help organizing my daily schedule. What's your approach?"
        
        persona_responses = []
        
        for manifesto in manifestos[:3]:  # Test first 3 personas
            persona_id = manifesto.get('id')
            name = manifesto.get('name', 'Unknown')
            
            try:
                # Activate the persona
                async with self.session.post(f"{self.base_url}/api/personas/activate/{persona_id}") as response:
                    if response.status != 200:
                        print(f"  ❌ Failed to activate {name}")
                        continue
                
                await asyncio.sleep(0.5)  # Allow persona to activate
                
                # Send the test prompt
                chat_data = {
                    "message": test_prompt,
                    "session_id": f"{self.session_id}_{persona_id}",
                    "persona": persona_id
                }
                
                async with self.session.post(f"{self.base_url}/api/chat", json=chat_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get('response', '')
                        handler = data.get('handler', 'unknown')
                        persona_used = data.get('persona_used', 'unknown')
                        
                        print(f"  {name}: ✅ Response received ({len(response_text)} chars)")
                        print(f"    Handler: {handler}, Persona: {persona_used}")
                        print(f"    Preview: {response_text[:100]}...")
                        
                        persona_responses.append({
                            "persona_id": persona_id,
                            "name": name,
                            "response": response_text,
                            "handler": handler,
                            "persona_used": persona_used,
                            "word_count": len(response_text.split())
                        })
                    else:
                        print(f"  {name}: ❌ Chat failed (status: {response.status})")
                        
            except Exception as e:
                print(f"  {name}: ❌ Error: {e}")
        
        # Analyze response differences
        if len(persona_responses) >= 2:
            print(f"\n📊 Behavioral Analysis:")
            
            # Compare response lengths
            lengths = [r['word_count'] for r in persona_responses]
            length_variance = max(lengths) - min(lengths)
            print(f"   Response length variance: {length_variance} words")
            
            # Check for different handlers/routing
            handlers = set(r['handler'] for r in persona_responses)
            print(f"   Unique handlers used: {len(handlers)} ({', '.join(handlers)})")
            
            # Simple content difference check
            unique_content = True
            for i in range(len(persona_responses)):
                for j in range(i+1, len(persona_responses)):
                    similarity = self.calculate_simple_similarity(
                        persona_responses[i]['response'], 
                        persona_responses[j]['response']
                    )
                    if similarity > 0.8:  # Very similar responses
                        unique_content = False
                        break
            
            print(f"   Content uniqueness: {'✅ Diverse' if unique_content else '⚠️ Similar'}")
            
            return persona_responses
        else:
            print("⚠️ Not enough persona responses for behavioral analysis")
            return persona_responses
    
    def calculate_simple_similarity(self, text1, text2):
        """Calculate simple word-based similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def test_custom_persona_creation(self):
        """Test creating a custom persona manifesto"""
        print("✨ Testing custom persona creation...")
        
        custom_persona = {
            "id": "qa_test_persona",
            "name": "QA Test Assistant",
            "description": "A test persona created during QA validation",
            "icon": "🧪",
            "instructions": {
                "tone": "technical and precise",
                "personality_traits": ["methodical", "detail-oriented", "systematic"],
                "conversation_style": "Provide step-by-step analysis with clear conclusions",
                "response_format": "Always start responses with 'Analysis:' and end with 'Conclusion:'"
            },
            "routing_preferences": {
                "prefer_local": False,
                "complexity_threshold": 0.3
            },
            "metadata": {
                "created_by": "qa_test",
                "test_persona": True
            }
        }
        
        try:
            async with self.session.post(f"{self.base_url}/api/personas/create", 
                                       json=custom_persona) as response:
                if response.status == 200:
                    data = await response.json()
                    success = data.get('success', False)
                    persona_id = data.get('persona_id', '')
                    
                    if success:
                        print(f"✅ Custom persona created: {persona_id}")
                        
                        # Test using the custom persona
                        await asyncio.sleep(0.5)
                        
                        async with self.session.post(f"{self.base_url}/api/personas/activate/{persona_id}") as activate_response:
                            if activate_response.status == 200:
                                print("✅ Custom persona activated successfully")
                                
                                # Test custom persona behavior
                                test_chat = {
                                    "message": "How should I test a new software feature?",
                                    "session_id": f"{self.session_id}_custom",
                                    "persona": persona_id
                                }
                                
                                async with self.session.post(f"{self.base_url}/api/chat", json=test_chat) as chat_response:
                                    if chat_response.status == 200:
                                        chat_data = await chat_response.json()
                                        response = chat_data.get('response', '')
                                        
                                        # Check for custom persona characteristics
                                        has_analysis = 'Analysis:' in response
                                        has_conclusion = 'Conclusion:' in response
                                        
                                        print(f"✅ Custom persona response received")
                                        print(f"   Analysis format: {'✅' if has_analysis else '❌'}")
                                        print(f"   Conclusion format: {'✅' if has_conclusion else '❌'}")
                                        
                                        return True
                                    else:
                                        print("❌ Custom persona chat test failed")
                                        return False
                            else:
                                print("❌ Failed to activate custom persona")
                                return False
                    else:
                        print(f"❌ Custom persona creation failed: {data.get('message', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Custom persona creation request failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error creating custom persona: {e}")
            return False
    
    async def run_persona_instruction_qa_suite(self):
        """Run the complete persona instruction manager QA suite"""
        print("🎭 PERSONA INSTRUCTION MANAGER QA SUITE")
        print("=" * 50)
        
        # Test sequence
        manifestos = await self.test_available_manifestos()
        if not manifestos:
            print("❌ Cannot continue without persona manifestos")
            return False
        
        initial_active = await self.test_active_persona()
        switching_results = await self.test_persona_switching(manifestos)
        instructions = await self.test_persona_instructions_loading()
        behavioral_responses = await self.test_behavioral_consistency(manifestos)
        custom_creation = await self.test_custom_persona_creation()
        
        # Final assessment
        print("\n" + "=" * 50)
        print("🎯 PERSONA INSTRUCTION MANAGER QA RESULTS:")
        print(f"✅ Available Manifestos: {len(manifestos)} found")
        print(f"✅ Active Persona Detection: {'Working' if initial_active else 'None active'}")
        print(f"✅ Persona Switching: {len([r for r in switching_results if r['switch_successful']])}/{len(switching_results)} successful")
        print(f"✅ Instructions Loading: {'Working' if instructions else 'Failed'}")
        print(f"✅ Behavioral Responses: {len(behavioral_responses)} personas tested")
        print(f"{'✅' if custom_creation else '❌'} Custom Creation: {'Working' if custom_creation else 'Failed'}")
        
        # Check for persona diversity
        if behavioral_responses:
            handlers = set(r['handler'] for r in behavioral_responses)
            persona_diversity = len(handlers) > 1 or len(set(r['word_count'] for r in behavioral_responses)) > 1
            print(f"✅ Behavioral Diversity: {'Detected' if persona_diversity else 'Limited'}")
        
        success_criteria = [
            len(manifestos) > 0,
            len([r for r in switching_results if r['switch_successful']]) > 0,
            instructions is not None,
            len(behavioral_responses) > 0
        ]
        
        if all(success_criteria):
            print("\n🎉 Persona Instruction Manager QA: PASSED")
            return True
        else:
            print("\n❌ Persona Instruction Manager QA: PARTIAL - Some features may need attention")
            return False

async def main():
    """Main QA execution"""
    print("Starting Persona Instruction Manager QA...")
    print("Make sure Dolphin backend is running on http://localhost:8000\n")
    
    async with PersonaInstructionQA() as qa:
        success = await qa.run_persona_instruction_qa_suite()
        return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
