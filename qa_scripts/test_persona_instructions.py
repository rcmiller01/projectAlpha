#!/usr/bin/env python3
"""
🎭 Persona Instruction Manager - Targeted QA Script
Tests persona manifestos, routing changes, and behavioral adaptation
"""

import requests
import json
import time
from datetime import datetime

class PersonaInstructionQA:
    """Targeted tests for the Persona Instruction Manager"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_session_id = f"persona_qa_{int(time.time())}"
        self.test_prompt = "I'm feeling overwhelmed with my coding project and need guidance on how to approach debugging a complex issue."
        
    def make_request(self, method, endpoint, **kwargs):
        """Make HTTP request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(method, url, timeout=15, **kwargs)
            return response
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None
    
    def get_available_manifestos(self):
        """Get list of available persona manifestos"""
        print("🎭 Getting available persona manifestos...")
        
        response = self.make_request("GET", "/api/personas/manifestos")
        if response and response.status_code == 200:
            manifestos = response.json()
            
            print(f"✅ Found {len(manifestos)} persona manifestos:")
            
            personas = {}
            for manifesto in manifestos:
                persona_id = manifesto.get('id', 'unknown')
                name = manifesto.get('name', 'Unknown')
                description = manifesto.get('description', 'No description')
                is_active = manifesto.get('is_active', False)
                
                active_indicator = "🎯" if is_active else "  "
                print(f"   {active_indicator} {name} ({persona_id})")
                print(f"      📝 {description}")
                
                personas[persona_id] = {
                    'name': name,
                    'description': description,
                    'is_active': is_active,
                    'full_data': manifesto
                }
            
            return personas
        else:
            print("❌ Failed to get persona manifestos")
            return {}
    
    def test_persona_activation(self, persona_id, persona_name):
        """Test activating a specific persona"""
        print(f"🎯 Testing activation of persona: {persona_name}")
        
        response = self.make_request("POST", f"/api/personas/activate/{persona_id}")
        if response and response.status_code == 200:
            data = response.json()
            activated_id = data.get('activated_persona', '')
            message = data.get('message', '')
            
            print(f"   ✅ Activation response: {message}")
            
            if activated_id == persona_id:
                print(f"   ✅ Correctly activated: {persona_id}")
                return True
            else:
                print(f"   ❌ Activation mismatch: expected {persona_id}, got {activated_id}")
                return False
        else:
            print(f"   ❌ Failed to activate persona: {response.status_code if response else 'no response'}")
            return False
    
    def get_active_persona_instructions(self):
        """Get current active persona instructions"""
        print("📋 Getting active persona instructions...")
        
        response = self.make_request("GET", "/api/personas/active-instructions")
        if response and response.status_code == 200:
            data = response.json()
            
            name = data.get('name', 'Unknown')
            instructions = data.get('instructions', {})
            
            print(f"✅ Active persona: {name}")
            print(f"   📝 Instruction sections: {list(instructions.keys())}")
            
            return {
                'name': name,
                'instructions': instructions,
                'full_data': data
            }
        else:
            print("❌ Failed to get active instructions")
            return None
    
    def test_persona_response_differences(self, personas_to_test):
        """Test that different personas produce different responses"""
        print("🔄 Testing persona response differences...")
        
        responses = {}
        
        for persona_id, persona_data in personas_to_test.items():
            persona_name = persona_data['name']
            print(f"   🎭 Testing {persona_name}...")
            
            # Activate persona
            if self.test_persona_activation(persona_id, persona_name):
                
                # Wait for activation to take effect
                time.sleep(1)
                
                # Send test prompt
                chat_data = {
                    "message": self.test_prompt,
                    "session_id": f"{self.test_session_id}_{persona_id}",
                    "persona": persona_id
                }
                
                response = self.make_request("POST", "/api/chat", json=chat_data)
                if response and response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    handler = data.get('handler', 'unknown')
                    persona_used = data.get('persona_used', 'unknown')
                    
                    responses[persona_id] = {
                        'name': persona_name,
                        'response': response_text,
                        'handler': handler,
                        'persona_used': persona_used,
                        'length': len(response_text),
                        'word_count': len(response_text.split())
                    }
                    
                    print(f"      ✅ Response: {len(response_text)} chars, Handler: {handler}")
                else:
                    print(f"      ❌ Chat failed for {persona_name}")
            else:
                print(f"      ❌ Activation failed for {persona_name}")
        
        return responses
    
    def analyze_persona_differences(self, responses):
        """Analyze differences between persona responses"""
        print("📊 Analyzing persona response differences...")
        
        if len(responses) < 2:
            print("❌ Need at least 2 personas to compare")
            return False
        
        # Analyze response characteristics
        analysis = {}
        
        for persona_id, response_data in responses.items():
            name = response_data['name']
            text = response_data['response'].lower()
            word_count = response_data['word_count']
            
            # Analyze tone/style indicators
            characteristics = {
                'empathetic_words': sum(1 for word in ['feel', 'understand', 'support', 'help', 'care'] if word in text),
                'technical_words': sum(1 for word in ['code', 'debug', 'function', 'algorithm', 'syntax'] if word in text),
                'analytical_words': sum(1 for word in ['analyze', 'evaluate', 'assess', 'examine', 'consider'] if word in text),
                'motivational_words': sum(1 for word in ['achieve', 'succeed', 'overcome', 'persist', 'goal'] if word in text),
                'creative_words': sum(1 for word in ['creative', 'innovative', 'imagine', 'explore', 'artistic'] if word in text),
                'question_count': text.count('?'),
                'exclamation_count': text.count('!'),
                'word_count': word_count,
                'formality_score': sum(1 for word in ['furthermore', 'however', 'consequently', 'therefore'] if word in text)
            }
            
            analysis[persona_id] = {
                'name': name,
                'characteristics': characteristics,
                'response_sample': response_data['response'][:200] + "..." if len(response_data['response']) > 200 else response_data['response']
            }
        
        # Display analysis
        print("\n📈 Persona Characteristic Analysis:")
        for persona_id, data in analysis.items():
            name = data['name']
            chars = data['characteristics']
            print(f"\n   🎭 {name}:")
            print(f"      💭 Empathetic words: {chars['empathetic_words']}")
            print(f"      ⚡ Technical words: {chars['technical_words']}")
            print(f"      📊 Analytical words: {chars['analytical_words']}")
            print(f"      🎯 Motivational words: {chars['motivational_words']}")
            print(f"      🎨 Creative words: {chars['creative_words']}")
            print(f"      📝 Word count: {chars['word_count']}")
            print(f"      ❓ Questions: {chars['question_count']}")
            print(f"      ❗ Exclamations: {chars['exclamation_count']}")
        
        # Check for meaningful differences
        word_counts = [data['characteristics']['word_count'] for data in analysis.values()]
        empathy_scores = [data['characteristics']['empathetic_words'] for data in analysis.values()]
        technical_scores = [data['characteristics']['technical_words'] for data in analysis.values()]
        
        # Calculate variance to see if personas differ
        word_count_variance = max(word_counts) - min(word_counts) if word_counts else 0
        empathy_variance = max(empathy_scores) - min(empathy_scores) if empathy_scores else 0
        technical_variance = max(technical_scores) - min(technical_scores) if technical_scores else 0
        
        print(f"\n📊 Variance Analysis:")
        print(f"   📝 Word count variance: {word_count_variance}")
        print(f"   💭 Empathy score variance: {empathy_variance}")
        print(f"   ⚡ Technical score variance: {technical_variance}")
        
        # Assess if personas are meaningfully different
        has_word_differences = word_count_variance > 50  # At least 50 word difference
        has_tone_differences = empathy_variance > 0 or technical_variance > 0
        
        if has_word_differences and has_tone_differences:
            print("✅ Personas show meaningful behavioral differences")
            return True
        elif has_tone_differences:
            print("⚠️ Personas show some differences but could be more distinct")
            return True
        else:
            print("❌ Personas appear to be too similar")
            return False
    
    def test_routing_adjustments(self, personas_to_test):
        """Test that different personas affect routing decisions"""
        print("🔄 Testing persona routing adjustments...")
        
        routing_test_prompt = "I need help building a complex machine learning model with TensorFlow and optimizing its performance."
        
        routing_results = {}
        
        for persona_id, persona_data in personas_to_test.items():
            persona_name = persona_data['name']
            print(f"   🎯 Testing routing for {persona_name}...")
            
            # Activate persona
            if self.test_persona_activation(persona_id, persona_name):
                time.sleep(1)
                
                # Send routing test prompt
                chat_data = {
                    "message": routing_test_prompt,
                    "session_id": f"{self.test_session_id}_routing_{persona_id}",
                    "persona": persona_id
                }
                
                response = self.make_request("POST", "/api/chat", json=chat_data)
                if response and response.status_code == 200:
                    data = response.json()
                    handler = data.get('handler', 'unknown')
                    routing_reason = data.get('routing_reason', '')
                    persona_used = data.get('persona_used', 'unknown')
                    
                    routing_results[persona_id] = {
                        'name': persona_name,
                        'handler': handler,
                        'routing_reason': routing_reason,
                        'persona_used': persona_used
                    }
                    
                    print(f"      ✅ Handler: {handler}, Reason: {routing_reason}")
                else:
                    print(f"      ❌ Routing test failed for {persona_name}")
        
        # Analyze routing differences
        handlers_used = [result['handler'] for result in routing_results.values()]
        unique_handlers = set(handlers_used)
        
        print(f"\n📊 Routing Analysis:")
        print(f"   🤖 Handlers used: {list(unique_handlers)}")
        print(f"   🔄 Total unique routes: {len(unique_handlers)}")
        
        for persona_id, result in routing_results.items():
            print(f"   🎭 {result['name']}: {result['handler']}")
        
        # Consider routing successful if we see different handlers or different reasoning
        return len(unique_handlers) > 1 or len(set(result['routing_reason'] for result in routing_results.values())) > 1
    
    def test_runtime_persona_modification(self):
        """Test modifying a persona at runtime"""
        print("🔧 Testing runtime persona modification...")
        
        # Create a custom test persona
        custom_persona = {
            "id": "qa_test_persona",
            "name": "QA Test Persona",
            "description": "A persona created for testing runtime modifications",
            "icon": "🧪",
            "routing_preferences": {
                "dolphin_bias": 0.9,
                "openrouter_threshold": 0.8,
                "n8n_threshold": 0.7
            },
            "prompt_style": {
                "tone": "extremely enthusiastic and energetic",
                "personality_traits": ["energetic", "optimistic", "helpful"],
                "conversation_style": "Use lots of exclamation points and positive language!",
                "prefix": "As your super enthusiastic coding buddy, "
            }
        }
        
        # Create the persona
        response = self.make_request("POST", "/api/personas/create", json=custom_persona)
        if response and response.status_code == 200:
            print("   ✅ Custom persona created successfully")
            
            # Activate and test it
            if self.test_persona_activation("qa_test_persona", "QA Test Persona"):
                time.sleep(1)
                
                # Test response with original persona
                chat_data = {
                    "message": "Help me with a simple Python function",
                    "session_id": f"{self.test_session_id}_custom",
                    "persona": "qa_test_persona"
                }
                
                response = self.make_request("POST", "/api/chat", json=chat_data)
                if response and response.status_code == 200:
                    data = response.json()
                    original_response = data.get('response', '')
                    exclamation_count_original = original_response.count('!')
                    
                    print(f"   ✅ Original response: {exclamation_count_original} exclamations")
                    
                    # Now modify the persona (make it very formal)
                    modified_persona = custom_persona.copy()
                    modified_persona["prompt_style"]["tone"] = "extremely formal and academic"
                    modified_persona["prompt_style"]["conversation_style"] = "Use formal language with no exclamation points or casual expressions."
                    modified_persona["prompt_style"]["prefix"] = "As your formal academic assistant, "
                    
                    modify_response = self.make_request("POST", "/api/personas/create", json=modified_persona)
                    if modify_response and modify_response.status_code == 200:
                        print("   ✅ Persona modified successfully")
                        
                        # Reactivate and test again
                        if self.test_persona_activation("qa_test_persona", "QA Test Persona"):
                            time.sleep(1)
                            
                            modified_response = self.make_request("POST", "/api/chat", json=chat_data)
                            if modified_response and modified_response.status_code == 200:
                                modified_data = modified_response.json()
                                new_response = modified_data.get('response', '')
                                exclamation_count_new = new_response.count('!')
                                
                                print(f"   ✅ Modified response: {exclamation_count_new} exclamations")
                                
                                # Check if behavior actually changed
                                if exclamation_count_original > exclamation_count_new:
                                    print("   ✅ Persona behavior successfully modified at runtime")
                                    return True
                                else:
                                    print("   ⚠️ Persona modification may not have taken effect")
                                    return False
                            else:
                                print("   ❌ Failed to get modified response")
                                return False
                        else:
                            print("   ❌ Failed to reactivate modified persona")
                            return False
                    else:
                        print("   ❌ Failed to modify persona")
                        return False
                else:
                    print("   ❌ Failed to get original response")
                    return False
            else:
                print("   ❌ Failed to activate custom persona")
                return False
        else:
            print("   ❌ Failed to create custom persona")
            return False
    
    def run_persona_instruction_qa_suite(self):
        """Run the complete persona instruction QA suite"""
        print("🎭 PERSONA INSTRUCTION MANAGER QA SUITE")
        print("=" * 55)
        
        test_results = {
            "manifesto_retrieval": False,
            "persona_activation": False,
            "instruction_retrieval": False,
            "response_differences": False,
            "routing_adjustments": False,
            "runtime_modification": False
        }
        
        # Step 1: Get available manifestos
        personas = self.get_available_manifestos()
        test_results["manifesto_retrieval"] = len(personas) > 0
        
        if not test_results["manifesto_retrieval"]:
            print("❌ Cannot proceed without persona manifestos")
            return test_results
        
        # Select test personas (limit to 3 for efficiency)
        test_personas = dict(list(personas.items())[:3])
        
        # Step 2: Test persona activation
        activation_results = []
        for persona_id, persona_data in test_personas.items():
            result = self.test_persona_activation(persona_id, persona_data['name'])
            activation_results.append(result)
        
        test_results["persona_activation"] = any(activation_results)
        
        # Step 3: Test instruction retrieval
        active_instructions = self.get_active_persona_instructions()
        test_results["instruction_retrieval"] = active_instructions is not None
        
        # Step 4: Test response differences
        responses = self.test_persona_response_differences(test_personas)
        test_results["response_differences"] = self.analyze_persona_differences(responses)
        
        # Step 5: Test routing adjustments
        test_results["routing_adjustments"] = self.test_routing_adjustments(test_personas)
        
        # Step 6: Test runtime modification
        test_results["runtime_modification"] = self.test_runtime_persona_modification()
        
        # Results summary
        print("\n" + "=" * 55)
        print("🏆 PERSONA INSTRUCTION MANAGER QA RESULTS")
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, passed in test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
        
        print(f"\n📊 Overall Score: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        
        if passed_tests == total_tests:
            print("🎉 ALL PERSONA INSTRUCTION TESTS PASSED!")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Persona instruction manager mostly functional")
        else:
            print("⚠️ Persona instruction manager needs attention")
        
        return test_results

def main():
    """Run the persona instruction QA suite"""
    print("🎭 Starting Persona Instruction Manager QA Script...")
    print("Ensure Dolphin backend is running on http://localhost:8000\n")
    
    qa = PersonaInstructionQA()
    results = qa.run_persona_instruction_qa_suite()
    
    return results

if __name__ == "__main__":
    main()
