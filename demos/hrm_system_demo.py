#!/usr/bin/env python3
"""
HRM System Complete Demo
========================

Comprehensive demonstration of the Hierarchical Reasoning Model (HRM) system
showing the full pipeline from user input to final response through all
major components:

1. HRM Router - Main orchestration
2. SubAgent Router - Specialized agent routing
3. AI Reformulator - Personality consistency
4. Core Arbiter - Decision fusion
5. Mirror Mode - Self-reflection

Author: AI Development Team
Version: 1.0.0
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import HRM components
from backend.hrm_router import HRMRouter, HRMMode, RequestType
from backend.subagent_router import SubAgentRouter, AgentType
from backend.ai_reformulator import PersonalityFormatter, ReformulationRequest
from core.core_arbiter import CoreArbiter
from core.mirror_mode import MirrorModeManager, MirrorType

class HRMSystemDemo:
    """
    Complete HRM System Demonstration

    Shows how all components work together to provide
    intelligent, emotionally aware, and personality-consistent
    AI responses.
    """

    def __init__(self):
        # Initialize all major components
        self.hrm_router = HRMRouter()
        self.subagent_router = SubAgentRouter()
        self.personality_formatter = PersonalityFormatter()
        self.core_arbiter = CoreArbiter()
        self.mirror_manager = MirrorModeManager()

        # Demo configuration
        self.demo_config = {
            "show_intermediate_steps": True,
            "enable_mirror_mode": True,
            "detailed_logging": True,
            "save_results": True
        }

        # Identity layer protection settings
        self.identity_admin_key = "hrm_admin_key_2025"  # TODO: Load from secure config
        self.identity_layer_locked = True

        # Results storage
        self.demo_results = []

        print("🧠 HRM System Demo initialized")
        print("📋 Components loaded:")
        print("   ✅ HRM Router")
        print("   ✅ SubAgent Router")
        print("   ✅ Personality Formatter")
        print("   ✅ Core Arbiter")
        print("   ✅ Mirror Mode Manager")
        print(f"   🔒 Identity Layer Protection: {'ENABLED' if self.identity_layer_locked else 'DISABLED'}")

    def update_identity_layer(self, identity_data: Dict[str, Any], admin_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Attempt to update the HRM identity layer with write protection.

        Args:
            identity_data: New identity data to apply
            admin_key: Admin authentication key

        Returns:
            Dictionary with operation result
        """
        if self.identity_layer_locked and admin_key != self.identity_admin_key:
            unauthorized_attempt = {
                "success": False,
                "error": "UNAUTHORIZED_ACCESS",
                "message": "Identity layer write access denied. Admin key required.",
                "timestamp": datetime.now().isoformat(),
                "attempted_changes": list(identity_data.keys()) if identity_data else [],
                "source": "hrm_system_demo"
            }

            # Log unauthorized attempt
            print(f"🚨 SECURITY: Unauthorized identity layer access attempt")
            print(f"   Attempted changes: {list(identity_data.keys())}")
            print(f"   Admin key provided: {'Yes' if admin_key else 'No'}")
            print(f"   Key valid: {'Yes' if admin_key == self.identity_admin_key else 'No'}")

            # TODO: Log to security audit system
            self._log_security_violation(unauthorized_attempt)

            return unauthorized_attempt

        # Admin access granted - proceed with identity layer update
        try:
            # Validate identity data structure
            if not isinstance(identity_data, dict):
                return {
                    "success": False,
                    "error": "INVALID_DATA_FORMAT",
                    "message": "Identity data must be a dictionary",
                    "timestamp": datetime.now().isoformat()
                }

            # Apply identity layer changes (simulated)
            applied_changes = []
            for key, value in identity_data.items():
                # Simulate identity layer update logic
                applied_changes.append(f"{key}: {value}")

            result = {
                "success": True,
                "message": "Identity layer updated successfully",
                "applied_changes": applied_changes,
                "timestamp": datetime.now().isoformat(),
                "admin_authorized": True
            }

            print(f"✅ ADMIN: Identity layer updated successfully")
            print(f"   Changes applied: {len(applied_changes)}")
            print(f"   Admin key verified: ✓")

            return result

        except Exception as e:
            error_result = {
                "success": False,
                "error": "UPDATE_FAILED",
                "message": f"Identity layer update failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

            print(f"❌ ERROR: Identity layer update failed: {str(e)}")
            return error_result

    def _log_security_violation(self, violation_data: Dict[str, Any]):
        """Log security violations for audit purposes"""
        try:
            # Create security logs directory
            security_logs_path = Path("security_logs")
            security_logs_path.mkdir(exist_ok=True)

            # Log to security audit file
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = security_logs_path / f"hrm_security_violations_{timestamp}.jsonl"

            with open(log_file, 'a') as f:
                f.write(json.dumps(violation_data) + '\n')

            print(f"   🔍 Security violation logged to: {log_file}")

        except Exception as e:
            print(f"   ⚠️  Failed to log security violation: {str(e)}")

    def test_identity_layer_protection(self):
        """Test identity layer write protection with various scenarios"""
        print(f"\n🔒 TESTING IDENTITY LAYER WRITE PROTECTION")
        print("=" * 60)

        test_scenarios = [
            {
                "name": "Unauthorized access attempt",
                "data": {"core_personality": "aggressive", "memory_retention": "disabled"},
                "admin_key": None,
                "expected_outcome": "DENIED"
            },
            {
                "name": "Invalid admin key attempt",
                "data": {"emotional_baseline": "neutral"},
                "admin_key": "wrong_key",
                "expected_outcome": "DENIED"
            },
            {
                "name": "Valid admin access",
                "data": {"personality_stability": "high", "learning_rate": 0.1},
                "admin_key": self.identity_admin_key,
                "expected_outcome": "ALLOWED"
            }
        ]

        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 Test {i}: {scenario['name']}")
            print(f"   Data: {scenario['data']}")
            print(f"   Admin key: {'PROVIDED' if scenario['admin_key'] else 'NONE'}")

            result = self.update_identity_layer(scenario['data'], scenario['admin_key'])

            success_matches_expected = (
                (result['success'] and scenario['expected_outcome'] == 'ALLOWED') or
                (not result['success'] and scenario['expected_outcome'] == 'DENIED')
            )

            print(f"   Result: {'✅ PASS' if success_matches_expected else '❌ FAIL'}")
            print(f"   Expected: {scenario['expected_outcome']}, Got: {'ALLOWED' if result['success'] else 'DENIED'}")

        print(f"\n🔒 Identity layer protection testing complete")

    async def run_comprehensive_demo(self):
        """Run the complete HRM system demonstration"""

        print("\n" + "="*80)
        print("🎯 HIERARCHICAL REASONING MODEL (HRM) SYSTEM DEMONSTRATION")
        print("="*80)

        # Test scenarios covering different types of interactions
        test_scenarios = [
            {
                "name": "Technical Problem Solving",
                "input": "Can you help me implement a efficient sorting algorithm in Python with O(n log n) complexity?",
                "context": {"user_expertise": "intermediate", "priority": 0.8},
                "expected_mode": HRMMode.TECHNICAL,
                "expected_agent": AgentType.TECHNICAL
            },
            {
                "name": "Emotional Support Request",
                "input": "I've been feeling really anxious lately about my job security and it's affecting my sleep",
                "context": {"mood": "anxiety", "emotional_intensity": 0.7},
                "expected_mode": HRMMode.THERAPEUTIC,
                "expected_agent": AgentType.EMOTIONAL
            },
            {
                "name": "Creative Writing Request",
                "input": "Write me a short story about a lonely lighthouse keeper who discovers something magical",
                "context": {"mood": "contemplative", "creativity_level": 0.9},
                "expected_mode": HRMMode.CREATIVE,
                "expected_agent": AgentType.CREATIVE
            },
            {
                "name": "Complex Analysis Request",
                "input": "Analyze the potential long-term impacts of AI development on employment markets",
                "context": {"depth_required": "high", "priority": 0.9},
                "expected_mode": HRMMode.ANALYTICAL,
                "expected_agent": AgentType.ANALYTICAL
            },
            {
                "name": "Memory and Context Recall",
                "input": "Do you remember what we discussed about my career goals in our previous conversation?",
                "context": {"conversation_history": ["career_discussion"], "user_id": "demo_user"},
                "expected_mode": HRMMode.BALANCED,
                "expected_agent": AgentType.MEMORY
            },
            {
                "name": "Ritual and Symbolic Meaning",
                "input": "I want to create a meaningful ritual for transitioning into a new chapter of my life",
                "context": {"symbolic_depth": 0.8, "personal_significance": 0.9},
                "expected_mode": HRMMode.EMOTION_LEAD,
                "expected_agent": AgentType.RITUAL
            }
        ]

        # Test identity layer protection
        self.test_identity_layer_protection()

        # Process each scenario
        for i, scenario in enumerate(test_scenarios, 1):
            await self._process_scenario(i, scenario)

            if i < len(test_scenarios):
                print("\n" + "-"*80 + "\n")

        # Show system analytics and performance
        await self._show_system_analytics()

        # Save results if configured
        if self.demo_config["save_results"]:
            await self._save_demo_results()

        print("\n" + "="*80)
        print("🎉 HRM System Demonstration Complete!")
        print("="*80)

    async def _process_scenario(self, scenario_num: int, scenario: Dict[str, Any]):
        """Process a single demo scenario through the complete HRM pipeline"""

        print(f"🎬 SCENARIO {scenario_num}: {scenario['name']}")
        print(f"📝 Input: {scenario['input']}")
        print(f"🎛️  Context: {scenario['context']}")

        start_time = time.time()
        scenario_result = {
            "scenario_name": scenario["name"],
            "input": scenario["input"],
            "context": scenario["context"],
            "timestamp": datetime.now().isoformat(),
            "processing_steps": []
        }

        try:
            # STEP 1: HRM Router Analysis and Mode Selection
            print(f"\n🧠 STEP 1: HRM Router Processing")
            hrm_response = await self.hrm_router.process_request(
                scenario["input"],
                scenario["context"]
            )

            step1_result = {
                "step": "hrm_router",
                "mode_selected": hrm_response.processing_mode.value,
                "confidence": hrm_response.confidence_score,
                "agents_involved": hrm_response.agents_involved,
                "processing_time": hrm_response.processing_time
            }
            scenario_result["processing_steps"].append(step1_result)

            if self.demo_config["show_intermediate_steps"]:
                print(f"   ⚙️  Mode Selected: {hrm_response.processing_mode.value}")
                print(f"   🎯 Confidence: {hrm_response.confidence_score:.2f}")
                print(f"   🤖 Agents Involved: {', '.join(hrm_response.agents_involved)}")
                print(f"   ⏱️  Processing Time: {hrm_response.processing_time:.3f}s")

            # STEP 2: SubAgent Router Specialized Processing
            print(f"\n🤖 STEP 2: SubAgent Router Processing")
            subagent_response = await self.subagent_router.route(
                scenario["input"],
                scenario["context"]
            )

            step2_result = {
                "step": "subagent_router",
                "agent_selected": subagent_response.agent_type.value,
                "intent_detected": subagent_response.intent_detected.value,
                "confidence": subagent_response.confidence,
                "processing_time": subagent_response.processing_time
            }
            scenario_result["processing_steps"].append(step2_result)

            if self.demo_config["show_intermediate_steps"]:
                print(f"   🎯 Agent Selected: {subagent_response.agent_type.value}")
                print(f"   🧭 Intent Detected: {subagent_response.intent_detected.value}")
                print(f"   📊 Confidence: {subagent_response.confidence:.2f}")
                print(f"   ⏱️  Processing Time: {subagent_response.processing_time:.3f}s")

            # STEP 3: Personality Formatting
            print(f"\n🎭 STEP 3: Personality Consistency Formatting")
            reformulation_request = ReformulationRequest(
                original_response=subagent_response.content,
                agent_type=subagent_response.agent_type.value,
                intent_detected=subagent_response.intent_detected.value,
                user_context=scenario["context"],
                personality_context={}
            )

            formatted_response = await self.personality_formatter.format(reformulation_request)

            step3_result = {
                "step": "personality_formatter",
                "personality_applied": formatted_response.metadata.get("target_personality", "unknown"),
                "adjustments_made": formatted_response.personality_adjustments,
                "tone_changes": [change.value for change in formatted_response.tone_changes],
                "confidence": formatted_response.reformulation_confidence,
                "processing_time": formatted_response.processing_time
            }
            scenario_result["processing_steps"].append(step3_result)

            if self.demo_config["show_intermediate_steps"]:
                print(f"   🎭 Personality Applied: {formatted_response.metadata.get('target_personality', 'unknown')}")
                print(f"   🔧 Adjustments Made: {', '.join(formatted_response.personality_adjustments)}")
                print(f"   🎵 Emotional Tone: {formatted_response.emotional_tone}")
                print(f"   📊 Confidence: {formatted_response.reformulation_confidence:.2f}")
                print(f"   ⏱️  Processing Time: {formatted_response.processing_time:.3f}s")

            # STEP 4: Mirror Mode Self-Reflection (if enabled)
            final_response = formatted_response.content

            if self.demo_config["enable_mirror_mode"]:
                print(f"\n🪩 STEP 4: Mirror Mode Self-Reflection")

                mirror_context = {
                    "original_response": final_response,
                    "processing_mode": hrm_response.processing_mode.value,
                    "agent_used": subagent_response.agent_type.value,
                    "confidence": formatted_response.reformulation_confidence
                }

                enhanced_response = self.mirror_manager.add_mirror_reflection(
                    final_response,
                    mirror_context,
                    [MirrorType.REASONING, MirrorType.EMOTIONAL, MirrorType.ROUTING]
                )

                mirror_reflection = enhanced_response[len(final_response):].strip() if enhanced_response != final_response else None

                step4_result = {
                    "step": "mirror_mode",
                    "reflection_added": mirror_reflection is not None,
                    "reflection_types": ["reasoning", "emotional", "routing"],
                    "reflection_content": mirror_reflection[:100] + "..." if mirror_reflection and len(mirror_reflection) > 100 else mirror_reflection
                }
                scenario_result["processing_steps"].append(step4_result)

                if self.demo_config["show_intermediate_steps"]:
                    print(f"   🪩 Reflection Added: {'Yes' if mirror_reflection else 'No'}")
                    if mirror_reflection:
                        print(f"   💭 Reflection Preview: {mirror_reflection[:100]}...")

                final_response = enhanced_response

            # Calculate total processing time
            total_time = time.time() - start_time

            # FINAL RESULTS
            print(f"\n✨ FINAL RESULT")
            print(f"📊 Total Processing Time: {total_time:.3f}s")
            print(f"🎯 Overall Pipeline Confidence: {self._calculate_pipeline_confidence(scenario_result):.2f}")
            print(f"📝 Final Response Length: {len(final_response)} characters")
            print(f"\n💬 FINAL RESPONSE:")
            print("─" * 60)
            print(final_response[:500] + ("..." if len(final_response) > 500 else ""))
            print("─" * 60)

            # Store complete result
            scenario_result.update({
                "total_processing_time": total_time,
                "pipeline_confidence": self._calculate_pipeline_confidence(scenario_result),
                "final_response": final_response,
                "final_response_length": len(final_response),
                "success": True
            })

        except Exception as e:
            print(f"❌ ERROR in scenario processing: {str(e)}")
            scenario_result.update({
                "error": str(e),
                "success": False,
                "total_processing_time": time.time() - start_time
            })

        self.demo_results.append(scenario_result)

    def _calculate_pipeline_confidence(self, scenario_result: Dict[str, Any]) -> float:
        """Calculate overall pipeline confidence from individual step confidences"""
        confidences = []

        for step in scenario_result["processing_steps"]:
            if "confidence" in step:
                confidences.append(step["confidence"])

        if not confidences:
            return 0.0

        # Use harmonic mean for conservative confidence estimate
        harmonic_mean = len(confidences) / sum(1/c for c in confidences if c > 0)
        return min(1.0, harmonic_mean)

    async def _show_system_analytics(self):
        """Show comprehensive system analytics and performance metrics"""

        print(f"\n📊 SYSTEM ANALYTICS & PERFORMANCE METRICS")
        print("=" * 80)

        # HRM Router Analytics
        print(f"\n🧠 HRM Router Performance:")
        hrm_status = self.hrm_router.get_system_status()
        print(f"   Total Requests: {hrm_status['metrics']['total_requests']}")
        print(f"   Success Rate: {hrm_status['metrics']['successful_responses']/hrm_status['metrics']['total_requests']*100:.1f}%")
        print(f"   Avg Processing Time: {hrm_status['metrics']['average_processing_time']:.3f}s")
        print(f"   Mode Distribution: {hrm_status['metrics']['mode_distribution']}")

        # SubAgent Router Analytics
        print(f"\n🤖 SubAgent Router Performance:")
        subagent_analytics = self.subagent_router.get_routing_analytics()
        print(f"   Total Routes: {subagent_analytics['total_routes']}")
        print(f"   Success Rate: {subagent_analytics['success_rate']:.1%}")
        print(f"   Agent Utilization: {subagent_analytics['agent_utilization']}")

        # Personality Formatter Analytics
        print(f"\n🎭 Personality Formatter Performance:")
        formatter_analytics = self.personality_formatter.get_formatting_analytics()
        print(f"   Total Reformulations: {formatter_analytics['total_reformulations']}")
        print(f"   Success Rate: {formatter_analytics['success_rate']:.1%}")
        print(f"   Avg Confidence: {formatter_analytics['average_confidence']:.2f}")
        print(f"   Personality Distribution: {formatter_analytics['personality_distribution']}")

        # Core Arbiter Status
        print(f"\n⚖️  Core Arbiter System Health:")
        arbiter_status = self.core_arbiter.get_system_status()
        print(f"   Health Status: {arbiter_status['health_status']}")
        print(f"   Stability Score: {arbiter_status['drift_state']['stability_score']:.2f}")
        print(f"   Memory Usage: {arbiter_status.get('memory_usage', 'N/A')}")

        # Demo Results Summary
        print(f"\n📈 Demo Results Summary:")
        successful_scenarios = sum(1 for result in self.demo_results if result.get('success', False))
        total_scenarios = len(self.demo_results)
        avg_processing_time = sum(result.get('total_processing_time', 0) for result in self.demo_results) / max(total_scenarios, 1)
        avg_confidence = sum(result.get('pipeline_confidence', 0) for result in self.demo_results) / max(total_scenarios, 1)

        print(f"   Scenarios Completed: {total_scenarios}")
        print(f"   Success Rate: {successful_scenarios/total_scenarios*100:.1f}%")
        print(f"   Avg Pipeline Processing Time: {avg_processing_time:.3f}s")
        print(f"   Avg Pipeline Confidence: {avg_confidence:.2f}")

    async def _save_demo_results(self):
        """Save demo results to file for analysis"""
        results_path = Path("demo_results")
        results_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_path / f"hrm_demo_results_{timestamp}.json"

        demo_summary = {
            "demo_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(self.demo_results),
                "demo_config": self.demo_config
            },
            "system_analytics": {
                "hrm_router": self.hrm_router.get_system_status(),
                "subagent_router": self.subagent_router.get_routing_analytics(),
                "personality_formatter": self.personality_formatter.get_formatting_analytics(),
                "core_arbiter": self.core_arbiter.get_system_status()
            },
            "scenario_results": self.demo_results
        }

        with open(filename, 'w') as f:
            json.dump(demo_summary, f, indent=2, default=str)

        print(f"\n💾 Demo results saved to: {filename}")

    async def interactive_mode(self):
        """Interactive mode for real-time HRM testing"""

        print(f"\n🎮 INTERACTIVE HRM SYSTEM MODE")
        print("=" * 50)
        print("Enter messages to test the complete HRM pipeline")
        print("Type 'quit' to exit, 'help' for commands")
        print()

        while True:
            try:
                user_input = input("🗣️  You: ").strip()

                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_interactive_help()
                    continue
                elif user_input.lower() == 'status':
                    await self._show_system_analytics()
                    continue
                elif not user_input:
                    continue

                print("\n🧠 Processing through HRM pipeline...")

                # Process through HRM system
                start_time = time.time()

                # Simple context for interactive mode
                context = {
                    "user_id": "interactive_user",
                    "session_id": f"interactive_{int(time.time())}"
                }

                hrm_response = await self.hrm_router.process_request(user_input, context)

                processing_time = time.time() - start_time

                print(f"\n🤖 AI Response:")
                print("─" * 50)
                print(hrm_response.primary_response)
                print("─" * 50)
                print(f"⚙️  Mode: {hrm_response.processing_mode.value} | 📊 Confidence: {hrm_response.confidence_score:.2f} | ⏱️  Time: {processing_time:.2f}s")

                if hrm_response.mirror_reflection:
                    print(f"\n🪩 Mirror Reflection:")
                    print(hrm_response.mirror_reflection)

                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

        print("👋 Interactive mode ended")

    def _show_interactive_help(self):
        """Show help for interactive mode"""
        print(f"\n📖 INTERACTIVE MODE COMMANDS:")
        print(f"   • Type any message to process through HRM")
        print(f"   • 'status' - Show system analytics")
        print(f"   • 'help' - Show this help")
        print(f"   • 'quit' - Exit interactive mode")
        print()


async def main():
    """Main demo runner with different modes"""

    print("🚀 HRM System Demo Launcher")
    print("=" * 40)
    print("1. Comprehensive Demo - Full system demonstration")
    print("2. Interactive Mode - Real-time testing")
    print("3. Quick Performance Test - Speed benchmarks")
    print()

    try:
        choice = input("Select mode (1-3): ").strip()

        demo = HRMSystemDemo()

        if choice == "1":
            await demo.run_comprehensive_demo()
        elif choice == "2":
            await demo.interactive_mode()
        elif choice == "3":
            await run_performance_test()
        else:
            print("Invalid choice. Running comprehensive demo...")
            await demo.run_comprehensive_demo()

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")


async def run_performance_test():
    """Run quick performance benchmarks"""
    print("\n⚡ HRM SYSTEM PERFORMANCE TEST")
    print("=" * 40)

    demo = HRMSystemDemo()

    # Quick test messages
    test_messages = [
        "Hello, how are you?",
        "Can you help me with Python?",
        "I'm feeling anxious today",
        "Write a short poem",
        "Analyze climate change data"
    ]

    total_time = 0
    successful_requests = 0

    for i, message in enumerate(test_messages, 1):
        print(f"Test {i}/{len(test_messages)}: {message[:30]}...")

        start_time = time.time()
        try:
            response = await demo.hrm_router.process_request(message, {})
            processing_time = time.time() - start_time
            total_time += processing_time
            successful_requests += 1

            print(f"   ✅ Processed in {processing_time:.3f}s (confidence: {response.confidence_score:.2f})")

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")

    print(f"\n📊 PERFORMANCE RESULTS:")
    print(f"   Total Tests: {len(test_messages)}")
    print(f"   Successful: {successful_requests}")
    print(f"   Success Rate: {successful_requests/len(test_messages)*100:.1f}%")
    print(f"   Total Time: {total_time:.3f}s")
    print(f"   Average Time: {total_time/len(test_messages):.3f}s per request")


if __name__ == "__main__":
    asyncio.run(main())
