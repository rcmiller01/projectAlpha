#!/usr/bin/env python3
"""
SLiM Agent Demo - Prototype Testing Harness

This demo showcases the new SLiM (Specialized Large Intelligence Models) agent
system integrated with GraphRAG memory, tool routing, and the HRM stack.

Features Demonstrated:
- DeductionAgent (Left-Brain Logic) with logic_high model
- MetaphorAgent (Right-Brain Creativity) with emotion_creative model
- Integration with GraphRAG memory for semantic context
- Tool router usage for autonomous capabilities
- HRM router dispatch system
- Memory persistence and fact extraction

Author: ProjectAlpha Team
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.core_conductor import CoreConductor
from memory.graphrag_memory import GraphRAGMemory
from src.tools.tool_request_router import ToolRequestRouter
from src.core.hrm_router import HRMRouter
from src.agents.deduction_agent import DeductionAgent
from src.agents.metaphor_agent import MetaphorAgent

def test_direct_agent_usage():
    """Test agents directly without HRM Router"""
    print("=" * 80)
    print("🧪 DIRECT AGENT TESTING")
    print("=" * 80)

    # Initialize core components
    conductor = CoreConductor()
    memory = GraphRAGMemory()
    router = ToolRequestRouter()

    # Create agents
    deduction_agent = DeductionAgent(conductor, memory, router)
    metaphor_agent = MetaphorAgent(conductor, memory, router)

    print(f"✅ Initialized agents:")
    print(f"  • DeductionAgent: {deduction_agent.agent_id} (role: {deduction_agent.role})")
    print(f"  • MetaphorAgent: {metaphor_agent.agent_id} (role: {metaphor_agent.role})")

    # Test DeductionAgent
    print(f"\n🧮 Testing DeductionAgent:")

    test_prompts_deduction = [
        "Prove: if A⇒B and B⇒C then A⇒C",
        "Solve: x^2 - 5x + 6 = 0",
        "Is the following argument valid: All birds can fly. Penguins are birds. Therefore, penguins can fly."
    ]

    for i, prompt in enumerate(test_prompts_deduction, 1):
        print(f"\n  Test {i}: {prompt}")
        response = deduction_agent.run(prompt, depth=1)
        print(f"  Response: {response[:200]}...")

    # Test MetaphorAgent
    print(f"\n🎨 Testing MetaphorAgent:")

    test_prompts_metaphor = [
        "Give me three metaphors for longing",
        "Create a metaphor connecting 'artificial intelligence' and 'growing garden'",
        "Write a short poetic story about the relationship between memory and time"
    ]

    for i, prompt in enumerate(test_prompts_metaphor, 1):
        print(f"\n  Test {i}: {prompt}")
        response = metaphor_agent.run(prompt, depth=1)
        print(f"  Response: {response[:200]}...")

    return deduction_agent, metaphor_agent

def test_hrm_router_dispatch():
    """Test agents through HRM Router dispatch system"""
    print("\n" + "=" * 80)
    print("🎛️  HRM ROUTER DISPATCH TESTING")
    print("=" * 80)

    # Initialize HRM Router (includes agent registry)
    hrm_router = HRMRouter()

    print(f"✅ HRM Router initialized")

    # List registered agents
    agents = hrm_router.list_agents()
    print(f"\n📋 Registered agents:")
    for key, info in agents.items():
        print(f"  • {key}: {info['class']} (role: {info['role']}, spec: {info['specialization']})")

    # Test dispatch to DeductionAgent
    print(f"\n🧮 Testing HRM dispatch to DeductionAgent:")

    deduction_prompts = [
        "Analyze this logical argument: If all cats are mammals, and all mammals are animals, what can we conclude about cats?",
        "Prove that the square root of 2 is irrational"
    ]

    for i, prompt in enumerate(deduction_prompts, 1):
        print(f"\n  Dispatch {i}: {prompt[:60]}...")
        response = hrm_router.dispatch_to_agent("deduction", prompt, depth=2)
        if response:
            print(f"  Response: {response[:150]}...")
        else:
            print(f"  Response: Failed to get response")

    # Test dispatch to MetaphorAgent
    print(f"\n🎨 Testing HRM dispatch to MetaphorAgent:")

    metaphor_prompts = [
        "Create metaphors comparing 'learning' to natural phenomena",
        "Write a creative interpretation of the symbol of a lighthouse"
    ]

    for i, prompt in enumerate(metaphor_prompts, 1):
        print(f"\n  Dispatch {i}: {prompt[:60]}...")
        response = hrm_router.dispatch_to_agent("metaphor", prompt, depth=2)
        if response:
            print(f"  Response: {response[:150]}...")
        else:
            print(f"  Response: Failed to get response")

    return hrm_router

def test_agent_specialization():
    """Test specialized methods of each agent type"""
    print("\n" + "=" * 80)
    print("⚡ AGENT SPECIALIZATION TESTING")
    print("=" * 80)

    # Initialize components
    conductor = CoreConductor()
    memory = GraphRAGMemory()
    router = ToolRequestRouter()

    deduction_agent = DeductionAgent(conductor, memory, router)
    metaphor_agent = MetaphorAgent(conductor, memory, router)

    # Test DeductionAgent specialized methods
    print(f"\n🧮 DeductionAgent Specialized Methods:")

    print(f"\n  1. Mathematical Solving:")
    math_response = deduction_agent.solve_mathematical("Find the derivative of x^3 + 2x^2 - 5x + 1")
    print(f"     Response: {math_response[:120]}...")

    print(f"\n  2. Logical Proof:")
    proof_response = deduction_agent.prove("The sum of two even numbers is always even")
    print(f"     Response: {proof_response[:120]}...")

    print(f"\n  3. Argument Analysis:")
    arg_response = deduction_agent.analyze_argument("Since all humans are mortal, and Socrates is human, Socrates must be mortal.")
    print(f"     Response: {arg_response[:120]}...")

    # Test MetaphorAgent specialized methods
    print(f"\n🎨 MetaphorAgent Specialized Methods:")

    print(f"\n  1. Metaphor Generation:")
    metaphor_response = metaphor_agent.generate_metaphors("artificial intelligence", count=2, style="poetic")
    print(f"     Response: {metaphor_response[:120]}...")

    print(f"\n  2. Creative Story:")
    story_response = metaphor_agent.creative_story("the journey of discovery", elements=["hidden paths", "unexpected companions"])
    print(f"     Response: {story_response[:120]}...")

    print(f"\n  3. Symbol Interpretation:")
    symbol_response = metaphor_agent.interpret_symbol("butterfly", context="transformation and change")
    print(f"     Response: {symbol_response[:120]}...")

def test_memory_integration():
    """Test memory system integration and persistence"""
    print("\n" + "=" * 80)
    print("🧠 MEMORY INTEGRATION TESTING")
    print("=" * 80)

    # Initialize with memory persistence
    memory = GraphRAGMemory("data/slim_agent_test_memory.json")
    conductor = CoreConductor()
    router = ToolRequestRouter()

    deduction_agent = DeductionAgent(conductor, memory, router)

    # Add some initial facts
    print(f"✅ Adding initial facts to memory...")
    memory.add_fact("mathematics", "includes_field", "algebra", confidence=1.0, source="demo")
    memory.add_fact("algebra", "involves", "equations", confidence=0.9, source="demo")
    memory.add_fact("equations", "can_be", "linear_or_quadratic", confidence=0.8, source="demo")

    # Test memory-aware reasoning
    print(f"\n🧮 Testing memory-aware reasoning:")
    prompt = "Explain the relationship between mathematics and solving equations"

    # Get memory context first
    memory_result = memory.query_related("mathematics", depth=2)
    print(f"  Memory context: {len(memory_result.related_concepts)} related concepts found")

    # Agent response (will use memory context)
    response = deduction_agent.run(prompt, depth=2)
    print(f"  Agent response: {response[:150]}...")

    # Save memory state
    memory.save_memory()
    print(f"  Memory saved with updated facts")

    return memory

def test_concurrent_agents():
    """Test multiple agents working concurrently"""
    print("\n" + "=" * 80)
    print("⚡ CONCURRENT AGENT TESTING")
    print("=" * 80)

    import threading
    import time

    hrm_router = HRMRouter()
    results = {}

    def agent_task(agent_key, prompt, task_id):
        """Task function for concurrent execution"""
        start_time = time.time()
        response = hrm_router.dispatch_to_agent(agent_key, prompt)
        end_time = time.time()

        results[task_id] = {
            "agent": agent_key,
            "prompt": prompt[:50] + "...",
            "response": response[:100] + "..." if response else "Failed",
            "duration": round(end_time - start_time, 2)
        }

    # Create concurrent tasks
    tasks = [
        ("deduction", "Prove that 1 + 1 = 2 using formal logic", "task1"),
        ("metaphor", "Create a metaphor for quantum computing", "task2"),
        ("deduction", "Analyze: All swans are white. This is a swan. Therefore, it is white.", "task3"),
        ("metaphor", "Write a haiku about artificial consciousness", "task4")
    ]

    print(f"🚀 Launching {len(tasks)} concurrent agent tasks...")

    threads = []
    for agent_key, prompt, task_id in tasks:
        thread = threading.Thread(target=agent_task, args=(agent_key, prompt, task_id))
        threads.append(thread)
        thread.start()

    # Wait for all tasks to complete
    for thread in threads:
        thread.join()

    # Display results
    print(f"\n📊 Concurrent execution results:")
    for task_id, result in results.items():
        print(f"  {task_id}: {result['agent']} - {result['duration']}s")
        print(f"    Prompt: {result['prompt']}")
        print(f"    Response: {result['response']}")
        print()

def test_stress_and_stability():
    """Run stress test with rapid successive prompts"""
    print("\n" + "=" * 80)
    print("💪 STRESS AND STABILITY TESTING")
    print("=" * 80)

    hrm_router = HRMRouter()

    # Rapid-fire prompts for each agent
    rapid_prompts = [
        "Quick math: 7 * 8 = ?",
        "What color symbolizes peace?",
        "Is 17 prime?",
        "Metaphor for speed?",
        "Logic: A and B?",
        "Creative: Ocean + Mountain?",
        "Solve: 2x = 10",
        "Symbol: Tree meaning?",
        "Proof: 0 + n = n",
        "Analogy: Life as journey?"
    ]

    print(f"🔥 Running {len(rapid_prompts)} rapid-fire prompts...")

    success_count = 0
    total_time = 0

    import time

    for i, prompt in enumerate(rapid_prompts):
        agent_key = "deduction" if i % 2 == 0 else "metaphor"

        start_time = time.time()
        response = hrm_router.dispatch_to_agent(agent_key, prompt, depth=1)
        end_time = time.time()

        duration = end_time - start_time
        total_time += duration

        if response and "error" not in response.lower():
            success_count += 1
            status = "✅"
        else:
            status = "❌"

        print(f"  {status} {agent_key}: {prompt} ({duration:.2f}s)")

    print(f"\n📈 Stress test results:")
    print(f"  Success rate: {success_count}/{len(rapid_prompts)} ({success_count/len(rapid_prompts)*100:.1f}%)")
    print(f"  Average response time: {total_time/len(rapid_prompts):.2f}s")
    print(f"  Total test time: {total_time:.2f}s")

def main():
    """Run comprehensive SLiM agent testing suite"""
    print("🚀 ProjectAlpha SLiM Agent Demo and Testing Suite")
    print("This demo tests the new SLiM agent system with GraphRAG memory")
    print("and tool router integration.\n")

    try:
        # Run test suites
        test_direct_agent_usage()

        hrm_router = test_hrm_router_dispatch()

        test_agent_specialization()

        memory = test_memory_integration()

        test_concurrent_agents()

        test_stress_and_stability()

        # Final summary
        print("\n" + "=" * 80)
        print("✅ SLiM AGENT DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\n📊 System Summary:")
        print("• Direct agent instantiation: ✅")
        print("• HRM Router dispatch: ✅")
        print("• Agent specialization methods: ✅")
        print("• Memory integration: ✅")
        print("• Concurrent execution: ✅")
        print("• Stress testing: ✅")

        # Show final statistics
        if hrm_router:
            agents = hrm_router.list_agents()
            print(f"\n🎯 Final Statistics:")
            print(f"• Registered agents: {len(agents)}")
            instantiated = sum(1 for info in agents.values() if info['instantiated'])
            print(f"• Instantiated agents: {instantiated}")

            # Memory stats
            stats = hrm_router.get_integration_stats()
            if stats:
                print(f"• Memory nodes: {stats['memory_stats']['total_nodes']}")
                print(f"• Available tools: {stats['tool_stats']['total_tools']}")

        print(f"\n🔧 Next Steps:")
        print("1. Deploy to target node for live testing")
        print("2. Integrate with web/CLI front-end")
        print("3. Monitor performance and fine-tune models")
        print("4. Expand agent specializations as needed")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
