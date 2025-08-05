#!/usr/bin/env python3
"""
SubAgent Router System - Example Usage
Demonstrates practical usage of the multi-LLM orchestration system
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.subagent_router import SubAgentRouter
from backend.ai_reformulator import PersonalityFormatter, ReformulationRequest

class SubAgentDemo:
    """Demo class showing SubAgent Router usage"""
    
    def __init__(self):
        self.router = SubAgentRouter()
        self.formatter = PersonalityFormatter()
        
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a message through the complete SubAgent pipeline
        
        Args:
            message: User's input message
            context: Context including mood, conversation history, etc.
            
        Returns:
            Complete response with metadata
        """
        if context is None:
            context = {}
            
        start_time = time.time()
        
        # Step 1: Route to appropriate agent
        print(f"🎯 Routing message: '{message[:50]}...'")
        agent_response = await self.router.route(message, context)
        
        print(f"   → Routed to {agent_response.agent_type} agent")
        print(f"   → Intent detected: {agent_response.intent_detected.value}")
        print(f"   → Confidence: {agent_response.confidence:.2f}")
        
        # Step 2: Format for personality consistency
        print(f"🎭 Formatting for personality consistency...")
        format_request = ReformulationRequest(
            original_response=agent_response.content,
            agent_type=agent_response.agent_type,
            intent_detected=agent_response.intent_detected.value,
            user_context=context,
            personality_context={}
        )
        
        formatted_response = await self.formatter.format(format_request)
        
        print(f"   → Tone: {formatted_response.emotional_tone}")
        print(f"   → Formatting confidence: {formatted_response.reformulation_confidence:.2f}")
        
        total_time = time.time() - start_time
        
        return {
            "final_response": formatted_response.content,
            "agent_used": agent_response.agent_type,
            "intent_detected": agent_response.intent_detected.value,
            "routing_confidence": agent_response.confidence,
            "formatting_confidence": formatted_response.reformulation_confidence,
            "emotional_tone": formatted_response.emotional_tone,
            "processing_time": total_time,
            "metadata": {
                "original_response": agent_response.content,
                "personality_adjustments": formatted_response.personality_adjustments,
                "agent_metadata": agent_response.metadata
            }
        }
    
    async def demo_conversation(self):
        """Run a demo conversation showing different agent types"""
        
        print("🤖 SubAgent Router System Demo")
        print("=" * 60)
        print("This demo shows how different types of messages are routed to specialized agents\n")
        
        # Demo scenarios
        scenarios = [
            {
                "message": "Help me implement a function to calculate fibonacci numbers in Python",
                "context": {"mood": "focused", "project_type": "python"},
                "description": "🔧 Technical/Code Request"
            },
            {
                "message": "Write me a short story about a lighthouse keeper who finds meaning in solitude",
                "context": {"mood": "contemplative", "conversation_depth": 0.7},
                "description": "🎨 Creative/Artistic Request"
            },
            {
                "message": "Do you remember what we discussed about personal growth in our last conversation?",
                "context": {
                    "conversation_history": ["We talked about overcoming challenges"],
                    "total_interactions": 8,
                    "mood": "reflective"
                },
                "description": "🧠 Memory/Recall Request"
            },
            {
                "message": "I'm feeling overwhelmed with all the changes happening in my life",
                "context": {"mood": "anxious", "conversation_depth": 0.6},
                "description": "💭 Emotional Support Request"
            },
            {
                "message": "Create a meaningful ritual for starting fresh after a difficult period",
                "context": {"mood": "hopeful", "conversation_depth": 0.8},
                "description": "🕯️ Ritual/Symbolic Request"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{scenario['description']}")
            print("-" * 40)
            
            result = await self.process_message(
                scenario["message"], 
                scenario["context"]
            )
            
            print(f"\n💬 Final Response:")
            print(f"{result['final_response'][:200]}...")
            if len(result['final_response']) > 200:
                print("   [Response truncated for demo]")
            
            print(f"\n📊 Processing Stats:")
            print(f"   Processing time: {result['processing_time']:.2f}s")
            print(f"   Agent: {result['agent_used']}")
            print(f"   Emotional tone: {result['emotional_tone']}")
            
            if i < len(scenarios):
                print("\n" + "=" * 60 + "\n")
        
        # Show analytics
        print("\n📈 System Analytics")
        print("-" * 40)
        analytics = self.router.get_routing_analytics()
        print(f"Total routes processed: {analytics['total_routes']}")
        print(f"Intent distribution: {json.dumps(analytics['intent_distribution'], indent=2)}")
        print(f"Available agents: {', '.join(analytics['available_agents'])}")

async def interactive_demo():
    """Interactive demo where user can input messages"""
    
    demo = SubAgentDemo()
    
    print("\n🎮 Interactive SubAgent Demo")
    print("=" * 60)
    print("Enter messages to see how they're routed to different agents")
    print("Type 'quit' to exit, 'help' for examples\n")
    
    while True:
        try:
            message = input("👤 You: ").strip()
            
            if message.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif message.lower() == 'help':
                print("\n💡 Example messages to try:")
                print("  • 'Debug this Python error: NameError'")
                print("  • 'Tell me a story about courage'") 
                print("  • 'What did we talk about before?'")
                print("  • 'I need emotional support today'")
                print("  • 'Create a ritual for new beginnings'\n")
                continue
            elif not message:
                continue
            
            # Determine context based on message content
            context: Dict[str, Any] = {"mood": "neutral"}
            if any(word in message.lower() for word in ["sad", "overwhelmed", "anxious", "difficult"]):
                context["mood"] = "anxious"
            elif any(word in message.lower() for word in ["creative", "story", "beautiful", "imagine"]):
                context["mood"] = "creative"
            elif any(word in message.lower() for word in ["remember", "recall", "previous", "before"]):
                context["conversation_history"] = ["Previous conversations"]
                context["total_interactions"] = 5
            
            print(f"\n🤖 AI: Processing...")
            result = await demo.process_message(message, context)
            
            print(f"🤖 AI ({result['agent_used']}): {result['final_response']}\n")
            print(f"   [Intent: {result['intent_detected']}, Tone: {result['emotional_tone']}, Time: {result['processing_time']:.2f}s]\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo runner"""
    
    print("🚀 SubAgent Router System - Complete Demo")
    print("=" * 60)
    print("Choose a demo mode:")
    print("1. Automated conversation demo")
    print("2. Interactive demo")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    async def run_demos():
        demo = SubAgentDemo()
        
        if choice in ['1', '3']:
            await demo.demo_conversation()
        
        if choice in ['2', '3']:
            if choice == '3':
                input("\nPress Enter to start interactive demo...")
            await interactive_demo()
    
    asyncio.run(run_demos())

if __name__ == "__main__":
    main()
