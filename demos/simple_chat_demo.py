#!/usr/bin/env python3
"""
Simple integration example showing how to use CoreArbiter 
with the existing Dolphin AI chat system.
"""

import asyncio
import json
from pathlib import Path
from core_arbiter import CoreArbiter, WeightingStrategy

class SimpleEAIDemo:
    """Simple Emotional AI demonstration using CoreArbiter"""
    
    def __init__(self):
        self.arbiter = CoreArbiter()
        self.conversation_history = []
    
    async def chat(self, user_message: str, context: str = "general"):
        """Simple chat interface using CoreArbiter"""
        
        # Prepare state from conversation history
        state = {
            "context": context,
            "conversation_length": len(self.conversation_history),
            "recent_messages": self.conversation_history[-3:] if self.conversation_history else [],
            "emotional_continuity": self._analyze_emotional_continuity()
        }
        
        # Process through CoreArbiter
        response = await self.arbiter.process_input(user_message, state)
        
        # Store in conversation history
        self.conversation_history.append({
            "user": user_message,
            "assistant": response.final_output,
            "metadata": {
                "tone": response.tone,
                "confidence": response.confidence,
                "strategy": response.resolution_strategy,
                "emotional_override": response.emotional_override
            }
        })
        
        return response
    
    def _analyze_emotional_continuity(self) -> float:
        """Analyze emotional continuity in recent conversation"""
        if len(self.conversation_history) < 2:
            return 0.5
        
        # Simple heuristic: count emotional vs logical responses
        recent = self.conversation_history[-5:]
        emotional_count = sum(1 for msg in recent 
                             if msg.get("metadata", {}).get("emotional_override", False))
        
        return emotional_count / len(recent)
    
    def print_response(self, response, user_input: str):
        """Pretty print response"""
        print(f"\n👤 You: {user_input}")
        
        # Choose emoji based on tone
        emoji_map = {
            "emotional": "💖",
            "balanced": "💭", 
            "objective": "🤔"
        }
        emoji = emoji_map.get(response.tone, "🤖")
        
        print(f"{emoji} AI: {response.final_output}")
        
        # Show metadata
        if response.reflection:
            print(f"✨ Reflection: {response.reflection}")
        
        print(f"📊 {response.tone.title()} tone • {response.confidence:.0%} confidence • {response.resolution_strategy}")
        
        if response.emotional_override:
            print("🌊 Emotional override active")

async def main():
    """Run simple chat demo"""
    print("🌟 Simple Emotional AI Chat Demo")
    print("Using CoreArbiter for intelligent emotional responses")
    print("Type 'quit' to exit, 'status' for system info, 'strategy <name>' to change strategy\n")
    
    demo = SimpleEAIDemo()
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'status':
                status = demo.arbiter.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Health: {status['health_status'].upper()}")
                print(f"   Strategy: {status['weighting_strategy']}")
                print(f"   Decisions: {status['decision_count']}")
                print(f"   Stability: {status['drift_state']['stability_score']:.1%}")
                continue
            elif user_input.lower().startswith('strategy '):
                strategy_name = user_input.split(' ', 1)[1]
                try:
                    strategy = WeightingStrategy(strategy_name)
                    demo.arbiter.set_weighting_strategy(strategy)
                    print(f"✅ Strategy changed to: {strategy.value}")
                except ValueError:
                    print(f"❌ Invalid strategy. Options: logic_dominant, emotional_priority, harmonic, adaptive")
                continue
            elif not user_input:
                continue
            
            # Determine context based on input
            context = "general"
            if any(word in user_input.lower() for word in ["sad", "upset", "scared", "worried", "anxious"]):
                context = "emotional_support"
            elif any(word in user_input.lower() for word in ["analyze", "logic", "data", "calculate"]):
                context = "analytical_task"
            elif any(word in user_input.lower() for word in ["meaning", "purpose", "spiritual", "deep"]):
                context = "philosophical_inquiry"
            
            # Get response from CoreArbiter
            response = await demo.chat(user_input, context)
            demo.print_response(response, user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show final stats
    print(f"\n📈 Final Statistics:")
    print(f"   Conversations: {len(demo.conversation_history)}")
    
    if demo.conversation_history:
        emotional_responses = sum(1 for msg in demo.conversation_history 
                                if msg.get("metadata", {}).get("emotional_override", False))
        print(f"   Emotional responses: {emotional_responses}/{len(demo.conversation_history)}")
        
        avg_confidence = sum(msg.get("metadata", {}).get("confidence", 0) 
                           for msg in demo.conversation_history) / len(demo.conversation_history)
        print(f"   Average confidence: {avg_confidence:.1%}")
    
    status = demo.arbiter.get_system_status()
    print(f"   Final stability: {status['drift_state']['stability_score']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())
