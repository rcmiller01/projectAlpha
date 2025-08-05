"""
SubAgent Router Integration with FastAPI Backend
Demonstrates how to integrate the SubAgent Router system with the existing backend
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import time

# Import our SubAgent Router system
from backend.subagent_router import SubAgentRouter
from backend.ai_reformulator import PersonalityFormatter, ReformulationRequest

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

class SubAgentResponse(BaseModel):
    response: str
    agent_used: str
    intent_detected: str
    routing_confidence: float
    formatting_confidence: float
    emotional_tone: str
    processing_time: float
    metadata: Dict[str, Any]

class AnalyticsResponse(BaseModel):
    total_routes: int
    intent_distribution: Dict[str, int]
    agent_performance: Dict[str, Dict[str, float]]
    available_agents: List[str]

# FastAPI app setup
app = FastAPI(title="SubAgent Router API", version="1.0.0")

# Initialize SubAgent Router system
router = SubAgentRouter()
formatter = PersonalityFormatter()

@app.on_event("startup")
async def startup_event():
    """Initialize the SubAgent Router system on startup"""
    print("🚀 Initializing SubAgent Router System...")
    print(f"✅ Available agents: {list(router.agents.keys())}")

@app.post("/chat/subagent", response_model=SubAgentResponse)
async def chat_with_subagents(request: ChatRequest):
    """
    Process a chat message through the SubAgent Router system
    
    This endpoint:
    1. Routes the message to the appropriate specialized agent
    2. Formats the response for personality consistency
    3. Returns the final response with metadata
    """
    try:
        start_time = time.time()
        
        # Prepare context
        context = request.context or {}
        if request.user_id:
            context["user_id"] = request.user_id
        
        # Step 1: Route to appropriate agent
        agent_response = await router.route(request.message, context)
        
        # Step 2: Format for personality consistency
        format_request = ReformulationRequest(
            original_response=agent_response.content,
            agent_type=agent_response.agent_type,
            intent_detected=agent_response.intent_detected.value,
            user_context=context,
            personality_context={}
        )
        
        formatted_response = await formatter.format(format_request)
        
        total_time = time.time() - start_time
        
        return SubAgentResponse(
            response=formatted_response.content,
            agent_used=agent_response.agent_type,
            intent_detected=agent_response.intent_detected.value,
            routing_confidence=agent_response.confidence,
            formatting_confidence=formatted_response.reformulation_confidence,
            emotional_tone=formatted_response.emotional_tone,
            processing_time=total_time,
            metadata={
                "original_response": agent_response.content,
                "personality_adjustments": formatted_response.personality_adjustments,
                "agent_metadata": agent_response.metadata,
                "processing_notes": formatted_response.processing_notes
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SubAgent processing failed: {str(e)}")

@app.get("/analytics/routing", response_model=AnalyticsResponse)
async def get_routing_analytics():
    """Get analytics about routing decisions and agent performance"""
    try:
        analytics = router.get_routing_analytics()
        
        return AnalyticsResponse(
            total_routes=analytics["total_routes"],
            intent_distribution=analytics["intent_distribution"],
            agent_performance=analytics["agent_performance"],
            available_agents=analytics["available_agents"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics retrieval failed: {str(e)}")

@app.get("/agents/capabilities")
async def get_agent_capabilities():
    """Get information about available agents and their capabilities"""
    try:
        capabilities = {}
        
        for agent_name, agent in router.agents.items():
            if agent is not None and hasattr(agent, 'get_capabilities'):
                capabilities[agent_name] = agent.get_capabilities()
            else:
                capabilities[agent_name] = {
                    "agent_type": agent_name,
                    "status": "available" if agent_name == "conversational" else "mock",
                    "specialties": ["General conversation and emotional support"]
                }
        
        return capabilities
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capabilities retrieval failed: {str(e)}")

@app.post("/chat/intent-classify")
async def classify_intent(request: ChatRequest):
    """Classify the intent of a message without processing it"""
    try:
        intent, confidence = router.classifier.classify_intent(
            request.message, 
            request.context or {}
        )
        
        return {
            "message": request.message,
            "intent": intent.value,
            "confidence": confidence,
            "suggested_agent": router._make_routing_decision(intent, confidence, request.context or {}).agent_chosen
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intent classification failed: {str(e)}")

@app.get("/personality/profile")
async def get_personality_profile():
    """Get current personality profile and modifiers"""
    try:
        profile = formatter.get_current_personality_profile()
        
        return {
            "personality_traits": profile,
            "available_tones": list(formatter.tone_mappings.values()),
            "emotional_filters": list(formatter.emotional_filters.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Personality profile retrieval failed: {str(e)}")

@app.post("/personality/feedback")
async def apply_personality_feedback(user_response: str, ai_message: str, context: Optional[Dict[str, Any]] = None):
    """Apply user feedback to evolve personality preferences"""
    try:
        formatter.apply_personality_evolution_feedback(
            user_response=user_response,
            reformulated_response=ai_message,
            context=context or {}
        )
        
        return {"status": "feedback_applied", "message": "Personality evolution updated"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback application failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for the SubAgent Router system"""
    try:
        # Test basic functionality
        test_intent, test_confidence = router.classifier.classify_intent("Hello")
        
        return {
            "status": "healthy",
            "available_agents": list(router.agents.keys()),
            "total_routes_processed": len(router.routing_history),
            "test_classification": {
                "intent": test_intent.value,
                "confidence": test_confidence
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Example usage function
async def example_usage():
    """Example of how to use the API endpoints"""
    
    print("🔧 SubAgent Router API Example Usage")
    print("=" * 50)
    
    # Example requests
    examples = [
        {
            "message": "Help me debug this Python function",
            "context": {"mood": "focused", "project_type": "python"}
        },
        {
            "message": "Tell me a story about courage",
            "context": {"mood": "creative", "conversation_depth": 0.6}
        },
        {
            "message": "I'm feeling overwhelmed",
            "context": {"mood": "anxious"}
        }
    ]
    
    for example in examples:
        print(f"\n📝 Example: {example['message']}")
        
        # Simulate API call
        request = ChatRequest(**example)
        response = await chat_with_subagents(request)
        
        print(f"   Agent: {response.agent_used}")
        print(f"   Intent: {response.intent_detected}")
        print(f"   Response: {response.response[:100]}...")

if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())
    
    # To run the FastAPI server:
    # uvicorn subagent_integration:app --reload --port 8000
