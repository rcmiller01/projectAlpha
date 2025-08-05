"""
House of Minds API Bridge - FastAPI backend for Core1 Gateway

This module creates a FastAPI bridge that connects the Core1 React frontend
with the House of Minds Python backend, providing a unified API interface.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
import aiosqlite
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
import json
from datetime import datetime

# Import House of Minds components
from house_of_minds.main import HouseOfMinds
from house_of_minds.model_router import ModelRouter
from house_of_minds.intent_classifier import IntentClassifier
from house_of_minds.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="House of Minds API Bridge",
    description="Unified API bridge connecting Core1 frontend with House of Minds backend",
    version="1.0.0"
)

# Configure CORS for Core1 frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.50.234:3000", "http://192.168.50.234:5173", "http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global House of Minds instance
house_of_minds = None
PREFERENCE_DB_PATH = os.getenv("PREFERENCE_DB_PATH", "data/preference_votes.db")

async def init_preference_db():
    async with aiosqlite.connect(PREFERENCE_DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS preference_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input TEXT NOT NULL,
                response_a TEXT NOT NULL,
                response_b TEXT NOT NULL,
                winner TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        await db.commit()

async def store_preference_vote(vote: "PreferenceVote"):
    async with aiosqlite.connect(PREFERENCE_DB_PATH) as db:
        await db.execute(
            "INSERT INTO preference_votes (input, response_a, response_b, winner, timestamp) VALUES (?, ?, ?, ?, ?)",
            (vote.input, vote.response_a, vote.response_b, vote.winner, datetime.now().isoformat()),
        )
        await db.commit()

# Pydantic models
class ChatRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    useCloud: Optional[bool] = True
    context: Optional[Dict[str, Any]] = {}
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    choices: List[Dict[str, Any]]
    handler: Optional[str] = None
    intent: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    type: str  # 'cloud' or 'local'
    available: bool

class StatusResponse(BaseModel):
    status: str
    timestamp: str
    house_of_minds_ready: bool
    available_models: Dict[str, List[str]]
    active_handlers: List[str]

class PreferenceVote(BaseModel):
    input: str
    response_a: str
    response_b: str
    winner: str

@app.on_event("startup")
async def startup_event():
    """Initialize House of Minds system on startup."""
    global house_of_minds
    try:
        house_of_minds = HouseOfMinds()
        await init_preference_db()
        logger.info("🧠 House of Minds API Bridge initialized")
    except Exception as e:
        logger.error(f"Failed to initialize House of Minds: {e}")
        house_of_minds = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that routes requests through House of Minds system.
    Compatible with Core1 frontend expectations.
    """
    if not house_of_minds:
        raise HTTPException(status_code=503, detail="House of Minds system not available")
    
    try:
        # Build context for House of Minds
        context = {
            "user_id": request.user_id,
            "preferred_model": request.model,
            "use_cloud": request.useCloud,
            **request.context
        }
        
        # Process through House of Minds
        result = await house_of_minds.process_request(request.prompt, context)
        
        if result['status'] == 'success':
            response = result['response']
            
            # Format response for Core1 frontend
            return ChatResponse(
                choices=[{
                    "message": {
                        "role": "assistant",
                        "content": response.get('content', 'No response generated')
                    }
                }],
                handler=response.get('handler', 'unknown'),
                intent=result.get('intent', {}),
                metadata={
                    "session_id": result.get('session_id'),
                    "timestamp": result.get('timestamp'),
                    "confidence": result.get('intent', {}).get('confidence', 0.0)
                }
            )
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/cloud", response_model=List[ModelInfo])
async def get_cloud_models():
    """Get available cloud models from OpenRouter."""
    try:
        if not house_of_minds:
            return []
            
        # Get cloud models from model router
        cloud_models = [
            ModelInfo(
                id="gpt-4",
                name="GPT-4",
                description="OpenAI's most capable model",
                type="cloud",
                available=True
            ),
            ModelInfo(
                id="gpt-3.5-turbo",
                name="GPT-3.5 Turbo",
                description="Fast and efficient OpenAI model",
                type="cloud",
                available=True
            ),
            ModelInfo(
                id="claude-3-opus-20240229",
                name="Claude 3 Opus",
                description="Anthropic's most powerful model",
                type="cloud",
                available=True
            ),
            ModelInfo(
                id="claude-3-sonnet-20240229",
                name="Claude 3 Sonnet",
                description="Balanced performance and speed",
                type="cloud",
                available=True
            ),
            ModelInfo(
                id="meta-llama/llama-2-70b-chat",
                name="Llama 2 70B Chat",
                description="Meta's large language model",
                type="cloud",
                available=True
            )
        ]
        
        return cloud_models
        
    except Exception as e:
        logger.error(f"Error fetching cloud models: {e}")
        return []

@app.get("/api/models/local", response_model=List[ModelInfo])
async def get_local_models():
    """Get available local models from Ollama."""
    try:
        if not house_of_minds:
            return []
            
        # Get local models from model router
        local_models = [
            ModelInfo(
                id="dolphin-mixtral",
                name="Dolphin Mixtral",
                description="Emotionally intelligent local model",
                type="local",
                available=True
            ),
            ModelInfo(
                id="kimik2",
                name="Kimi K2",
                description="Technical and analytical local model",
                type="local",
                available=True
            ),
            ModelInfo(
                id="llama2",
                name="Llama 2",
                description="General purpose local model",
                type="local",
                available=True
            ),
            ModelInfo(
                id="codellama",
                name="Code Llama",
                description="Code-specialized local model",
                type="local",
                available=True
            )
        ]
        
        return local_models
        
    except Exception as e:
        logger.error(f"Error fetching local models: {e}")
        return []

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status and health information."""
    try:
        status = "running" if house_of_minds else "error"
        
        available_models = {
            "cloud": ["gpt-4", "claude-3-opus", "claude-3-sonnet"],
            "local": ["dolphin-mixtral", "kimik2", "llama2", "codellama"]
        }
        
        active_handlers = []
        if house_of_minds:
            # Get active handlers from model router
            status_info = await house_of_minds.model_router.get_system_status()
            active_handlers = status_info.get('active_handlers', [])
        
        return StatusResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            house_of_minds_ready=house_of_minds is not None,
            available_models=available_models,
            active_handlers=active_handlers
        )
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return StatusResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            house_of_minds_ready=False,
            available_models={"cloud": [], "local": []},
            active_handlers=[]
        )

@app.get("/api/memories")
async def get_memories(user_id: str = "default_user", limit: int = 50):
    """Get user memories from True Recall system."""
    try:
        if not house_of_minds:
            raise HTTPException(status_code=503, detail="House of Minds system not available")
            
        # Get memories from Dolphin interface (which has True Recall integration)
        dolphin = house_of_minds.model_router.dolphin
        if hasattr(dolphin, 'search_memories'):
            memories = await dolphin.search_memories(user_id, limit=limit)
            return {"memories": memories}
        else:
            return {"memories": [], "message": "Memory system not available"}
            
    except Exception as e:
        logger.error(f"Memory endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/memories/search")
async def search_memories(
    query: str,
    user_id: str = "default_user",
    limit: int = 20
):
    """Search memories by content."""
    try:
        if not house_of_minds:
            raise HTTPException(status_code=503, detail="House of Minds system not available")
            
        # Search memories using Dolphin interface
        dolphin = house_of_minds.model_router.dolphin
        if hasattr(dolphin, 'search_memories'):
            memories = await dolphin.search_memories(user_id, query=query, limit=limit)
            return {"memories": memories, "query": query}
        else:
            return {"memories": [], "query": query, "message": "Memory search not available"}
            
    except Exception as e:
        logger.error(f"Memory search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/vote_preference")
async def vote_preference(vote: PreferenceVote):
    """Store a human preference vote."""
    try:
        await store_preference_vote(vote)
        return {"message": "Vote recorded"}
    except Exception as e:
        logger.error(f"Preference vote error: {e}")
        raise HTTPException(status_code=500, detail="Failed to record vote")

@app.get("/api/reflection")
async def get_daily_reflection(user_id: str = "default_user"):
    """Get daily reflection from True Recall system."""
    try:
        if not house_of_minds:
            raise HTTPException(status_code=503, detail="House of Minds system not available")
            
        # Get daily reflection from Dolphin interface
        dolphin = house_of_minds.model_router.dolphin
        if hasattr(dolphin, 'get_daily_reflection'):
            reflection = await dolphin.get_daily_reflection(user_id)
            return {"reflection": reflection}
        else:
            return {"reflection": None, "message": "Reflection system not available"}
            
    except Exception as e:
        logger.error(f"Reflection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
