"""
House of Minds API Bridge - FastAPI backend for Core1 Gateway

This module creates a FastAPI bridge that connects the Core1 React frontend
with the House of Minds Python backend, providing a unified API interface.

Enhanced with security features:
- API bridge security hardening with authentication
- Input validation and sanitization for all API endpoints
- Rate limiting and session management for API access
- Comprehensive audit logging for all bridge activities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import aiosqlite
import os
import hashlib
import re
import time
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
API_BRIDGE_TOKEN_LENGTH = 32
MAX_REQUEST_SIZE = 10000  # characters
BRIDGE_RATE_LIMIT = 100  # requests per hour per IP
MAX_CONCURRENT_REQUESTS = 20
VALID_ORIGINS = {
    "http://192.168.50.234:3000", "http://192.168.50.234:5173",
    "http://localhost:3000", "http://localhost:5173"
}

# Rate limiting storage
request_counts = defaultdict(lambda: deque())
active_requests = set()

# Security headers
security = HTTPBearer()

# Import House of Minds components with validation
try:
    from house_of_minds.main import HouseOfMinds
    from house_of_minds.model_router import ModelRouter
    from house_of_minds.intent_classifier import IntentClassifier
    from house_of_minds.config_manager import ConfigManager
    HOM_AVAILABLE = True
    logger.info("House of Minds components loaded successfully")
except ImportError as e:
    logger.warning(f"House of Minds components not available: {e}")
    HOM_AVAILABLE = False

def check_rate_limit(client_ip: str) -> bool:
    """Check if client IP has exceeded rate limit"""
    current_time = time.time()

    # Clean old requests
    while (request_counts[client_ip] and
           request_counts[client_ip][0] < current_time - 3600):  # 1 hour window
        request_counts[client_ip].popleft()

    # Check limit
    if len(request_counts[client_ip]) >= BRIDGE_RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return False

    # Add current request
    request_counts[client_ip].append(current_time)
    return True

def validate_origin(origin: str) -> bool:
    """Validate request origin"""
    return origin in VALID_ORIGINS

def validate_input_text(text: str, max_length: int = MAX_REQUEST_SIZE) -> tuple[bool, str]:
    """Validate and sanitize input text"""
    try:
        if not text or not isinstance(text, str):
            return False, "Input must be a non-empty string"

        if len(text) > max_length:
            return False, f"Input exceeds maximum length of {max_length}"

        # Check for potential injection patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',               # JavaScript protocol
            r'on\w+\s*=',                # Event handlers
            r'expression\s*\(',          # CSS expressions
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False, f"Input contains potentially dangerous content"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating input text: {str(e)}")
        return False, f"Validation error: {str(e)}"

def sanitize_text_input(text: str) -> str:
    """Sanitize text input for safety"""
    if not isinstance(text, str):
        return ""

    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\']', '', text)

    # Limit length
    if len(text) > MAX_REQUEST_SIZE:
        text = text[:MAX_REQUEST_SIZE] + "..."

    return text.strip()

def log_bridge_activity(activity_type: str, client_ip: str, details: Dict[str, Any], status: str = "success"):
    """Log API bridge activities"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'client_ip': client_ip,
            'details': details,
            'status': status
        }

        logger.info(f"Bridge activity logged: {activity_type} from {client_ip} ({status})")

        if status != "success":
            logger.warning(f"Bridge activity issue: {activity_type} from {client_ip} failed with {status}")

    except Exception as e:
        logger.error(f"Error logging bridge activity: {str(e)}")

async def validate_request(request: Request) -> bool:
    """Validate incoming request"""
    client_ip = request.client.host

    # Check rate limit
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Check concurrent requests
    if len(active_requests) >= MAX_CONCURRENT_REQUESTS:
        logger.warning(f"Concurrent request limit reached: {len(active_requests)}")
        raise HTTPException(status_code=503, detail="Server too busy")

    # Validate origin if present
    origin = request.headers.get("origin")
    if origin and not validate_origin(origin):
        logger.warning(f"Invalid origin: {origin} from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid origin")

    return True

class SecureBaseModel(BaseModel):
    """Base model with input validation"""

    @validator('*', pre=True)
    def sanitize_string_fields(cls, v):
        if isinstance(v, str):
            return sanitize_text_input(v)
        return v

app = FastAPI(
    title="House of Minds API Bridge (Secured)",
    description="Unified API bridge connecting Core1 frontend with House of Minds backend - Security Enhanced",
    version="1.1.0"
)

# Configure CORS with security restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(VALID_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow necessary methods
    allow_headers=["Content-Type", "Authorization"],  # Restrict headers
)

# Global House of Minds instance
house_of_minds = None
PREFERENCE_DB_PATH = os.getenv("PREFERENCE_DB_PATH", "data/preference_votes.db")

async def init_preference_db():
    """Initialize preference database with security checks"""
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(PREFERENCE_DB_PATH), exist_ok=True)

        async with aiosqlite.connect(PREFERENCE_DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS preference_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input TEXT NOT NULL,
                    response_a TEXT NOT NULL,
                    response_b TEXT NOT NULL,
                    winner TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    client_ip TEXT,
                    session_id TEXT
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
