"""
Unified Companion API

FastAPI application for the unified companion system providing
endpoints for interaction processing and system management.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging
import asyncio

# Internal imports (will be available when system is complete)
from modules.core.unified_companion import UnifiedCompanion
from modules.database.database_interface import create_database_interface, DatabaseInterface

# Pydantic models for API
class UserInteractionRequest(BaseModel):
    """Request model for user interaction"""
    user_input: str = Field(..., description="User's input message")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")

class UserInteractionResponse(BaseModel):
    """Response model for user interaction"""
    companion_response: str = Field(..., description="Companion's response")
    interaction_id: str = Field(..., description="Unique interaction identifier")
    context_analysis: Dict[str, Any] = Field(..., description="Analysis of interaction context")
    metadata: Dict[str, Any] = Field(..., description="Interaction metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class UserProfileRequest(BaseModel):
    """Request model for user profile creation/update"""
    display_name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

class UserProfileResponse(BaseModel):
    """Response model for user profile"""
    user_id: str
    display_name: Optional[str]
    preferences: Dict[str, Any]
    adaptive_profile: Dict[str, Any]
    created_at: datetime
    last_active: datetime

class SessionSummaryResponse(BaseModel):
    """Response model for session summary"""
    session_id: str
    user_id: str
    total_interactions: int
    primary_focuses: List[str]
    emotional_trajectory: List[Dict[str, Any]]
    session_summary: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime]

class SystemStatusResponse(BaseModel):
    """Response model for system status"""
    system_name: str = "Unified Companion"
    version: str = "1.0.0"
    status: str
    components: Dict[str, str]
    uptime: str
    active_users: int

# Global system components
companion_system: Optional[UnifiedCompanion] = None
database: Optional[DatabaseInterface] = None
system_config: Dict[str, Any] = {}

# Create FastAPI app
app = FastAPI(
    title="Unified Companion API",
    description="API for the adaptive companion system providing seamless support across personal, technical, and creative contexts",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global companion_system, database, system_config
    
    try:
        logger.info("Initializing Unified Companion System...")
        
        # Load system configuration
        system_config = {
            "mythomax": {
                "model_name": "TheBloke/MythoMax-L2-13B-GPTQ",
                "quantization": True,
                "max_length": 2048,
                "temperature": 0.7
            },
            "database": {
                "type": "inmemory",  # Will be mongodb in production
                "connection_string": None
            },
            "safety": {
                "crisis_detection": True,
                "content_filtering": True
            }
        }
        
        # Initialize database
        database = create_database_interface(
            connection_string=system_config["database"]["connection_string"],
            database_type=system_config["database"]["type"]
        )
        await database.initialize()
        
        # Initialize companion system
        companion_system = UnifiedCompanion(system_config)
        await companion_system.initialize()
        
        logger.info("Unified Companion System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup system components on shutdown"""
    global database
    
    try:
        if database:
            await database.close()
        logger.info("System shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

def get_companion_system() -> UnifiedCompanion:
    """Dependency to get companion system instance"""
    if companion_system is None:
        raise HTTPException(status_code=503, detail="Companion system not initialized")
    return companion_system

def get_database() -> DatabaseInterface:
    """Dependency to get database instance"""
    if database is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return database

@app.post("/api/v1/interact", response_model=UserInteractionResponse)
async def process_interaction(
    request: UserInteractionRequest,
    background_tasks: BackgroundTasks,
    companion: UnifiedCompanion = Depends(get_companion_system),
    db: DatabaseInterface = Depends(get_database)
):
    """
    Process user interaction and return companion response
    """
    try:
        # Generate unique interaction ID
        interaction_id = str(uuid.uuid4())
        
        # Build session context
        session_context = {
            "session_id": request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "interaction_id": interaction_id,
            **request.context
        }
        
        # Process interaction with companion system
        response_data = await companion.process_interaction(
            user_id=request.user_id,
            user_input=request.user_input,
            session_context=session_context
        )
        
        # Schedule background tasks for data persistence
        background_tasks.add_task(
            save_interaction_data,
            interaction_id,
            request,
            response_data,
            db
        )
        
        # Return response
        return UserInteractionResponse(
            companion_response=response_data["companion_response"],
            interaction_id=interaction_id,
            context_analysis=response_data["context_analysis"],
            metadata=response_data["interaction_metadata"]
        )
        
    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process interaction: {str(e)}")

async def save_interaction_data(
    interaction_id: str,
    request: UserInteractionRequest,
    response_data: Dict[str, Any],
    db: DatabaseInterface
):
    """Background task to save interaction data to database"""
    try:
        from modules.database.database_interface import InteractionRecord, InteractionType
        
        # Create interaction record
        interaction = InteractionRecord(
            interaction_id=interaction_id,
            user_id=request.user_id,
            session_id=request.session_id or response_data["interaction_metadata"]["session_id"],
            timestamp=datetime.now(),
            user_input=request.user_input,
            companion_response=response_data["companion_response"],
            interaction_type=InteractionType(response_data["context_analysis"]["primary_focus"]) if response_data["context_analysis"]["primary_focus"] in [t.value for t in InteractionType] else InteractionType.GENERAL_CONVERSATION,
            context_analysis=response_data["context_analysis"],
            emotional_state={},  # Will be populated by psychological modules
            technical_context={},
            creative_context={},
            guidance_used={},
            response_metrics={}
        )
        
        # Save to database
        await db.save_interaction(interaction)
        
    except Exception as e:
        logger.error(f"Error saving interaction data: {e}")

@app.post("/api/v1/users/{user_id}/profile", response_model=UserProfileResponse)
async def create_user_profile(
    user_id: str,
    profile_data: UserProfileRequest,
    db: DatabaseInterface = Depends(get_database)
):
    """Create or update user profile"""
    try:
        from modules.database.database_interface import UserProfile
        
        # Check if user exists
        existing_user = await db.get_user_profile(user_id)
        
        if existing_user:
            # Update existing user
            updates = {}
            if profile_data.display_name is not None:
                updates["display_name"] = profile_data.display_name
            if profile_data.preferences is not None:
                updates["preferences"] = profile_data.preferences
            
            success = await db.update_user_profile(user_id, updates)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update user profile")
            
            updated_user = await db.get_user_profile(user_id)
            if not updated_user:
                raise HTTPException(status_code=500, detail="Failed to retrieve updated profile")
            
            return UserProfileResponse(
                user_id=updated_user.user_id,
                display_name=updated_user.display_name,
                preferences=updated_user.preferences,
                adaptive_profile=updated_user.adaptive_profile,
                created_at=updated_user.created_at,
                last_active=updated_user.last_active
            )
        else:
            # Create new user
            new_user = UserProfile(
                user_id=user_id,
                created_at=datetime.now(),
                last_active=datetime.now(),
                display_name=profile_data.display_name,
                preferences=profile_data.preferences or {},
                adaptive_profile={}
            )
            
            success = await db.create_user_profile(new_user)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create user profile")
            
            return UserProfileResponse(
                user_id=new_user.user_id,
                display_name=new_user.display_name,
                preferences=new_user.preferences,
                adaptive_profile=new_user.adaptive_profile,
                created_at=new_user.created_at,
                last_active=new_user.last_active
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating/updating user profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process user profile: {str(e)}")

@app.get("/api/v1/users/{user_id}/profile", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: str,
    db: DatabaseInterface = Depends(get_database)
):
    """Get user profile"""
    try:
        user = await db.get_user_profile(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return UserProfileResponse(
            user_id=user.user_id,
            display_name=user.display_name,
            preferences=user.preferences,
            adaptive_profile=user.adaptive_profile,
            created_at=user.created_at,
            last_active=user.last_active
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

@app.get("/api/v1/users/{user_id}/summary")
async def get_user_summary(
    user_id: str,
    companion: UnifiedCompanion = Depends(get_companion_system)
):
    """Get user interaction summary and adaptive profile"""
    try:
        summary = await companion.get_interaction_summary(user_id)
        return summary
        
    except Exception as e:
        logger.error(f"Error getting user summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user summary: {str(e)}")

@app.get("/api/v1/users/{user_id}/interactions")
async def get_user_interactions(
    user_id: str,
    limit: int = 20,
    db: DatabaseInterface = Depends(get_database)
):
    """Get recent user interactions"""
    try:
        interactions = await db.get_recent_interactions(user_id, limit)
        
        # Convert to response format
        return [
            {
                "interaction_id": interaction.interaction_id,
                "session_id": interaction.session_id,
                "timestamp": interaction.timestamp.isoformat(),
                "user_input": interaction.user_input,
                "companion_response": interaction.companion_response,
                "interaction_type": interaction.interaction_type.value,
                "context_analysis": interaction.context_analysis
            }
            for interaction in interactions
        ]
        
    except Exception as e:
        logger.error(f"Error getting user interactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user interactions: {str(e)}")

@app.get("/api/v1/sessions/{session_id}", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    db: DatabaseInterface = Depends(get_database)
):
    """Get session summary"""
    try:
        session = await db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionSummaryResponse(
            session_id=session.session_id,
            user_id=session.user_id,
            total_interactions=session.total_interactions,
            primary_focuses=session.primary_focuses,
            emotional_trajectory=session.emotional_trajectory,
            session_summary=session.session_summary,
            start_time=session.start_time,
            end_time=session.end_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session summary: {str(e)}")

@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and health information"""
    try:
        # Check component status
        components = {
            "companion_system": "operational" if companion_system else "not_initialized",
            "database": "operational" if database else "not_initialized",
            "mythomax_interface": "operational" if companion_system and companion_system.mythomax else "not_initialized"
        }
        
        overall_status = "operational" if all(status == "operational" for status in components.values()) else "degraded"
        
        return SystemStatusResponse(
            status=overall_status,
            components=components,
            uptime="system_startup",  # Would be calculated from startup time in production
            active_users=0  # Would be tracked in production
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/api/v1/system/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
