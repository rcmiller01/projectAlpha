#!/usr/bin/env python3
"""
HRM API Integration
===================

FastAPI integration layer for the Hierarchical Reasoning Model (HRM) system.
Provides REST API endpoints to integrate HRM functionality with existing
projectAlpha backend services.

Endpoints:
- POST /hrm/process - Process a message through HRM pipeline
- GET /hrm/status - Get HRM system status and health
- POST /hrm/config - Update HRM configuration
- GET /hrm/analytics - Get system analytics and metrics

Author: AI Development Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Import HRM components
from backend.hrm_router import HRMRouter, HRMMode, RequestType, HRMResponse
from backend.subagent_router import SubAgentRouter, AgentType
from backend.ai_reformulator import PersonalityFormatter, PersonalityProfile
from core.core_arbiter import CoreArbiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="HRM System API",
    description="Hierarchical Reasoning Model API for projectAlpha",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API

class HRMProcessRequest(BaseModel):
    """Request model for HRM processing"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message to process")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    user_id: Optional[str] = Field(default="default_user", description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    personality_preference: Optional[str] = Field(default=None, description="Preferred personality profile")
    processing_mode: Optional[str] = Field(default=None, description="Preferred processing mode")

class HRMProcessResponse(BaseModel):
    """Response model for HRM processing"""
    response: str = Field(..., description="Final AI response")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    mode_used: str = Field(..., description="Processing mode that was used")
    agents_involved: List[str] = Field(..., description="List of agents involved in processing")
    personality_applied: str = Field(..., description="Personality profile applied")
    success: bool = Field(..., description="Whether processing was successful")

class HRMStatusResponse(BaseModel):
    """Response model for system status"""
    system_health: str = Field(..., description="Overall system health status")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Individual component status")
    performance_metrics: Dict[str, Any] = Field(..., description="System performance metrics")
    active_sessions: int = Field(..., description="Number of active sessions")
    uptime: str = Field(..., description="System uptime")
    timestamp: str = Field(..., description="Status timestamp")

class HRMConfigRequest(BaseModel):
    """Request model for configuration updates"""
    component: str = Field(..., description="Component to configure (hrm_router, subagent_router, etc.)")
    config: Dict[str, Any] = Field(..., description="Configuration parameters")

class HRMAnalyticsResponse(BaseModel):
    """Response model for system analytics"""
    total_requests: int = Field(..., description="Total requests processed")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Overall success rate")
    average_processing_time: float = Field(..., ge=0.0, description="Average processing time")
    mode_distribution: Dict[str, int] = Field(..., description="Distribution of processing modes")
    agent_utilization: Dict[str, int] = Field(..., description="Agent utilization statistics")
    personality_distribution: Dict[str, int] = Field(..., description="Personality profile usage")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate")

# Global HRM system components
hrm_router = None
subagent_router = None
personality_formatter = None
core_arbiter = None

# System metrics
system_metrics = {
    "start_time": datetime.now(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_processing_time": 0.0
}

@app.on_event("startup")
async def startup_event():
    """Initialize HRM system components on startup"""
    global hrm_router, subagent_router, personality_formatter, core_arbiter
    
    logger.info("🚀 Initializing HRM System API...")
    
    try:
        # Initialize core components
        hrm_router = HRMRouter()
        subagent_router = SubAgentRouter()
        personality_formatter = PersonalityFormatter()
        core_arbiter = CoreArbiter()
        
        logger.info("✅ HRM System API initialized successfully")
        logger.info(f"   🧠 HRM Router: Ready")
        logger.info(f"   🤖 SubAgent Router: {len(subagent_router.agents)} agents loaded")
        logger.info(f"   🎭 Personality Formatter: Ready")
        logger.info(f"   ⚖️  Core Arbiter: Ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize HRM System: {str(e)}")
        raise

@app.post("/hrm/process", response_model=HRMProcessResponse)
async def process_message(request: HRMProcessRequest, background_tasks: BackgroundTasks):
    """
    Process a message through the complete HRM pipeline
    
    This endpoint orchestrates the full HRM processing:
    1. Route through HRM Router for mode selection
    2. Process through specialized SubAgents
    3. Apply personality formatting
    4. Return complete response with metadata
    """
    global system_metrics
    
    start_time = datetime.now()
    system_metrics["total_requests"] += 1
    
    try:
        # Prepare context
        context = request.context.copy()
        context.update({
            "user_id": request.user_id,
            "session_id": request.session_id or f"session_{int(start_time.timestamp())}",
            "api_request": True,
            "timestamp": start_time.isoformat()
        })
        
        # Add personality preference if specified
        if request.personality_preference:
            context["personality_preference"] = request.personality_preference
        
        # Process through HRM Router
        logger.info(f"🧠 Processing message from {request.user_id}: {request.message[:50]}...")
        
        hrm_response = await hrm_router.process_request(request.message, context)
        
        # Calculate total processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        system_metrics["successful_requests"] += 1
        system_metrics["total_processing_time"] += processing_time
        
        # Prepare response
        response = HRMProcessResponse(
            response=hrm_response.primary_response,
            metadata={
                "reasoning_trace": hrm_response.reasoning_trace,
                "emotional_insights": hrm_response.emotional_insights,
                "mirror_reflection": hrm_response.mirror_reflection,
                "source_weights": hrm_response.source_weights,
                "request_context": context
            },
            confidence=hrm_response.confidence_score,
            processing_time=processing_time,
            mode_used=hrm_response.processing_mode.value,
            agents_involved=hrm_response.agents_involved,
            personality_applied=hrm_response.metadata.get("personality_applied", "unknown"),
            success=True
        )
        
        # Log successful processing
        logger.info(f"✅ Successfully processed message in {processing_time:.3f}s "
                   f"(mode: {hrm_response.processing_mode.value}, confidence: {hrm_response.confidence_score:.2f})")
        
        return response
        
    except Exception as e:
        system_metrics["failed_requests"] += 1
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.error(f"❌ Failed to process message: {str(e)}")
        
        # Return error response
        return HRMProcessResponse(
            response=f"I apologize, but I encountered an error processing your request: {str(e)}",
            metadata={"error": str(e), "error_type": type(e).__name__},
            confidence=0.0,
            processing_time=processing_time,
            mode_used="error",
            agents_involved=["error_handler"],
            personality_applied="error",
            success=False
        )

@app.get("/hrm/status", response_model=HRMStatusResponse)
async def get_system_status():
    """Get comprehensive HRM system status and health information"""
    
    try:
        # Collect status from all components
        components_status = {}
        
        # HRM Router status
        if hrm_router:
            hrm_status = hrm_router.get_system_status()
            components_status["hrm_router"] = {
                "status": "healthy",
                "metrics": hrm_status["metrics"],
                "active_sessions": hrm_status["active_sessions"]
            }
        
        # SubAgent Router status
        if subagent_router:
            subagent_analytics = subagent_router.get_routing_analytics()
            components_status["subagent_router"] = {
                "status": "healthy",
                "total_routes": subagent_analytics["total_routes"],
                "success_rate": subagent_analytics["success_rate"],
                "available_agents": subagent_analytics["available_agents"]
            }
        
        # Personality Formatter status
        if personality_formatter:
            formatter_analytics = personality_formatter.get_formatting_analytics()
            components_status["personality_formatter"] = {
                "status": "healthy",
                "total_reformulations": formatter_analytics["total_reformulations"],
                "success_rate": formatter_analytics["success_rate"],
                "average_confidence": formatter_analytics["average_confidence"]
            }
        
        # Core Arbiter status
        if core_arbiter:
            arbiter_status = core_arbiter.get_system_status()
            components_status["core_arbiter"] = {
                "status": arbiter_status["health_status"],
                "stability_score": arbiter_status["drift_state"]["stability_score"],
                "memory_usage": arbiter_status.get("memory_usage", "unknown")
            }
        
        # Calculate overall system health
        component_healths = [comp.get("status", "unknown") for comp in components_status.values()]
        if all(health == "healthy" for health in component_healths):
            overall_health = "healthy"
        elif any(health == "healthy" for health in component_healths):
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"
        
        # Performance metrics
        total_requests = system_metrics["total_requests"]
        performance_metrics = {
            "total_requests": total_requests,
            "successful_requests": system_metrics["successful_requests"],
            "failed_requests": system_metrics["failed_requests"],
            "success_rate": system_metrics["successful_requests"] / max(total_requests, 1),
            "average_processing_time": system_metrics["total_processing_time"] / max(total_requests, 1),
            "uptime_seconds": (datetime.now() - system_metrics["start_time"]).total_seconds()
        }
        
        # Calculate uptime
        uptime_seconds = performance_metrics["uptime_seconds"]
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return HRMStatusResponse(
            system_health=overall_health,
            components=components_status,
            performance_metrics=performance_metrics,
            active_sessions=components_status.get("hrm_router", {}).get("active_sessions", 0),
            uptime=uptime_str,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.post("/hrm/config")
async def update_configuration(request: HRMConfigRequest):
    """Update HRM system configuration"""
    
    try:
        component = request.component.lower()
        config = request.config
        
        if component == "hrm_router" and hrm_router:
            hrm_router.update_config(config)
        elif component == "subagent_router" and subagent_router:
            # SubAgent router config update would go here
            logger.info(f"SubAgent router config update requested (not implemented)")
        elif component == "personality_formatter" and personality_formatter:
            # Personality formatter config update would go here
            logger.info(f"Personality formatter config update requested (not implemented)")
        elif component == "core_arbiter" and core_arbiter:
            # Core arbiter config update would go here
            logger.info(f"Core arbiter config update requested (not implemented)")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown component: {component}")
        
        logger.info(f"✅ Configuration updated for {component}")
        
        return {
            "success": True,
            "message": f"Configuration updated for {component}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error updating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.get("/hrm/analytics", response_model=HRMAnalyticsResponse)
async def get_analytics():
    """Get comprehensive system analytics and performance metrics"""
    
    try:
        # Collect analytics from all components
        analytics_data = {
            "total_requests": system_metrics["total_requests"],
            "success_rate": system_metrics["successful_requests"] / max(system_metrics["total_requests"], 1),
            "average_processing_time": system_metrics["total_processing_time"] / max(system_metrics["total_requests"], 1),
            "error_rate": system_metrics["failed_requests"] / max(system_metrics["total_requests"], 1),
            "mode_distribution": {},
            "agent_utilization": {},
            "personality_distribution": {}
        }
        
        # Get HRM Router analytics
        if hrm_router:
            hrm_status = hrm_router.get_system_status()
            analytics_data["mode_distribution"] = hrm_status["metrics"].get("mode_distribution", {})
        
        # Get SubAgent Router analytics
        if subagent_router:
            subagent_analytics = subagent_router.get_routing_analytics()
            analytics_data["agent_utilization"] = subagent_analytics.get("agent_utilization", {})
        
        # Get Personality Formatter analytics
        if personality_formatter:
            formatter_analytics = personality_formatter.get_formatting_analytics()
            analytics_data["personality_distribution"] = formatter_analytics.get("personality_distribution", {})
        
        return HRMAnalyticsResponse(**analytics_data)
        
    except Exception as e:
        logger.error(f"❌ Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@app.get("/hrm/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy" if all([hrm_router, subagent_router, personality_formatter, core_arbiter]) else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "hrm_router": hrm_router is not None,
            "subagent_router": subagent_router is not None,
            "personality_formatter": personality_formatter is not None,
            "core_arbiter": core_arbiter is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "HRM System API",
        "version": "1.0.0",
        "description": "Hierarchical Reasoning Model API for projectAlpha",
        "endpoints": {
            "POST /hrm/process": "Process a message through HRM pipeline",
            "GET /hrm/status": "Get system status and health",
            "POST /hrm/config": "Update system configuration",
            "GET /hrm/analytics": "Get system analytics",
            "GET /hrm/health": "Simple health check"
        },
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting HRM System API...")
    print("📖 Documentation available at: http://localhost:8001/docs")
    print("🔍 Health check at: http://localhost:8001/hrm/health")
    
    uvicorn.run(
        "hrm_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
