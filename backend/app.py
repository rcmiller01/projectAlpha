"""
FastAPI backend for ProjectAlpha with centralized configuration.
"""

import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Try to load settings, fall back to defaults if required env vars are missing
try:
    from config.settings import settings, log_configuration_summary
    config_loaded = True
except Exception as e:
    print(f"Warning: Could not load full configuration: {e}")
    print("Running with default settings - some features may be disabled")
    settings = None
    config_loaded = False

# Configure logging
log_level = settings.LOG_LEVEL if settings else "INFO"
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log configuration on startup
logger.info("🚀 Starting ProjectAlpha Backend")
if config_loaded:
    log_configuration_summary(settings)
else:
    logger.warning("⚠️  Running with minimal configuration - set required environment variables for full functionality")

# Check for kill switch
if settings and settings.SAFE_MODE_FORCE:
    logger.warning("⚠️  SAFE MODE FORCE is enabled - system operating in restricted mode")

app = FastAPI(
    title="ProjectAlpha Backend",
    version="1.0.0",
    debug=settings.DEBUG if settings else False
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("🔧 ProjectAlpha Backend startup complete")

    if settings:
        # Log feature status
        features_status = {
            "emotion_loop": settings.EMOTION_LOOP_ENABLED,
            "autopilot": settings.AUTOPILOT_ENABLED,
            "safe_mode": settings.SAFE_MODE_FORCE,
            "debug": settings.DEBUG,
            "gpu": settings.GPU_ENABLED,
            "cluster": settings.CLUSTER_ENABLED,
        }
        logger.info(f"📊 Feature flags: {features_status}")
    else:
        logger.info("📊 Feature flags: Using defaults (config not fully loaded)")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("⏹️  ProjectAlpha Backend shutting down")

@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "name": "ProjectAlpha Backend",
        "version": "1.0.0",
        "status": "running",
        "safe_mode": settings.SAFE_MODE_FORCE if settings else False,
        "features": {
            "emotion_loop": settings.EMOTION_LOOP_ENABLED if settings else True,
            "autopilot": settings.AUTOPILOT_ENABLED if settings else True,
            "debug": settings.DEBUG if settings else False,
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2025-08-07T00:00:00Z",
        "config": {
            "port": settings.PORT if settings else 8000,
            "server_role": settings.SERVER_ROLE if settings else "unknown",
            "server_id": settings.SERVER_ID if settings else "default",
            "safe_mode": settings.SAFE_MODE_FORCE if settings else False,
        }
    }

@app.get("/config")
async def get_config_summary():
    """Get non-sensitive configuration summary."""
    if not settings:
        return {
            "error": "Configuration not fully loaded",
            "message": "Set required environment variables (MONGO_ROOT_USERNAME, MONGO_ROOT_PASSWORD, etc.)"
        }

    return {
        "server": {
            "role": settings.SERVER_ROLE,
            "id": settings.SERVER_ID,
            "port": settings.PORT,
            "cluster_enabled": settings.CLUSTER_ENABLED,
            "gpu_enabled": settings.GPU_ENABLED,
        },
        "features": {
            "emotion_loop": settings.EMOTION_LOOP_ENABLED,
            "autopilot": settings.AUTOPILOT_ENABLED,
            "safe_mode": settings.SAFE_MODE_FORCE,
            "debug": settings.DEBUG,
        },
        "rate_limiting": {
            "window_seconds": settings.RATE_LIMIT_WINDOW,
            "max_requests": settings.RATE_LIMIT_MAX,
        },
        "emotional_processing": {
            "drift_scaling_factor": settings.DRIFT_SCALING_FACTOR,
            "max_penalty_threshold": settings.MAX_PENALTY_THRESHOLD,
        }
    }

if __name__ == "__main__":
    import uvicorn

    port = settings.PORT if settings else 8000
    debug = settings.DEBUG if settings else False
    log_level = settings.LOG_LEVEL.lower() if settings else "info"

    logger.info(f"🌐 Starting server on port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        log_level=log_level
    )
