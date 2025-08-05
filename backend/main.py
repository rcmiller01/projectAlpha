"""
FastAPI backend for Unified AI Companion with Anchor Settings integration.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="Unified AI Companion Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anchor settings configuration
ANCHOR_SETTINGS_PATH = Path("config/anchor_settings.json")

class AnchorSettings(BaseModel):
    """Model for anchor settings configuration."""
    weights: Dict[str, float]
    signature: str = "Emberveil-01"
    locked: bool = False
    last_updated: str = None

def load_anchor_settings() -> Dict[str, Any]:
    """Load anchor settings from configuration file."""
    try:
        if ANCHOR_SETTINGS_PATH.exists():
            with open(ANCHOR_SETTINGS_PATH, 'r') as f:
                settings = json.load(f)
                logger.info(f"Loaded anchor settings from {ANCHOR_SETTINGS_PATH}")
                return settings
        else:
            logger.warning(f"Anchor settings file not found at {ANCHOR_SETTINGS_PATH}")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load anchor settings: {e}")
    
    # Return default settings
    default_settings = {
        "weights": {
            "persona_continuity": 0.4,
            "expression_accuracy": 0.3,
            "response_depth": 0.2,
            "memory_alignment": 0.1
        },
        "signature": "Emberveil-01", 
        "locked": False,
        "last_updated": None
    }
    logger.info("Using default anchor settings")
    return default_settings

def save_anchor_settings(settings: Dict[str, Any]) -> None:
    """Save anchor settings to configuration file."""
    try:
        # Ensure config directory exists
        ANCHOR_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        settings["last_updated"] = datetime.utcnow().isoformat()
        
        with open(ANCHOR_SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Saved anchor settings to {ANCHOR_SETTINGS_PATH}")
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save anchor settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to save anchor settings")

@app.get("/api/anchor/settings")
async def get_anchor_settings() -> Dict[str, Any]:
    """Get current anchor settings configuration."""
    try:
        settings = load_anchor_settings()
        return settings
    except Exception as e:
        logger.error(f"Error retrieving anchor settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve anchor settings")

@app.post("/api/anchor/settings")
async def update_anchor_settings(settings: AnchorSettings) -> Dict[str, Any]:
    """Update anchor settings configuration."""
    try:
        settings_dict = settings.dict()
        save_anchor_settings(settings_dict)
        return {"success": True, "settings": settings_dict}
    except Exception as e:
        logger.error(f"Error updating anchor settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update anchor settings")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
