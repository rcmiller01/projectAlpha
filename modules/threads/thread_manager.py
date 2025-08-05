"""
Thread Management API for AI Companion System

Provides REST endpoints for managing conversation threads, folders,
and persistent memory state per thread.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Data Models
class ThreadMessage(BaseModel):
    """Individual message in a thread"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotion: Optional[str] = None
    intensity: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ThreadState(BaseModel):
    """Emotional and memory state for a thread"""
    emotional_context: Dict[str, Any] = Field(default_factory=dict)
    memory_symbols: List[Dict[str, Any]] = Field(default_factory=list)
    relationship_context: Dict[str, Any] = Field(default_factory=dict)
    integration_state: Dict[str, Any] = Field(default_factory=dict)
    voice_preferences: Dict[str, Any] = Field(default_factory=dict)
    ritual_history: List[Dict[str, Any]] = Field(default_factory=list)

class Thread(BaseModel):
    """Conversation thread with metadata and state"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    folder_id: Optional[str] = None
    folder_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: List[ThreadMessage] = Field(default_factory=list)
    thread_state: ThreadState = Field(default_factory=ThreadState)
    is_archived: bool = False
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    last_emotion: Optional[str] = None
    message_count: int = 0

class Folder(BaseModel):
    """Folder for organizing threads"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    color: Optional[str] = "#6366f1"  # Default indigo
    icon: Optional[str] = "💭"
    thread_count: int = 0

class CreateThreadRequest(BaseModel):
    """Request to create a new thread"""
    title: str
    folder_id: Optional[str] = None
    folder_name: Optional[str] = None
    initial_message: Optional[str] = None
    trigger_symbolic_rebirth: bool = True

class UpdateThreadRequest(BaseModel):
    """Request to update thread metadata"""
    title: Optional[str] = None
    folder_id: Optional[str] = None
    folder_name: Optional[str] = None
    is_archived: Optional[bool] = None
    tags: Optional[List[str]] = None

class CreateFolderRequest(BaseModel):
    """Request to create a new folder"""
    name: str
    description: Optional[str] = None
    color: Optional[str] = "#6366f1"
    icon: Optional[str] = "💭"

class ThreadManager:
    """Manages thread persistence and operations"""
    
    def __init__(self, data_dir: str = "data/threads"):
        self.data_dir = data_dir
        self.threads_file = os.path.join(data_dir, "threads.json")
        self.folders_file = os.path.join(data_dir, "folders.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize storage files if they don't exist
        self._init_storage()
        
    def _init_storage(self):
        """Initialize storage files with empty data"""
        if not os.path.exists(self.threads_file):
            with open(self.threads_file, 'w') as f:
                json.dump({}, f)
                
        if not os.path.exists(self.folders_file):
            with open(self.folders_file, 'w') as f:
                json.dump({}, f)
    
    def _load_threads(self) -> Dict[str, Dict]:
        """Load threads from storage"""
        try:
            with open(self.threads_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading threads: {e}")
            return {}
    
    def _save_threads(self, threads: Dict[str, Dict]):
        """Save threads to storage"""
        try:
            with open(self.threads_file, 'w') as f:
                json.dump(threads, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving threads: {e}")
    
    def _load_folders(self) -> Dict[str, Dict]:
        """Load folders from storage"""
        try:
            with open(self.folders_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading folders: {e}")
            return {}
    
    def _save_folders(self, folders: Dict[str, Dict]):
        """Save folders to storage"""
        try:
            with open(self.folders_file, 'w') as f:
                json.dump(folders, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving folders: {e}")
    
    def create_thread(self, request: CreateThreadRequest) -> Thread:
        """Create a new thread"""
        thread = Thread(
            title=request.title,
            folder_id=request.folder_id,
            folder_name=request.folder_name
        )
        
        # Add initial message if provided
        if request.initial_message:
            initial_msg = ThreadMessage(
                role="user",
                content=request.initial_message
            )
            thread.messages.append(initial_msg)
            thread.message_count = 1
        
        # Save thread
        threads = self._load_threads()
        threads[thread.id] = thread.dict()
        self._save_threads(threads)
        
        # Update folder thread count if folder specified
        if request.folder_id or request.folder_name:
            self._update_folder_thread_count(request.folder_id or request.folder_name, 1)
        
        logger.info(f"Created new thread: {thread.title} ({thread.id})")
        return thread
    
    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get a specific thread by ID"""
        threads = self._load_threads()
        thread_data = threads.get(thread_id)
        
        if not thread_data:
            return None
            
        try:
            return Thread(**thread_data)
        except Exception as e:
            logger.error(f"Error deserializing thread {thread_id}: {e}")
            return None
    
    def update_thread(self, thread_id: str, request: UpdateThreadRequest) -> Optional[Thread]:
        """Update thread metadata"""
        threads = self._load_threads()
        thread_data = threads.get(thread_id)
        
        if not thread_data:
            return None
        
        # Update fields
        if request.title is not None:
            thread_data['title'] = request.title
        if request.folder_id is not None:
            thread_data['folder_id'] = request.folder_id
        if request.folder_name is not None:
            thread_data['folder_name'] = request.folder_name
        if request.is_archived is not None:
            thread_data['is_archived'] = request.is_archived
        if request.tags is not None:
            thread_data['tags'] = request.tags
            
        thread_data['updated_at'] = datetime.now().isoformat()
        
        threads[thread_id] = thread_data
        self._save_threads(threads)
        
        return Thread(**thread_data)
    
    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        threads = self._load_threads()
        thread_data = threads.get(thread_id)
        
        if not thread_data:
            return False
        
        # Update folder thread count
        folder_id = thread_data.get('folder_id') or thread_data.get('folder_name')
        if folder_id:
            self._update_folder_thread_count(folder_id, -1)
        
        del threads[thread_id]
        self._save_threads(threads)
        
        logger.info(f"Deleted thread: {thread_id}")
        return True
    
    def list_threads(self, folder_id: Optional[str] = None, 
                    include_archived: bool = False) -> List[Thread]:
        """List threads, optionally filtered by folder"""
        threads = self._load_threads()
        result = []
        
        for thread_data in threads.values():
            # Skip archived threads unless requested
            if thread_data.get('is_archived', False) and not include_archived:
                continue
                
            # Filter by folder if specified
            if folder_id:
                thread_folder = thread_data.get('folder_id') or thread_data.get('folder_name')
                if thread_folder != folder_id:
                    continue
            
            try:
                thread = Thread(**thread_data)
                result.append(thread)
            except Exception as e:
                logger.error(f"Error deserializing thread: {e}")
                continue
        
        # Sort by updated_at (most recent first)
        result.sort(key=lambda t: t.updated_at, reverse=True)
        return result
    
    def add_message_to_thread(self, thread_id: str, message: ThreadMessage) -> Optional[Thread]:
        """Add a message to a thread"""
        threads = self._load_threads()
        thread_data = threads.get(thread_id)
        
        if not thread_data:
            return None
        
        # Add message
        thread_data['messages'].append(message.dict())
        thread_data['message_count'] = len(thread_data['messages'])
        thread_data['updated_at'] = datetime.now().isoformat()
        
        # Update last emotion if message has emotion
        if message.emotion:
            thread_data['last_emotion'] = message.emotion
        
        threads[thread_id] = thread_data
        self._save_threads(threads)
        
        return Thread(**thread_data)
    
    def update_thread_state(self, thread_id: str, state: ThreadState) -> Optional[Thread]:
        """Update the emotional/memory state of a thread"""
        threads = self._load_threads()
        thread_data = threads.get(thread_id)
        
        if not thread_data:
            return None
        
        thread_data['thread_state'] = state.dict()
        thread_data['updated_at'] = datetime.now().isoformat()
        
        threads[thread_id] = thread_data
        self._save_threads(threads)
        
        return Thread(**thread_data)
    
    def create_folder(self, request: CreateFolderRequest) -> Folder:
        """Create a new folder"""
        folder = Folder(
            name=request.name,
            description=request.description,
            color=request.color,
            icon=request.icon
        )
        
        folders = self._load_folders()
        folders[folder.id] = folder.dict()
        self._save_folders(folders)
        
        logger.info(f"Created new folder: {folder.name} ({folder.id})")
        return folder
    
    def list_folders(self) -> List[Folder]:
        """List all folders"""
        folders = self._load_folders()
        result = []
        
        for folder_data in folders.values():
            try:
                folder = Folder(**folder_data)
                result.append(folder)
            except Exception as e:
                logger.error(f"Error deserializing folder: {e}")
                continue
        
        # Sort by name
        result.sort(key=lambda f: f.name.lower())
        return result
    
    def _update_folder_thread_count(self, folder_id: str, delta: int):
        """Update thread count for a folder"""
        folders = self._load_folders()
        
        for folder_data in folders.values():
            if folder_data['id'] == folder_id or folder_data['name'] == folder_id:
                folder_data['thread_count'] = max(0, folder_data.get('thread_count', 0) + delta)
                self._save_folders(folders)
                break

# Global thread manager instance
thread_manager = ThreadManager()

# FastAPI routes (to be integrated into main app)
def setup_thread_routes(app: FastAPI):
    """Set up thread management routes"""
    
    @app.get("/api/threads", response_model=List[Thread])
    async def list_threads(folder_id: Optional[str] = None, 
                          include_archived: bool = False):
        """List threads, optionally filtered by folder"""
        try:
            threads = thread_manager.list_threads(folder_id, include_archived)
            return threads
        except Exception as e:
            logger.error(f"Error listing threads: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/thread/{thread_id}", response_model=Thread)
    async def get_thread(thread_id: str):
        """Get a specific thread by ID"""
        thread = thread_manager.get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return thread
    
    @app.post("/api/thread", response_model=Thread)
    async def create_thread(request: CreateThreadRequest):
        """Create a new thread"""
        try:
            thread = thread_manager.create_thread(request)
            
            # Trigger symbolic rebirth if requested
            if request.trigger_symbolic_rebirth:
                await _trigger_symbolic_rebirth(thread.id)
            
            return thread
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/api/thread/{thread_id}", response_model=Thread)
    async def update_thread(thread_id: str, request: UpdateThreadRequest):
        """Update thread metadata"""
        thread = thread_manager.update_thread(thread_id, request)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return thread
    
    @app.delete("/api/thread/{thread_id}")
    async def delete_thread(thread_id: str):
        """Delete a thread"""
        success = thread_manager.delete_thread(thread_id)
        if not success:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"message": "Thread deleted successfully"}
    
    @app.post("/api/thread/{thread_id}/message")
    async def add_message(thread_id: str, message: ThreadMessage):
        """Add a message to a thread"""
        thread = thread_manager.add_message_to_thread(thread_id, message)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"message": "Message added successfully", "thread": thread}
    
    @app.put("/api/thread/{thread_id}/state")
    async def update_thread_state(thread_id: str, state: ThreadState):
        """Update thread emotional/memory state"""
        thread = thread_manager.update_thread_state(thread_id, state)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"message": "Thread state updated successfully", "thread": thread}
    
    @app.get("/api/folders", response_model=List[Folder])
    async def list_folders():
        """List all folders"""
        try:
            folders = thread_manager.list_folders()
            return folders
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/folder", response_model=Folder)
    async def create_folder(request: CreateFolderRequest):
        """Create a new folder"""
        try:
            folder = thread_manager.create_folder(request)
            return folder
        except Exception as e:
            logger.error(f"Error creating folder: {e}")
            raise HTTPException(status_code=500, detail=str(e))

async def _trigger_symbolic_rebirth(thread_id: str):
    """Trigger symbolic rebirth for new thread"""
    try:
        # This would integrate with your existing AI companion system
        # to reset emotional state and trigger symbolic rebirth
        logger.info(f"Triggering symbolic rebirth for thread: {thread_id}")
        
        # You can add integration with:
        # - unified_companion.py for state reset
        # - symbolic_ritual_manager.py for ritual initiation
        # - emotion_reflector.py for fresh emotional baseline
        
    except Exception as e:
        logger.error(f"Error triggering symbolic rebirth: {e}")
