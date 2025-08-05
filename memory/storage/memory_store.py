"""
True Recall - Memory Store

This module provides persistent storage for memory events using both
JSONL (for human readability) and SQLite (for fast querying) backends.
"""

import asyncio
import aiosqlite
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, date, timedelta
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MemoryStore:
    """
    Persistent storage manager for True Recall memory events.
    
    Supports both JSONL (human-readable logs) and SQLite (fast queries)
    storage backends with automatic data synchronization.
    """
    
    def __init__(self, storage_path: str = "memory_data", config: Optional[Dict[str, Any]] = None):
        """Initialize the memory store."""
        self.storage_path = Path(storage_path)
        self.config = config or {}
        
        # Storage configuration
        self.use_sqlite = self.config.get('use_sqlite', True)
        self.use_jsonl = self.config.get('use_jsonl', True)
        self.auto_backup = self.config.get('auto_backup', True)
        self.max_jsonl_file_size = self.config.get('max_jsonl_file_size', 50 * 1024 * 1024)  # 50MB
        
        # File paths
        self.sqlite_path = self.storage_path / "true_recall.db"
        self.jsonl_path = self.storage_path / "events"
        self.backup_path = self.storage_path / "backups"
        
        # Ensure directories exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.mkdir(parents=True, exist_ok=True)
        
        if self.auto_backup:
            self.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Database connection pool
        self._db = None
        self._db_lock = asyncio.Lock()
        
        # JSONL file management
        self._current_jsonl_file = None
        self._jsonl_file_size = 0
        
        logger.info(f"📁 Memory Store initialized at {self.storage_path}")
    
    async def initialize(self):
        """Initialize the storage backends."""
        try:
            if self.use_sqlite:
                await self._initialize_sqlite()
            
            if self.use_jsonl:
                await self._initialize_jsonl()
            
            logger.info("✅ Memory Store initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory store: {e}")
            raise
    
    async def _initialize_sqlite(self):
        """Initialize SQLite database and create tables."""
        async with aiosqlite.connect(str(self.sqlite_path)) as db:
            # Create events table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tone TEXT,
                    emotion_tags TEXT,  -- JSON array
                    salience REAL NOT NULL,
                    related_ids TEXT,  -- JSON array
                    metadata TEXT,     -- JSON object
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_events_actor ON events(actor)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_events_salience ON events(salience)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at)")
            
            # Create daily summaries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS daily_reflections (
                    date TEXT PRIMARY KEY,
                    reflection_data TEXT NOT NULL,  -- JSON object
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create weekly summaries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS weekly_reflections (
                    week_start TEXT PRIMARY KEY,
                    reflection_data TEXT NOT NULL,  -- JSON object
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create metadata table for store configuration
            await db.execute("""
                CREATE TABLE IF NOT EXISTS store_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
            logger.info("🗄️ SQLite database initialized")
    
    async def _initialize_jsonl(self):
        """Initialize JSONL file management."""
        today = date.today()
        self._current_jsonl_file = self.jsonl_path / f"events_{today.isoformat()}.jsonl"
        
        # Check current file size if it exists
        if self._current_jsonl_file.exists():
            self._jsonl_file_size = self._current_jsonl_file.stat().st_size
        else:
            self._jsonl_file_size = 0
        
        logger.info(f"📝 JSONL storage initialized: {self._current_jsonl_file}")
    
    async def store_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Store a memory event in the configured backends.
        
        Args:
            event_data: Event data dictionary with all required fields
            
        Returns:
            bool: True if successfully stored
        """
        try:
            success = True
            
            if self.use_sqlite:
                success &= await self._store_event_sqlite(event_data)
            
            if self.use_jsonl:
                success &= await self._store_event_jsonl(event_data)
            
            if success:
                logger.debug(f"📥 Stored event {event_data.get('id', 'unknown')}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to store event: {e}")
            return False
    
    async def _store_event_sqlite(self, event_data: Dict[str, Any]) -> bool:
        """Store event in SQLite database."""
        try:
            async with aiosqlite.connect(str(self.sqlite_path)) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO events 
                    (id, timestamp, actor, event_type, content, tone, emotion_tags, 
                     salience, related_ids, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data['id'],
                    event_data['timestamp'],
                    event_data['actor'],
                    event_data['event_type'],
                    event_data['content'],
                    event_data.get('tone', ''),
                    json.dumps(event_data.get('emotion_tags', [])),
                    event_data['salience'],
                    json.dumps(event_data.get('related_ids', [])),
                    json.dumps(event_data.get('metadata', {})),
                    datetime.now().isoformat()
                ))
                await db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ SQLite storage failed: {e}")
            return False
    
    async def _store_event_jsonl(self, event_data: Dict[str, Any]) -> bool:
        """Store event in JSONL file."""
        try:
            # Check if we need to rotate the file
            await self._check_jsonl_rotation()
            
            # Write event to JSONL file
            with open(self._current_jsonl_file, 'a', encoding='utf-8') as f:
                json.dump(event_data, f, ensure_ascii=False)
                f.write('\n')
            
            # Update file size tracking
            self._jsonl_file_size += len(json.dumps(event_data, ensure_ascii=False)) + 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ JSONL storage failed: {e}")
            return False
    
    async def _check_jsonl_rotation(self):
        """Check if JSONL file needs rotation."""
        if self._jsonl_file_size > self.max_jsonl_file_size:
            # Rotate to new file
            old_file = self._current_jsonl_file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            today = date.today()
            self._current_jsonl_file = self.jsonl_path / f"events_{today.isoformat()}_{timestamp}.jsonl"
            self._jsonl_file_size = 0
            
            logger.info(f"🔄 Rotated JSONL file: {old_file} -> {self._current_jsonl_file}")
    
    async def retrieve_events(
        self,
        event_ids: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        actor: Optional[str] = None,
        event_type: Optional[str] = None,
        min_salience: Optional[float] = None,
        emotion_tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: str = 'timestamp',
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events from storage with flexible filtering.
        
        Args:
            event_ids: Specific event IDs to retrieve
            time_range: Tuple of (start_time, end_time) for filtering
            actor: Filter by actor
            event_type: Filter by event type
            min_salience: Minimum salience score
            emotion_tags: Filter by emotion tags
            limit: Maximum number of events to return
            offset: Number of events to skip
            order_by: Field to order by
            order_desc: Whether to order in descending order
            
        Returns:
            List of event dictionaries
        """
        try:
            if self.use_sqlite:
                return await self._retrieve_events_sqlite(
                    event_ids, time_range, actor, event_type, min_salience,
                    emotion_tags, limit, offset, order_by, order_desc
                )
            elif self.use_jsonl:
                return await self._retrieve_events_jsonl(
                    event_ids, time_range, actor, event_type, min_salience,
                    emotion_tags, limit, offset, order_by, order_desc
                )
            else:
                logger.warning("⚠️ No storage backend configured for retrieval")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve events: {e}")
            return []
    
    async def _retrieve_events_sqlite(
        self, event_ids, time_range, actor, event_type, min_salience,
        emotion_tags, limit, offset, order_by, order_desc
    ) -> List[Dict[str, Any]]:
        """Retrieve events from SQLite database."""
        try:
            query_parts = ["SELECT * FROM events WHERE 1=1"]
            params = []
            
            # Build query based on filters
            if event_ids:
                placeholders = ','.join(['?' for _ in event_ids])
                query_parts.append(f"AND id IN ({placeholders})")
                params.extend(event_ids)
            
            if time_range:
                query_parts.append("AND timestamp >= ? AND timestamp <= ?")
                params.extend([time_range[0].isoformat(), time_range[1].isoformat()])
            
            if actor:
                query_parts.append("AND actor = ?")
                params.append(actor)
            
            if event_type:
                query_parts.append("AND event_type = ?")
                params.append(event_type)
            
            if min_salience is not None:
                query_parts.append("AND salience >= ?")
                params.append(min_salience)
            
            if emotion_tags:
                # Simple emotion tag filtering (could be improved with full-text search)
                for emotion in emotion_tags:
                    query_parts.append("AND emotion_tags LIKE ?")
                    params.append(f'%"{emotion}"%')
            
            # Add ordering
            order_direction = "DESC" if order_desc else "ASC"
            query_parts.append(f"ORDER BY {order_by} {order_direction}")
            
            # Add limit and offset
            if limit:
                query_parts.append("LIMIT ?")
                params.append(limit)
            
            if offset > 0:
                query_parts.append("OFFSET ?")
                params.append(offset)
            
            query = " ".join(query_parts)
            
            async with aiosqlite.connect(str(self.sqlite_path)) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    events = []
                    for row in rows:
                        event_dict = dict(zip(columns, row))
                        
                        # Parse JSON fields
                        event_dict['emotion_tags'] = json.loads(event_dict['emotion_tags'] or '[]')
                        event_dict['related_ids'] = json.loads(event_dict['related_ids'] or '[]')
                        event_dict['metadata'] = json.loads(event_dict['metadata'] or '{}')
                        
                        events.append(event_dict)
                    
                    return events
            
        except Exception as e:
            logger.error(f"❌ SQLite retrieval failed: {e}")
            return []
    
    async def _retrieve_events_jsonl(
        self, event_ids, time_range, actor, event_type, min_salience,
        emotion_tags, limit, offset, order_by, order_desc
    ) -> List[Dict[str, Any]]:
        """Retrieve events from JSONL files."""
        try:
            events = []
            
            # Read all JSONL files in the events directory
            for jsonl_file in self.jsonl_path.glob("events_*.jsonl"):
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            events.append(event)
                        except json.JSONDecodeError:
                            continue
            
            # Apply filters
            filtered_events = []
            for event in events:
                # Event ID filter
                if event_ids and event.get('id') not in event_ids:
                    continue
                
                # Time range filter
                if time_range:
                    event_time = datetime.fromisoformat(event['timestamp'])
                    if not (time_range[0] <= event_time <= time_range[1]):
                        continue
                
                # Actor filter
                if actor and event.get('actor') != actor:
                    continue
                
                # Event type filter
                if event_type and event.get('event_type') != event_type:
                    continue
                
                # Salience filter
                if min_salience is not None and event.get('salience', 0) < min_salience:
                    continue
                
                # Emotion tags filter
                if emotion_tags:
                    event_emotions = event.get('emotion_tags', [])
                    if not any(emotion in event_emotions for emotion in emotion_tags):
                        continue
                
                filtered_events.append(event)
            
            # Sort events
            if order_by in ['timestamp', 'salience']:
                reverse = order_desc
                if order_by == 'timestamp':
                    filtered_events.sort(key=lambda e: e.get('timestamp', ''), reverse=reverse)
                else:
                    filtered_events.sort(key=lambda e: e.get('salience', 0), reverse=reverse)
            
            # Apply offset and limit
            start_idx = offset
            end_idx = offset + limit if limit else len(filtered_events)
            
            return filtered_events[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"❌ JSONL retrieval failed: {e}")
            return []
    
    async def store_reflection(self, reflection_type: str, date_key: str, reflection_data: Dict[str, Any]) -> bool:
        """
        Store a reflection summary (daily, weekly, etc.).
        
        Args:
            reflection_type: 'daily' or 'weekly'
            date_key: Date or week start date as ISO string
            reflection_data: Complete reflection data
            
        Returns:
            bool: True if successfully stored
        """
        try:
            if not self.use_sqlite:
                logger.warning("⚠️ Reflection storage requires SQLite backend")
                return False
            
            table_name = f"{reflection_type}_reflections"
            
            async with aiosqlite.connect(str(self.sqlite_path)) as db:
                await db.execute(f"""
                    INSERT OR REPLACE INTO {table_name}
                    (date, reflection_data, updated_at)
                    VALUES (?, ?, ?)
                """, (
                    date_key,
                    json.dumps(reflection_data, ensure_ascii=False),
                    datetime.now().isoformat()
                ))
                await db.commit()
            
            logger.info(f"📔 Stored {reflection_type} reflection for {date_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store {reflection_type} reflection: {e}")
            return False
    
    async def retrieve_reflection(self, reflection_type: str, date_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a reflection summary.
        
        Args:
            reflection_type: 'daily' or 'weekly'
            date_key: Date or week start date as ISO string
            
        Returns:
            Reflection data or None if not found
        """
        try:
            if not self.use_sqlite:
                logger.warning("⚠️ Reflection retrieval requires SQLite backend")
                return None
            
            table_name = f"{reflection_type}_reflections"
            
            async with aiosqlite.connect(str(self.sqlite_path)) as db:
                async with db.execute(f"""
                    SELECT reflection_data FROM {table_name} WHERE date = ?
                """, (date_key,)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        return json.loads(row[0])
                    else:
                        return None
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve {reflection_type} reflection: {e}")
            return None
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage system."""
        try:
            stats = {
                'storage_path': str(self.storage_path),
                'backends_enabled': {
                    'sqlite': self.use_sqlite,
                    'jsonl': self.use_jsonl
                },
                'total_events': 0,
                'daily_reflections': 0,
                'weekly_reflections': 0,
                'storage_size_mb': 0
            }
            
            if self.use_sqlite and self.sqlite_path.exists():
                async with aiosqlite.connect(str(self.sqlite_path)) as db:
                    # Count events
                    async with db.execute("SELECT COUNT(*) FROM events") as cursor:
                        row = await cursor.fetchone()
                        stats['total_events'] = row[0] if row else 0
                    
                    # Count daily reflections
                    async with db.execute("SELECT COUNT(*) FROM daily_reflections") as cursor:
                        row = await cursor.fetchone()
                        stats['daily_reflections'] = row[0] if row else 0
                    
                    # Count weekly reflections
                    async with db.execute("SELECT COUNT(*) FROM weekly_reflections") as cursor:
                        row = await cursor.fetchone()
                        stats['weekly_reflections'] = row[0] if row else 0
                
                # SQLite file size
                stats['sqlite_size_mb'] = self.sqlite_path.stat().st_size / (1024 * 1024)
            
            if self.use_jsonl and self.jsonl_path.exists():
                # Count JSONL files and size
                jsonl_files = list(self.jsonl_path.glob("events_*.jsonl"))
                jsonl_size = sum(f.stat().st_size for f in jsonl_files)
                stats['jsonl_files'] = len(jsonl_files)
                stats['jsonl_size_mb'] = jsonl_size / (1024 * 1024)
            
            # Total storage size
            stats['storage_size_mb'] = sum([
                stats.get('sqlite_size_mb', 0),
                stats.get('jsonl_size_mb', 0)
            ])
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get storage stats: {e}")
            return {'error': str(e)}
    
    async def backup_data(self, backup_name: Optional[str] = None) -> bool:
        """Create a backup of all memory data."""
        try:
            if not self.auto_backup:
                logger.warning("⚠️ Auto backup is disabled")
                return False
            
            if not backup_name:
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_dir = self.backup_path / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup SQLite database
            if self.use_sqlite and self.sqlite_path.exists():
                import shutil
                backup_db_path = backup_dir / "true_recall.db"
                shutil.copy2(self.sqlite_path, backup_db_path)
                logger.info(f"📦 Backed up SQLite database to {backup_db_path}")
            
            # Backup JSONL files
            if self.use_jsonl and self.jsonl_path.exists():
                import shutil
                backup_events_path = backup_dir / "events"
                shutil.copytree(self.jsonl_path, backup_events_path, dirs_exist_ok=True)
                logger.info(f"📦 Backed up JSONL files to {backup_events_path}")
            
            logger.info(f"✅ Backup completed: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    async def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """Clean up old data beyond the retention period."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            if self.use_sqlite:
                async with aiosqlite.connect(str(self.sqlite_path)) as db:
                    # Delete old events
                    await db.execute("""
                        DELETE FROM events WHERE timestamp < ?
                    """, (cutoff_date.isoformat(),))
                    
                    # Delete old reflections
                    await db.execute("""
                        DELETE FROM daily_reflections WHERE date < ?
                    """, (cutoff_date.date().isoformat(),))
                    
                    await db.commit()
            
            if self.use_jsonl:
                # Remove old JSONL files
                cutoff_date_str = cutoff_date.date().isoformat()
                for jsonl_file in self.jsonl_path.glob("events_*.jsonl"):
                    # Extract date from filename
                    filename = jsonl_file.stem
                    if filename.startswith("events_"):
                        date_part = filename.split("_")[1]
                        if date_part < cutoff_date_str:
                            jsonl_file.unlink()
                            logger.info(f"🗑️ Removed old JSONL file: {jsonl_file}")
            
            logger.info(f"🧹 Cleanup completed: removed data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    async def close(self):
        """Close the memory store and clean up resources."""
        try:
            if self._db:
                await self._db.close()
                self._db = None
            
            logger.info("🔒 Memory Store closed")
            
        except Exception as e:
            logger.error(f"❌ Failed to close memory store: {e}")

# Convenience functions for quick access
async def create_memory_store(storage_path: str = "memory_data", **config) -> MemoryStore:
    """Create and initialize a memory store."""
    store = MemoryStore(storage_path, config)
    await store.initialize()
    return store
