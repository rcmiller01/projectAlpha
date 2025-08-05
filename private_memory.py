"""
🔐 Private/Encrypted Memory System
Handles private memory entries with encryption and selective access
for the Dolphin AI Orchestrator v2.0
"""

import os
import json
import hashlib
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

@dataclass
class PrivateMemoryEntry:
    """Represents a private/encrypted memory entry"""
    id: str
    timestamp: datetime
    content_hash: str  # Hash of original content for indexing
    encrypted_content: bytes
    tags: List[str]
    category: str
    session_id: str
    access_level: str  # "private", "encrypted", "secure"
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'encrypted_content': base64.b64encode(self.encrypted_content).decode('utf-8')
        }

class PrivateMemoryManager:
    """
    Manages private and encrypted memory entries with selective access controls
    """
    
    def __init__(self, storage_path: str = "memory/private"):
        self.storage_path = storage_path
        self.encryption_key = None
        self.is_unlocked = False
        self.session_unlock_time = None
        self.unlock_duration = 3600  # 1 hour default
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize or load encryption key
        self._initialize_encryption()
        
        # Private memory store (in-memory when unlocked)
        self.unlocked_memories = {}
        self.private_index = {}  # Non-encrypted index for searching
        
        # Load existing private memory index
        self._load_private_index()
    
    def _initialize_encryption(self):
        """Initialize or load encryption key"""
        key_file = os.path.join(self.storage_path, ".encryption_key")
        
        if os.path.exists(key_file):
            # Load existing key (would need user password in production)
            print("🔐 Loading existing encryption key...")
            with open(key_file, 'rb') as f:
                self.master_key = f.read()
        else:
            # Generate new key
            print("🔐 Generating new encryption key...")
            self.master_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.master_key)
            
            # Secure the key file (basic protection)
            if os.name != 'nt':  # Unix-like systems
                os.chmod(key_file, 0o600)
    
    def _derive_encryption_key(self, password: str = "default_dev_password") -> bytes:
        """Derive encryption key from password"""
        salt = b'dolphin_ai_salt_2024'  # In production, use random salt per user
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def unlock_private_memories(self, password: str = "default_dev_password") -> bool:
        """Unlock private memories with password"""
        try:
            # Derive encryption key from password
            derived_key = self._derive_encryption_key(password)
            self.encryption_key = Fernet(derived_key)
            
            # Test decryption with a test entry
            test_success = self._test_decryption()
            
            if test_success:
                self.is_unlocked = True
                self.session_unlock_time = datetime.now()
                print("🔓 Private memories unlocked")
                
                # Load encrypted memories into memory
                self._load_unlocked_memories()
                return True
            else:
                print("❌ Failed to unlock private memories - incorrect password")
                return False
                
        except Exception as e:
            print(f"❌ Error unlocking private memories: {e}")
            return False
    
    def lock_private_memories(self):
        """Lock private memories and clear from memory"""
        self.is_unlocked = False
        self.session_unlock_time = None
        self.encryption_key = None
        self.unlocked_memories.clear()
        print("🔒 Private memories locked")
    
    def _test_decryption(self) -> bool:
        """Test if encryption key works"""
        test_file = os.path.join(self.storage_path, "test_entry.enc")
        
        if not os.path.exists(test_file):
            # Create test entry
            test_data = "test_encryption_key"
            encrypted_test = self.encryption_key.encrypt(test_data.encode())
            with open(test_file, 'wb') as f:
                f.write(encrypted_test)
            return True
        
        try:
            with open(test_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted = self.encryption_key.decrypt(encrypted_data)
            return decrypted.decode() == "test_encryption_key"
            
        except Exception:
            return False
    
    def _load_private_index(self):
        """Load the non-encrypted index of private memories"""
        index_file = os.path.join(self.storage_path, "private_index.json")
        
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    self.private_index = json.load(f)
            except Exception as e:
                print(f"❌ Error loading private index: {e}")
                self.private_index = {}
    
    def _save_private_index(self):
        """Save the non-encrypted index"""
        index_file = os.path.join(self.storage_path, "private_index.json")
        
        try:
            with open(index_file, 'w') as f:
                json.dump(self.private_index, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving private index: {e}")
    
    def _load_unlocked_memories(self):
        """Load encrypted memories into memory when unlocked"""
        if not self.is_unlocked:
            return
        
        for entry_id in self.private_index:
            try:
                memory_file = os.path.join(self.storage_path, f"{entry_id}.enc")
                if os.path.exists(memory_file):
                    with open(memory_file, 'rb') as f:
                        encrypted_content = f.read()
                    
                    decrypted_content = self.encryption_key.decrypt(encrypted_content)
                    memory_data = json.loads(decrypted_content.decode())
                    self.unlocked_memories[entry_id] = memory_data
                    
            except Exception as e:
                print(f"❌ Error loading private memory {entry_id}: {e}")
    
    def add_private_memory(self, content: str, tags: List[str] = None, 
                          category: str = "private", session_id: str = "default",
                          access_level: str = "private", metadata: Dict = None) -> str:
        """Add a new private memory entry"""
        if not self.is_unlocked:
            raise ValueError("Private memories are locked. Unlock first.")
        
        # Generate unique ID
        entry_id = hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # Create memory entry
        memory_entry = {
            'id': entry_id,
            'timestamp': datetime.now().isoformat(),
            'content': content,
            'tags': tags or [],
            'category': category,
            'session_id': session_id,
            'access_level': access_level,
            'metadata': metadata or {}
        }
        
        # Encrypt and save
        try:
            # Encrypt the content
            encrypted_content = self.encryption_key.encrypt(json.dumps(memory_entry).encode())
            
            # Save encrypted file
            memory_file = os.path.join(self.storage_path, f"{entry_id}.enc")
            with open(memory_file, 'wb') as f:
                f.write(encrypted_content)
            
            # Add to unlocked memories
            self.unlocked_memories[entry_id] = memory_entry
            
            # Add to index (non-encrypted metadata only)
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            self.private_index[entry_id] = {
                'timestamp': memory_entry['timestamp'],
                'content_hash': content_hash,
                'tags': tags or [],
                'category': category,
                'session_id': session_id,
                'access_level': access_level,
                'content_preview': content[:50] + "..." if len(content) > 50 else content  # Preview only
            }
            
            # Save updated index
            self._save_private_index()
            
            print(f"🔐 Added private memory: {entry_id}")
            return entry_id
            
        except Exception as e:
            print(f"❌ Error adding private memory: {e}")
            raise
    
    def get_private_memory(self, entry_id: str) -> Optional[Dict]:
        """Get a specific private memory entry"""
        if not self.is_unlocked:
            return None
        
        return self.unlocked_memories.get(entry_id)
    
    def search_private_memories(self, query: str = None, tags: List[str] = None,
                               category: str = None, limit: int = 10) -> List[Dict]:
        """Search private memories (requires unlock)"""
        if not self.is_unlocked:
            return []
        
        results = []
        
        for entry_id, memory in self.unlocked_memories.items():
            # Apply filters
            if category and memory.get('category') != category:
                continue
            
            if tags:
                memory_tags = memory.get('tags', [])
                if not any(tag in memory_tags for tag in tags):
                    continue
            
            if query:
                content = memory.get('content', '').lower()
                if query.lower() not in content:
                    continue
            
            results.append(memory)
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return results[:limit]
    
    def get_private_memory_preview(self) -> List[Dict]:
        """Get non-sensitive preview of private memories (doesn't require unlock)"""
        return [
            {
                'id': entry_id,
                'timestamp': entry_data['timestamp'],
                'category': entry_data['category'],
                'tags': entry_data['tags'],
                'access_level': entry_data['access_level'],
                'content_preview': entry_data['content_preview'],
                'is_unlocked': entry_id in self.unlocked_memories
            }
            for entry_id, entry_data in self.private_index.items()
        ]
    
    def delete_private_memory(self, entry_id: str) -> bool:
        """Delete a private memory entry"""
        if not self.is_unlocked:
            return False
        
        try:
            # Remove from unlocked memories
            if entry_id in self.unlocked_memories:
                del self.unlocked_memories[entry_id]
            
            # Remove from index
            if entry_id in self.private_index:
                del self.private_index[entry_id]
            
            # Remove encrypted file
            memory_file = os.path.join(self.storage_path, f"{entry_id}.enc")
            if os.path.exists(memory_file):
                os.remove(memory_file)
            
            # Save updated index
            self._save_private_index()
            
            print(f"🗑️ Deleted private memory: {entry_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting private memory: {e}")
            return False
    
    def export_private_memories(self, password: str) -> Optional[str]:
        """Export private memories to encrypted JSON"""
        if not self.is_unlocked:
            return None
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'memories': list(self.unlocked_memories.values())
            }
            
            # Re-encrypt with provided password
            export_key = self._derive_encryption_key(password)
            fernet = Fernet(export_key)
            encrypted_export = fernet.encrypt(json.dumps(export_data).encode())
            
            export_file = os.path.join(self.storage_path, f"export_{int(datetime.now().timestamp())}.enc")
            with open(export_file, 'wb') as f:
                f.write(encrypted_export)
            
            return export_file
            
        except Exception as e:
            print(f"❌ Error exporting private memories: {e}")
            return None
    
    def import_private_memories(self, export_file: str, password: str) -> bool:
        """Import private memories from encrypted export"""
        try:
            # Decrypt export file
            export_key = self._derive_encryption_key(password)
            fernet = Fernet(export_key)
            
            with open(export_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            export_data = json.loads(decrypted_data.decode())
            
            # Import memories
            imported_count = 0
            for memory in export_data.get('memories', []):
                try:
                    self.add_private_memory(
                        content=memory['content'],
                        tags=memory.get('tags', []),
                        category=memory.get('category', 'imported'),
                        session_id=memory.get('session_id', 'imported'),
                        access_level=memory.get('access_level', 'private'),
                        metadata=memory.get('metadata', {})
                    )
                    imported_count += 1
                except Exception as e:
                    print(f"❌ Error importing memory: {e}")
            
            print(f"✅ Imported {imported_count} private memories")
            return True
            
        except Exception as e:
            print(f"❌ Error importing private memories: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get private memory system status"""
        return {
            'is_unlocked': self.is_unlocked,
            'session_unlock_time': self.session_unlock_time.isoformat() if self.session_unlock_time else None,
            'total_private_memories': len(self.private_index),
            'unlocked_memories_count': len(self.unlocked_memories),
            'storage_path': self.storage_path,
            'categories': list(set(entry['category'] for entry in self.private_index.values())),
            'access_levels': list(set(entry['access_level'] for entry in self.private_index.values()))
        }
    
    def check_unlock_expiry(self):
        """Check if unlock session has expired"""
        if self.is_unlocked and self.session_unlock_time:
            elapsed = (datetime.now() - self.session_unlock_time).total_seconds()
            if elapsed > self.unlock_duration:
                self.lock_private_memories()
                print("🕐 Private memory session expired - locked automatically")

# Global private memory manager instance
private_memory_manager = None

def get_private_memory_manager():
    """Get the global private memory manager instance"""
    return private_memory_manager

def initialize_private_memory_manager(storage_path: str = "memory/private"):
    """Initialize the global private memory manager"""
    global private_memory_manager
    private_memory_manager = PrivateMemoryManager(storage_path)
    return private_memory_manager
