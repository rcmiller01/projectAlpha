// MongoDB Initialization Script for EmotionalAI

// Create database and collections
db = db.getSiblingDB('emotionalai');

// Create collections with proper indexes
db.createCollection('users');
db.createCollection('memories');
db.createCollection('conversations');
db.createCollection('rituals');
db.createCollection('private_content');
db.createCollection('biometric_sessions');
db.createCollection('symbolic_fusions');
db.createCollection('touch_journals');
db.createCollection('scene_templates');

// Create indexes for better performance
db.users.createIndex({ "user_id": 1 }, { unique: true });
db.memories.createIndex({ "user_id": 1, "timestamp": -1 });
db.memories.createIndex({ "emotion": 1 });
db.conversations.createIndex({ "user_id": 1, "timestamp": -1 });
db.rituals.createIndex({ "user_id": 1, "ritual_id": 1 });
db.private_content.createIndex({ "user_id": 1, "content_id": 1 });
db.biometric_sessions.createIndex({ "user_id": 1, "session_id": 1 });
db.symbolic_fusions.createIndex({ "user_id": 1, "timestamp": -1 });
db.touch_journals.createIndex({ "user_id": 1, "timestamp": -1 });

// Insert initial configuration
db.config.insertOne({
    "system": "emotionalai",
    "version": "1.0.0",
    "initialized": new Date(),
    "features": {
        "symbolic_fusion": true,
        "scene_initiation": true,
        "touch_journal": true,
        "dynamic_wake_word": true,
        "mirror_ritual": true,
        "private_scenes": true,
        "biometric_integration": true
    }
});

print("✅ EmotionalAI database initialized successfully"); 