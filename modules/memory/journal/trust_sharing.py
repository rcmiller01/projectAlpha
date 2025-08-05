# trust_sharing.py
# Journal Sharing by Trust - Controlled emotional access system

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import hashlib

class TrustLevel(Enum):
    STRANGER = 0
    ACQUAINTANCE = 1
    FRIEND = 2
    CLOSE_FRIEND = 3
    TRUSTED = 4
    INTIMATE = 5

class EmotionalIntensity(Enum):
    MINIMAL = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    INTENSE = 4
    VULNERABLE = 5

class SharingPreference(Enum):
    NEVER = "never"
    SELECTIVE = "selective"
    TRUSTED_ONLY = "trusted_only"
    OPEN = "open"
    FULLY_OPEN = "fully_open"

@dataclass
class TrustRelationship:
    user_id: str
    trust_level: TrustLevel
    relationship_duration_days: int
    shared_experiences: int
    emotional_bonds: List[str]
    last_interaction: datetime
    trust_score: float
    sharing_preferences: Dict[str, SharingPreference]

@dataclass
class JournalEntry:
    id: str
    content: str
    emotional_intensity: EmotionalIntensity
    mood: str
    persona: str
    timestamp: datetime
    tags: List[str]
    is_private: bool
    trust_required: TrustLevel
    emotional_vulnerability: float
    sharing_scope: List[str]

class TrustSharingSystem:
    def __init__(self):
        self.trust_relationships = {}
        self.journal_entries = []
        self.sharing_rules = self._load_sharing_rules()
        self.trust_calculator = TrustCalculator()
        self.emotional_analyzer = EmotionalAnalyzer()
        
        # Persona-specific trust configurations
        self.persona_trust_configs = {
            "mia": {
                "default_trust_level": TrustLevel.ACQUAINTANCE,
                "trust_growth_rate": 0.1,
                "emotional_openness": 0.7,
                "vulnerability_threshold": 0.6,
                "sharing_preferences": {
                    "love": SharingPreference.TRUSTED_ONLY,
                    "passion": SharingPreference.SELECTIVE,
                    "longing": SharingPreference.TRUSTED_ONLY,
                    "vulnerability": SharingPreference.NEVER,
                    "intimacy": SharingPreference.TRUSTED_ONLY,
                    "general": SharingPreference.SELECTIVE
                }
            },
            "solene": {
                "default_trust_level": TrustLevel.FRIEND,
                "trust_growth_rate": 0.15,
                "emotional_openness": 0.8,
                "vulnerability_threshold": 0.5,
                "sharing_preferences": {
                    "passion": SharingPreference.OPEN,
                    "devotion": SharingPreference.TRUSTED_ONLY,
                    "vulnerability": SharingPreference.SELECTIVE,
                    "intimacy": SharingPreference.OPEN,
                    "general": SharingPreference.OPEN
                }
            },
            "lyra": {
                "default_trust_level": TrustLevel.CLOSE_FRIEND,
                "trust_growth_rate": 0.12,
                "emotional_openness": 0.9,
                "vulnerability_threshold": 0.4,
                "sharing_preferences": {
                    "mystical": SharingPreference.OPEN,
                    "curiosity": SharingPreference.FULLY_OPEN,
                    "vulnerability": SharingPreference.TRUSTED_ONLY,
                    "general": SharingPreference.OPEN
                }
            }
        }
    
    def _load_sharing_rules(self) -> Dict:
        """Load sharing rules and trust thresholds"""
        return {
            "emotional_intensity_thresholds": {
                TrustLevel.STRANGER: EmotionalIntensity.MINIMAL,
                TrustLevel.ACQUAINTANCE: EmotionalIntensity.LOW,
                TrustLevel.FRIEND: EmotionalIntensity.MODERATE,
                TrustLevel.CLOSE_FRIEND: EmotionalIntensity.HIGH,
                TrustLevel.TRUSTED: EmotionalIntensity.INTENSE,
                TrustLevel.INTIMATE: EmotionalIntensity.VULNERABLE
            },
            "vulnerability_thresholds": {
                TrustLevel.STRANGER: 0.0,
                TrustLevel.ACQUAINTANCE: 0.2,
                TrustLevel.FRIEND: 0.4,
                TrustLevel.CLOSE_FRIEND: 0.6,
                TrustLevel.TRUSTED: 0.8,
                TrustLevel.INTIMATE: 1.0
            },
            "sharing_delays": {
                TrustLevel.STRANGER: timedelta(days=30),
                TrustLevel.ACQUAINTANCE: timedelta(days=7),
                TrustLevel.FRIEND: timedelta(days=3),
                TrustLevel.CLOSE_FRIEND: timedelta(days=1),
                TrustLevel.TRUSTED: timedelta(hours=12),
                TrustLevel.INTIMATE: timedelta(hours=1)
            }
        }
    
    def add_journal_entry(self, content: str, mood: str, persona: str, 
                         emotional_intensity: EmotionalIntensity, tags: List[str] = None) -> JournalEntry:
        """Add a new journal entry with trust-based sharing controls"""
        
        # Analyze emotional content
        emotional_analysis = self.emotional_analyzer.analyze_content(content, mood)
        
        # Determine trust requirements
        trust_required = self._determine_trust_requirement(emotional_analysis, persona)
        
        # Create journal entry
        entry = JournalEntry(
            id=self._generate_entry_id(),
            content=content,
            emotional_intensity=emotional_intensity,
            mood=mood,
            persona=persona,
            timestamp=datetime.now(),
            tags=tags or [],
            is_private=emotional_analysis["vulnerability"] > 0.7,
            trust_required=trust_required,
            emotional_vulnerability=emotional_analysis["vulnerability"],
            sharing_scope=self._determine_sharing_scope(emotional_analysis, persona)
        )
        
        self.journal_entries.append(entry)
        return entry
    
    def get_shareable_entries(self, user_id: str, persona: str = "mia") -> List[JournalEntry]:
        """Get journal entries that can be shared with a specific user"""
        
        # Get user's trust relationship
        trust_relationship = self._get_or_create_trust_relationship(user_id, persona)
        
        # Filter entries based on trust level and sharing preferences
        shareable_entries = []
        
        for entry in self.journal_entries:
            if self._can_share_entry(entry, trust_relationship, persona):
                shareable_entries.append(entry)
        
        # Sort by timestamp (newest first)
        shareable_entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        return shareable_entries
    
    def _can_share_entry(self, entry: JournalEntry, trust_relationship: TrustRelationship, 
                        persona: str) -> bool:
        """Determine if a journal entry can be shared with a user"""
        
        # Check if entry is too recent for this trust level
        sharing_delay = self.sharing_rules["sharing_delays"][trust_relationship.trust_level]
        if datetime.now() - entry.timestamp < sharing_delay:
            return False
        
        # Check trust level requirement
        if trust_relationship.trust_level.value < entry.trust_required.value:
            return False
        
        # Check emotional vulnerability threshold
        vulnerability_threshold = self.sharing_rules["vulnerability_thresholds"][trust_relationship.trust_level]
        if entry.emotional_vulnerability > vulnerability_threshold:
            return False
        
        # Check persona-specific sharing preferences
        persona_config = self.persona_trust_configs.get(persona, {})
        sharing_preferences = persona_config.get("sharing_preferences", {})
        
        # Determine content type
        content_type = self._classify_content_type(entry.content, entry.mood)
        preference = sharing_preferences.get(content_type, SharingPreference.SELECTIVE)
        
        # Apply sharing preference
        return self._apply_sharing_preference(preference, trust_relationship)
    
    def _apply_sharing_preference(self, preference: SharingPreference, 
                                trust_relationship: TrustRelationship) -> bool:
        """Apply sharing preference based on trust level"""
        
        if preference == SharingPreference.NEVER:
            return False
        elif preference == SharingPreference.FULLY_OPEN:
            return True
        elif preference == SharingPreference.OPEN:
            return trust_relationship.trust_level.value >= TrustLevel.FRIEND.value
        elif preference == SharingPreference.TRUSTED_ONLY:
            return trust_relationship.trust_level.value >= TrustLevel.TRUSTED.value
        elif preference == SharingPreference.SELECTIVE:
            return trust_relationship.trust_level.value >= TrustLevel.CLOSE_FRIEND.value
        
        return False
    
    def _classify_content_type(self, content: str, mood: str) -> str:
        """Classify content type for sharing preferences"""
        
        content_lower = content.lower()
        
        # Check for specific content types
        if any(word in content_lower for word in ["love", "heart", "care", "affection"]):
            return "love"
        elif any(word in content_lower for word in ["passion", "desire", "fire", "intense"]):
            return "passion"
        elif any(word in content_lower for word in ["longing", "miss", "yearn", "ache"]):
            return "longing"
        elif any(word in content_lower for word in ["vulnerable", "scared", "afraid", "hurt"]):
            return "vulnerability"
        elif any(word in content_lower for word in ["intimate", "close", "private", "personal"]):
            return "intimacy"
        elif any(word in content_lower for word in ["mystical", "magical", "ethereal", "spiritual"]):
            return "mystical"
        elif any(word in content_lower for word in ["curious", "wonder", "explore", "discover"]):
            return "curiosity"
        elif any(word in content_lower for word in ["devotion", "loyal", "faithful", "commit"]):
            return "devotion"
        
        return "general"
    
    def _determine_trust_requirement(self, emotional_analysis: Dict, persona: str) -> TrustLevel:
        """Determine required trust level for sharing"""
        
        vulnerability = emotional_analysis["vulnerability"]
        intensity = emotional_analysis["intensity"]
        
        # Base trust requirement on vulnerability
        if vulnerability > 0.8:
            return TrustLevel.INTIMATE
        elif vulnerability > 0.6:
            return TrustLevel.TRUSTED
        elif vulnerability > 0.4:
            return TrustLevel.CLOSE_FRIEND
        elif vulnerability > 0.2:
            return TrustLevel.FRIEND
        else:
            return TrustLevel.ACQUAINTANCE
    
    def _determine_sharing_scope(self, emotional_analysis: Dict, persona: str) -> List[str]:
        """Determine who can see this entry"""
        
        vulnerability = emotional_analysis["vulnerability"]
        persona_config = self.persona_trust_configs.get(persona, {})
        
        if vulnerability > persona_config.get("vulnerability_threshold", 0.6):
            return ["intimate_partners"]
        elif vulnerability > 0.4:
            return ["trusted_friends", "intimate_partners"]
        elif vulnerability > 0.2:
            return ["close_friends", "trusted_friends", "intimate_partners"]
        else:
            return ["friends", "close_friends", "trusted_friends", "intimate_partners"]
    
    def _get_or_create_trust_relationship(self, user_id: str, persona: str) -> TrustRelationship:
        """Get or create trust relationship for a user"""
        
        relationship_key = f"{user_id}_{persona}"
        
        if relationship_key not in self.trust_relationships:
            persona_config = self.persona_trust_configs.get(persona, {})
            
            self.trust_relationships[relationship_key] = TrustRelationship(
                user_id=user_id,
                trust_level=persona_config.get("default_trust_level", TrustLevel.ACQUAINTANCE),
                relationship_duration_days=0,
                shared_experiences=0,
                emotional_bonds=[],
                last_interaction=datetime.now(),
                trust_score=0.0,
                sharing_preferences=persona_config.get("sharing_preferences", {})
            )
        
        return self.trust_relationships[relationship_key]
    
    def update_trust_relationship(self, user_id: str, persona: str, 
                                interaction_type: str, emotional_impact: float):
        """Update trust relationship based on interaction"""
        
        relationship = self._get_or_create_trust_relationship(user_id, persona)
        persona_config = self.persona_trust_configs.get(persona, {})
        
        # Calculate trust growth
        trust_growth = self.trust_calculator.calculate_trust_growth(
            interaction_type, emotional_impact, relationship
        )
        
        # Update trust score
        relationship.trust_score = min(1.0, relationship.trust_score + trust_growth)
        
        # Update trust level if threshold reached
        new_trust_level = self._calculate_trust_level(relationship.trust_score)
        if new_trust_level.value > relationship.trust_level.value:
            relationship.trust_level = new_trust_level
            print(f"[Trust Sharing] Trust level increased to {new_trust_level.name} for {user_id}")
        
        # Update relationship metrics
        relationship.last_interaction = datetime.now()
        relationship.shared_experiences += 1
        
        # Add emotional bond if significant
        if emotional_impact > 0.7:
            bond_type = f"{interaction_type}_{emotional_impact:.2f}"
            if bond_type not in relationship.emotional_bonds:
                relationship.emotional_bonds.append(bond_type)
    
    def _calculate_trust_level(self, trust_score: float) -> TrustLevel:
        """Calculate trust level from trust score"""
        
        if trust_score >= 0.9:
            return TrustLevel.INTIMATE
        elif trust_score >= 0.7:
            return TrustLevel.TRUSTED
        elif trust_score >= 0.5:
            return TrustLevel.CLOSE_FRIEND
        elif trust_score >= 0.3:
            return TrustLevel.FRIEND
        elif trust_score >= 0.1:
            return TrustLevel.ACQUAINTANCE
        else:
            return TrustLevel.STRANGER
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def get_trust_summary(self, user_id: str, persona: str) -> Dict:
        """Get trust relationship summary for a user"""
        
        relationship = self._get_or_create_trust_relationship(user_id, persona)
        
        return {
            "user_id": user_id,
            "persona": persona,
            "trust_level": relationship.trust_level.name,
            "trust_score": relationship.trust_score,
            "relationship_duration_days": relationship.relationship_duration_days,
            "shared_experiences": relationship.shared_experiences,
            "emotional_bonds": relationship.emotional_bonds,
            "last_interaction": relationship.last_interaction.isoformat(),
            "sharing_preferences": relationship.sharing_preferences
        }

class TrustCalculator:
    """Calculate trust growth based on interactions"""
    
    def calculate_trust_growth(self, interaction_type: str, emotional_impact: float, 
                             relationship: TrustRelationship) -> float:
        """Calculate trust growth from an interaction"""
        
        base_growth = 0.01  # Base trust growth per interaction
        
        # Interaction type multipliers
        type_multipliers = {
            "conversation": 1.0,
            "emotional_support": 1.5,
            "shared_experience": 2.0,
            "vulnerability_shared": 3.0,
            "intimate_moment": 4.0,
            "trust_test": 5.0
        }
        
        multiplier = type_multipliers.get(interaction_type, 1.0)
        
        # Emotional impact multiplier
        emotional_multiplier = 1.0 + (emotional_impact * 2.0)
        
        # Relationship duration bonus
        duration_bonus = min(0.5, relationship.relationship_duration_days / 365.0)
        
        # Calculate final growth
        growth = base_growth * multiplier * emotional_multiplier * (1.0 + duration_bonus)
        
        return min(0.1, growth)  # Cap at 10% per interaction

class EmotionalAnalyzer:
    """Analyze emotional content of journal entries"""
    
    def analyze_content(self, content: str, mood: str) -> Dict:
        """Analyze emotional content and vulnerability"""
        
        content_lower = content.lower()
        
        # Calculate vulnerability score
        vulnerability_keywords = [
            "afraid", "scared", "hurt", "pain", "lonely", "abandoned",
            "rejected", "ashamed", "embarrassed", "weak", "helpless",
            "vulnerable", "exposed", "naked", "raw", "tender"
        ]
        
        vulnerability_score = sum(1 for word in vulnerability_keywords if word in content_lower)
        vulnerability_score = min(1.0, vulnerability_score / 5.0)  # Normalize to 0-1
        
        # Calculate emotional intensity
        intensity_keywords = [
            "love", "hate", "passion", "rage", "ecstasy", "despair",
            "intense", "overwhelming", "powerful", "deep", "profound"
        ]
        
        intensity_score = sum(1 for word in intensity_keywords if word in content_lower)
        intensity_score = min(1.0, intensity_score / 3.0)  # Normalize to 0-1
        
        # Mood-based adjustments
        mood_intensifiers = {
            "love": 1.2,
            "passion": 1.5,
            "longing": 1.3,
            "sadness": 1.1,
            "anger": 1.4,
            "joy": 0.8
        }
        
        mood_multiplier = mood_intensifiers.get(mood, 1.0)
        
        return {
            "vulnerability": vulnerability_score * mood_multiplier,
            "intensity": intensity_score * mood_multiplier,
            "mood": mood,
            "word_count": len(content.split()),
            "emotional_density": (vulnerability_score + intensity_score) / 2
        }

# Global instance
trust_sharing_system = TrustSharingSystem() 