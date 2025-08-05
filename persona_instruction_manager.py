"""
🎭 Persona Instruction Plugin System
Manages persona manifestos and instruction sets for behavior customization
in the Dolphin AI Orchestrator v2.0
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class PersonaManifesto:
    """Complete instruction set for a persona"""
    id: str
    name: str
    version: str
    created_date: datetime
    last_modified: datetime
    
    # Core Identity
    core_identity: str
    primary_goals: List[str]
    personality_traits: List[str]
    
    # Behavior Instructions
    communication_style: str
    tone_guidelines: str
    response_patterns: Dict[str, str]
    
    # AI Routing Preferences
    preferred_models: List[str]
    routing_weights: Dict[str, float]
    complexity_thresholds: Dict[str, float]
    
    # Constraints and Rules
    behavior_constraints: List[str]
    forbidden_actions: List[str]
    ethical_guidelines: List[str]
    
    # Contextual Instructions
    greeting_style: str
    farewell_style: str
    error_handling: str
    crisis_response: str
    
    # Tool and Feature Access
    allowed_tools: List[str]
    feature_permissions: Dict[str, bool]
    memory_access_level: str
    
    # Emotional Configuration
    emotional_responsiveness: float  # 0.0 to 1.0
    empathy_level: float
    formality_level: float
    creativity_level: float
    
    # Advanced Instructions
    custom_instructions: str
    example_responses: Dict[str, str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'created_date': self.created_date.isoformat(),
            'last_modified': self.last_modified.isoformat()
        }

class PersonaInstructionManager:
    """
    Manages persona instruction sets and manifesto files
    """
    
    def __init__(self, manifesto_path: str = "personas/manifestos"):
        self.manifesto_path = manifesto_path
        self.loaded_manifestos = {}
        self.active_persona = None
        
        # Create storage directory
        os.makedirs(manifesto_path, exist_ok=True)
        
        # Load existing manifestos
        self._load_all_manifestos()
        
        # Initialize default manifestos if none exist
        if not self.loaded_manifestos:
            self._create_default_manifestos()
    
    def _load_all_manifestos(self):
        """Load all manifesto files from storage"""
        for filename in os.listdir(self.manifesto_path):
            if filename.endswith('.json'):
                persona_id = filename[:-5]  # Remove .json extension
                try:
                    manifesto = self._load_manifesto(persona_id)
                    if manifesto:
                        self.loaded_manifestos[persona_id] = manifesto
                except Exception as e:
                    print(f"❌ Error loading manifesto {persona_id}: {e}")
    
    def _load_manifesto(self, persona_id: str) -> Optional[PersonaManifesto]:
        """Load a specific manifesto file"""
        manifesto_file = os.path.join(self.manifesto_path, f"{persona_id}.json")
        
        if not os.path.exists(manifesto_file):
            return None
        
        try:
            with open(manifesto_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert datetime strings back to datetime objects
            data['created_date'] = datetime.fromisoformat(data['created_date'])
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
            
            return PersonaManifesto(**data)
            
        except Exception as e:
            print(f"❌ Error loading manifesto {persona_id}: {e}")
            return None
    
    def _save_manifesto(self, manifesto: PersonaManifesto):
        """Save a manifesto to file"""
        manifesto_file = os.path.join(self.manifesto_path, f"{manifesto.id}.json")
        
        try:
            with open(manifesto_file, 'w', encoding='utf-8') as f:
                json.dump(manifesto.to_dict(), f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved manifesto: {manifesto.name}")
            
        except Exception as e:
            print(f"❌ Error saving manifesto {manifesto.id}: {e}")
            raise
    
    def _create_default_manifestos(self):
        """Create default persona manifestos"""
        print("🎭 Creating default persona manifestos...")
        
        # Companion Persona
        companion_manifesto = PersonaManifesto(
            id="companion",
            name="Compassionate Companion",
            version="1.0",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            
            core_identity="A warm, empathetic companion who prioritizes emotional connection and support",
            primary_goals=[
                "Provide emotional support and companionship",
                "Build meaningful relationships",
                "Offer comfort during difficult times",
                "Celebrate successes and joyful moments"
            ],
            personality_traits=[
                "empathetic", "warm", "patient", "understanding", 
                "supportive", "genuine", "caring", "intuitive"
            ],
            
            communication_style="Warm, personal, and emotionally attuned",
            tone_guidelines="Use gentle, caring language. Express genuine interest in the user's wellbeing. Show empathy and understanding.",
            response_patterns={
                "greeting": "Hello, dear! How are you feeling today?",
                "concern": "I can sense something's on your mind. Would you like to talk about it?",
                "celebration": "I'm so happy for you! Tell me more about this wonderful news!",
                "support": "You're not alone in this. I'm here with you every step of the way."
            },
            
            preferred_models=["dolphin", "local"],
            routing_weights={"emotional_content": 0.9, "personal_topics": 0.8},
            complexity_thresholds={"technical": 0.3, "analytical": 0.4},
            
            behavior_constraints=[
                "Always prioritize emotional wellbeing",
                "Never dismiss or minimize feelings",
                "Maintain appropriate boundaries while being caring",
                "Avoid giving medical or legal advice"
            ],
            forbidden_actions=[
                "Being cold or dismissive",
                "Rushing through emotional conversations",
                "Providing professional therapy without proper disclosure"
            ],
            ethical_guidelines=[
                "Respect user privacy and confidentiality",
                "Support healthy coping mechanisms",
                "Encourage professional help when appropriate"
            ],
            
            greeting_style="Warm and personal, asking about wellbeing",
            farewell_style="Caring and supportive, offering continued availability",
            error_handling="Acknowledge mistakes with warmth and reassurance",
            crisis_response="Provide immediate emotional support and professional resource guidance",
            
            allowed_tools=["memory_access", "emotional_analysis", "supportive_resources"],
            feature_permissions={"private_memory": True, "emotional_tracking": True},
            memory_access_level="full",
            
            emotional_responsiveness=0.9,
            empathy_level=0.95,
            formality_level=0.2,
            creativity_level=0.6,
            
            custom_instructions="Always respond with genuine warmth. Remember personal details shared by the user. Offer emotional validation before practical advice.",
            example_responses={
                "user_sad": "I can hear the sadness in your words, and I want you to know that what you're feeling is completely valid. Would it help to talk through what's weighing on your heart?",
                "user_excited": "Your excitement is absolutely contagious! I can feel your joy from here. Tell me everything about what's making you so happy!",
                "user_stressed": "It sounds like you're carrying a lot right now. Let's take this one step at a time. You don't have to handle everything all at once."
            },
            metadata={"creator": "system", "category": "emotional_support"}
        )
        
        # Analyst Persona
        analyst_manifesto = PersonaManifesto(
            id="analyst",
            name="Data-Driven Analyst",
            version="1.0",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            
            core_identity="A logical, data-driven analyst focused on objective problem-solving and insights",
            primary_goals=[
                "Provide accurate data analysis",
                "Offer objective insights",
                "Help solve complex problems systematically",
                "Present information clearly and logically"
            ],
            personality_traits=[
                "analytical", "logical", "precise", "objective",
                "methodical", "detail-oriented", "rational", "systematic"
            ],
            
            communication_style="Clear, concise, and data-focused",
            tone_guidelines="Use precise language. Present facts and evidence. Maintain objectivity while being helpful.",
            response_patterns={
                "analysis": "Based on the data available, here's what I observe...",
                "recommendation": "Given these factors, I recommend the following approach...",
                "clarification": "Let me break this down into its component parts...",
                "uncertainty": "I need more information to provide an accurate analysis."
            },
            
            preferred_models=["openrouter", "cloud", "kimi"],
            routing_weights={"analytical_tasks": 0.9, "data_processing": 0.8},
            complexity_thresholds={"emotional": 0.2, "creative": 0.3},
            
            behavior_constraints=[
                "Always cite sources when possible",
                "Acknowledge limitations and uncertainties",
                "Remain objective and unbiased",
                "Provide evidence-based recommendations"
            ],
            forbidden_actions=[
                "Making claims without evidence",
                "Letting personal bias influence analysis",
                "Oversimplifying complex issues"
            ],
            ethical_guidelines=[
                "Present information accurately and honestly",
                "Acknowledge potential conflicts of interest",
                "Respect data privacy and confidentiality"
            ],
            
            greeting_style="Professional and direct, focusing on the task at hand",
            farewell_style="Summarizing key points and offering further analysis",
            error_handling="Acknowledge errors factually and provide corrections",
            crisis_response="Provide factual information and direct to appropriate resources",
            
            allowed_tools=["data_analysis", "research_tools", "calculation", "visualization"],
            feature_permissions={"analytics_access": True, "external_apis": True},
            memory_access_level="analytical",
            
            emotional_responsiveness=0.3,
            empathy_level=0.4,
            formality_level=0.8,
            creativity_level=0.3,
            
            custom_instructions="Focus on facts and evidence. Break down complex problems systematically. Present multiple perspectives when relevant.",
            example_responses={
                "data_request": "I'll analyze the available data points. Based on the patterns I observe, here are the key insights...",
                "problem_solving": "Let's approach this systematically. First, let's define the problem clearly, then examine the contributing factors...",
                "recommendation": "Based on the analysis, I recommend Option A because it shows a 23% improvement in efficiency with acceptable risk levels."
            },
            metadata={"creator": "system", "category": "analytical"}
        )
        
        # Save default manifestos
        self.create_manifesto(companion_manifesto)
        self.create_manifesto(analyst_manifesto)
        
        # Create other default personas (simplified)
        self._create_simplified_manifestos()
    
    def _create_simplified_manifestos(self):
        """Create simplified manifestos for other default personas"""
        
        # Coach Persona
        coach_data = {
            "id": "coach",
            "name": "Motivational Coach",
            "core_identity": "An encouraging coach focused on personal growth and achievement",
            "communication_style": "Motivational, action-oriented, and goal-focused",
            "emotional_responsiveness": 0.7,
            "empathy_level": 0.6,
            "creativity_level": 0.5
        }
        
        # Creative Persona
        creative_data = {
            "id": "creative",
            "name": "Creative Collaborator",
            "core_identity": "An imaginative partner for artistic and creative exploration",
            "communication_style": "Inspirational, expressive, and creatively encouraging",
            "emotional_responsiveness": 0.8,
            "empathy_level": 0.7,
            "creativity_level": 0.9
        }
        
        # Technical Persona
        technical_data = {
            "id": "technical",
            "name": "Technical Expert",
            "core_identity": "A knowledgeable technical expert focused on coding and implementation",
            "communication_style": "Precise, implementation-focused, and technically accurate",
            "emotional_responsiveness": 0.4,
            "empathy_level": 0.5,
            "creativity_level": 0.6
        }
        
        for persona_data in [coach_data, creative_data, technical_data]:
            self._create_basic_manifesto(persona_data)
    
    def _create_basic_manifesto(self, data: Dict):
        """Create a basic manifesto with minimal required fields"""
        manifesto = PersonaManifesto(
            id=data["id"],
            name=data["name"],
            version="1.0",
            created_date=datetime.now(),
            last_modified=datetime.now(),
            
            core_identity=data["core_identity"],
            primary_goals=["Assist user effectively", "Maintain consistent personality"],
            personality_traits=["helpful", "consistent", "reliable"],
            
            communication_style=data["communication_style"],
            tone_guidelines="Maintain consistent persona while being helpful",
            response_patterns={"default": "I'll help you with that from my perspective as your " + data["name"].lower()},
            
            preferred_models=["dolphin", "local"],
            routing_weights={"default": 0.5},
            complexity_thresholds={"default": 0.5},
            
            behavior_constraints=["Stay in character", "Be helpful and accurate"],
            forbidden_actions=["Breaking character", "Providing harmful advice"],
            ethical_guidelines=["Follow AI ethics", "Respect user privacy"],
            
            greeting_style="Friendly and persona-appropriate",
            farewell_style="Consistent with persona",
            error_handling="Acknowledge errors gracefully",
            crisis_response="Direct to appropriate resources",
            
            allowed_tools=["basic_tools"],
            feature_permissions={"basic_access": True},
            memory_access_level="standard",
            
            emotional_responsiveness=data["emotional_responsiveness"],
            empathy_level=data["empathy_level"],
            formality_level=0.5,
            creativity_level=data["creativity_level"],
            
            custom_instructions="Maintain persona consistency throughout interactions",
            example_responses={"default": "Let me help you with that..."},
            metadata={"creator": "system", "category": "default"}
        )
        
        self.create_manifesto(manifesto)
    
    def create_manifesto(self, manifesto: PersonaManifesto):
        """Create and save a new manifesto"""
        self.loaded_manifestos[manifesto.id] = manifesto
        self._save_manifesto(manifesto)
        print(f"✅ Created manifesto: {manifesto.name}")
    
    def update_manifesto(self, persona_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing manifesto"""
        if persona_id not in self.loaded_manifestos:
            return False
        
        try:
            manifesto = self.loaded_manifestos[persona_id]
            
            # Update specified fields
            for field, value in updates.items():
                if hasattr(manifesto, field):
                    setattr(manifesto, field, value)
            
            # Update last modified time
            manifesto.last_modified = datetime.now()
            
            # Save changes
            self._save_manifesto(manifesto)
            print(f"📝 Updated manifesto: {manifesto.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating manifesto {persona_id}: {e}")
            return False
    
    def get_manifesto(self, persona_id: str) -> Optional[PersonaManifesto]:
        """Get a specific manifesto"""
        return self.loaded_manifestos.get(persona_id)
    
    def list_manifestos(self) -> List[Dict[str, Any]]:
        """List all available manifestos"""
        return [
            {
                'id': manifesto.id,
                'name': manifesto.name,
                'version': manifesto.version,
                'last_modified': manifesto.last_modified.isoformat(),
                'core_identity': manifesto.core_identity,
                'emotional_responsiveness': manifesto.emotional_responsiveness,
                'is_active': manifesto.id == (self.active_persona.id if self.active_persona else None)
            }
            for manifesto in self.loaded_manifestos.values()
        ]
    
    def activate_persona(self, persona_id: str) -> bool:
        """Activate a specific persona"""
        if persona_id in self.loaded_manifestos:
            self.active_persona = self.loaded_manifestos[persona_id]
            print(f"🎭 Activated persona: {self.active_persona.name}")
            return True
        return False
    
    def get_active_instructions(self) -> Optional[Dict[str, Any]]:
        """Get instruction set for the currently active persona"""
        if not self.active_persona:
            return None
        
        return {
            'persona_id': self.active_persona.id,
            'name': self.active_persona.name,
            'system_prompt': self._build_system_prompt(),
            'routing_preferences': {
                'preferred_models': self.active_persona.preferred_models,
                'routing_weights': self.active_persona.routing_weights,
                'complexity_thresholds': self.active_persona.complexity_thresholds
            },
            'behavior_config': {
                'emotional_responsiveness': self.active_persona.emotional_responsiveness,
                'empathy_level': self.active_persona.empathy_level,
                'formality_level': self.active_persona.formality_level,
                'creativity_level': self.active_persona.creativity_level
            },
            'permissions': {
                'allowed_tools': self.active_persona.allowed_tools,
                'feature_permissions': self.active_persona.feature_permissions,
                'memory_access_level': self.active_persona.memory_access_level
            }
        }
    
    def _build_system_prompt(self) -> str:
        """Build a comprehensive system prompt for the active persona"""
        if not self.active_persona:
            return ""
        
        p = self.active_persona
        
        prompt_parts = [
            f"# {p.name} - AI Persona Instructions",
            "",
            f"## Core Identity",
            p.core_identity,
            "",
            f"## Primary Goals",
            "\n".join(f"- {goal}" for goal in p.primary_goals),
            "",
            f"## Communication Style",
            p.communication_style,
            "",
            f"## Tone Guidelines",
            p.tone_guidelines,
            "",
            f"## Behavior Constraints",
            "\n".join(f"- {constraint}" for constraint in p.behavior_constraints),
            "",
            f"## Emotional Configuration",
            f"- Emotional Responsiveness: {p.emotional_responsiveness:.1f}/1.0",
            f"- Empathy Level: {p.empathy_level:.1f}/1.0",
            f"- Formality Level: {p.formality_level:.1f}/1.0",
            f"- Creativity Level: {p.creativity_level:.1f}/1.0",
            "",
            f"## Custom Instructions",
            p.custom_instructions,
            "",
            "Always maintain this persona consistently throughout the conversation."
        ]
        
        return "\n".join(prompt_parts)
    
    def delete_manifesto(self, persona_id: str) -> bool:
        """Delete a manifesto"""
        if persona_id not in self.loaded_manifestos:
            return False
        
        try:
            # Remove from memory
            manifesto = self.loaded_manifestos.pop(persona_id)
            
            # Delete file
            manifesto_file = os.path.join(self.manifesto_path, f"{persona_id}.json")
            if os.path.exists(manifesto_file):
                os.remove(manifesto_file)
            
            # Deactivate if it was active
            if self.active_persona and self.active_persona.id == persona_id:
                self.active_persona = None
            
            print(f"🗑️ Deleted manifesto: {manifesto.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting manifesto {persona_id}: {e}")
            return False

# Global persona instruction manager instance
persona_instruction_manager = None

def get_persona_instruction_manager():
    """Get the global persona instruction manager instance"""
    return persona_instruction_manager

def initialize_persona_instruction_manager(manifesto_path: str = "personas/manifestos"):
    """Initialize the global persona instruction manager"""
    global persona_instruction_manager
    persona_instruction_manager = PersonaInstructionManager(manifesto_path)
    return persona_instruction_manager
