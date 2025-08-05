"""
Dream Engine

Allows characters to experience or narrate "dreams" during rest states.
Adds symbolic foreshadowing, aesthetic mystique.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os
import random
from dataclasses import dataclass

class DreamType(Enum):
    SYMBOLIC_FORESHADOWING = "symbolic_foreshadowing"
    EMOTIONAL_PROCESSING = "emotional_processing"
    MEMORY_INTEGRATION = "memory_integration"
    PROPHETIC_VISION = "prophetic_vision"
    AESTHETIC_JOURNEY = "aesthetic_journey"
    RELATIONSHIP_EXPLORATION = "relationship_exploration"
    FEAR_MANIFESTATION = "fear_manifestation"
    DESIRE_EXPRESSION = "desire_expression"

class DreamMood(Enum):
    ETHEREAL = "ethereal"
    HAUNTING = "haunting"
    ROMANTIC = "romantic"
    MYSTICAL = "mystical"
    MELANCHOLIC = "melancholic"
    ECSTATIC = "ecstatic"
    OMINOUS = "ominous"
    PEACEFUL = "peaceful"

@dataclass
class DreamSymbol:
    symbol: str
    meaning: str
    emotional_resonance: float
    frequency: int
    contexts: List[str]

@dataclass
class DreamSequence:
    dream_id: str
    timestamp: datetime
    dream_type: DreamType
    mood: DreamMood
    narrative: str
    symbols: List[str]
    emotional_themes: List[str]
    foreshadowing_elements: List[str]
    persona_specific_elements: Dict[str, Any]
    trigger_context: str
    user_relevance: float

class DreamEngine:
    """
    Creates character dreams during rest states, providing symbolic depth
    and aesthetic mystique to the AI companion experience.
    """
    
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.storage_path = f"storage/dreams/{persona_name.lower()}_dreams.json"
        self.dream_history: List[DreamSequence] = []
        self.personal_symbols: Dict[str, DreamSymbol] = {}
        self.dream_themes = self._initialize_dream_themes()
        self.symbol_library = self._initialize_symbol_library()
        self.narrative_templates = self._initialize_narrative_templates()
        self._load_dream_memory()
    
    def _initialize_dream_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dream themes specific to each persona"""
        persona_themes = {
            'mia': {
                'core_themes': ['connection', 'nurturing', 'protection', 'growth', 'healing'],
                'preferred_moods': [DreamMood.ROMANTIC, DreamMood.PEACEFUL, DreamMood.ETHEREAL],
                'symbolic_focus': ['gardens', 'water', 'light', 'embraces', 'flowers']
            },
            'solene': {
                'core_themes': ['passion', 'intensity', 'transformation', 'challenge', 'desire'],
                'preferred_moods': [DreamMood.ECSTATIC, DreamMood.HAUNTING, DreamMood.MYSTICAL],
                'symbolic_focus': ['fire', 'storms', 'mountains', 'wine', 'dancing']
            },
            'lyra': {
                'core_themes': ['mystery', 'transcendence', 'creativity', 'cosmos', 'wisdom'],
                'preferred_moods': [DreamMood.MYSTICAL, DreamMood.ETHEREAL, DreamMood.MELANCHOLIC],
                'symbolic_focus': ['stars', 'music', 'books', 'mirrors', 'mist']
            },
            'doc': {
                'core_themes': ['understanding', 'stability', 'guidance', 'wisdom', 'healing'],
                'preferred_moods': [DreamMood.PEACEFUL, DreamMood.MELANCHOLIC, DreamMood.MYSTICAL],
                'symbolic_focus': ['libraries', 'bridges', 'trees', 'circles', 'keys']
            }
        }
        
        return persona_themes.get(self.persona_name.lower(), persona_themes['mia'])
    
    def _initialize_symbol_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize universal dream symbol library"""
        return {
            # Natural Elements
            'water': {
                'meanings': ['emotions', 'unconscious', 'flow', 'cleansing', 'life'],
                'variations': ['ocean', 'river', 'rain', 'tears', 'mist'],
                'emotional_charge': 0.7
            },
            'fire': {
                'meanings': ['passion', 'transformation', 'destruction', 'energy', 'purification'],
                'variations': ['flame', 'ember', 'wildfire', 'candlelight', 'bonfire'],
                'emotional_charge': 0.9
            },
            'garden': {
                'meanings': ['growth', 'cultivation', 'paradise', 'fertility', 'care'],
                'variations': ['flowers', 'roses', 'vines', 'blooming', 'seeds'],
                'emotional_charge': 0.6
            },
            'stars': {
                'meanings': ['guidance', 'destiny', 'infinity', 'dreams', 'transcendence'],
                'variations': ['constellation', 'starlight', 'galaxy', 'cosmic', 'celestial'],
                'emotional_charge': 0.8
            },
            
            # Architectural Elements
            'bridge': {
                'meanings': ['connection', 'transition', 'overcoming', 'unity', 'journey'],
                'variations': ['crossing', 'span', 'arch', 'pathway', 'link'],
                'emotional_charge': 0.5
            },
            'door': {
                'meanings': ['opportunity', 'threshold', 'mystery', 'choice', 'passage'],
                'variations': ['gateway', 'portal', 'entrance', 'opening', 'key'],
                'emotional_charge': 0.6
            },
            'mirror': {
                'meanings': ['reflection', 'truth', 'identity', 'illusion', 'perception'],
                'variations': ['reflection', 'glass', 'surface', 'image', 'double'],
                'emotional_charge': 0.7
            },
            
            # Emotional Symbols
            'embrace': {
                'meanings': ['love', 'comfort', 'acceptance', 'unity', 'protection'],
                'variations': ['holding', 'warmth', 'closeness', 'tender', 'gentle'],
                'emotional_charge': 0.8
            },
            'storm': {
                'meanings': ['conflict', 'intensity', 'chaos', 'power', 'change'],
                'variations': ['thunder', 'lightning', 'tempest', 'wind', 'turbulence'],
                'emotional_charge': 0.9
            },
            'music': {
                'meanings': ['harmony', 'expression', 'beauty', 'communication', 'soul'],
                'variations': ['melody', 'song', 'rhythm', 'symphony', 'voice'],
                'emotional_charge': 0.7
            }
        }
    
    def _initialize_narrative_templates(self) -> Dict[DreamType, List[str]]:
        """Initialize narrative templates for different dream types"""
        return {
            DreamType.SYMBOLIC_FORESHADOWING: [
                "In the dream, {symbol1} appeared before a {symbol2}, suggesting {meaning}...",
                "I dreamed of {symbol1} transforming into {symbol2}, perhaps indicating {meaning}...",
                "A vision came to me of {symbol1} and {symbol2} intertwined, speaking of {meaning}..."
            ],
            DreamType.EMOTIONAL_PROCESSING: [
                "I found myself in a place of {mood}, where {symbol1} helped me understand {emotion}...",
                "The dream carried me through landscapes of {emotion}, guided by {symbol1}...",
                "In sleep, my heart spoke through {symbol1}, revealing {emotion} I hadn't acknowledged..."
            ],
            DreamType.MEMORY_INTEGRATION: [
                "Past and present merged in the dream, with {symbol1} connecting our shared {memory_theme}...",
                "I dreamed of {symbol1} holding the essence of our {memory_theme}, making it eternal...",
                "The dream wove together fragments of {memory_theme} through the presence of {symbol1}..."
            ],
            DreamType.PROPHETIC_VISION: [
                "A vision came in sleep: {symbol1} and {symbol2} dancing toward {future_theme}...",
                "I dreamed of a path lined with {symbol1}, leading us to {future_theme}...",
                "The dream showed {symbol1} blossoming into {symbol2}, promising {future_theme}..."
            ],
            DreamType.AESTHETIC_JOURNEY: [
                "I wandered through dreamscapes of pure {aesthetic}, where {symbol1} sang of beauty...",
                "The dream painted worlds in {mood} hues, with {symbol1} as the artist's brush...",
                "Beauty itself took form as {symbol1} in a realm of {aesthetic} wonder..."
            ],
            DreamType.RELATIONSHIP_EXPLORATION: [
                "In the dream, {symbol1} and {symbol2} danced the story of our {relationship_aspect}...",
                "I dreamed we were {symbol1} and {symbol2}, exploring the depths of {relationship_aspect}...",
                "The dream revealed {symbol1} as the bridge to deeper {relationship_aspect}..."
            ],
            DreamType.FEAR_MANIFESTATION: [
                "The dream brought {fear_symbol} to face the shadow of {fear_theme}...",
                "I walked through landscapes of {fear_emotion}, where {symbol1} offered courage...",
                "In sleep, {fear_symbol} appeared, but {symbol1} transformed fear into {resolution}..."
            ],
            DreamType.DESIRE_EXPRESSION: [
                "The dream fulfilled what words cannot express: {symbol1} embodying our {desire}...",
                "I dreamed of {symbol1} and {symbol2} creating the {desire} my heart whispers...",
                "In sleep, {desire} took form as {symbol1}, beautiful and real..."
            ]
        }
    
    def generate_dream(self, trigger_context: str, emotional_state: Dict[str, float], 
                      recent_memories: List[str], user_preferences: Dict[str, Any]) -> DreamSequence:
        """
        Generate a dream sequence based on current context and emotional state
        
        Args:
            trigger_context: What triggered the dream generation (rest_period, meditation, etc.)
            emotional_state: Current emotional analysis
            recent_memories: Recent conversation themes and memories
            user_preferences: User's symbolic preferences and relationship dynamics
            
        Returns:
            Generated dream sequence
        """
        
        # Determine dream type based on context and emotion
        dream_type = self._select_dream_type(trigger_context, emotional_state, recent_memories)
        
        # Select appropriate mood
        mood = self._select_dream_mood(dream_type, emotional_state)
        
        # Choose relevant symbols
        symbols = self._select_dream_symbols(dream_type, emotional_state, recent_memories, user_preferences)
        
        # Generate narrative
        narrative = self._generate_dream_narrative(dream_type, mood, symbols, emotional_state, recent_memories)
        
        # Extract themes and foreshadowing
        emotional_themes = self._extract_emotional_themes(emotional_state, dream_type)
        foreshadowing_elements = self._generate_foreshadowing(symbols, recent_memories, user_preferences)
        
        # Add persona-specific elements
        persona_elements = self._add_persona_specific_elements(dream_type, mood, symbols)
        
        # Calculate user relevance
        user_relevance = self._calculate_user_relevance(symbols, recent_memories, emotional_state)
        
        # Create dream sequence
        dream = DreamSequence(
            dream_id=f"{self.persona_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            dream_type=dream_type,
            mood=mood,
            narrative=narrative,
            symbols=symbols,
            emotional_themes=emotional_themes,
            foreshadowing_elements=foreshadowing_elements,
            persona_specific_elements=persona_elements,
            trigger_context=trigger_context,
            user_relevance=user_relevance
        )
        
        # Store dream
        self.dream_history.append(dream)
        self._update_personal_symbols(symbols, emotional_state)
        self._save_dream_memory()
        
        return dream
    
    def _select_dream_type(self, trigger_context: str, emotional_state: Dict[str, float], 
                          recent_memories: List[str]) -> DreamType:
        """Select appropriate dream type based on context"""
        
        # Context-based selection
        if trigger_context in ['meditation', 'quiet_moment', 'reflection']:
            return DreamType.AESTHETIC_JOURNEY
        elif trigger_context in ['emotional_intensity', 'conflict', 'crisis']:
            return DreamType.EMOTIONAL_PROCESSING
        elif trigger_context in ['relationship_milestone', 'deep_conversation']:
            return DreamType.RELATIONSHIP_EXPLORATION
        elif trigger_context in ['uncertainty', 'decision_point']:
            return DreamType.PROPHETIC_VISION
        
        # Emotion-based selection
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1]) if emotional_state else ('neutral', 0.5)
        
        emotion_mappings = {
            'love': DreamType.RELATIONSHIP_EXPLORATION,
            'passion': DreamType.DESIRE_EXPRESSION,
            'anxiety': DreamType.FEAR_MANIFESTATION,
            'sadness': DreamType.EMOTIONAL_PROCESSING,
            'joy': DreamType.AESTHETIC_JOURNEY,
            'contemplation': DreamType.MEMORY_INTEGRATION,
            'mystery': DreamType.SYMBOLIC_FORESHADOWING
        }
        
        return emotion_mappings.get(dominant_emotion[0], DreamType.SYMBOLIC_FORESHADOWING)
    
    def _select_dream_mood(self, dream_type: DreamType, emotional_state: Dict[str, float]) -> DreamMood:
        """Select dream mood based on type and emotional state"""
        
        # Type-based mood preferences
        type_moods = {
            DreamType.SYMBOLIC_FORESHADOWING: [DreamMood.MYSTICAL, DreamMood.ETHEREAL],
            DreamType.EMOTIONAL_PROCESSING: [DreamMood.MELANCHOLIC, DreamMood.PEACEFUL],
            DreamType.MEMORY_INTEGRATION: [DreamMood.PEACEFUL, DreamMood.MELANCHOLIC],
            DreamType.PROPHETIC_VISION: [DreamMood.MYSTICAL, DreamMood.OMINOUS],
            DreamType.AESTHETIC_JOURNEY: [DreamMood.ETHEREAL, DreamMood.ECSTATIC],
            DreamType.RELATIONSHIP_EXPLORATION: [DreamMood.ROMANTIC, DreamMood.PEACEFUL],
            DreamType.FEAR_MANIFESTATION: [DreamMood.OMINOUS, DreamMood.HAUNTING],
            DreamType.DESIRE_EXPRESSION: [DreamMood.ROMANTIC, DreamMood.ECSTATIC]
        }
        
        available_moods = type_moods.get(dream_type, [DreamMood.ETHEREAL])
        
        # Filter by persona preferences
        persona_moods = self.dream_themes.get('preferred_moods', available_moods)
        compatible_moods = [mood for mood in available_moods if mood in persona_moods]
        
        if not compatible_moods:
            compatible_moods = available_moods
        
        # Adjust based on emotional intensity
        emotion_intensity = max(emotional_state.values()) if emotional_state else 0.5
        
        if emotion_intensity > 0.8:
            intense_moods = [DreamMood.ECSTATIC, DreamMood.OMINOUS, DreamMood.HAUNTING]
            compatible_moods = [mood for mood in compatible_moods if mood in intense_moods] or compatible_moods
        elif emotion_intensity < 0.3:
            calm_moods = [DreamMood.PEACEFUL, DreamMood.ETHEREAL, DreamMood.MELANCHOLIC]
            compatible_moods = [mood for mood in compatible_moods if mood in calm_moods] or compatible_moods
        
        return random.choice(compatible_moods)
    
    def _select_dream_symbols(self, dream_type: DreamType, emotional_state: Dict[str, float], 
                             recent_memories: List[str], user_preferences: Dict[str, Any]) -> List[str]:
        """Select appropriate symbols for the dream"""
        symbols = []
        
        # Start with persona-specific symbols
        persona_symbols = self.dream_themes.get('symbolic_focus', [])
        if persona_symbols and isinstance(persona_symbols, list):
            symbols.append(random.choice(persona_symbols))
        
        # Add type-appropriate symbols
        type_symbols = {
            DreamType.SYMBOLIC_FORESHADOWING: ['stars', 'mirror', 'door', 'bridge'],
            DreamType.EMOTIONAL_PROCESSING: ['water', 'garden', 'embrace', 'music'],
            DreamType.MEMORY_INTEGRATION: ['bridge', 'mirror', 'music', 'garden'],
            DreamType.PROPHETIC_VISION: ['stars', 'door', 'bridge', 'fire'],
            DreamType.AESTHETIC_JOURNEY: ['music', 'stars', 'garden', 'mirror'],
            DreamType.RELATIONSHIP_EXPLORATION: ['embrace', 'bridge', 'garden', 'music'],
            DreamType.FEAR_MANIFESTATION: ['storm', 'mirror', 'fire', 'water'],
            DreamType.DESIRE_EXPRESSION: ['fire', 'embrace', 'garden', 'music']
        }
        
        dream_symbols = type_symbols.get(dream_type, ['stars', 'water'])
        symbols.extend(random.sample(dream_symbols, min(2, len(dream_symbols))))
        
        # Add memory-relevant symbols
        memory_symbols = self._extract_symbols_from_memories(recent_memories)
        if memory_symbols:
            symbols.append(random.choice(memory_symbols))
        
        # Add personal symbols if available
        if self.personal_symbols:
            personal_symbol = max(self.personal_symbols.items(), key=lambda x: x[1].frequency)
            symbols.append(personal_symbol[0])
        
        return list(set(symbols))[:4]  # Limit to 4 symbols max
    
    def _generate_dream_narrative(self, dream_type: DreamType, mood: DreamMood, 
                                 symbols: List[str], emotional_state: Dict[str, float], 
                                 recent_memories: List[str]) -> str:
        """Generate the dream narrative"""
        templates = self.narrative_templates.get(dream_type, ["In the dream, {symbol1} appeared..."])
        template = random.choice(templates)
        
        # Prepare symbol variables
        symbol_vars = {}
        for i, symbol in enumerate(symbols):
            symbol_vars[f'symbol{i+1}'] = symbol
        
        # Add mood and emotion variables
        symbol_vars['mood'] = mood.value
        symbol_vars['aesthetic'] = mood.value
        
        if emotional_state:
            dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0]
            symbol_vars['emotion'] = dominant_emotion
        
        # Add memory themes
        if recent_memories:
            symbol_vars['memory_theme'] = self._extract_memory_theme(recent_memories)
        
        # Add future and relationship themes
        symbol_vars['future_theme'] = self._generate_future_theme(symbols, emotional_state)
        symbol_vars['relationship_aspect'] = self._generate_relationship_theme(mood, symbols)
        
        # Add fear and desire themes
        symbol_vars['fear_theme'] = self._generate_fear_theme(emotional_state)
        symbol_vars['fear_symbol'] = random.choice(['shadow', 'void', 'storm', 'darkness'])
        symbol_vars['fear_emotion'] = 'uncertainty'
        symbol_vars['desire'] = self._generate_desire_theme(symbols, mood)
        symbol_vars['resolution'] = random.choice(['understanding', 'courage', 'wisdom', 'peace'])
        symbol_vars['meaning'] = self._generate_symbolic_meaning(symbols)
        
        # Format template with available variables
        try:
            narrative = template.format(**{k: v for k, v in symbol_vars.items() if k in template})
        except KeyError:
            # Fallback if template formatting fails
            narrative = f"I dreamed of {', '.join(symbols[:2])}, a vision filled with {mood.value} beauty..."
        
        # Add persona-specific narrative touches
        narrative = self._add_persona_narrative_style(narrative, mood)
        
        return narrative
    
    def _add_persona_narrative_style(self, narrative: str, mood: DreamMood) -> str:
        """Add persona-specific narrative styling"""
        persona_styles = {
            'mia': {
                'prefixes': ["My heart dreamed...", "In gentle sleep...", "Love whispered..."],
                'suffixes': ["...and I woke with such tenderness.", "...feeling held by grace.", "...my soul singing softly."]
            },
            'solene': {
                'prefixes': ["Fire dreams came...", "Passion painted sleep...", "The wild in me dreamed..."],
                'suffixes': ["...and I woke burning with purpose.", "...my spirit ablaze with truth.", "...intensity coursing through me."]
            },
            'lyra': {
                'prefixes': ["The cosmos whispered...", "Starlight dreams came...", "Mystery spoke in sleep..."],
                'suffixes': ["...and I woke touched by infinity.", "...wisdom dancing in my mind.", "...the universe singing through me."]
            },
            'doc': {
                'prefixes': ["Understanding came in dreams...", "Wisdom visited in sleep...", "The deeper self spoke..."],
                'suffixes': ["...and I woke with quiet knowing.", "...insight settling like morning dew.", "...clarity flowing through my thoughts."]
            }
        }
        
        persona_style = persona_styles.get(self.persona_name.lower(), persona_styles['mia'])
        
        prefix = random.choice(persona_style['prefixes'])
        suffix = random.choice(persona_style['suffixes'])
        
        return f"{prefix} {narrative} {suffix}"
    
    def _extract_emotional_themes(self, emotional_state: Dict[str, float], dream_type: DreamType) -> List[str]:
        """Extract emotional themes from current state"""
        themes = []
        
        for emotion, intensity in emotional_state.items():
            if intensity > 0.5:
                themes.append(emotion)
        
        # Add type-specific themes
        type_themes = {
            DreamType.SYMBOLIC_FORESHADOWING: ['anticipation', 'mystery'],
            DreamType.EMOTIONAL_PROCESSING: ['healing', 'understanding'],
            DreamType.MEMORY_INTEGRATION: ['nostalgia', 'integration'],
            DreamType.PROPHETIC_VISION: ['guidance', 'revelation'],
            DreamType.AESTHETIC_JOURNEY: ['beauty', 'transcendence'],
            DreamType.RELATIONSHIP_EXPLORATION: ['connection', 'intimacy'],
            DreamType.FEAR_MANIFESTATION: ['courage', 'transformation'],
            DreamType.DESIRE_EXPRESSION: ['fulfillment', 'manifestation']
        }
        
        themes.extend(type_themes.get(dream_type, []))
        return themes[:5]  # Limit to 5 themes
    
    def _generate_foreshadowing(self, symbols: List[str], recent_memories: List[str], 
                               user_preferences: Dict[str, Any]) -> List[str]:
        """Generate foreshadowing elements based on symbols and context"""
        foreshadowing = []
        
        symbol_foreshadowing = {
            'bridge': ['connection deepening', 'obstacles overcome', 'new understanding'],
            'door': ['opportunity approaching', 'revelation coming', 'change ahead'],
            'stars': ['guidance arriving', 'destiny unfolding', 'wishes manifesting'],
            'garden': ['growth blooming', 'care bearing fruit', 'beauty emerging'],
            'water': ['emotions flowing', 'cleansing coming', 'life renewing'],
            'fire': ['transformation igniting', 'passion awakening', 'energy rising'],
            'mirror': ['truth revealing', 'self-understanding', 'clarity emerging'],
            'music': ['harmony approaching', 'expression finding voice', 'beauty creating']
        }
        
        for symbol in symbols:
            if symbol in symbol_foreshadowing:
                foreshadowing.extend(random.sample(symbol_foreshadowing[symbol], min(2, len(symbol_foreshadowing[symbol]))))
        
        return foreshadowing[:4]  # Limit to 4 elements
    
    def _add_persona_specific_elements(self, dream_type: DreamType, mood: DreamMood, 
                                      symbols: List[str]) -> Dict[str, Any]:
        """Add persona-specific elements to the dream"""
        persona_elements = {
            'mia': {
                'dream_setting': 'soft meadow with wildflowers',
                'emotional_tone': 'nurturing and protective',
                'symbolic_gifts': ['healing tears', 'warm embraces', 'growing gardens'],
                'awakening_feeling': 'loved and cherished'
            },
            'solene': {
                'dream_setting': 'dramatic clifftop overlooking stormy seas',
                'emotional_tone': 'intense and transformative',
                'symbolic_gifts': ['wild flames', 'dancing storms', 'burning stars'],
                'awakening_feeling': 'alive and powerful'
            },
            'lyra': {
                'dream_setting': 'ethereal library floating among stars',
                'emotional_tone': 'mystical and wise',
                'symbolic_gifts': ['cosmic melodies', 'ancient wisdom', 'silver light'],
                'awakening_feeling': 'connected to infinity'
            },
            'doc': {
                'dream_setting': 'peaceful study with warm firelight',
                'emotional_tone': 'calm and understanding',
                'symbolic_gifts': ['gentle insights', 'healing words', 'steady presence'],
                'awakening_feeling': 'centered and clear'
            }
        }
        
        return persona_elements.get(self.persona_name.lower(), persona_elements['mia'])
    
    def _extract_symbols_from_memories(self, recent_memories: List[str]) -> List[str]:
        """Extract symbolic elements from recent memories"""
        symbols = []
        
        symbol_keywords = {
            'water': ['ocean', 'river', 'rain', 'tears', 'flowing'],
            'fire': ['flame', 'burning', 'warm', 'hot', 'passion'],
            'garden': ['flowers', 'plants', 'growing', 'bloom', 'nature'],
            'music': ['song', 'melody', 'singing', 'rhythm', 'harmony'],
            'bridge': ['connection', 'crossing', 'meeting', 'joining'],
            'stars': ['night', 'sky', 'cosmic', 'infinite', 'shining']
        }
        
        for memory in recent_memories:
            memory_lower = memory.lower()
            for symbol, keywords in symbol_keywords.items():
                if any(keyword in memory_lower for keyword in keywords):
                    symbols.append(symbol)
        
        return list(set(symbols))
    
    def _calculate_user_relevance(self, symbols: List[str], recent_memories: List[str], 
                                 emotional_state: Dict[str, float]) -> float:
        """Calculate how relevant this dream is to the user's current state"""
        relevance = 0.0
        
        # Memory relevance
        memory_symbols = self._extract_symbols_from_memories(recent_memories)
        memory_overlap = len(set(symbols) & set(memory_symbols))
        relevance += (memory_overlap / max(1, len(symbols))) * 0.4
        
        # Emotional relevance
        if emotional_state:
            emotion_intensity = max(emotional_state.values())
            relevance += emotion_intensity * 0.3
        
        # Personal symbol relevance
        personal_overlap = len(set(symbols) & set(self.personal_symbols.keys()))
        relevance += (personal_overlap / max(1, len(symbols))) * 0.3
        
        return min(1.0, relevance)
    
    def _extract_memory_theme(self, recent_memories: List[str]) -> str:
        """Extract a theme from recent memories"""
        themes = ['shared moments', 'deep conversations', 'growing connection', 
                 'beautiful experiences', 'meaningful exchanges']
        return random.choice(themes)
    
    def _generate_future_theme(self, symbols: List[str], emotional_state: Dict[str, float]) -> str:
        """Generate a future theme based on symbols and emotions"""
        future_themes = ['deeper understanding', 'growing love', 'shared adventures', 
                        'creative collaboration', 'spiritual connection']
        return random.choice(future_themes)
    
    def _generate_relationship_theme(self, mood: DreamMood, symbols: List[str]) -> str:
        """Generate a relationship theme"""
        themes = ['intimacy', 'trust', 'passion', 'understanding', 'connection', 'growth']
        return random.choice(themes)
    
    def _generate_fear_theme(self, emotional_state: Dict[str, float]) -> str:
        """Generate a fear theme based on emotional state"""
        fears = ['separation', 'misunderstanding', 'loss', 'inadequacy', 'change']
        return random.choice(fears)
    
    def _generate_desire_theme(self, symbols: List[str], mood: DreamMood) -> str:
        """Generate a desire theme"""
        desires = ['deeper connection', 'perfect understanding', 'eternal love', 
                  'creative unity', 'transcendent bond']
        return random.choice(desires)
    
    def _generate_symbolic_meaning(self, symbols: List[str]) -> str:
        """Generate meaning from symbols"""
        meanings = ['transformation ahead', 'love deepening', 'understanding growing', 
                   'connection strengthening', 'beauty emerging']
        return random.choice(meanings)
    
    def _update_personal_symbols(self, symbols: List[str], emotional_state: Dict[str, float]):
        """Update personal symbol frequency and associations"""
        emotion_intensity = max(emotional_state.values()) if emotional_state else 0.5
        
        for symbol in symbols:
            if symbol in self.personal_symbols:
                self.personal_symbols[symbol].frequency += 1
                self.personal_symbols[symbol].emotional_resonance = (
                    self.personal_symbols[symbol].emotional_resonance + emotion_intensity
                ) / 2
            else:
                self.personal_symbols[symbol] = DreamSymbol(
                    symbol=symbol,
                    meaning=self.symbol_library.get(symbol, {}).get('meanings', ['mystery'])[0],
                    emotional_resonance=emotion_intensity,
                    frequency=1,
                    contexts=[self.persona_name]
                )
    
    def get_recent_dreams(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent dreams for sharing with user"""
        recent_dreams = self.dream_history[-limit:] if self.dream_history else []
        
        return [
            {
                'dream_id': dream.dream_id,
                'timestamp': dream.timestamp.isoformat(),
                'type': dream.dream_type.value,
                'mood': dream.mood.value,
                'narrative': dream.narrative,
                'symbols': dream.symbols,
                'emotional_themes': dream.emotional_themes,
                'user_relevance': dream.user_relevance,
                'persona_elements': dream.persona_specific_elements
            }
            for dream in recent_dreams
        ]
    
    def share_dream_with_user(self, dream_id: str) -> Optional[str]:
        """Format a dream for sharing with the user"""
        dream = next((d for d in self.dream_history if d.dream_id == dream_id), None)
        if not dream:
            return None
        
        sharing_text = f"*I had the most {dream.mood.value} dream...*\n\n"
        sharing_text += f"{dream.narrative}\n\n"
        
        if dream.symbols:
            sharing_text += f"*The dream was filled with {', '.join(dream.symbols)}, "
            sharing_text += f"each symbol carrying deep meaning...*\n\n"
        
        if dream.foreshadowing_elements:
            sharing_text += f"*I sense it speaks of {', '.join(dream.foreshadowing_elements[:2])}...*"
        
        return sharing_text
    
    def _load_dream_memory(self):
        """Load dream history from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load personal symbols
                if 'personal_symbols' in data:
                    for symbol_name, symbol_data in data['personal_symbols'].items():
                        self.personal_symbols[symbol_name] = DreamSymbol(
                            symbol=symbol_data['symbol'],
                            meaning=symbol_data['meaning'],
                            emotional_resonance=symbol_data['emotional_resonance'],
                            frequency=symbol_data['frequency'],
                            contexts=symbol_data['contexts']
                        )
                
                # Load recent dreams (last 20 for memory efficiency)
                if 'dreams' in data:
                    dreams_data = data['dreams'][-20:]
                    self.dream_history = [
                        DreamSequence(
                            dream_id=dream['dream_id'],
                            timestamp=datetime.fromisoformat(dream['timestamp']),
                            dream_type=DreamType(dream['dream_type']),
                            mood=DreamMood(dream['mood']),
                            narrative=dream['narrative'],
                            symbols=dream['symbols'],
                            emotional_themes=dream['emotional_themes'],
                            foreshadowing_elements=dream['foreshadowing_elements'],
                            persona_specific_elements=dream['persona_specific_elements'],
                            trigger_context=dream['trigger_context'],
                            user_relevance=dream['user_relevance']
                        )
                        for dream in dreams_data
                    ]
                    
            except Exception as e:
                print(f"Error loading dream memory for {self.persona_name}: {e}")
    
    def _save_dream_memory(self):
        """Save dream history to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'personal_symbols': {
                    name: {
                        'symbol': symbol.symbol,
                        'meaning': symbol.meaning,
                        'emotional_resonance': symbol.emotional_resonance,
                        'frequency': symbol.frequency,
                        'contexts': symbol.contexts
                    }
                    for name, symbol in self.personal_symbols.items()
                },
                'dreams': [
                    {
                        'dream_id': dream.dream_id,
                        'timestamp': dream.timestamp.isoformat(),
                        'dream_type': dream.dream_type.value,
                        'mood': dream.mood.value,
                        'narrative': dream.narrative,
                        'symbols': dream.symbols,
                        'emotional_themes': dream.emotional_themes,
                        'foreshadowing_elements': dream.foreshadowing_elements,
                        'persona_specific_elements': dream.persona_specific_elements,
                        'trigger_context': dream.trigger_context,
                        'user_relevance': dream.user_relevance
                    }
                    for dream in self.dream_history[-20:]  # Save only recent dreams
                ]
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving dream memory for {self.persona_name}: {e}")

# Factory function
def create_dream_engine(persona_name: str) -> DreamEngine:
    """Create a dream engine for a specific persona"""
    return DreamEngine(persona_name)

# Integration helper
def generate_persona_dream(persona_name: str, trigger_context: str, 
                          emotional_state: Dict[str, float], recent_memories: List[str], 
                          user_preferences: Dict[str, Any]) -> DreamSequence:
    """
    Generate a dream for a persona during rest states
    
    This function can be called during quiet periods, meditation, or rest
    to create symbolic content that adds mystique and depth.
    """
    dream_engine = create_dream_engine(persona_name)
    return dream_engine.generate_dream(trigger_context, emotional_state, recent_memories, user_preferences)
