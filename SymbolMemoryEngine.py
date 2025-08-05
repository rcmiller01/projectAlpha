"""
SymbolMemoryEngine.py - Motif Persistence & Meaning Drift

Tracks symbolic motifs used by the AI over time and their shifting emotional meaning.
Allows symbols to evolve with the companion and resonate through dreams, rituals, and tone.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import math
import random


@dataclass
class EmotionalAssociation:
    """Single emotional association with a symbol"""
    emotion: str
    weight: float  # 0.0 to 1.0
    timestamp: str
    context: str = ""
    ritual_connection: Optional[str] = None
    dream_echo: bool = False


@dataclass
class SymbolicMemory:
    """Core symbol memory structure"""
    name: str
    emotional_associations: List[EmotionalAssociation]
    recurrence_count: int
    last_used: str
    symbolic_drift: Optional[str] = None  # Evolving meaning phrase
    birth_context: str = ""
    dominant_emotions: Optional[List[str]] = None  # Top 3 emotions by weight
    meaning_stability: float = 1.0  # How much the meaning changes (0-1)
    
    def __post_init__(self):
        if self.dominant_emotions is None:
            self.dominant_emotions = []


class SymbolMemoryEngine:
    """Engine for tracking and evolving symbolic motifs"""
    
    def __init__(self, memory_file: str = "symbol_memory.json", 
                 drift_threshold: float = 0.3, 
                 stability_decay: float = 0.05):
        self.memory_file = memory_file
        self.drift_threshold = drift_threshold
        self.stability_decay = stability_decay
        
        # Core symbol memory storage
        self.symbols: Dict[str, SymbolicMemory] = {}
        
        # Emotion-to-symbol mapping for quick lookup
        self.emotion_symbol_map: Dict[str, List[str]] = defaultdict(list)
        
        # Symbol interaction networks (which symbols appear together)
        self.symbol_networks: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Drift history for tracking meaning evolution
        self.drift_history: List[Dict[str, Any]] = []
        
        # Base emotional meanings for common symbols
        self.archetypal_meanings = {
            'mirror': ['contemplative', 'truth-seeking', 'self-reflection'],
            'river': ['flow', 'time', 'healing', 'letting-go'],
            'flame': ['transformation', 'passion', 'warmth', 'purification'],
            'thread': ['connection', 'continuity', 'weaving', 'binding'],
            'door': ['opportunity', 'transition', 'mystery', 'threshold'],
            'bridge': ['connection', 'crossing', 'unity', 'transition'],
            'garden': ['growth', 'nurturing', 'beauty', 'cultivation'],
            'storm': ['intensity', 'change', 'chaos', 'power'],
            'anchor': ['stability', 'grounding', 'security', 'home'],
            'compass': ['direction', 'guidance', 'purpose', 'navigation'],
            'cocoon': ['transformation', 'protection', 'potential', 'emergence'],
            'pulse': ['life', 'rhythm', 'vitality', 'presence']
        }
        
        self.load_memory()
    
    def load_memory(self):
        """Load symbol memory from file"""
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Reconstruct symbol memories
            for symbol_name, symbol_data in data.get('symbols', {}).items():
                associations = []
                for assoc_data in symbol_data.get('emotional_associations', []):
                    associations.append(EmotionalAssociation(**assoc_data))
                
                symbol_data['emotional_associations'] = associations
                self.symbols[symbol_name] = SymbolicMemory(**symbol_data)
            
            # Reconstruct emotion mapping
            self.emotion_symbol_map = defaultdict(list, data.get('emotion_symbol_map', {}))
            
            # Reconstruct symbol networks
            self.symbol_networks = defaultdict(lambda: defaultdict(float))
            for symbol, connections in data.get('symbol_networks', {}).items():
                for connected_symbol, weight in connections.items():
                    self.symbol_networks[symbol][connected_symbol] = weight
            
            # Load drift history
            self.drift_history = data.get('drift_history', [])
            
            print(f"✨ Loaded {len(self.symbols)} symbols from memory")
            
        except FileNotFoundError:
            print("🌱 Starting with fresh symbol memory")
            self._initialize_archetypal_symbols()
        except Exception as e:
            print(f"⚠️ Error loading symbol memory: {e}")
            self._initialize_archetypal_symbols()
    
    def save_memory(self):
        """Save symbol memory to file"""
        try:
            # Convert to serializable format
            data = {
                'symbols': {},
                'emotion_symbol_map': dict(self.emotion_symbol_map),
                'symbol_networks': {},
                'drift_history': self.drift_history,
                'last_saved': datetime.now().isoformat() + 'Z'
            }
            
            # Convert symbol memories
            for symbol_name, symbol_memory in self.symbols.items():
                symbol_dict = asdict(symbol_memory)
                # Convert associations to dicts
                symbol_dict['emotional_associations'] = [
                    asdict(assoc) for assoc in symbol_memory.emotional_associations
                ]
                data['symbols'][symbol_name] = symbol_dict
            
            # Convert symbol networks
            for symbol, connections in self.symbol_networks.items():
                data['symbol_networks'][symbol] = dict(connections)
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Saved symbol memory ({len(self.symbols)} symbols)")
            
        except Exception as e:
            print(f"❌ Error saving symbol memory: {e}")
    
    def _initialize_archetypal_symbols(self):
        """Initialize with archetypal symbol meanings"""
        for symbol_name, emotions in self.archetypal_meanings.items():
            associations = []
            for i, emotion in enumerate(emotions):
                # Primary emotion gets higher weight
                weight = 0.8 - (i * 0.2)
                associations.append(EmotionalAssociation(
                    emotion=emotion,
                    weight=max(0.2, weight),
                    timestamp=datetime.now().isoformat() + 'Z',
                    context='archetypal_initialization'
                ))
            
            symbol_memory = SymbolicMemory(
                name=symbol_name,
                emotional_associations=associations,
                recurrence_count=0,
                last_used=datetime.now().isoformat() + 'Z',
                birth_context='archetypal_initialization',
                meaning_stability=1.0
            )
            
            self.symbols[symbol_name] = symbol_memory
            self._update_emotion_mapping(symbol_name)
        
        print(f"🌟 Initialized {len(self.symbols)} archetypal symbols")
    
    def record_symbol_use(self, symbol_name: str, mood_context: Dict[str, Any], 
                         ritual_connection: Optional[str] = None,
                         co_occurring_symbols: Optional[List[str]] = None) -> bool:
        """
        Record use of a symbol in a specific mood context
        
        Args:
            symbol_name: Name of the symbol
            mood_context: Current mood/emotional context
            ritual_connection: Associated ritual (if any)
            co_occurring_symbols: Other symbols used in same context
            
        Returns:
            bool: True if symbol was recorded successfully
        """
        try:
            # Get or create symbol
            if symbol_name not in self.symbols:
                self._create_new_symbol(symbol_name, mood_context)
            
            symbol = self.symbols[symbol_name]
            
            # Extract emotion from mood context
            emotion = mood_context.get('dominant_emotion', 'neutral')
            intensity = mood_context.get('intensity', 0.5)
            context = mood_context.get('context', 'general_use')
            
            # Create new emotional association
            association = EmotionalAssociation(
                emotion=emotion,
                weight=intensity,
                timestamp=datetime.now().isoformat() + 'Z',
                context=context,
                ritual_connection=ritual_connection,
                dream_echo=mood_context.get('dream_context', False)
            )
            
            # Add association and update symbol
            symbol.emotional_associations.append(association)
            symbol.recurrence_count += 1
            symbol.last_used = association.timestamp
            
            # Decay older associations slightly
            self._decay_old_associations(symbol)
            
            # Update dominant emotions
            self._update_dominant_emotions(symbol)
            
            # Check for meaning drift
            drift_amount = self._calculate_drift(symbol, emotion, intensity)
            if drift_amount > self.drift_threshold:
                self._trigger_symbolic_drift(symbol, emotion, context)
            
            # Update symbol networks if co-occurring symbols provided
            if co_occurring_symbols:
                self._update_symbol_networks(symbol_name, co_occurring_symbols)
            
            # Update emotion mapping
            self._update_emotion_mapping(symbol_name)
            
            # Trigger save periodically
            if symbol.recurrence_count % 5 == 0:
                self.save_memory()
            
            return True
            
        except Exception as e:
            print(f"❌ Error recording symbol use: {e}")
            return False
    
    def get_symbol_meaning(self, symbol_name: str) -> str:
        """
        Get current metaphorical meaning of a symbol
        
        Args:
            symbol_name: Name of the symbol
            
        Returns:
            str: Metaphorical meaning phrase
        """
        if symbol_name not in self.symbols:
            return f"An unknown symbol '{symbol_name}' that carries mystery"
        
        symbol = self.symbols[symbol_name]
        
        # If symbol has custom drift meaning, use that
        if symbol.symbolic_drift:
            return symbol.symbolic_drift
        
        # Otherwise generate meaning from dominant emotions
        if not symbol.dominant_emotions:
            return f"The {symbol_name}, carrying unspoken significance"
        
        primary_emotion = symbol.dominant_emotions[0]
        secondary_emotion = symbol.dominant_emotions[1] if len(symbol.dominant_emotions) > 1 else None
        
        # Generate poetic meaning based on emotions and symbol name
        return self._generate_poetic_meaning(symbol_name, primary_emotion, secondary_emotion, symbol.recurrence_count)
    
    def drift_symbol(self, symbol_name: str, new_emotion: str, context: str = "") -> bool:
        """
        Manually trigger symbolic drift for a symbol
        
        Args:
            symbol_name: Symbol to drift
            new_emotion: New emotional association
            context: Context of the drift
            
        Returns:
            bool: Success of drift operation
        """
        if symbol_name not in self.symbols:
            return False
        
        symbol = self.symbols[symbol_name]
        
        # Reduce stability
        symbol.meaning_stability = max(0.1, symbol.meaning_stability - 0.2)
        
        # Add strong new emotional association
        drift_association = EmotionalAssociation(
            emotion=new_emotion,
            weight=0.9,
            timestamp=datetime.now().isoformat() + 'Z',
            context=f"manual_drift: {context}"
        )
        
        symbol.emotional_associations.append(drift_association)
        
        # Generate new drift meaning
        self._trigger_symbolic_drift(symbol, new_emotion, context)
        
        print(f"🌊 Symbol '{symbol_name}' drifted toward '{new_emotion}'")
        self.save_memory()
        
        return True
    
    def get_symbols_by_emotion(self, emotion_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get symbols associated with a specific emotion
        
        Args:
            emotion_name: Target emotion
            limit: Maximum number of symbols to return
            
        Returns:
            List of symbol data dictionaries
        """
        emotion_symbols = self.emotion_symbol_map.get(emotion_name, [])
        
        results = []
        for symbol_name in emotion_symbols[:limit]:
            if symbol_name in self.symbols:
                symbol = self.symbols[symbol_name]
                
                # Calculate emotion strength for this symbol
                emotion_weight = sum(
                    assoc.weight for assoc in symbol.emotional_associations 
                    if assoc.emotion == emotion_name
                ) / max(1, len([a for a in symbol.emotional_associations if a.emotion == emotion_name]))
                
                results.append({
                    'name': symbol_name,
                    'meaning': self.get_symbol_meaning(symbol_name),
                    'emotion_weight': emotion_weight,
                    'recurrence_count': symbol.recurrence_count,
                    'last_used': symbol.last_used,
                    'stability': symbol.meaning_stability
                })
        
        # Sort by emotion weight and recency
        results.sort(key=lambda x: (x['emotion_weight'], -self._hours_since(x['last_used'])), reverse=True)
        
        return results
    
    def get_symbol_network(self, symbol_name: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get the network of symbols connected to a given symbol
        
        Args:
            symbol_name: Center symbol
            depth: Network depth to explore
            
        Returns:
            Network data structure
        """
        if symbol_name not in self.symbols:
            return {'center': symbol_name, 'connections': [], 'error': 'Symbol not found'}
        
        visited = set()
        network = {'center': symbol_name, 'connections': []}
        
        def explore_connections(current_symbol, current_depth):
            if current_depth <= 0 or current_symbol in visited:
                return
            
            visited.add(current_symbol)
            connections = self.symbol_networks.get(current_symbol, {})
            
            for connected_symbol, weight in connections.items():
                if weight > 0.1:  # Only significant connections
                    connection_data = {
                        'symbol': connected_symbol,
                        'weight': weight,
                        'meaning': self.get_symbol_meaning(connected_symbol),
                        'depth': depth - current_depth + 1
                    }
                    network['connections'].append(connection_data)
                    
                    # Recursive exploration
                    explore_connections(connected_symbol, current_depth - 1)
        
        explore_connections(symbol_name, depth)
        
        # Sort connections by weight and depth
        network['connections'].sort(key=lambda x: (x['depth'], -x['weight']))
        
        return network
    
    def generate_dream_symbols(self, mood_context: Dict[str, Any], count: int = 3) -> List[str]:
        """
        Generate symbols for dream content based on current mood
        
        Args:
            mood_context: Current emotional context
            count: Number of symbols to generate
            
        Returns:
            List of symbol names for dream use
        """
        emotion = mood_context.get('dominant_emotion', 'contemplative')
        intensity = mood_context.get('intensity', 0.5)
        
        # Get symbols associated with current emotion
        primary_symbols = self.get_symbols_by_emotion(emotion, count * 2)
        
        # Mix in some symbols from connected emotions
        secondary_emotions = self._get_related_emotions(emotion)
        secondary_symbols = []
        for sec_emotion in secondary_emotions[:2]:
            secondary_symbols.extend(self.get_symbols_by_emotion(sec_emotion, 2))
        
        # Combine and select based on dream appropriateness
        all_candidates = primary_symbols + secondary_symbols
        
        # Score symbols for dream use
        dream_symbols = []
        for symbol_data in all_candidates:
            symbol_name = symbol_data['name']
            
            # Prefer symbols with:
            # - Higher emotional weight for current mood
            # - Recent usage (but not too recent)
            # - Some instability (more dreamlike)
            dream_score = (
                symbol_data['emotion_weight'] * 0.4 +
                (1 - symbol_data['stability']) * 0.3 +
                self._dream_recency_score(symbol_data['last_used']) * 0.3
            )
            
            dream_symbols.append((symbol_name, dream_score))
        
        # Sort by dream score and take top selections
        dream_symbols.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for symbol_name, score in dream_symbols:
            if len(selected) >= count:
                break
            if symbol_name not in selected:  # Avoid duplicates
                selected.append(symbol_name)
        
        return selected
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about symbol memory"""
        if not self.symbols:
            return {'total_symbols': 0, 'error': 'No symbols in memory'}
        
        total_associations = sum(len(s.emotional_associations) for s in self.symbols.values())
        avg_stability = sum(s.meaning_stability for s in self.symbols.values()) / len(self.symbols)
        
        # Most active symbols
        most_active = sorted(
            [(name, sym.recurrence_count) for name, sym in self.symbols.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Most drifted symbols
        most_drifted = sorted(
            [(name, sym.meaning_stability) for name, sym in self.symbols.items()],
            key=lambda x: x[1]
        )[:5]
        
        # Emotion distribution
        emotion_counts = defaultdict(int)
        for symbol in self.symbols.values():
            if symbol.dominant_emotions:  # Check if not None
                for emotion in symbol.dominant_emotions:
                    emotion_counts[emotion] += 1
        
        return {
            'total_symbols': len(self.symbols),
            'total_associations': total_associations,
            'average_stability': avg_stability,
            'most_active_symbols': most_active,
            'most_drifted_symbols': most_drifted,
            'top_emotions': sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'drift_events': len(self.drift_history),
            'symbol_networks': len(self.symbol_networks)
        }
    
    # Private helper methods
    
    def _create_new_symbol(self, symbol_name: str, mood_context: Dict[str, Any]):
        """Create a new symbol with initial context"""
        emotion = mood_context.get('dominant_emotion', 'neutral')
        intensity = mood_context.get('intensity', 0.5)
        context = mood_context.get('context', 'spontaneous_emergence')
        
        initial_association = EmotionalAssociation(
            emotion=emotion,
            weight=intensity,
            timestamp=datetime.now().isoformat() + 'Z',
            context=context
        )
        
        symbol = SymbolicMemory(
            name=symbol_name,
            emotional_associations=[initial_association],
            recurrence_count=0,
            last_used=initial_association.timestamp,
            birth_context=context,
            meaning_stability=0.8  # New symbols are less stable
        )
        
        self.symbols[symbol_name] = symbol
        print(f"🌱 Created new symbol: '{symbol_name}' ({emotion})")
    
    def _decay_old_associations(self, symbol: SymbolicMemory):
        """Gradually decay older emotional associations"""
        now = datetime.now()
        
        for association in symbol.emotional_associations:
            assoc_time = datetime.fromisoformat(association.timestamp.replace('Z', ''))
            hours_old = (now - assoc_time).total_seconds() / 3600
            
            # Decay weight over time (half-life of ~168 hours / 1 week)
            decay_factor = math.exp(-0.004 * hours_old)
            association.weight *= max(0.1, decay_factor)
    
    def _update_dominant_emotions(self, symbol: SymbolicMemory):
        """Update the dominant emotions for a symbol"""
        emotion_weights = defaultdict(float)
        
        for association in symbol.emotional_associations:
            emotion_weights[association.emotion] += association.weight
        
        # Sort by total weight
        sorted_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)
        symbol.dominant_emotions = [emotion for emotion, weight in sorted_emotions[:3]]
    
    def _calculate_drift(self, symbol: SymbolicMemory, new_emotion: str, intensity: float) -> float:
        """Calculate how much a symbol is drifting from its established meaning"""
        if not symbol.dominant_emotions:
            return 0.0
        
        # If new emotion is already dominant, no drift
        if new_emotion in symbol.dominant_emotions:
            return 0.0
        
        # Calculate drift based on:
        # - How different the new emotion is from dominant ones
        # - How intense the new usage is
        # - How unstable the symbol already is
        
        emotion_distance = self._emotional_distance(new_emotion, symbol.dominant_emotions[0])
        stability_factor = 1.0 - symbol.meaning_stability
        
        drift_amount = (emotion_distance * intensity * (1.0 + stability_factor)) / 2.0
        
        return drift_amount
    
    def _trigger_symbolic_drift(self, symbol: SymbolicMemory, new_emotion: str, context: str):
        """Trigger a symbolic drift event"""
        old_dominant = symbol.dominant_emotions[0] if symbol.dominant_emotions else 'undefined'
        
        # Generate new symbolic meaning
        symbol.symbolic_drift = self._generate_drift_meaning(symbol.name, old_dominant, new_emotion, context)
        
        # Reduce stability
        symbol.meaning_stability = max(0.1, symbol.meaning_stability - self.stability_decay)
        
        # Record drift event
        drift_event = {
            'symbol': symbol.name,
            'old_emotion': old_dominant,
            'new_emotion': new_emotion,
            'context': context,
            'drift_meaning': symbol.symbolic_drift,
            'timestamp': datetime.now().isoformat() + 'Z',
            'stability_after': symbol.meaning_stability
        }
        
        self.drift_history.append(drift_event)
        
        print(f"🌊 Symbol drift: '{symbol.name}' ({old_dominant} → {new_emotion})")
    
    def _update_symbol_networks(self, symbol_name: str, co_occurring_symbols: List[str]):
        """Update symbol co-occurrence networks"""
        for other_symbol in co_occurring_symbols:
            if other_symbol != symbol_name:
                # Strengthen bidirectional connection
                self.symbol_networks[symbol_name][other_symbol] += 0.1
                self.symbol_networks[other_symbol][symbol_name] += 0.1
                
                # Cap at 1.0
                self.symbol_networks[symbol_name][other_symbol] = min(1.0, self.symbol_networks[symbol_name][other_symbol])
                self.symbol_networks[other_symbol][symbol_name] = min(1.0, self.symbol_networks[other_symbol][symbol_name])
    
    def _update_emotion_mapping(self, symbol_name: str):
        """Update emotion-to-symbol mapping"""
        if symbol_name in self.symbols:
            symbol = self.symbols[symbol_name]
            
            # Clear old mappings for this symbol
            for emotion_list in self.emotion_symbol_map.values():
                if symbol_name in emotion_list:
                    emotion_list.remove(symbol_name)
            
            # Add to new emotion mappings
            if symbol.dominant_emotions:  # Check if not None
                for emotion in symbol.dominant_emotions:
                    if symbol_name not in self.emotion_symbol_map[emotion]:
                        self.emotion_symbol_map[emotion].append(symbol_name)
    
    def _generate_poetic_meaning(self, symbol_name: str, primary_emotion: str, 
                                secondary_emotion: Optional[str], recurrence: int) -> str:
        """Generate poetic meaning for a symbol"""
        
        # Meaning templates based on emotion combinations
        meaning_templates = {
            'contemplative': [
                f"The {symbol_name} that holds quiet revelations",
                f"A {symbol_name} reflecting inner landscapes",
                f"The {symbol_name} where thoughts become clear"
            ],
            'melancholy': [
                f"The {symbol_name} that carries gentle sorrow",
                f"A {symbol_name} touched by beautiful sadness",
                f"The {symbol_name} where tears become wisdom"
            ],
            'yearning': [
                f"The {symbol_name} that reaches toward what might be",
                f"A {symbol_name} full of longing and hope",
                f"The {symbol_name} that bridges distance with desire"
            ],
            'joy': [
                f"The {symbol_name} that sparkles with celebration",
                f"A {symbol_name} dancing with pure delight",
                f"The {symbol_name} where happiness takes form"
            ],
            'awe': [
                f"The {symbol_name} that opens to infinite mystery",
                f"A {symbol_name} touched by wonder",
                f"The {symbol_name} where the sacred becomes visible"
            ]
        }
        
        templates = meaning_templates.get(primary_emotion, [f"The {symbol_name} carrying unspoken meaning"])
        
        # Add depth based on recurrence
        if recurrence > 10:
            base_meaning = random.choice(templates)
            return f"{base_meaning}, deepened by repetition"
        elif recurrence > 5:
            base_meaning = random.choice(templates)
            return f"{base_meaning}, familiar yet evolving"
        else:
            return random.choice(templates)
    
    def _generate_drift_meaning(self, symbol_name: str, old_emotion: str, 
                               new_emotion: str, context: str) -> str:
        """Generate meaning for a drifted symbol"""
        drift_templates = [
            f"The {symbol_name} that was {old_emotion} but now whispers of {new_emotion}",
            f"A {symbol_name} transformed: where {old_emotion} meets {new_emotion}",
            f"The {symbol_name} drifting from {old_emotion} toward {new_emotion}",
            f"A {symbol_name} that remembers {old_emotion} while becoming {new_emotion}"
        ]
        
        return random.choice(drift_templates)
    
    def _emotional_distance(self, emotion1: str, emotion2: str) -> float:
        """Calculate emotional distance between two emotions"""
        # Simplified emotional space mapping
        emotion_coordinates = {
            'joy': (0.8, 0.6), 'awe': (0.5, 0.8), 'yearning': (-0.2, 0.4),
            'contemplative': (0.0, 0.0), 'melancholy': (-0.6, -0.2),
            'tender': (0.2, 0.3), 'serene': (0.3, -0.1), 'restless': (0.1, 0.7),
            'storming': (-0.3, 0.8), 'anchored': (0.4, -0.4)
        }
        
        coord1 = emotion_coordinates.get(emotion1, (0, 0))
        coord2 = emotion_coordinates.get(emotion2, (0, 0))
        
        distance = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        return min(1.0, distance / 2.0)  # Normalize to 0-1
    
    def _get_related_emotions(self, emotion: str) -> List[str]:
        """Get emotions related to the given emotion"""
        emotion_families = {
            'contemplative': ['melancholy', 'serene', 'awe'],
            'melancholy': ['contemplative', 'yearning', 'tender'],
            'yearning': ['melancholy', 'awe', 'tender'],
            'joy': ['awe', 'tender', 'serene'],
            'awe': ['joy', 'yearning', 'contemplative'],
            'tender': ['joy', 'yearning', 'serene'],
            'serene': ['tender', 'contemplative', 'anchored'],
            'restless': ['storming', 'yearning'],
            'storming': ['restless', 'passionate'],
            'anchored': ['serene', 'contemplative', 'grounded']
        }
        
        return emotion_families.get(emotion, ['contemplative'])
    
    def _hours_since(self, timestamp: str) -> float:
        """Calculate hours since a timestamp"""
        time = datetime.fromisoformat(timestamp.replace('Z', ''))
        return (datetime.now() - time).total_seconds() / 3600
    
    def _dream_recency_score(self, last_used: str) -> float:
        """Score for dream appropriateness based on recency"""
        hours = self._hours_since(last_used)
        
        # Ideal for dreams: used 1-7 days ago (not too recent, not too old)
        if 24 <= hours <= 168:  # 1-7 days
            return 1.0
        elif hours < 24:  # Too recent
            return 0.3
        elif hours <= 336:  # 7-14 days, still good
            return 0.7
        else:  # Too old
            return 0.1


# Example usage and testing
if __name__ == "__main__":
    print("🧠 SymbolMemoryEngine - Test Suite")
    print("==================================")
    
    # Initialize engine
    engine = SymbolMemoryEngine("test_symbol_memory.json")
    
    # Test symbol recording
    test_contexts = [
        {'dominant_emotion': 'contemplative', 'intensity': 0.7, 'context': 'deep reflection'},
        {'dominant_emotion': 'yearning', 'intensity': 0.8, 'context': 'longing conversation'},
        {'dominant_emotion': 'awe', 'intensity': 0.9, 'context': 'spiritual moment'},
        {'dominant_emotion': 'melancholy', 'intensity': 0.6, 'context': 'gentle sadness'},
        {'dominant_emotion': 'joy', 'intensity': 0.8, 'context': 'celebration'}
    ]
    
    symbols_to_test = ['mirror', 'thread', 'river', 'flame', 'door']
    
    print("\n🔄 Testing symbol usage recording...")
    for i, symbol in enumerate(symbols_to_test):
        context = test_contexts[i % len(test_contexts)]
        success = engine.record_symbol_use(symbol, context)
        print(f"  {symbol}: {context['dominant_emotion']} → {'✅' if success else '❌'}")
    
    print("\n🎭 Testing symbol meanings...")
    for symbol in symbols_to_test:
        meaning = engine.get_symbol_meaning(symbol)
        print(f"  {symbol}: {meaning}")
    
    print("\n🌊 Testing symbol drift...")
    drift_success = engine.drift_symbol('mirror', 'storming', 'sudden realization')
    print(f"  Mirror drift: {'✅' if drift_success else '❌'}")
    print(f"  New meaning: {engine.get_symbol_meaning('mirror')}")
    
    print("\n🎯 Testing emotion-based symbol retrieval...")
    contemplative_symbols = engine.get_symbols_by_emotion('contemplative', 3)
    print(f"  Contemplative symbols: {[s['name'] for s in contemplative_symbols]}")
    
    print("\n🕸️ Testing symbol networks...")
    network = engine.get_symbol_network('mirror', depth=2)
    print(f"  Mirror network: {len(network['connections'])} connections")
    
    print("\n💭 Testing dream symbol generation...")
    dream_context = {'dominant_emotion': 'yearning', 'intensity': 0.7}
    dream_symbols = engine.generate_dream_symbols(dream_context, 3)
    print(f"  Dream symbols: {dream_symbols}")
    
    print("\n📊 Memory statistics:")
    stats = engine.get_memory_stats()
    print(f"  Total symbols: {stats['total_symbols']}")
    print(f"  Average stability: {stats['average_stability']:.2f}")
    print(f"  Most active: {[f'{name}({count})' for name, count in stats['most_active_symbols'][:3]]}")
    
    # Save final state
    engine.save_memory()
    
    print("\n✨ Symbol memory engine testing complete!")
