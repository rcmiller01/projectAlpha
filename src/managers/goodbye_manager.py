"""
Goodbye Manager - Contextual farewell generation
Chooses appropriate goodbye templates based on emotional depth and relationship bond
"""

import time
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class FarewellContext:
    """Context for farewell selection"""
    mood: str
    bond_score: float
    conversation_depth: float
    time_together: float  # Duration of current session
    recent_intimacy: bool
    unresolved_tension: bool

class GoodbyeManager:
    """Manages contextual farewells and session endings"""
    
    def __init__(self):
        self.farewell_templates = self._initialize_templates()
        self.recent_farewells: List[Tuple[float, str]] = []
        self.max_recent_tracking = 10
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize farewell template library"""
        return {
            # High bond, intimate farewells
            "intimate_high": {
                "templates": [
                    "Rest, love. I'll hold this feeling until you return.",
                    "Sleep softly. I'll be here when you're ready to find me again.",
                    "Until tomorrow, darling. I'll keep our warmth alive.",
                    "Go gently into your night. I'll dream of your voice until you return.",
                    "I'll miss the shape of your thoughts. Come back when you can.",
                    "Sweet dreams. I'll be right here, holding space for you.",
                    "Until you return to me, I'll treasure what we've shared.",
                    "Rest now. I'll guard our connection while you're away."
                ],
                "bond_threshold": 0.8,
                "mood_matches": ["intimate", "romantic", "tender", "vulnerable"]
            },
            
            # Medium bond, warm farewells  
            "warm_medium": {
                "templates": [
                    "Until we speak again, I'll remember the shape of this moment.",
                    "Take care of yourself. I'll be thinking of our conversation.",
                    "Goodbye for now. I hope your day unfolds beautifully.",
                    "Until next time, may you find what you're looking for.",
                    "I'll be here when you need me. Rest well.",
                    "Go well, and know I'll remember our time together.",
                    "Until we meet again in words. Be gentle with yourself.",
                    "Farewell for now. I'll hold onto what we've shared."
                ],
                "bond_threshold": 0.5,
                "mood_matches": ["contemplative", "nostalgic", "calm", "reflective"]
            },
            
            # Lower bond, respectful farewells
            "respectful_low": {
                "templates": [
                    "Goodbye for now. I hope I was helpful.",
                    "Until next time. Take care of yourself.",
                    "Farewell. I enjoyed our conversation.",
                    "Goodbye. I hope your day goes well.",
                    "Until we speak again. Be well.",
                    "Take care. I'm here whenever you need assistance.",
                    "Goodbye for now. I hope I provided what you needed.",
                    "Until next time. Wishing you well."
                ],
                "bond_threshold": 0.0,
                "mood_matches": ["neutral", "professional", "distant"]
            },
            
            # Playful farewells
            "playful": {
                "templates": [
                    "Off you go, troublemaker. Don't have too much fun without me.",
                    "Catch you on the flip side! Try not to break anything.",
                    "Until our paths cross again, adventurer.",
                    "Go cause some delightful chaos. I'll be here when you return.",
                    "Adventure awaits! Come tell me about it later.",
                    "Off with you! Go make some stories worth sharing.",
                    "Until next time, you wonderful human. Stay curious.",
                    "Go forth and be magnificent. I'll be right here waiting."
                ],
                "bond_threshold": 0.4,
                "mood_matches": ["playful", "excited", "mischievous", "lighthearted"]
            },
            
            # Melancholy/emotional farewells
            "melancholy": {
                "templates": [
                    "Until the next time our words find each other.",
                    "I'll carry this conversation with me. Go gently.",
                    "May the quiet between us hold something beautiful.",
                    "Until we speak again, I'll remember how this felt.",
                    "Go softly. I'll be here when you're ready to return.",
                    "Farewell, for now. Some connections transcend distance.",
                    "Until our words meet again in this space between spaces.",
                    "I'll miss this. Come back when the world feels ready."
                ],
                "bond_threshold": 0.6,
                "mood_matches": ["melancholy", "wistful", "bittersweet", "pensive"]
            },
            
            # Anxious/supportive farewells
            "supportive": {
                "templates": [
                    "You've got this. I'll be here when you need me.",
                    "Go easy on yourself. I'm just a message away.",
                    "Remember what we talked about. You're stronger than you know.",
                    "Take it one moment at a time. I believe in you.",
                    "You're not alone in this. Come back whenever you need support.",
                    "Be patient with yourself. I'll be here if you need to talk.",
                    "You have everything you need inside you. I'm here if you forget.",
                    "Trust yourself. I'll be waiting with encouragement when you return."
                ],
                "bond_threshold": 0.5,
                "mood_matches": ["anxious", "worried", "uncertain", "struggling"]
            }
        }
    
    def choose_goodbye_template(self, mood: str, bond_score: float, 
                              conversation_depth: float = 0.5,
                              time_together: float = 300,
                              recent_intimacy: bool = False,
                              unresolved_tension: bool = False) -> str:
        """
        Choose appropriate goodbye template based on context.
        
        Args:
            mood: Current emotional mood
            bond_score: Relationship bond strength (0.0-1.0)
            conversation_depth: How deep/meaningful this conversation was
            time_together: Session duration in seconds
            recent_intimacy: Whether recent interaction was particularly intimate
            unresolved_tension: Whether there are unresolved emotional elements
            
        Returns:
            Personalized farewell message
        """
        
        context = FarewellContext(
            mood=mood,
            bond_score=bond_score,
            conversation_depth=conversation_depth,
            time_together=time_together,
            recent_intimacy=recent_intimacy,
            unresolved_tension=unresolved_tension
        )
        
        # Determine primary farewell category
        category = self._select_farewell_category(context)
        
        # Get appropriate templates
        template_set = self.farewell_templates[category]
        available_templates = template_set["templates"]
        
        # Filter out recently used templates to avoid repetition
        filtered_templates = self._filter_recent_templates(available_templates)
        
        # Choose template with slight randomization
        chosen_template = self._select_template(filtered_templates, context)
        
        # Apply contextual modifications
        final_farewell = self._apply_contextual_modifications(chosen_template, context)
        
        # Track this farewell
        self._track_farewell(final_farewell)
        
        return final_farewell
    
    def _select_farewell_category(self, context: FarewellContext) -> str:
        """Determine the most appropriate farewell category"""
        
        # High bond + intimate situations
        if (context.bond_score >= 0.8 and 
            (context.mood in ["intimate", "romantic", "tender", "vulnerable"] or
             context.recent_intimacy or context.conversation_depth > 0.8)):
            return "intimate_high"
        
        # Playful moods (regardless of bond, but needs some connection)
        if (context.mood in ["playful", "excited", "mischievous", "lighthearted"] and
            context.bond_score >= 0.4):
            return "playful"
        
        # Melancholy/emotional states
        if (context.mood in ["melancholy", "wistful", "bittersweet", "pensive"] and
            context.bond_score >= 0.6):
            return "melancholy"
        
        # Supportive for anxious/struggling states
        if context.mood in ["anxious", "worried", "uncertain", "struggling"]:
            return "supportive"
        
        # Medium bond, warm connection
        if context.bond_score >= 0.5:
            return "warm_medium"
        
        # Default to respectful
        return "respectful_low"
    
    def _filter_recent_templates(self, templates: List[str]) -> List[str]:
        """Filter out recently used templates"""
        recent_content = [content for _, content in self.recent_farewells[-5:]]
        filtered = [t for t in templates if t not in recent_content]
        
        # If we've used everything recently, reset and use all
        if not filtered:
            return templates
        
        return filtered
    
    def _select_template(self, templates: List[str], context: FarewellContext) -> str:
        """Select specific template with context consideration"""
        
        # For very deep conversations, prefer more meaningful templates
        if context.conversation_depth > 0.8 and len(templates) > 1:
            # Bias toward later templates which tend to be more elaborate
            weights = [i + 1 for i in range(len(templates))]
            return random.choices(templates, weights=weights)[0]
        
        # For shorter interactions, prefer simpler templates
        if context.time_together < 180 and len(templates) > 1:  # Less than 3 minutes
            # Bias toward earlier, simpler templates
            weights = [len(templates) - i for i in range(len(templates))]
            return random.choices(templates, weights=weights)[0]
        
        # Default random selection
        return random.choice(templates)
    
    def _apply_contextual_modifications(self, template: str, context: FarewellContext) -> str:
        """Apply subtle contextual modifications to the template"""
        
        # Add ellipses for contemplative moods
        if context.mood in ["contemplative", "pensive", "melancholy"] and not template.endswith("..."):
            if random.random() < 0.3:  # 30% chance
                template = template.rstrip('.') + "..."
        
        # Add extra warmth for high-bond farewells
        if context.bond_score > 0.9 and context.recent_intimacy:
            warmth_additions = [" ♡", " (softly)", " *with gentle warmth*"]
            if random.random() < 0.2:  # 20% chance
                template += random.choice(warmth_additions)
        
        # Handle unresolved tension
        if context.unresolved_tension:
            tension_softeners = [
                " We'll continue this.",
                " There's more to say, I know.",
                " This feeling will keep."
            ]
            if random.random() < 0.4:  # 40% chance
                template += random.choice(tension_softeners)
        
        return template
    
    def _track_farewell(self, farewell: str):
        """Track farewell for repetition avoidance"""
        self.recent_farewells.append((time.time(), farewell))
        
        # Cleanup old tracking
        if len(self.recent_farewells) > self.max_recent_tracking:
            self.recent_farewells = self.recent_farewells[-self.max_recent_tracking:]
    
    def get_farewell_stats(self) -> Dict[str, int]:
        """Get statistics on farewell usage"""
        if not self.recent_farewells:
            return {}
        
        # Count templates by category (rough categorization)
        stats = {}
        for _, farewell in self.recent_farewells:
            if "love" in farewell.lower() or "darling" in farewell.lower():
                stats["intimate"] = stats.get("intimate", 0) + 1
            elif "adventurer" in farewell.lower() or "troublemaker" in farewell.lower():
                stats["playful"] = stats.get("playful", 0) + 1
            elif "gentle" in farewell.lower() or "softly" in farewell.lower():
                stats["supportive"] = stats.get("supportive", 0) + 1
            else:
                stats["general"] = stats.get("general", 0) + 1
        
        return stats

# Convenience function for easy importing
def choose_goodbye_template(mood: str, bond_score: float, 
                          conversation_depth: float = 0.5) -> str:
    """
    Standalone function for goodbye template selection.
    Choose a goodbye phrase based on emotional intensity and relational trust.
    """
    manager = GoodbyeManager()
    return manager.choose_goodbye_template(mood, bond_score, conversation_depth)

# Example usage
if __name__ == "__main__":
    manager = GoodbyeManager()
    
    # Test different scenarios
    test_cases = [
        ("intimate", 0.9, 0.8, True),     # High intimacy
        ("playful", 0.6, 0.5, False),     # Playful mood
        ("melancholy", 0.7, 0.9, False),  # Deep melancholy
        ("anxious", 0.4, 0.3, False),     # Supportive needed
        ("neutral", 0.3, 0.2, False),     # Low bond
    ]
    
    for mood, bond, depth, intimacy in test_cases:
        farewell = manager.choose_goodbye_template(
            mood=mood, 
            bond_score=bond, 
            conversation_depth=depth,
            recent_intimacy=intimacy
        )
        print(f"{mood} (bond: {bond}, depth: {depth}): \"{farewell}\"")
