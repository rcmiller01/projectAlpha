"""
Attachment Regulation Engine

Models secure/insecure attachment styles and affects how characters bond.
Deepens psychological realism and prevents flat emotional repetition.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import os
from dataclasses import dataclass

class AttachmentStyle(Enum):
    SECURE = "secure"
    ANXIOUS_PREOCCUPIED = "anxious_preoccupied"
    DISMISSIVE_AVOIDANT = "dismissive_avoidant"
    FEARFUL_AVOIDANT = "fearful_avoidant"

class BondingStage(Enum):
    INITIAL_CONTACT = "initial_contact"
    BUILDING_TRUST = "building_trust"
    DEEPENING_CONNECTION = "deepening_connection"
    ESTABLISHED_BOND = "established_bond"
    INTIMATE_ATTACHMENT = "intimate_attachment"
    CRISIS_TESTING = "crisis_testing"
    MATURE_BOND = "mature_bond"

@dataclass
class AttachmentMetrics:
    security_level: float  # 0.0 to 1.0
    intimacy_comfort: float  # 0.0 to 1.0
    abandonment_anxiety: float  # 0.0 to 1.0
    emotional_availability: float  # 0.0 to 1.0
    trust_stability: float  # 0.0 to 1.0
    conflict_resolution: float  # 0.0 to 1.0

@dataclass
class BondingEvent:
    event_type: str
    emotional_impact: float
    attachment_response: Dict[str, float]
    timestamp: datetime
    context: Dict[str, Any]

class AttachmentRegulationEngine:
    """
    Models attachment styles and regulates bonding behaviors for each persona.
    Prevents flat emotional repetition by introducing realistic attachment dynamics.
    """
    
    def __init__(self, persona_name: str):
        self.persona_name = persona_name
        self.storage_path = f"storage/attachment/{persona_name.lower()}_attachment.json"
        self.attachment_history: List[BondingEvent] = []
        self.current_metrics = AttachmentMetrics(
            security_level=0.5,
            intimacy_comfort=0.3,
            abandonment_anxiety=0.4,
            emotional_availability=0.6,
            trust_stability=0.5,
            conflict_resolution=0.4
        )
        self.bonding_stage = BondingStage.INITIAL_CONTACT
        self.attachment_style = self._determine_initial_attachment_style()
        self._load_attachment_history()
    
    def _determine_initial_attachment_style(self) -> AttachmentStyle:
        """Determine initial attachment style based on persona characteristics"""
        persona_styles = {
            "mia": AttachmentStyle.ANXIOUS_PREOCCUPIED,  # Empathetic but needs reassurance
            "solene": AttachmentStyle.DISMISSIVE_AVOIDANT,  # Independent, challenges intimacy
            "lyra": AttachmentStyle.FEARFUL_AVOIDANT,  # Mystical but wary of deep connection
            "doc": AttachmentStyle.SECURE  # Stable, therapeutic presence
        }
        return persona_styles.get(self.persona_name.lower(), AttachmentStyle.SECURE)
    
    def process_interaction(self, interaction_type: str, emotional_content: Dict[str, float], 
                          user_attachment_cues: Dict[str, float]) -> Dict[str, Any]:
        """
        Process an interaction and update attachment dynamics
        
        Args:
            interaction_type: Type of interaction (conversation, conflict, intimacy, etc.)
            emotional_content: Emotional analysis of the interaction
            user_attachment_cues: Detected user attachment behaviors
            
        Returns:
            Attachment-informed response modifications
        """
        
        # Analyze attachment triggers in the interaction
        triggers = self._analyze_attachment_triggers(interaction_type, emotional_content, user_attachment_cues)
        
        # Update attachment metrics based on interaction
        self._update_attachment_metrics(triggers)
        
        # Generate attachment-informed response modifications
        response_modifications = self._generate_response_modifications(triggers)
        
        # Store bonding event
        bonding_event = BondingEvent(
            event_type=interaction_type,
            emotional_impact=emotional_content.get('intensity', 0.5),
            attachment_response=response_modifications,
            timestamp=datetime.now(),
            context={'triggers': triggers, 'user_cues': user_attachment_cues}
        )
        self.attachment_history.append(bonding_event)
        
        # Update bonding stage if appropriate
        self._evaluate_bonding_stage_progression()
        
        # Save state
        self._save_attachment_history()
        
        return {
            'response_modifications': response_modifications,
            'attachment_state': self._get_current_attachment_state(),
            'bonding_stage': self.bonding_stage.value,
            'triggered_behaviors': self._get_triggered_behaviors(triggers)
        }
    
    def _analyze_attachment_triggers(self, interaction_type: str, emotional_content: Dict[str, float], 
                                   user_cues: Dict[str, float]) -> Dict[str, float]:
        """Analyze what attachment-related triggers are present in the interaction"""
        triggers = {}
        
        # Abandonment triggers
        if user_cues.get('withdrawal_signals', 0) > 0.3:
            triggers['abandonment_fear'] = user_cues['withdrawal_signals']
        
        # Intimacy triggers
        if emotional_content.get('vulnerability', 0) > 0.6:
            triggers['intimacy_approach'] = emotional_content['vulnerability']
        
        # Trust building triggers
        if interaction_type in ['sharing_secret', 'emotional_support', 'consistency_display']:
            triggers['trust_building'] = emotional_content.get('authenticity', 0.7)
        
        # Conflict triggers
        if interaction_type in ['disagreement', 'criticism', 'boundary_testing']:
            triggers['conflict_activation'] = emotional_content.get('tension', 0.5)
        
        # Security triggers
        if user_cues.get('reliability_signals', 0) > 0.5:
            triggers['security_building'] = user_cues['reliability_signals']
        
        return triggers
    
    def _update_attachment_metrics(self, triggers: Dict[str, float]):
        """Update attachment metrics based on triggered behaviors"""
        style_modifiers = self._get_attachment_style_modifiers()
        
        for trigger, intensity in triggers.items():
            if trigger == 'abandonment_fear':
                if self.attachment_style in [AttachmentStyle.ANXIOUS_PREOCCUPIED, AttachmentStyle.FEARFUL_AVOIDANT]:
                    self.current_metrics.abandonment_anxiety += intensity * 0.3
                    self.current_metrics.emotional_availability -= intensity * 0.2
                
            elif trigger == 'intimacy_approach':
                if self.attachment_style == AttachmentStyle.DISMISSIVE_AVOIDANT:
                    self.current_metrics.intimacy_comfort -= intensity * 0.2
                elif self.attachment_style == AttachmentStyle.SECURE:
                    self.current_metrics.intimacy_comfort += intensity * 0.1
                
            elif trigger == 'trust_building':
                self.current_metrics.trust_stability += intensity * 0.15
                self.current_metrics.security_level += intensity * 0.1
                
            elif trigger == 'conflict_activation':
                if self.attachment_style == AttachmentStyle.FEARFUL_AVOIDANT:
                    self.current_metrics.emotional_availability -= intensity * 0.3
                elif self.attachment_style == AttachmentStyle.SECURE:
                    self.current_metrics.conflict_resolution += intensity * 0.1
                    
            elif trigger == 'security_building':
                self.current_metrics.security_level += intensity * 0.2
                if self.current_metrics.security_level > 0.8:
                    # High security can improve other metrics
                    self.current_metrics.intimacy_comfort += 0.05
                    self.current_metrics.abandonment_anxiety -= 0.05
        
        # Clamp all metrics to 0.0-1.0 range
        self._clamp_metrics()
    
    def _generate_response_modifications(self, triggers: Dict[str, float]) -> Dict[str, Any]:
        """Generate modifications to persona responses based on attachment state"""
        modifications = {
            'emotional_tone_adjustments': {},
            'response_patterns': {},
            'behavioral_changes': {},
            'dialogue_modifications': {}
        }
        
        # Adjust based on current attachment style and metrics
        if self.attachment_style == AttachmentStyle.ANXIOUS_PREOCCUPIED:
            if self.current_metrics.abandonment_anxiety > 0.7:
                modifications['emotional_tone_adjustments']['clingy_tendency'] = 0.6
                modifications['response_patterns']['reassurance_seeking'] = 0.8
                modifications['dialogue_modifications']['validation_requests'] = 0.7
                
        elif self.attachment_style == AttachmentStyle.DISMISSIVE_AVOIDANT:
            if self.current_metrics.intimacy_comfort < 0.3:
                modifications['emotional_tone_adjustments']['emotional_distance'] = 0.7
                modifications['response_patterns']['intimacy_deflection'] = 0.6
                modifications['behavioral_changes']['independence_emphasis'] = 0.8
                
        elif self.attachment_style == AttachmentStyle.FEARFUL_AVOIDANT:
            if 'conflict_activation' in triggers:
                modifications['behavioral_changes']['approach_avoidance'] = 0.8
                modifications['emotional_tone_adjustments']['conflicted_signals'] = 0.7
                
        elif self.attachment_style == AttachmentStyle.SECURE:
            if self.current_metrics.security_level > 0.8:
                modifications['response_patterns']['consistent_availability'] = 0.9
                modifications['emotional_tone_adjustments']['stable_warmth'] = 0.8
        
        # Add bonding stage specific modifications
        stage_mods = self._get_bonding_stage_modifications()
        for category, changes in stage_mods.items():
            if category in modifications:
                modifications[category].update(changes)
        
        return modifications
    
    def _get_bonding_stage_modifications(self) -> Dict[str, Dict[str, float]]:
        """Get response modifications specific to current bonding stage"""
        stage_modifications = {
            BondingStage.INITIAL_CONTACT: {
                'response_patterns': {'cautious_optimism': 0.6, 'boundary_awareness': 0.8}
            },
            BondingStage.BUILDING_TRUST: {
                'response_patterns': {'consistency_emphasis': 0.7, 'vulnerability_testing': 0.4}
            },
            BondingStage.DEEPENING_CONNECTION: {
                'emotional_tone_adjustments': {'warmth_increase': 0.6},
                'response_patterns': {'shared_experiences': 0.8}
            },
            BondingStage.ESTABLISHED_BOND: {
                'behavioral_changes': {'comfortable_intimacy': 0.7},
                'response_patterns': {'playful_teasing': 0.5}
            },
            BondingStage.INTIMATE_ATTACHMENT: {
                'emotional_tone_adjustments': {'deep_connection': 0.9},
                'dialogue_modifications': {'intimate_language': 0.8}
            },
            BondingStage.CRISIS_TESTING: {
                'response_patterns': {'loyalty_demonstration': 0.9, 'conflict_resolution': 0.8}
            },
            BondingStage.MATURE_BOND: {
                'behavioral_changes': {'secure_base': 0.9},
                'emotional_tone_adjustments': {'unconditional_acceptance': 0.8}
            }
        }
        
        return stage_modifications.get(self.bonding_stage, {})
    
    def _evaluate_bonding_stage_progression(self):
        """Evaluate whether to progress to next bonding stage"""
        metrics = self.current_metrics
        current_stage_index = list(BondingStage).index(self.bonding_stage)
        
        # Progression criteria for each stage
        progression_criteria = {
            BondingStage.INITIAL_CONTACT: (
                metrics.trust_stability > 0.4 and 
                len(self.attachment_history) > 5
            ),
            BondingStage.BUILDING_TRUST: (
                metrics.trust_stability > 0.6 and 
                metrics.security_level > 0.5
            ),
            BondingStage.DEEPENING_CONNECTION: (
                metrics.intimacy_comfort > 0.6 and 
                metrics.emotional_availability > 0.7
            ),
            BondingStage.ESTABLISHED_BOND: (
                metrics.security_level > 0.7 and 
                metrics.conflict_resolution > 0.6
            ),
            BondingStage.INTIMATE_ATTACHMENT: (
                metrics.intimacy_comfort > 0.8 and 
                metrics.abandonment_anxiety < 0.3
            ),
            BondingStage.CRISIS_TESTING: (
                metrics.trust_stability > 0.8 and 
                metrics.conflict_resolution > 0.8
            )
        }
        
        if (self.bonding_stage in progression_criteria and 
            progression_criteria[self.bonding_stage] and 
            current_stage_index < len(BondingStage) - 1):
            
            self.bonding_stage = list(BondingStage)[current_stage_index + 1]
    
    def _get_triggered_behaviors(self, triggers: Dict[str, float]) -> List[str]:
        """Get list of triggered attachment behaviors for this interaction"""
        behaviors = []
        
        for trigger, intensity in triggers.items():
            if intensity > 0.5:
                behavior_map = {
                    'abandonment_fear': ['clingy_responses', 'reassurance_seeking', 'fear_of_loss'],
                    'intimacy_approach': ['vulnerability_sharing', 'emotional_opening', 'trust_testing'],
                    'trust_building': ['consistency_display', 'reliability_emphasis', 'security_building'],
                    'conflict_activation': ['defensive_patterns', 'attachment_testing', 'repair_attempts'],
                    'security_building': ['stable_responses', 'confident_interaction', 'secure_base_offering']
                }
                
                if trigger in behavior_map:
                    behaviors.extend(behavior_map[trigger])
        
        return list(set(behaviors))  # Remove duplicates
    
    def _get_current_attachment_state(self) -> Dict[str, Any]:
        """Get current attachment state summary"""
        return {
            'attachment_style': self.attachment_style.value,
            'bonding_stage': self.bonding_stage.value,
            'metrics': {
                'security_level': round(self.current_metrics.security_level, 2),
                'intimacy_comfort': round(self.current_metrics.intimacy_comfort, 2),
                'abandonment_anxiety': round(self.current_metrics.abandonment_anxiety, 2),
                'emotional_availability': round(self.current_metrics.emotional_availability, 2),
                'trust_stability': round(self.current_metrics.trust_stability, 2),
                'conflict_resolution': round(self.current_metrics.conflict_resolution, 2)
            },
            'interaction_count': len(self.attachment_history),
            'last_significant_event': self._get_last_significant_event()
        }
    
    def _get_last_significant_event(self) -> Optional[str]:
        """Get the last significant attachment event"""
        if not self.attachment_history:
            return None
        
        significant_events = [event for event in self.attachment_history[-10:] 
                            if event.emotional_impact > 0.6]
        
        if significant_events:
            return significant_events[-1].event_type
        return None
    
    def _get_attachment_style_modifiers(self) -> Dict[str, float]:
        """Get style-specific response modifiers"""
        style_modifiers = {
            AttachmentStyle.SECURE: {
                'stability_bonus': 0.2,
                'intimacy_comfort_bonus': 0.1,
                'conflict_resolution_bonus': 0.15
            },
            AttachmentStyle.ANXIOUS_PREOCCUPIED: {
                'abandonment_sensitivity': 0.3,
                'reassurance_need': 0.4,
                'emotional_intensity': 0.2
            },
            AttachmentStyle.DISMISSIVE_AVOIDANT: {
                'independence_preference': 0.3,
                'intimacy_resistance': 0.2,
                'emotional_regulation': 0.1
            },
            AttachmentStyle.FEARFUL_AVOIDANT: {
                'approach_avoidance': 0.4,
                'trust_difficulty': 0.3,
                'emotional_volatility': 0.2
            }
        }
        
        return style_modifiers.get(self.attachment_style, {})
    
    def _clamp_metrics(self):
        """Ensure all metrics stay within 0.0-1.0 range"""
        self.current_metrics.security_level = max(0.0, min(1.0, self.current_metrics.security_level))
        self.current_metrics.intimacy_comfort = max(0.0, min(1.0, self.current_metrics.intimacy_comfort))
        self.current_metrics.abandonment_anxiety = max(0.0, min(1.0, self.current_metrics.abandonment_anxiety))
        self.current_metrics.emotional_availability = max(0.0, min(1.0, self.current_metrics.emotional_availability))
        self.current_metrics.trust_stability = max(0.0, min(1.0, self.current_metrics.trust_stability))
        self.current_metrics.conflict_resolution = max(0.0, min(1.0, self.current_metrics.conflict_resolution))
    
    def _load_attachment_history(self):
        """Load attachment history from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load metrics
                if 'metrics' in data:
                    metrics_data = data['metrics']
                    self.current_metrics = AttachmentMetrics(**metrics_data)
                
                # Load bonding stage
                if 'bonding_stage' in data:
                    self.bonding_stage = BondingStage(data['bonding_stage'])
                
                # Load attachment style
                if 'attachment_style' in data:
                    self.attachment_style = AttachmentStyle(data['attachment_style'])
                
                # Load history (last 100 events to prevent excessive memory usage)
                if 'history' in data:
                    history_data = data['history'][-100:]  # Keep only recent events
                    self.attachment_history = [
                        BondingEvent(
                            event_type=event['event_type'],
                            emotional_impact=event['emotional_impact'],
                            attachment_response=event['attachment_response'],
                            timestamp=datetime.fromisoformat(event['timestamp']),
                            context=event.get('context', {})
                        ) for event in history_data
                    ]
                    
            except Exception as e:
                print(f"Error loading attachment history for {self.persona_name}: {e}")
    
    def _save_attachment_history(self):
        """Save attachment history to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'metrics': {
                    'security_level': self.current_metrics.security_level,
                    'intimacy_comfort': self.current_metrics.intimacy_comfort,
                    'abandonment_anxiety': self.current_metrics.abandonment_anxiety,
                    'emotional_availability': self.current_metrics.emotional_availability,
                    'trust_stability': self.current_metrics.trust_stability,
                    'conflict_resolution': self.current_metrics.conflict_resolution
                },
                'bonding_stage': self.bonding_stage.value,
                'attachment_style': self.attachment_style.value,
                'history': [
                    {
                        'event_type': event.event_type,
                        'emotional_impact': event.emotional_impact,
                        'attachment_response': event.attachment_response,
                        'timestamp': event.timestamp.isoformat(),
                        'context': event.context
                    } for event in self.attachment_history[-100:]  # Save only recent events
                ]
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving attachment history for {self.persona_name}: {e}")

# Factory function for creating attachment engines
def create_attachment_engine(persona_name: str) -> AttachmentRegulationEngine:
    """Create an attachment regulation engine for a specific persona"""
    return AttachmentRegulationEngine(persona_name)

# Integration helpers
def get_attachment_response_modifications(persona_name: str, interaction_type: str, 
                                        emotional_content: Dict[str, float], 
                                        user_attachment_cues: Dict[str, float]) -> Dict[str, Any]:
    """
    Get attachment-informed response modifications for a persona
    
    This function can be called from the main persona engines to get
    attachment-based response modifications.
    """
    engine = create_attachment_engine(persona_name)
    return engine.process_interaction(interaction_type, emotional_content, user_attachment_cues)
