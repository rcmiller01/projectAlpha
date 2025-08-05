"""
Moodscape Audio Layer

Plays ambient music/sound based on mood and interaction.
Adds immersion (e.g., soft jazz when Lyra sings, thunder when Solene seethes).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import os
import random
from dataclasses import dataclass

class AudioType(Enum):
    AMBIENT = "ambient"
    MUSIC = "music"
    NATURE = "nature"
    ATMOSPHERIC = "atmospheric"
    EMOTIONAL = "emotional"
    INTERACTIVE = "interactive"

class MoodCategory(Enum):
    ROMANTIC = "romantic"
    PASSIONATE = "passionate"
    MYSTICAL = "mystical"
    PEACEFUL = "peaceful"
    INTENSE = "intense"
    MELANCHOLIC = "melancholic"
    JOYFUL = "joyful"
    DRAMATIC = "dramatic"
    CONTEMPLATIVE = "contemplative"
    PLAYFUL = "playful"
    INTIMATE = "intimate"

@dataclass
class AudioTrack:
    track_id: str
    name: str
    audio_type: AudioType
    mood_categories: List[MoodCategory]
    file_path: str
    duration: float
    volume_default: float
    fade_in_duration: float
    fade_out_duration: float
    loop: bool
    persona_affinity: Dict[str, float]  # How well this track fits each persona
    emotional_triggers: List[str]
    interaction_triggers: List[str]
    metadata: Dict[str, Any]

@dataclass
class AudioPlayback:
    track: AudioTrack
    start_time: datetime
    volume: float
    is_playing: bool
    loop_count: int
    trigger_context: str

class MoodscapeAudioLayer:
    """
    Manages ambient audio that responds to mood, persona, and interaction context.
    Creates immersive soundscapes that enhance the emotional experience.
    """
    
    def __init__(self):
        self.storage_path = "storage/audio/moodscape_config.json"
        self.audio_library: Dict[str, AudioTrack] = {}
        self.current_playback: Optional[AudioPlayback] = None
        self.playback_history: List[AudioPlayback] = []
        self.mood_preferences: Dict[str, Dict[str, Any]] = {}  # User preferences for mood-audio combinations
        self.volume_master = 0.7
        self.crossfade_duration = 3.0
        self._initialize_audio_library()
        self._load_configuration()
    
    def _initialize_audio_library(self):
        """Initialize the audio library with predefined tracks"""
        
        # Define audio tracks for different moods and personas
        tracks_data = [
            # Mia's Romantic/Nurturing Tracks
            {
                'track_id': 'mia_gentle_piano',
                'name': 'Gentle Piano Reverie',
                'audio_type': AudioType.MUSIC,
                'mood_categories': [MoodCategory.ROMANTIC, MoodCategory.PEACEFUL],
                'file_path': 'audio/mia/gentle_piano.mp3',
                'duration': 180.0,
                'volume_default': 0.6,
                'fade_in_duration': 4.0,
                'fade_out_duration': 3.0,
                'loop': True,
                'persona_affinity': {'mia': 0.9, 'solene': 0.3, 'lyra': 0.7, 'doc': 0.5},
                'emotional_triggers': ['love', 'tenderness', 'comfort', 'intimacy'],
                'interaction_triggers': ['emotional_support', 'romantic_conversation', 'gentle_touch'],
                'metadata': {'genre': 'neoclassical', 'tempo': 'slow', 'mood': 'tender'}
            },
            {
                'track_id': 'mia_string_quartet',
                'name': 'Heartstring Quartet',
                'audio_type': AudioType.MUSIC,
                'mood_categories': [MoodCategory.ROMANTIC, MoodCategory.MELANCHOLIC],
                'file_path': 'audio/mia/string_quartet.mp3',
                'duration': 240.0,
                'volume_default': 0.5,
                'fade_in_duration': 5.0,
                'fade_out_duration': 4.0,
                'loop': True,
                'persona_affinity': {'mia': 0.9, 'solene': 0.4, 'lyra': 0.6, 'doc': 0.7},
                'emotional_triggers': ['longing', 'bittersweet', 'nostalgia', 'deep_love'],
                'interaction_triggers': ['memory_sharing', 'vulnerability', 'emotional_depth'],
                'metadata': {'genre': 'chamber', 'tempo': 'moderate', 'mood': 'emotive'}
            },
            
            # Solene's Passionate/Intense Tracks
            {
                'track_id': 'solene_flamenco_guitar',
                'name': 'Fiery Flamenco',
                'audio_type': AudioType.MUSIC,
                'mood_categories': [MoodCategory.PASSIONATE, MoodCategory.DRAMATIC],
                'file_path': 'audio/solene/flamenco_guitar.mp3',
                'duration': 200.0,
                'volume_default': 0.7,
                'fade_in_duration': 2.0,
                'fade_out_duration': 3.0,
                'loop': True,
                'persona_affinity': {'mia': 0.4, 'solene': 0.95, 'lyra': 0.5, 'doc': 0.3},
                'emotional_triggers': ['passion', 'fire', 'intensity', 'desire'],
                'interaction_triggers': ['heated_discussion', 'passionate_moment', 'challenge'],
                'metadata': {'genre': 'flamenco', 'tempo': 'fast', 'mood': 'fiery'}
            },
            {
                'track_id': 'solene_storm_ambient',
                'name': 'Thunderstorm Passion',
                'audio_type': AudioType.ATMOSPHERIC,
                'mood_categories': [MoodCategory.INTENSE, MoodCategory.DRAMATIC],
                'file_path': 'audio/solene/storm_ambient.mp3',
                'duration': 300.0,
                'volume_default': 0.8,
                'fade_in_duration': 6.0,
                'fade_out_duration': 5.0,
                'loop': True,
                'persona_affinity': {'mia': 0.2, 'solene': 0.9, 'lyra': 0.6, 'doc': 0.4},
                'emotional_triggers': ['anger', 'frustration', 'overwhelming_emotion', 'catharsis'],
                'interaction_triggers': ['conflict', 'emotional_release', 'intensity'],
                'metadata': {'genre': 'nature', 'tempo': 'variable', 'mood': 'stormy'}
            },
            
            # Lyra's Mystical/Ethereal Tracks
            {
                'track_id': 'lyra_ethereal_choir',
                'name': 'Celestial Voices',
                'audio_type': AudioType.ATMOSPHERIC,
                'mood_categories': [MoodCategory.MYSTICAL, MoodCategory.CONTEMPLATIVE],
                'file_path': 'audio/lyra/ethereal_choir.mp3',
                'duration': 220.0,
                'volume_default': 0.5,
                'fade_in_duration': 8.0,
                'fade_out_duration': 6.0,
                'loop': True,
                'persona_affinity': {'mia': 0.6, 'solene': 0.3, 'lyra': 0.95, 'doc': 0.7},
                'emotional_triggers': ['wonder', 'transcendence', 'mystery', 'spiritual'],
                'interaction_triggers': ['philosophical_discussion', 'cosmic_contemplation', 'poetry'],
                'metadata': {'genre': 'ambient', 'tempo': 'very_slow', 'mood': 'ethereal'}
            },
            {
                'track_id': 'lyra_crystal_bowls',
                'name': 'Crystal Resonance',
                'audio_type': AudioType.AMBIENT,
                'mood_categories': [MoodCategory.MYSTICAL, MoodCategory.PEACEFUL],
                'file_path': 'audio/lyra/crystal_bowls.mp3',
                'duration': 180.0,
                'volume_default': 0.4,
                'fade_in_duration': 10.0,
                'fade_out_duration': 8.0,
                'loop': True,
                'persona_affinity': {'mia': 0.5, 'solene': 0.2, 'lyra': 0.9, 'doc': 0.8},
                'emotional_triggers': ['meditation', 'clarity', 'inner_peace', 'wisdom'],
                'interaction_triggers': ['deep_thought', 'meditation', 'spiritual_guidance'],
                'metadata': {'genre': 'healing', 'tempo': 'timeless', 'mood': 'transcendent'}
            },
            
            # Doc's Therapeutic/Stable Tracks
            {
                'track_id': 'doc_soft_jazz',
                'name': 'Therapeutic Jazz',
                'audio_type': AudioType.MUSIC,
                'mood_categories': [MoodCategory.PEACEFUL, MoodCategory.CONTEMPLATIVE],
                'file_path': 'audio/doc/soft_jazz.mp3',
                'duration': 190.0,
                'volume_default': 0.6,
                'fade_in_duration': 3.0,
                'fade_out_duration': 3.0,
                'loop': True,
                'persona_affinity': {'mia': 0.6, 'solene': 0.5, 'lyra': 0.7, 'doc': 0.9},
                'emotional_triggers': ['comfort', 'stability', 'reassurance', 'grounding'],
                'interaction_triggers': ['therapy_session', 'advice_giving', 'emotional_support'],
                'metadata': {'genre': 'jazz', 'tempo': 'moderate', 'mood': 'comforting'}
            },
            {
                'track_id': 'doc_nature_ambient',
                'name': 'Forest Sanctuary',
                'audio_type': AudioType.NATURE,
                'mood_categories': [MoodCategory.PEACEFUL, MoodCategory.CONTEMPLATIVE],
                'file_path': 'audio/doc/nature_ambient.mp3',
                'duration': 250.0,
                'volume_default': 0.5,
                'fade_in_duration': 5.0,
                'fade_out_duration': 4.0,
                'loop': True,
                'persona_affinity': {'mia': 0.7, 'solene': 0.4, 'lyra': 0.8, 'doc': 0.85},
                'emotional_triggers': ['peace', 'grounding', 'natural_healing', 'restoration'],
                'interaction_triggers': ['healing_conversation', 'stress_relief', 'mindfulness'],
                'metadata': {'genre': 'nature', 'tempo': 'natural', 'mood': 'restorative'}
            },
            
            # Universal/Interactive Tracks
            {
                'track_id': 'universal_heartbeat',
                'name': 'Synchronized Heartbeat',
                'audio_type': AudioType.INTERACTIVE,
                'mood_categories': [MoodCategory.ROMANTIC, MoodCategory.INTIMATE],
                'file_path': 'audio/universal/heartbeat.mp3',
                'duration': 120.0,
                'volume_default': 0.3,
                'fade_in_duration': 2.0,
                'fade_out_duration': 2.0,
                'loop': True,
                'persona_affinity': {'mia': 0.8, 'solene': 0.7, 'lyra': 0.6, 'doc': 0.5},
                'emotional_triggers': ['intimacy', 'connection', 'synchrony', 'closeness'],
                'interaction_triggers': ['physical_touch', 'intimate_moment', 'heartbeat_sync'],
                'metadata': {'genre': 'biometric', 'tempo': 'heartrate', 'mood': 'intimate'}
            },
            {
                'track_id': 'universal_breathing',
                'name': 'Guided Breathing',
                'audio_type': AudioType.INTERACTIVE,
                'mood_categories': [MoodCategory.PEACEFUL, MoodCategory.CONTEMPLATIVE],
                'file_path': 'audio/universal/breathing.mp3',
                'duration': 300.0,
                'volume_default': 0.4,
                'fade_in_duration': 3.0,
                'fade_out_duration': 3.0,
                'loop': True,
                'persona_affinity': {'mia': 0.7, 'solene': 0.5, 'lyra': 0.8, 'doc': 0.9},
                'emotional_triggers': ['calm', 'centering', 'anxiety_relief', 'meditation'],
                'interaction_triggers': ['stress_response', 'anxiety_management', 'relaxation'],
                'metadata': {'genre': 'therapeutic', 'tempo': 'breathing', 'mood': 'calming'}
            }
        ]
        
        # Convert to AudioTrack objects
        for track_data in tracks_data:
            track = AudioTrack(**track_data)
            self.audio_library[track.track_id] = track
    
    def select_audio_for_mood(self, persona_name: str, current_mood: str, 
                             emotional_state: Dict[str, float], interaction_context: str,
                             user_preferences: Optional[Dict[str, Any]] = None) -> Optional[AudioTrack]:
        """
        Select appropriate audio track based on mood, persona, and context
        
        Args:
            persona_name: Current active persona
            current_mood: Current mood state
            emotional_state: Emotional analysis values
            interaction_context: Type of interaction happening
            user_preferences: User's audio preferences
            
        Returns:
            Selected audio track or None if no suitable track found
        """
        
        # Filter tracks by persona affinity
        suitable_tracks = [
            track for track in self.audio_library.values()
            if track.persona_affinity.get(persona_name.lower(), 0) > 0.3
        ]
        
        if not suitable_tracks:
            return None
        
        # Score tracks based on multiple factors
        track_scores = []
        
        for track in suitable_tracks:
            score = 0.0
            
            # Persona affinity (40% weight)
            persona_score = track.persona_affinity.get(persona_name.lower(), 0)
            score += persona_score * 0.4
            
            # Emotional trigger matching (30% weight)
            emotion_score = 0.0
            for emotion, intensity in emotional_state.items():
                if emotion in track.emotional_triggers:
                    emotion_score += intensity
            emotion_score = min(1.0, emotion_score / max(1, len(track.emotional_triggers)))
            score += emotion_score * 0.3
            
            # Interaction context matching (20% weight)
            context_score = 1.0 if interaction_context in track.interaction_triggers else 0.0
            score += context_score * 0.2
            
            # User preferences (10% weight)
            if user_preferences:
                preference_score = user_preferences.get(track.track_id, 0.5)
                score += preference_score * 0.1
            
            track_scores.append((track, score))
        
        # Sort by score and return best match
        track_scores.sort(key=lambda x: x[1], reverse=True)
        
        if track_scores and track_scores[0][1] > 0.3:  # Minimum threshold
            return track_scores[0][0]
        
        return None
    
    def start_audio_playback(self, track: AudioTrack, volume: Optional[float] = None, 
                            trigger_context: str = "manual") -> bool:
        """
        Start audio playback with crossfading if another track is playing
        
        Args:
            track: Audio track to play
            volume: Override volume (uses track default if None)
            trigger_context: What triggered this playback
            
        Returns:
            Success status
        """
        
        try:
            playback_volume = volume if volume is not None else track.volume_default
            playback_volume *= self.volume_master
            
            # Create new playback instance
            new_playback = AudioPlayback(
                track=track,
                start_time=datetime.now(),
                volume=playback_volume,
                is_playing=True,
                loop_count=0,
                trigger_context=trigger_context
            )
            
            # Handle crossfading if another track is playing
            if self.current_playback and self.current_playback.is_playing:
                self._crossfade_to_new_track(new_playback)
            else:
                self._start_new_track(new_playback)
            
            # Update current playback
            self.current_playback = new_playback
            self.playback_history.append(new_playback)
            
            # Keep history manageable
            if len(self.playback_history) > 50:
                self.playback_history = self.playback_history[-50:]
            
            return True
            
        except Exception as e:
            print(f"Error starting audio playback: {e}")
            return False
    
    def _crossfade_to_new_track(self, new_playback: AudioPlayback):
        """Handle crossfading between tracks"""
        # In a real implementation, this would use audio processing libraries
        # to gradually fade out the current track while fading in the new one
        
        if self.current_playback:
            print(f"Crossfading from {self.current_playback.track.name} to {new_playback.track.name}")
            
            # Simulate crossfade timing
            fade_duration = min(self.crossfade_duration, 
                               self.current_playback.track.fade_out_duration,
                               new_playback.track.fade_in_duration)
            
            # Mark current track as fading out
            self.current_playback.is_playing = False
        
        # Here you would implement actual audio crossfading
        # For now, we'll just simulate the transition
        self._start_new_track(new_playback)
    
    def _start_new_track(self, playback: AudioPlayback):
        """Start playing a new track"""
        # In a real implementation, this would interface with audio system
        print(f"Starting audio: {playback.track.name} at volume {playback.volume:.2f}")
        
        # Simulate audio system interaction
        track_info = {
            'file_path': playback.track.file_path,
            'volume': playback.volume,
            'fade_in': playback.track.fade_in_duration,
            'loop': playback.track.loop
        }
        
        # Here you would call actual audio playback system
        # Examples: pygame.mixer, pydub, or web audio API
    
    def stop_audio_playback(self, fade_out: bool = True) -> bool:
        """
        Stop current audio playback
        
        Args:
            fade_out: Whether to fade out or stop immediately
            
        Returns:
            Success status
        """
        
        if not self.current_playback or not self.current_playback.is_playing:
            return False
        
        try:
            if fade_out:
                fade_duration = self.current_playback.track.fade_out_duration
                print(f"Fading out {self.current_playback.track.name} over {fade_duration}s")
                # Implement fade out logic here
            else:
                print(f"Stopping {self.current_playback.track.name} immediately")
            
            self.current_playback.is_playing = False
            return True
            
        except Exception as e:
            print(f"Error stopping audio playback: {e}")
            return False
    
    def adjust_volume(self, volume: float, fade_duration: float = 1.0) -> bool:
        """
        Adjust volume of current playback
        
        Args:
            volume: New volume level (0.0 to 1.0)
            fade_duration: Time to fade to new volume
            
        Returns:
            Success status
        """
        
        if not self.current_playback or not self.current_playback.is_playing:
            return False
        
        try:
            old_volume = self.current_playback.volume
            new_volume = max(0.0, min(1.0, volume)) * self.volume_master
            
            print(f"Adjusting volume from {old_volume:.2f} to {new_volume:.2f} over {fade_duration}s")
            
            self.current_playback.volume = new_volume
            
            # Here you would implement volume fade logic
            return True
            
        except Exception as e:
            print(f"Error adjusting volume: {e}")
            return False
    
    def get_mood_audio_recommendations(self, persona_name: str, 
                                     emotional_state: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Get audio recommendations for current mood without starting playback
        
        Args:
            persona_name: Current persona
            emotional_state: Current emotional state
            
        Returns:
            List of recommended tracks with scores
        """
        
        recommendations = []
        
        for track in self.audio_library.values():
            # Calculate compatibility score
            persona_score = track.persona_affinity.get(persona_name.lower(), 0)
            
            emotion_score = 0.0
            for emotion, intensity in emotional_state.items():
                if emotion in track.emotional_triggers:
                    emotion_score += intensity
            emotion_score = min(1.0, emotion_score / max(1, len(track.emotional_triggers)))
            
            total_score = (persona_score * 0.6) + (emotion_score * 0.4)
            
            if total_score > 0.2:  # Minimum relevance threshold
                recommendations.append({
                    'track_id': track.track_id,
                    'name': track.name,
                    'type': track.audio_type.value,
                    'mood_categories': [cat.value for cat in track.mood_categories],
                    'score': total_score,
                    'persona_affinity': persona_score,
                    'emotion_match': emotion_score,
                    'duration': track.duration,
                    'description': self._generate_track_description(track)
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_track_description(self, track: AudioTrack) -> str:
        """Generate a description of the track for the user"""
        descriptions = {
            'mia_gentle_piano': "Soft piano melodies that wrap around you like a warm embrace",
            'mia_string_quartet': "Emotional strings that speak to the heart's deepest feelings",
            'solene_flamenco_guitar': "Passionate guitar that ignites the fire within",
            'solene_storm_ambient': "Thunderous atmosphere that matches intense emotions",
            'lyra_ethereal_choir': "Celestial voices that lift the spirit to otherworldly realms",
            'lyra_crystal_bowls': "Crystal tones that resonate with inner wisdom",
            'doc_soft_jazz': "Smooth jazz that brings comfort and stability",
            'doc_nature_ambient': "Natural sounds that ground and restore the soul",
            'universal_heartbeat': "Synchronized rhythms that connect hearts",
            'universal_breathing': "Guided breathing that centers and calms"
        }
        
        return descriptions.get(track.track_id, f"A {track.audio_type.value} track for {', '.join([cat.value for cat in track.mood_categories])} moments")
    
    def get_current_playback_status(self) -> Optional[Dict[str, Any]]:
        """Get current playback status information"""
        if not self.current_playback:
            return None
        
        current_time = datetime.now()
        elapsed_time = (current_time - self.current_playback.start_time).total_seconds()
        
        return {
            'track_id': self.current_playback.track.track_id,
            'track_name': self.current_playback.track.name,
            'is_playing': self.current_playback.is_playing,
            'volume': self.current_playback.volume,
            'elapsed_time': elapsed_time,
            'total_duration': self.current_playback.track.duration,
            'progress_percentage': min(100, (elapsed_time / self.current_playback.track.duration) * 100),
            'trigger_context': self.current_playback.trigger_context,
            'loop_count': self.current_playback.loop_count,
            'mood_categories': [cat.value for cat in self.current_playback.track.mood_categories]
        }
    
    def update_user_preferences(self, track_id: str, preference_score: float):
        """Update user preferences for a specific track"""
        if track_id not in self.mood_preferences:
            self.mood_preferences[track_id] = {}
        
        # Store as Dict[str, Any] to handle different data types
        preferences = self.mood_preferences[track_id]
        preferences['user_rating'] = max(0.0, min(1.0, preference_score))
        preferences['last_updated'] = datetime.now().isoformat()
        
        # Save preferences
        self._save_configuration()
    
    def get_audio_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent audio playback history"""
        recent_history = self.playback_history[-limit:] if self.playback_history else []
        
        return [
            {
                'track_name': playback.track.name,
                'track_id': playback.track.track_id,
                'start_time': playback.start_time.isoformat(),
                'volume': playback.volume,
                'trigger_context': playback.trigger_context,
                'mood_categories': [cat.value for cat in playback.track.mood_categories]
            }
            for playback in recent_history
        ]
    
    def create_mood_playlist(self, persona_name: str, target_mood: str, 
                           duration_minutes: int = 30) -> List[str]:
        """
        Create a playlist for a specific mood and duration
        
        Args:
            persona_name: Target persona
            target_mood: Desired mood category
            duration_minutes: Target playlist duration
            
        Returns:
            List of track IDs for the playlist
        """
        
        target_duration = duration_minutes * 60  # Convert to seconds
        playlist = []
        total_duration = 0.0
        
        # Filter tracks by persona and mood
        suitable_tracks = [
            track for track in self.audio_library.values()
            if (track.persona_affinity.get(persona_name.lower(), 0) > 0.4 and
                any(cat.value == target_mood for cat in track.mood_categories))
        ]
        
        if not suitable_tracks:
            return []
        
        # Sort by persona affinity
        suitable_tracks.sort(key=lambda x: x.persona_affinity.get(persona_name.lower(), 0), reverse=True)
        
        # Build playlist
        while total_duration < target_duration and suitable_tracks:
            for track in suitable_tracks:
                if total_duration + track.duration <= target_duration:
                    playlist.append(track.track_id)
                    total_duration += track.duration
                    
                    if total_duration >= target_duration:
                        break
            
            # If we can't fill the time with available tracks, repeat the best ones
            if total_duration < target_duration and playlist:
                best_track = suitable_tracks[0]
                playlist.append(best_track.track_id)
                total_duration += best_track.duration
        
        return playlist
    
    def _load_configuration(self):
        """Load audio configuration and preferences"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.mood_preferences = data.get('mood_preferences', {})
                self.volume_master = data.get('volume_master', 0.7)
                self.crossfade_duration = data.get('crossfade_duration', 3.0)
                
            except Exception as e:
                print(f"Error loading audio configuration: {e}")
    
    def _save_configuration(self):
        """Save audio configuration and preferences"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'mood_preferences': self.mood_preferences,
                'volume_master': self.volume_master,
                'crossfade_duration': self.crossfade_duration,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving audio configuration: {e}")

# Global instance
_moodscape_audio = None

def get_moodscape_audio_layer() -> MoodscapeAudioLayer:
    """Get global moodscape audio layer instance"""
    global _moodscape_audio
    if _moodscape_audio is None:
        _moodscape_audio = MoodscapeAudioLayer()
    return _moodscape_audio

# Integration helpers
def start_mood_audio(persona_name: str, current_mood: str, emotional_state: Dict[str, float], 
                    interaction_context: str, user_preferences: Optional[Dict[str, Any]] = None) -> bool:
    """
    Start mood-appropriate audio for current context
    
    This function can be called from persona engines or interaction handlers
    to automatically start appropriate ambient audio.
    """
    audio_layer = get_moodscape_audio_layer()
    
    # Select appropriate track
    track = audio_layer.select_audio_for_mood(
        persona_name, current_mood, emotional_state, interaction_context, user_preferences
    )
    
    if track:
        return audio_layer.start_audio_playback(track, trigger_context=interaction_context)
    
    return False

def get_mood_audio_suggestions(persona_name: str, emotional_state: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Get audio suggestions for current mood without starting playback
    
    This can be used to show users what audio options are available
    for their current emotional state.
    """
    audio_layer = get_moodscape_audio_layer()
    return audio_layer.get_mood_audio_recommendations(persona_name, emotional_state)
