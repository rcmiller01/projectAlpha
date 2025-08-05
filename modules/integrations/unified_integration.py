"""
Unified Integration Service for AI Companion

This module provides a single interface for all external integrations:
- Calendar (Google, Outlook, Apple)
- Email (Gmail, Outlook, Apple Mail)
- SMS (Twilio, native messaging)
- Apple Music (music control and recommendations)
- Social Media (Reddit, X, Facebook, Threads, Instagram)
- Apple HealthKit (biometrics and health data)

Focuses on unified persona with emotional context awareness.
"""

import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

# Import our integration modules
try:
    from .health_integration import EmotionalHealthIntegration
    from .social_integration import SocialMediaIntegration
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append('..')
    from health_integration import EmotionalHealthIntegration
    from social_integration import SocialMediaIntegration

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    CALENDAR = "calendar"
    EMAIL = "email"
    SMS = "sms"
    MUSIC = "music"
    SOCIAL = "social"
    HEALTH = "health"

@dataclass
class CalendarEvent:
    """Calendar event representation"""
    event_id: str
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime
    location: Optional[str]
    attendees: List[str]
    reminder_minutes: int
    emotional_context: Optional[str] = None  # meeting, personal, work, etc.

@dataclass
class EmailMessage:
    """Email message representation"""
    message_id: str
    subject: str
    sender: str
    recipients: List[str]
    timestamp: datetime
    content: str
    importance: str  # high, normal, low
    emotional_tone: Optional[str] = None  # urgent, casual, formal, etc.

@dataclass
class SMSMessage:
    """SMS message representation"""
    message_id: str
    phone_number: str
    content: str
    timestamp: datetime
    direction: str  # incoming, outgoing
    emotional_urgency: float = 0.0  # 0.0 to 1.0

@dataclass
class MusicContext:
    """Current music context"""
    currently_playing: Optional[str] = None
    playlist: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    volume: float = 0.5
    is_playing: bool = False

@dataclass
class UnifiedContext:
    """Unified context from all integrations"""
    calendar_events: List[CalendarEvent]
    recent_emails: List[EmailMessage]
    recent_messages: List[SMSMessage]
    music_context: MusicContext
    social_context: Dict[str, Any]
    health_context: Dict[str, Any]
    emotional_recommendations: List[str]
    last_updated: datetime

class CalendarIntegration:
    """Calendar integration service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authenticated_providers = set()
        
    async def authenticate(self, providers: List[str]) -> Dict[str, bool]:
        """Authenticate with calendar providers"""
        results = {}
        
        for provider in providers:
            try:
                # Simulate authentication
                if provider.lower() in ["google", "outlook", "apple"]:
                    results[provider] = True
                    self.authenticated_providers.add(provider.lower())
                    logger.info(f"Authenticated with {provider} Calendar")
                else:
                    results[provider] = False
                    
            except Exception as e:
                logger.error(f"Failed to authenticate with {provider}: {e}")
                results[provider] = False
        
        return results
    
    async def get_upcoming_events(self, days: int = 7) -> List[CalendarEvent]:
        """Get upcoming calendar events"""
        events = []
        
        try:
            # Simulate calendar API calls
            sample_events = [
                {
                    "id": "cal_001",
                    "title": "Team Meeting",
                    "description": "Weekly sync with the development team",
                    "start": datetime.now() + timedelta(hours=2),
                    "end": datetime.now() + timedelta(hours=3),
                    "location": "Conference Room A",
                    "attendees": ["team@company.com"]
                },
                {
                    "id": "cal_002",
                    "title": "Doctor Appointment",
                    "description": "Annual checkup",
                    "start": datetime.now() + timedelta(days=2),
                    "end": datetime.now() + timedelta(days=2, hours=1),
                    "location": "Medical Center",
                    "attendees": []
                }
            ]
            
            for event_data in sample_events:
                event = CalendarEvent(
                    event_id=event_data["id"],
                    title=event_data["title"],
                    description=event_data.get("description"),
                    start_time=event_data["start"],
                    end_time=event_data["end"],
                    location=event_data.get("location"),
                    attendees=event_data["attendees"],
                    reminder_minutes=15,
                    emotional_context=self._classify_event_emotion(event_data["title"])
                )
                events.append(event)
            
            logger.debug(f"Retrieved {len(events)} calendar events")
            return events
            
        except Exception as e:
            logger.error(f"Error getting calendar events: {e}")
            return []
    
    def _classify_event_emotion(self, title: str) -> str:
        """Classify emotional context of calendar event"""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ["meeting", "sync", "standup", "review"]):
            return "work"
        elif any(word in title_lower for word in ["doctor", "appointment", "medical", "dentist"]):
            return "health"
        elif any(word in title_lower for word in ["dinner", "lunch", "birthday", "party", "celebration"]):
            return "social"
        elif any(word in title_lower for word in ["workout", "gym", "exercise", "yoga"]):
            return "wellness"
        else:
            return "personal"

class EmailIntegration:
    """Email integration service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authenticated_providers = set()
        
    async def authenticate(self, providers: List[str]) -> Dict[str, bool]:
        """Authenticate with email providers"""
        results = {}
        
        for provider in providers:
            try:
                if provider.lower() in ["gmail", "outlook", "apple"]:
                    results[provider] = True
                    self.authenticated_providers.add(provider.lower())
                    logger.info(f"Authenticated with {provider} Email")
                else:
                    results[provider] = False
                    
            except Exception as e:
                logger.error(f"Failed to authenticate with {provider}: {e}")
                results[provider] = False
        
        return results
    
    async def get_recent_emails(self, limit: int = 10) -> List[EmailMessage]:
        """Get recent emails"""
        emails = []
        
        try:
            # Simulate email API calls
            sample_emails = [
                {
                    "id": "email_001",
                    "subject": "Project Update - Q4 Goals",
                    "sender": "manager@company.com",
                    "recipients": ["user@company.com"],
                    "timestamp": datetime.now() - timedelta(hours=1),
                    "content": "Hi there, wanted to update you on our Q4 progress...",
                    "importance": "normal"
                },
                {
                    "id": "email_002",
                    "subject": "Urgent: System Maintenance Tonight",
                    "sender": "admin@company.com",
                    "recipients": ["all@company.com"],
                    "timestamp": datetime.now() - timedelta(minutes=30),
                    "content": "Please be aware that we will be performing system maintenance...",
                    "importance": "high"
                }
            ]
            
            for email_data in sample_emails:
                email = EmailMessage(
                    message_id=email_data["id"],
                    subject=email_data["subject"],
                    sender=email_data["sender"],
                    recipients=email_data["recipients"],
                    timestamp=email_data["timestamp"],
                    content=email_data["content"],
                    importance=email_data["importance"],
                    emotional_tone=self._analyze_email_tone(email_data["subject"], email_data["content"])
                )
                emails.append(email)
            
            logger.debug(f"Retrieved {len(emails)} emails")
            return emails
            
        except Exception as e:
            logger.error(f"Error getting emails: {e}")
            return []
    
    def _analyze_email_tone(self, subject: str, content: str) -> str:
        """Analyze emotional tone of email"""
        text = (subject + " " + content).lower()
        
        if any(word in text for word in ["urgent", "asap", "immediate", "critical"]):
            return "urgent"
        elif any(word in text for word in ["thanks", "appreciate", "congratulations", "great"]):
            return "positive"
        elif any(word in text for word in ["issue", "problem", "error", "failure"]):
            return "concern"
        else:
            return "neutral"

class SMSIntegration:
    """SMS integration service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authenticated = False
        
    async def authenticate(self) -> bool:
        """Authenticate with SMS service"""
        try:
            # In real implementation, authenticate with Twilio or native SMS APIs
            self.authenticated = True
            logger.info("Authenticated with SMS service")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with SMS: {e}")
            return False
    
    async def get_recent_messages(self, limit: int = 10) -> List[SMSMessage]:
        """Get recent SMS messages"""
        messages = []
        
        try:
            # Simulate SMS API calls
            sample_messages = [
                {
                    "id": "sms_001",
                    "phone": "+1234567890",
                    "content": "Running a bit late for our meeting, be there in 10 minutes!",
                    "timestamp": datetime.now() - timedelta(minutes=15),
                    "direction": "incoming"
                },
                {
                    "id": "sms_002",
                    "phone": "+0987654321",
                    "content": "Thanks for the help today! Really appreciate it 😊",
                    "timestamp": datetime.now() - timedelta(hours=2),
                    "direction": "incoming"
                }
            ]
            
            for msg_data in sample_messages:
                message = SMSMessage(
                    message_id=msg_data["id"],
                    phone_number=msg_data["phone"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    direction=msg_data["direction"],
                    emotional_urgency=self._analyze_urgency(msg_data["content"])
                )
                messages.append(message)
            
            logger.debug(f"Retrieved {len(messages)} SMS messages")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting SMS messages: {e}")
            return []
    
    def _analyze_urgency(self, content: str) -> float:
        """Analyze urgency level of SMS content"""
        content_lower = content.lower()
        
        urgent_keywords = ["urgent", "asap", "emergency", "help", "now", "immediately"]
        moderate_keywords = ["soon", "quickly", "important", "need"]
        
        urgent_count = sum(1 for word in urgent_keywords if word in content_lower)
        moderate_count = sum(1 for word in moderate_keywords if word in content_lower)
        
        if urgent_count > 0:
            return min(1.0, urgent_count * 0.8)
        elif moderate_count > 0:
            return min(0.7, moderate_count * 0.3)
        else:
            return 0.2

class AppleMusicIntegration:
    """Apple Music integration service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.authenticated = False
        self.current_context = MusicContext()
        
    async def authenticate(self) -> bool:
        """Authenticate with Apple Music API"""
        try:
            # In real implementation, use Apple Music API or MusicKit
            self.authenticated = True
            logger.info("Authenticated with Apple Music")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Apple Music: {e}")
            return False
    
    async def get_music_context(self) -> MusicContext:
        """Get current music context"""
        try:
            # Simulate Apple Music API call
            self.current_context = MusicContext(
                currently_playing="Clair de Lune - Claude Debussy",
                playlist="Classical Focus",
                genre="Classical",
                mood="peaceful",
                volume=0.6,
                is_playing=True
            )
            
            return self.current_context
            
        except Exception as e:
            logger.error(f"Error getting music context: {e}")
            return MusicContext()
    
    async def control_playback(self, action: str, **kwargs) -> bool:
        """Control music playback"""
        try:
            if action == "play":
                self.current_context.is_playing = True
                logger.info("Music playback started")
            elif action == "pause":
                self.current_context.is_playing = False
                logger.info("Music playback paused")
            elif action == "volume":
                volume = kwargs.get("level", 0.5)
                self.current_context.volume = max(0.0, min(1.0, volume))
                logger.info(f"Volume set to {self.current_context.volume}")
            elif action == "play_mood":
                mood = kwargs.get("mood", "calm")
                await self._play_mood_based_music(mood)
                logger.info(f"Playing music for mood: {mood}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error controlling playback: {e}")
            return False
    
    async def _play_mood_based_music(self, mood: str):
        """Play music based on emotional mood"""
        mood_playlists = {
            "calm": "Peaceful Instrumentals",
            "energetic": "Upbeat Pop",
            "focused": "Concentration Classics",
            "sad": "Melancholic Melodies",
            "happy": "Feel Good Hits",
            "stressed": "Relaxation Sounds"
        }
        
        playlist = mood_playlists.get(mood, "Mixed Genres")
        self.current_context.playlist = playlist
        self.current_context.mood = mood
        self.current_context.is_playing = True

class UnifiedIntegrationService:
    """Main unified integration service"""
    
    def __init__(self, config_path: str = "data/unified_config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
        
        # Initialize integration services
        self.calendar = CalendarIntegration(self.config.get("calendar", {}))
        self.email = EmailIntegration(self.config.get("email", {}))
        self.sms = SMSIntegration(self.config.get("sms", {}))
        self.music = AppleMusicIntegration(self.config.get("music", {}))
        self.social = SocialMediaIntegration()
        self.health = EmotionalHealthIntegration()
        
        self.initialized_services = set()
        
    def load_config(self):
        """Load unified configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info("Loaded unified integration config")
            else:
                self.config = self._get_default_config()
                self.save_config()
        except Exception as e:
            logger.error(f"Error loading unified config: {e}")
            self.config = self._get_default_config()
    
    def save_config(self):
        """Save unified configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving unified config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default unified configuration"""
        return {
            "enabled_integrations": [],
            "emotional_awareness": True,
            "unified_persona": {
                "name": "AI Companion",
                "personality_traits": ["empathetic", "supportive", "intelligent", "adaptive"],
                "communication_style": "warm and understanding"
            },
            "calendar": {
                "providers": ["google", "outlook"],
                "sync_interval": 300
            },
            "email": {
                "providers": ["gmail", "outlook"],
                "check_interval": 600
            },
            "sms": {
                "provider": "native",
                "urgency_threshold": 0.7
            },
            "music": {
                "provider": "apple_music",
                "mood_based_control": True
            },
            "social": {
                "platforms": ["reddit", "twitter"],
                "privacy_mode": "read_only"
            },
            "health": {
                "metrics": ["heart_rate", "stress_level", "sleep_analysis"],
                "emotional_integration": True
            }
        }
    
    async def initialize_all(self) -> Dict[str, Dict[str, bool]]:
        """Initialize all integration services"""
        results = {}
        
        # Calendar
        if "calendar" in self.config.get("enabled_integrations", []):
            calendar_providers = self.config.get("calendar", {}).get("providers", [])
            results["calendar"] = await self.calendar.authenticate(calendar_providers)
            if any(results["calendar"].values()):
                self.initialized_services.add("calendar")
        
        # Email
        if "email" in self.config.get("enabled_integrations", []):
            email_providers = self.config.get("email", {}).get("providers", [])
            results["email"] = await self.email.authenticate(email_providers)
            if any(results["email"].values()):
                self.initialized_services.add("email")
        
        # SMS
        if "sms" in self.config.get("enabled_integrations", []):
            sms_success = await self.sms.authenticate()
            results["sms"] = {"sms": sms_success}
            if sms_success:
                self.initialized_services.add("sms")
        
        # Apple Music
        if "music" in self.config.get("enabled_integrations", []):
            music_success = await self.music.authenticate()
            results["music"] = {"apple_music": music_success}
            if music_success:
                self.initialized_services.add("music")
        
        # Social Media
        if "social" in self.config.get("enabled_integrations", []):
            social_platforms = self.config.get("social", {}).get("platforms", [])
            results["social"] = await self.social.initialize(social_platforms)
            if any(results["social"].values()):
                self.initialized_services.add("social")
        
        # Health
        if "health" in self.config.get("enabled_integrations", []):
            health_success = await self.health.initialize()
            results["health"] = {"healthkit": health_success}
            if health_success:
                self.initialized_services.add("health")
        
        logger.info(f"Initialized {len(self.initialized_services)} integration services")
        return results
    
    async def get_unified_context(self) -> UnifiedContext:
        """Get unified context from all integrations"""
        try:
            # Gather data from all initialized services
            calendar_events = []
            recent_emails = []
            recent_messages = []
            music_context = MusicContext()
            social_context = {}
            health_context = {}
            
            # Calendar
            if "calendar" in self.initialized_services:
                calendar_events = await self.calendar.get_upcoming_events()
            
            # Email
            if "email" in self.initialized_services:
                recent_emails = await self.email.get_recent_emails()
            
            # SMS
            if "sms" in self.initialized_services:
                recent_messages = await self.sms.get_recent_messages()
            
            # Music
            if "music" in self.initialized_services:
                music_context = await self.music.get_music_context()
            
            # Social Media
            if "social" in self.initialized_services:
                social_ctx = await self.social.get_social_context()
                social_context = asdict(social_ctx)
            
            # Health
            if "health" in self.initialized_services:
                health_context = await self.health.get_emotional_health_context()
            
            # Generate unified emotional recommendations
            emotional_recommendations = self._generate_unified_recommendations(
                calendar_events, recent_emails, recent_messages, music_context, 
                social_context, health_context
            )
            
            context = UnifiedContext(
                calendar_events=calendar_events,
                recent_emails=recent_emails,
                recent_messages=recent_messages,
                music_context=music_context,
                social_context=social_context,
                health_context=health_context,
                emotional_recommendations=emotional_recommendations,
                last_updated=datetime.now()
            )
            
            logger.info("Generated unified context from all integrations")
            return context
            
        except Exception as e:
            logger.error(f"Error getting unified context: {e}")
            return UnifiedContext(
                calendar_events=[],
                recent_emails=[],
                recent_messages=[],
                music_context=MusicContext(),
                social_context={},
                health_context={},
                emotional_recommendations=[],
                last_updated=datetime.now()
            )
    
    def _generate_unified_recommendations(self, calendar_events, emails, messages, 
                                        music_context, social_context, health_context) -> List[str]:
        """Generate emotional recommendations based on all available context"""
        recommendations = []
        
        try:
            # Calendar-based recommendations
            upcoming_meetings = [e for e in calendar_events if e.emotional_context == "work"]
            if len(upcoming_meetings) > 3:
                recommendations.append("You have several meetings coming up - would you like me to help you prepare or find some focus music?")
            
            # Email urgency recommendations
            urgent_emails = [e for e in emails if e.importance == "high"]
            if urgent_emails:
                recommendations.append(f"You have {len(urgent_emails)} urgent emails - would you like me to help prioritize your responses?")
            
            # SMS urgency recommendations
            urgent_messages = [m for m in messages if m.emotional_urgency > 0.7]
            if urgent_messages:
                recommendations.append("You have some urgent messages - it might be good to check those when you have a moment")
            
            # Health-based recommendations
            if health_context and "emotional_recommendations" in health_context:
                recommendations.extend(health_context["emotional_recommendations"][:2])
            
            # Social mood recommendations
            if social_context and "mood_indicators" in social_context:
                mood = social_context["mood_indicators"]
                if mood.get("negative", 0) > 0.6:
                    recommendations.append("I notice some challenging content in your social feeds - would you like to talk about anything or listen to some uplifting music?")
            
            # Music recommendations based on context
            if not music_context.is_playing:
                if len(upcoming_meetings) > 0:
                    recommendations.append("Since you have meetings coming up, would you like me to play some focus music?")
                elif health_context and health_context.get("health_context", {}).get("stress_level", 0) > 0.7:
                    recommendations.append("Your stress levels seem elevated - would some calming music help?")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating unified recommendations: {e}")
            return []
    
    async def execute_action(self, action_type: str, **kwargs) -> bool:
        """Execute an action across integrations"""
        try:
            if action_type == "play_mood_music":
                mood = kwargs.get("mood", "calm")
                if "music" in self.initialized_services:
                    return await self.music.control_playback("play_mood", mood=mood)
            
            elif action_type == "send_sms":
                # Would implement SMS sending capability
                logger.info("SMS sending not implemented in demo")
                return False
            
            elif action_type == "create_calendar_event":
                # Would implement calendar event creation
                logger.info("Calendar event creation not implemented in demo")
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {e}")
            return False
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all integration services"""
        return {
            "initialized_services": list(self.initialized_services),
            "total_available": len([
                "calendar", "email", "sms", "music", "social", "health"
            ]),
            "unified_persona": self.config.get("unified_persona", {}),
            "emotional_awareness": self.config.get("emotional_awareness", True),
            "last_config_update": datetime.now().isoformat()
        }

# Example usage and testing
async def demo_unified_integration():
    """Demonstrate unified integration service"""
    integration_service = UnifiedIntegrationService()
    
    print("=== Unified Integration Service Demo ===")
    
    # Set up enabled integrations for demo
    integration_service.config["enabled_integrations"] = [
        "calendar", "email", "sms", "music", "social", "health"
    ]
    
    # Initialize all services
    init_results = await integration_service.initialize_all()
    
    print("Service initialization results:")
    for service, results in init_results.items():
        print(f"\n{service.title()}:")
        for provider, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {provider}")
    
    # Get unified context
    context = await integration_service.get_unified_context()
    
    print(f"\nUnified Context Summary:")
    print(f"  📅 Calendar Events: {len(context.calendar_events)}")
    print(f"  📧 Recent Emails: {len(context.recent_emails)}")
    print(f"  💬 Recent Messages: {len(context.recent_messages)}")
    print(f"  🎵 Music Playing: {context.music_context.is_playing}")
    print(f"  📱 Social Topics: {len(context.social_context.get('active_topics', []))}")
    print(f"  🏥 Health Alerts: {len(context.health_context.get('health_context', {}).get('health_alerts', []))}")
    
    print(f"\nEmotional Recommendations:")
    for i, rec in enumerate(context.emotional_recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    # Test action execution
    print(f"\nTesting Actions:")
    music_result = await integration_service.execute_action("play_mood_music", mood="calm")
    print(f"  Play calm music: {'✓' if music_result else '✗'}")
    
    # Service status
    status = await integration_service.get_service_status()
    print(f"\nService Status: {status['initialized_services']}")
    print(f"Unified Persona: {status['unified_persona']['name']}")

if __name__ == "__main__":
    asyncio.run(demo_unified_integration())
