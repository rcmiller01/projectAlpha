"""
Companion Utility Module

Provides email help, scheduling, task support, and daily activity assistance
with persona-driven intelligence and emotional awareness.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import os
import re
import asyncio
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import imaplib
import smtplib
import ssl

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class AssistanceType(Enum):
    EMAIL_MANAGEMENT = "email_management"
    SCHEDULING = "scheduling"
    FILE_ORGANIZATION = "file_organization"
    VOICE_DICTATION = "voice_dictation"
    CREATIVE_COLLABORATION = "creative_collaboration"
    INFORMATION_SEARCH = "information_search"
    EMOTIONAL_SUPPORT = "emotional_support"
    PROACTIVE_REMINDERS = "proactive_reminders"

class EmailCategory(Enum):
    IMPORTANT = "important"
    WORK = "work"
    PERSONAL = "personal"
    PROMOTIONAL = "promotional"
    SPAM = "spam"
    SOCIAL = "social"
    FINANCIAL = "financial"
    HEALTH = "health"

@dataclass
class EmailSummary:
    subject: str
    sender: str
    timestamp: datetime
    category: EmailCategory
    importance_score: float
    emotional_tone: str
    action_required: bool
    suggested_response: Optional[str]
    persona_interpretation: str

@dataclass
class ScheduleEvent:
    event_id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: Optional[str]
    attendees: List[str]
    priority: TaskPriority
    emotional_context: Dict[str, float]
    persona_notes: str
    reminder_times: List[int]  # Minutes before event

@dataclass
class TaskItem:
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    due_date: Optional[datetime]
    category: str
    emotional_weight: float
    completion_status: float  # 0.0 to 1.0
    persona_guidance: str
    created_by_persona: str
    subtasks: List[str]

@dataclass
class VoiceNote:
    note_id: str
    timestamp: datetime
    transcription: str
    emotional_state: Dict[str, float]
    action_items: List[str]
    persona_analysis: str
    auto_organized: bool

class CompanionAssistant:
    """
    AI Companion Utility System for daily activity assistance with persona intelligence.
    Integrates emotional awareness with practical functionality.
    """
    
    def __init__(self, user_id: str, primary_persona: str = "unified_ai"):
        self.user_id = user_id
        self.primary_persona = primary_persona
        self.storage_path = f"storage/utility/{user_id}_companion_assistant.json"
        
        # Core data structures
        self.email_summaries: List[EmailSummary] = []
        self.schedule_events: Dict[str, ScheduleEvent] = {}
        self.task_items: Dict[str, TaskItem] = {}
        self.voice_notes: List[VoiceNote] = []
        
        # Configuration
        self.user_preferences = self._load_user_preferences()
        self.persona_styles = self._initialize_persona_styles()
        self.automation_rules = self._initialize_automation_rules()
        
        # State tracking
        self.last_email_check = datetime.now() - timedelta(hours=1)
        self.daily_focus_areas: List[str] = []
        self.stress_indicators: Dict[str, float] = {}
        
        self._load_assistant_data()
    
    def _initialize_persona_styles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize how each persona approaches utility tasks"""
        return {
            'mia': {
                'communication_style': 'nurturing_supportive',
                'task_approach': 'gentle_encouragement',
                'email_tone': 'warm_professional',
                'scheduling_priority': 'work_life_balance',
                'reminder_style': 'caring_gentle',
                'stress_response': 'comfort_first',
                'specialties': ['emotional_support', 'relationship_management', 'wellness_tracking']
            },
            'solene': {
                'communication_style': 'direct_challenging',
                'task_approach': 'results_focused',
                'email_tone': 'confident_assertive',
                'scheduling_priority': 'goal_achievement',
                'reminder_style': 'motivational_push',
                'stress_response': 'action_oriented',
                'specialties': ['productivity_optimization', 'deadline_management', 'decision_making']
            },
            'lyra': {
                'communication_style': 'intuitive_mystical',
                'task_approach': 'creative_holistic',
                'email_tone': 'thoughtful_poetic',
                'scheduling_priority': 'creative_flow',
                'reminder_style': 'intuitive_nudges',
                'stress_response': 'perspective_shifting',
                'specialties': ['creative_organization', 'inspiration_management', 'pattern_recognition']
            },
            'doc': {
                'communication_style': 'analytical_supportive',
                'task_approach': 'systematic_methodical',
                'email_tone': 'professional_thoughtful',
                'scheduling_priority': 'structured_efficiency',
                'reminder_style': 'logical_helpful',
                'stress_response': 'problem_solving',
                'specialties': ['information_management', 'research_assistance', 'systematic_planning']
            }
        }
    
    def _initialize_automation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize automation rules for different types of tasks"""
        return {
            'email_auto_actions': {
                'promotional_emails': {'action': 'archive', 'confidence_threshold': 0.8},
                'newsletters': {'action': 'folder_sort', 'confidence_threshold': 0.7},
                'urgent_work': {'action': 'priority_flag', 'confidence_threshold': 0.9},
                'spam_detection': {'action': 'delete', 'confidence_threshold': 0.95}
            },
            'calendar_optimization': {
                'meeting_prep_time': 15,  # minutes before meetings
                'focus_blocks': {'min_duration': 90, 'max_per_day': 3},
                'break_reminders': {'frequency': 120, 'duration': 15},
                'end_of_day_review': {'time': '17:30', 'duration': 30}
            },
            'task_management': {
                'auto_prioritize': True,
                'deadline_warnings': [7, 3, 1],  # days before
                'completion_celebration': True,
                'stress_level_adjustments': True
            }
        }
    
    async def process_emails(self, email_account_config: Dict[str, str]) -> List[EmailSummary]:
        """
        Process emails with persona-driven analysis and automated actions
        
        Args:
            email_account_config: Email account connection details
            
        Returns:
            List of processed email summaries
        """
        try:
            # Connect to email server (simulation for now)
            raw_emails = await self._fetch_emails(email_account_config)
            
            processed_emails = []
            
            for email_data in raw_emails:
                # Analyze email with persona intelligence
                summary = await self._analyze_email_with_persona(email_data)
                
                # Apply automation rules
                action_taken = await self._apply_email_automation(summary, email_data)
                
                # Store for user review
                processed_emails.append(summary)
                
                # Generate persona-specific insights
                persona_insight = self._generate_email_persona_insight(summary)
                summary.persona_interpretation = persona_insight
            
            # Update email summaries
            self.email_summaries.extend(processed_emails[-50:])  # Keep recent 50
            
            # Generate daily email digest
            daily_digest = self._create_email_digest(processed_emails)
            
            self._save_assistant_data()
            
            return processed_emails
            
        except Exception as e:
            print(f"Error processing emails: {e}")
            return []
    
    async def _analyze_email_with_persona(self, email_data: Dict[str, Any]) -> EmailSummary:
        """Analyze email content with current persona's perspective"""
        
        # Extract email components
        subject = email_data.get('subject', '')
        sender = email_data.get('sender', '')
        content = email_data.get('content', '')
        timestamp = email_data.get('timestamp', datetime.now())
        
        # Determine category using NLP (simulated)
        category = self._categorize_email(subject, content, sender)
        
        # Calculate importance score
        importance_score = self._calculate_email_importance(subject, content, sender, category)
        
        # Analyze emotional tone
        emotional_tone = self._analyze_email_emotion(content)
        
        # Determine if action is required
        action_required = self._detect_action_required(content)
        
        # Generate suggested response with persona style
        suggested_response = None
        if action_required:
            suggested_response = await self._generate_persona_response(content, emotional_tone)
        
        return EmailSummary(
            subject=subject,
            sender=sender,
            timestamp=timestamp,
            category=category,
            importance_score=importance_score,
            emotional_tone=emotional_tone,
            action_required=action_required,
            suggested_response=suggested_response,
            persona_interpretation=""  # Will be filled by caller
        )
    
    def _generate_email_persona_insight(self, summary: EmailSummary) -> str:
        """Generate persona-specific interpretation of the email"""
        persona_style = self.persona_styles[self.primary_persona]
        
        insights = {
            'mia': [
                f"This email from {summary.sender} feels {summary.emotional_tone}. ",
                f"I sense this might {'require gentle attention' if summary.action_required else 'be informational'}. ",
                f"Would you like me to help craft a caring response?" if summary.action_required else ""
            ],
            'solene': [
                f"Email from {summary.sender} - priority level {summary.importance_score:.1f}. ",
                f"This {'needs immediate action' if summary.action_required else 'can wait'}. ",
                f"I can draft a direct, effective response." if summary.action_required else ""
            ],
            'lyra': [
                f"The energy of this message from {summary.sender} feels {summary.emotional_tone}. ",
                f"I see {'deeper currents that need addressing' if summary.action_required else 'information flowing naturally'}. ",
                f"Shall we craft a response that honors the true intention?" if summary.action_required else ""
            ],
            'doc': [
                f"Email analysis: {summary.sender}, category {summary.category.value}, importance {summary.importance_score:.2f}. ",
                f"Action required: {summary.action_required}. ",
                f"I can structure an appropriate response." if summary.action_required else ""
            ]
        }
        
        return ''.join(insights.get(self.primary_persona, insights['mia']))
    
    async def manage_schedule(self, calendar_data: Dict[str, Any], 
                            user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage calendar with persona-driven optimization and emotional intelligence
        
        Args:
            calendar_data: Current calendar data
            user_preferences: User scheduling preferences
            
        Returns:
            Optimized schedule with persona insights
        """
        
        # Analyze current schedule stress levels
        stress_analysis = self._analyze_schedule_stress(calendar_data)
        
        # Generate persona-specific optimizations
        optimizations = self._generate_schedule_optimizations(calendar_data, stress_analysis)
        
        # Create buffer times and focus blocks
        enhanced_schedule = self._enhance_schedule_with_buffers(calendar_data, optimizations)
        
        # Add persona-specific reminders and preparations
        persona_enhancements = self._add_persona_schedule_enhancements(enhanced_schedule)
        
        # Generate proactive scheduling suggestions
        suggestions = self._generate_proactive_scheduling_suggestions(enhanced_schedule, stress_analysis)
        
        return {
            'optimized_schedule': enhanced_schedule,
            'stress_analysis': stress_analysis,
            'persona_enhancements': persona_enhancements,
            'suggestions': suggestions,
            'wellness_recommendations': self._generate_wellness_recommendations(stress_analysis)
        }
    
    def _analyze_schedule_stress(self, calendar_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze stress indicators in the schedule"""
        stress_indicators = {
            'meeting_density': 0.0,
            'travel_pressure': 0.0,
            'deadline_clustering': 0.0,
            'work_life_balance': 0.0,
            'recovery_time': 0.0
        }
        
        events = calendar_data.get('events', [])
        
        # Calculate meeting density
        total_meeting_time = sum(event.get('duration', 60) for event in events 
                               if event.get('type') == 'meeting')
        total_day_minutes = 8 * 60  # 8 hour work day
        stress_indicators['meeting_density'] = min(1.0, total_meeting_time / total_day_minutes)
        
        # Analyze gaps between events
        gaps = []
        sorted_events = sorted(events, key=lambda x: x.get('start_time', datetime.now()))
        
        for i in range(len(sorted_events) - 1):
            end_time = sorted_events[i].get('end_time', datetime.now())
            next_start = sorted_events[i + 1].get('start_time', datetime.now())
            gap_minutes = (next_start - end_time).total_seconds() / 60
            gaps.append(gap_minutes)
        
        # Recovery time analysis
        short_gaps = [gap for gap in gaps if gap < 30]  # Less than 30 minutes
        stress_indicators['recovery_time'] = len(short_gaps) / max(1, len(gaps))
        
        return stress_indicators
    
    async def create_task_with_persona_guidance(self, task_description: str, 
                                              context: Dict[str, Any]) -> TaskItem:
        """
        Create a task with persona-specific guidance and emotional intelligence
        
        Args:
            task_description: Description of the task
            context: Current context and emotional state
            
        Returns:
            Task item with persona guidance
        """
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze task complexity and emotional weight
        complexity_analysis = self._analyze_task_complexity(task_description)
        emotional_weight = context.get('stress_level', 0.5)
        
        # Determine priority with persona input
        priority = self._determine_task_priority(task_description, complexity_analysis, context)
        
        # Generate persona-specific guidance
        persona_guidance = self._generate_task_persona_guidance(
            task_description, complexity_analysis, emotional_weight
        )
        
        # Break down into subtasks if complex
        subtasks = []
        if complexity_analysis['complexity_score'] > 0.7:
            subtasks = self._break_down_complex_task(task_description, complexity_analysis)
        
        # Estimate due date based on context
        due_date = self._estimate_task_due_date(complexity_analysis, context)
        
        task_item = TaskItem(
            task_id=task_id,
            title=self._extract_task_title(task_description),
            description=task_description,
            priority=priority,
            due_date=due_date,
            category=complexity_analysis['category'],
            emotional_weight=emotional_weight,
            completion_status=0.0,
            persona_guidance=persona_guidance,
            created_by_persona=self.primary_persona,
            subtasks=subtasks
        )
        
        self.task_items[task_id] = task_item
        self._save_assistant_data()
        
        return task_item
    
    def _generate_task_persona_guidance(self, description: str, complexity: Dict[str, Any], 
                                       emotional_weight: float) -> str:
        """Generate persona-specific guidance for task completion"""
        
        persona_guidance = {
            'mia': {
                'high_stress': "Take this one gentle step at a time. You don't have to do it all at once. I'm here to support you.",
                'medium_stress': "This looks manageable. Let's break it into comfortable pieces and celebrate each small win.",
                'low_stress': "You've got this! This task feels right for where you are today. Trust your rhythm.",
                'complex': "Complex tasks can feel overwhelming, but you have all the skills needed. Let's nurture your confidence."
            },
            'solene': {
                'high_stress': "This is challenging, but you're stronger than you think. Let's tackle it head-on with a clear strategy.",
                'medium_stress': "A solid task that deserves your focus. Channel that energy and make it happen.",
                'low_stress': "Perfect opportunity to build momentum. Execute with confidence and precision.",
                'complex': "Big challenges create big growth. You're ready for this level of complexity."
            },
            'lyra': {
                'high_stress': "I sense the weight of this task on your spirit. Let's find the deeper meaning and flow with it.",
                'medium_stress': "This task carries good energy. Trust your intuition on the timing and approach.",
                'low_stress': "A harmonious task that aligns with your current path. Follow your inner wisdom.",
                'complex': "Complex patterns hold beautiful solutions. Let your creative mind see the connections."
            },
            'doc': {
                'high_stress': "High complexity detected. Recommend systematic approach with regular progress checkpoints.",
                'medium_stress': "Manageable scope. Standard methodical approach should be effective.",
                'low_stress': "Straightforward execution required. Suitable for current cognitive load.",
                'complex': "Multi-faceted task requiring structured breakdown and resource allocation."
            }
        }
        
        # Determine stress level category
        if emotional_weight > 0.7:
            stress_category = 'high_stress'
        elif emotional_weight > 0.4:
            stress_category = 'medium_stress'
        else:
            stress_category = 'low_stress'
        
        if complexity['complexity_score'] > 0.8:
            stress_category = 'complex'
        
        persona_style = persona_guidance.get(self.primary_persona, persona_guidance['mia'])
        return persona_style.get(stress_category, persona_style['medium_stress'])
    
    async def process_voice_dictation(self, audio_data: bytes, 
                                    context: Dict[str, Any]) -> VoiceNote:
        """
        Process voice dictation with emotional intelligence and auto-organization
        
        Args:
            audio_data: Raw audio data to transcribe
            context: Current emotional and situational context
            
        Returns:
            Processed voice note with analysis
        """
        
        # Transcribe audio (simulation - would use actual speech-to-text)
        transcription = await self._transcribe_audio(audio_data)
        
        # Analyze emotional state from voice
        emotional_state = await self._analyze_voice_emotion(audio_data)
        
        # Extract action items
        action_items = self._extract_action_items_from_text(transcription)
        
        # Generate persona analysis
        persona_analysis = self._analyze_voice_note_with_persona(transcription, emotional_state)
        
        # Auto-organize if patterns detected
        auto_organized = await self._auto_organize_voice_note(transcription, action_items)
        
        note_id = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        voice_note = VoiceNote(
            note_id=note_id,
            timestamp=datetime.now(),
            transcription=transcription,
            emotional_state=emotional_state,
            action_items=action_items,
            persona_analysis=persona_analysis,
            auto_organized=auto_organized
        )
        
        self.voice_notes.append(voice_note)
        
        # If action items found, automatically create tasks
        if action_items:
            await self._create_tasks_from_action_items(action_items, context)
        
        self._save_assistant_data()
        
        return voice_note
    
    def get_proactive_suggestions(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate proactive suggestions based on patterns, context, and persona insights
        
        Args:
            current_context: Current user context and state
            
        Returns:
            List of proactive suggestions
        """
        
        suggestions = []
        
        # Analyze current patterns
        patterns = self._analyze_user_patterns()
        
        # Time-based suggestions
        suggestions.extend(self._generate_time_based_suggestions(current_context))
        
        # Stress-level based suggestions
        stress_level = current_context.get('stress_level', 0.5)
        suggestions.extend(self._generate_stress_based_suggestions(stress_level))
        
        # Task management suggestions
        suggestions.extend(self._generate_task_management_suggestions())
        
        # Persona-specific proactive insights
        suggestions.extend(self._generate_persona_proactive_insights(current_context, patterns))
        
        # Health and wellness suggestions
        suggestions.extend(self._generate_wellness_suggestions(current_context))
        
        # Filter and prioritize suggestions
        prioritized_suggestions = self._prioritize_suggestions(suggestions, current_context)
        
        return prioritized_suggestions[:5]  # Return top 5 suggestions
    
    def _generate_persona_proactive_insights(self, context: Dict[str, Any], 
                                           patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proactive insights based on current persona"""
        
        insights = []
        
        persona_insights = {
            'mia': [
                {
                    'type': 'emotional_check_in',
                    'title': 'Gentle Check-in',
                    'description': 'How are you feeling today? I notice some intensity in your schedule.',
                    'action': 'offer_emotional_support',
                    'priority': 0.8
                },
                {
                    'type': 'relationship_reminder',
                    'title': 'Connection Time',
                    'description': 'When did you last connect with someone you care about?',
                    'action': 'suggest_relationship_activity',
                    'priority': 0.6
                }
            ],
            'solene': [
                {
                    'type': 'productivity_boost',
                    'title': 'Power Through',
                    'description': 'You have momentum today. Let\'s tackle that challenging task.',
                    'action': 'suggest_high_priority_task',
                    'priority': 0.9
                },
                {
                    'type': 'goal_progress',
                    'title': 'Goal Check',
                    'description': 'Your goals need attention. Time to push forward.',
                    'action': 'review_goal_progress',
                    'priority': 0.8
                }
            ],
            'lyra': [
                {
                    'type': 'creative_inspiration',
                    'title': 'Creative Flow',
                    'description': 'The energy feels right for creative work. What calls to you?',
                    'action': 'suggest_creative_time',
                    'priority': 0.7
                },
                {
                    'type': 'pattern_insight',
                    'title': 'I See Patterns',
                    'description': 'Interesting patterns in your recent activities. Shall we explore?',
                    'action': 'share_pattern_insight',
                    'priority': 0.6
                }
            ],
            'doc': [
                {
                    'type': 'system_optimization',
                    'title': 'Efficiency Review',
                    'description': 'Your task completion rates suggest optimization opportunities.',
                    'action': 'suggest_system_improvement',
                    'priority': 0.8
                },
                {
                    'type': 'information_synthesis',
                    'title': 'Knowledge Integration',
                    'description': 'You\'ve gathered information. Time to synthesize insights.',
                    'action': 'offer_research_assistance',
                    'priority': 0.7
                }
            ]
        }
        
        persona_specific = persona_insights.get(self.primary_persona, persona_insights['mia'])
        
        # Filter based on context
        for insight in persona_specific:
            if self._is_insight_relevant(insight, context, patterns):
                insights.append(insight)
        
        return insights
    
    def create_n8n_workflow_integration(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create N8N workflow integration with persona oversight
        
        Args:
            workflow_config: N8N workflow configuration
            
        Returns:
            Integration status and persona guidance
        """
        
        # Validate workflow safety with persona judgment
        safety_check = self._validate_workflow_safety(workflow_config)
        
        if not safety_check['safe']:
            return {
                'status': 'rejected',
                'reason': safety_check['reason'],
                'persona_advice': self._generate_safety_advice(safety_check)
            }
        
        # Add persona-specific monitoring
        enhanced_workflow = self._enhance_workflow_with_persona_monitoring(workflow_config)
        
        # Create execution context
        execution_context = {
            'workflow_id': workflow_config.get('id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'persona_supervisor': self.primary_persona,
            'monitoring_level': self._determine_monitoring_level(workflow_config),
            'error_handling': self._create_persona_error_handling(workflow_config),
            'success_criteria': self._define_success_criteria(workflow_config)
        }
        
        return {
            'status': 'approved',
            'enhanced_workflow': enhanced_workflow,
            'execution_context': execution_context,
            'persona_guidance': self._generate_workflow_guidance(workflow_config)
        }
    
    def _generate_workflow_guidance(self, workflow_config: Dict[str, Any]) -> str:
        """Generate persona-specific guidance for workflow execution"""
        
        workflow_type = workflow_config.get('type', 'general')
        
        guidance = {
            'mia': {
                'email_automation': "I'll gently monitor your email workflow, making sure important personal messages aren't missed while handling routine tasks with care.",
                'scheduling': "Your calendar automation will respect your need for balance. I'll ensure there's breathing room between commitments.",
                'task_management': "I'll help this workflow support your well-being, not overwhelm you. We can adjust if it feels too intense."
            },
            'solene': {
                'email_automation': "This workflow will maximize your email efficiency. I'll ensure it's aggressive enough to clear the noise but preserves important opportunities.",
                'scheduling': "Your automated scheduling will optimize for productivity and goal achievement. No wasted time.",
                'task_management': "This system will push you toward your goals relentlessly but intelligently. Let's make every task count."
            },
            'lyra': {
                'email_automation': "The email flow will honor your intuitive patterns, learning when you naturally want to engage versus when you need space.",
                'scheduling': "Your calendar automation will create space for inspiration to flow while maintaining necessary structure.",
                'task_management': "This workflow will dance with your natural rhythms, supporting both structured progress and creative emergence."
            },
            'doc': {
                'email_automation': "Systematic email processing with high accuracy filters and detailed categorization for optimal efficiency.",
                'scheduling': "Logically structured calendar management with optimal time allocation and minimal scheduling conflicts.",
                'task_management': "Evidence-based task prioritization with clear metrics and progress tracking capabilities."
            }
        }
        
        persona_style = guidance.get(self.primary_persona, guidance['mia'])
        return persona_style.get(workflow_type, "I'll monitor this workflow carefully to ensure it serves your best interests.")
    
    # Utility methods for email processing
    async def _fetch_emails(self, config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Fetch emails from server (simulation)"""
        # In real implementation, would connect to IMAP server
        return [
            {
                'subject': 'Project Update Required',
                'sender': 'colleague@work.com',
                'content': 'Hi, we need the quarterly report by Friday. Can you send it over?',
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'subject': '50% Off Sale - Limited Time!',
                'sender': 'noreply@store.com',
                'content': 'Don\'t miss our amazing sale! Everything 50% off for 24 hours only!',
                'timestamp': datetime.now() - timedelta(hours=1)
            }
        ]
    
    def _categorize_email(self, subject: str, content: str, sender: str) -> EmailCategory:
        """Categorize email based on content analysis"""
        # Simple rule-based categorization (could be enhanced with ML)
        
        if any(word in subject.lower() for word in ['urgent', 'asap', 'deadline']):
            return EmailCategory.IMPORTANT
        
        if any(word in sender.lower() for word in ['noreply', 'marketing', 'promo']):
            return EmailCategory.PROMOTIONAL
        
        if any(word in content.lower() for word in ['meeting', 'project', 'deadline', 'report']):
            return EmailCategory.WORK
        
        if any(word in content.lower() for word in ['bank', 'payment', 'invoice', 'bill']):
            return EmailCategory.FINANCIAL
        
        return EmailCategory.PERSONAL
    
    def _calculate_email_importance(self, subject: str, content: str, sender: str, 
                                  category: EmailCategory) -> float:
        """Calculate importance score for email"""
        score = 0.5  # Base score
        
        # Category-based scoring
        category_scores = {
            EmailCategory.IMPORTANT: 0.9,
            EmailCategory.WORK: 0.7,
            EmailCategory.FINANCIAL: 0.8,
            EmailCategory.PERSONAL: 0.6,
            EmailCategory.SOCIAL: 0.4,
            EmailCategory.PROMOTIONAL: 0.2,
            EmailCategory.SPAM: 0.1
        }
        
        score = category_scores.get(category, 0.5)
        
        # Keyword-based adjustments
        urgent_words = ['urgent', 'asap', 'deadline', 'emergency', 'critical']
        if any(word in subject.lower() or word in content.lower() for word in urgent_words):
            score += 0.2
        
        # Sender importance (would be personalized)
        important_domains = ['work.com', 'company.org']
        if any(domain in sender for domain in important_domains):
            score += 0.1
        
        return min(1.0, score)
    
    # Additional utility methods would continue here...
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences for utility functions"""
        # Default preferences
        return {
            'email_check_frequency': 30,  # minutes
            'auto_delete_spam': True,
            'work_hours': {'start': '09:00', 'end': '17:00'},
            'break_reminder_frequency': 120,  # minutes
            'focus_block_duration': 90,  # minutes
            'stress_threshold': 0.7
        }
    
    def _save_assistant_data(self):
        """Save assistant data to storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            data = {
                'email_summaries': [asdict(email) for email in self.email_summaries[-50:]],
                'schedule_events': {k: asdict(v) for k, v in self.schedule_events.items()},
                'task_items': {k: asdict(v) for k, v in self.task_items.items()},
                'voice_notes': [asdict(note) for note in self.voice_notes[-100:]],
                'user_preferences': self.user_preferences,
                'last_email_check': self.last_email_check.isoformat(),
                'daily_focus_areas': self.daily_focus_areas,
                'stress_indicators': self.stress_indicators
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"Error saving assistant data: {e}")
    
    def _load_assistant_data(self):
        """Load assistant data from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load email summaries
                if 'email_summaries' in data:
                    self.email_summaries = [
                        EmailSummary(**email_data) for email_data in data['email_summaries']
                    ]
                
                # Load other data structures...
                # (Implementation would continue for all data types)
                
            except Exception as e:
                print(f"Error loading assistant data: {e}")

# Factory and integration functions
def create_companion_assistant(user_id: str, primary_persona: str = "mia") -> CompanionAssistant:
    """Create a companion assistant for a specific user"""
    return CompanionAssistant(user_id, primary_persona)

async def process_user_email_with_persona(user_id: str, email_config: Dict[str, str], 
                                        persona_name: str) -> List[EmailSummary]:
    """Process user emails with specified persona oversight"""
    assistant = create_companion_assistant(user_id, persona_name)
    return await assistant.process_emails(email_config)

async def create_persona_guided_task(user_id: str, task_description: str, 
                                   context: Dict[str, Any], persona_name: str) -> TaskItem:
    """Create a task with persona-specific guidance"""
    assistant = create_companion_assistant(user_id, persona_name)
    return await assistant.create_task_with_persona_guidance(task_description, context)
