"""
Utility Assistant - Calendar, Tasks, and Gentle Reminders
Provides non-invasive support while maintaining emotional tone
"""

import json
import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class CalendarEvent:
    """A calendar event with emotional context"""
    title: str
    start_time: datetime
    end_time: datetime
    description: str = ""
    location: str = ""
    priority: str = "normal"  # low, normal, high, urgent
    emotional_tone: str = "neutral"  # excited, anxious, relaxed, important
    reminder_sent: bool = False
    created_at: float = 0.0

@dataclass
class Task:
    """A task with emotional awareness"""
    id: str
    title: str
    description: str = ""
    priority: str = "normal"
    status: str = "pending"  # pending, in_progress, completed, cancelled
    due_date: Optional[datetime] = None
    created_at: float = 0.0
    completed_at: Optional[float] = None
    emotional_weight: float = 0.0  # How emotionally significant this task is
    difficulty_level: str = "medium"  # easy, medium, hard
    estimated_duration: int = 30  # minutes
    context_tags: List[str] = None  # type: ignore

    def __post_init__(self):
        if self.context_tags is None:
            self.context_tags = []
        if self.created_at == 0.0:
            self.created_at = time.time()

@dataclass
class GentleReminder:
    """A emotionally-aware reminder"""
    content: str
    related_item: str  # task_id or event_title
    urgency: float  # 0.0 to 1.0
    tone: str  # gentle, encouraging, urgent, supportive
    suggested_response: str = ""
    created_at: float = 0.0

class UtilityAssistant:
    """
    Utility assistant for calendar, tasks, and gentle reminders
    Maintains emotional awareness while providing practical support
    """
    
    def __init__(self, data_path: str = "data/utility"):
        self.data_path = data_path
        self.logger = logging.getLogger(f"{__name__}")
        
        # Data storage
        self.calendar_events: List[CalendarEvent] = []
        self.tasks: Dict[str, Task] = {}
        self.personal_preferences: Dict[str, Any] = {}
        self.reminder_history: List[GentleReminder] = []
        
        # Settings
        self.reminder_style = "gentle"  # gentle, direct, encouraging
        self.time_awareness_level = "medium"  # low, medium, high
        self.task_check_frequency = 1800  # 30 minutes
        self.last_task_check = time.time()
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        # Load existing data
        self.load_data()
    
    def read_calendar(self, source: str, source_type: str = "file") -> List[CalendarEvent]:
        """
        Read calendar from various sources with emotional context analysis
        """
        events = []
        
        try:
            if source_type == "file":
                events = self._read_calendar_file(source)
            elif source_type == "json":
                events = self._read_json_calendar(source)
            else:
                self.logger.warning(f"Unsupported calendar source type: {source_type}")
                return events
            
            # Add emotional context to events
            for event in events:
                event.emotional_tone = self._analyze_event_emotion(event)
            
            self.calendar_events.extend(events)
            self.save_data()
            
            self.logger.info(f"Loaded {len(events)} calendar events from {source}")
            
        except Exception as e:
            self.logger.error(f"Error reading calendar from {source}: {e}")
        
        return events
    
    def read_task_list(self, source: str, source_type: str = "txt") -> List[Task]:
        """
        Read tasks from various sources (txt, markdown, Notion-style JSON)
        """
        tasks = []
        
        try:
            if source_type == "txt":
                tasks = self._read_txt_tasks(source)
            elif source_type == "markdown":
                tasks = self._read_markdown_tasks(source)
            elif source_type == "json":
                tasks = self._read_json_tasks(source)
            else:
                self.logger.warning(f"Unsupported task source type: {source_type}")
                return tasks
            
            # Add tasks to our system
            for task in tasks:
                self.tasks[task.id] = task
                # Analyze emotional weight
                task.emotional_weight = self._analyze_task_emotion(task)
            
            self.save_data()
            self.logger.info(f"Loaded {len(tasks)} tasks from {source}")
            
        except Exception as e:
            self.logger.error(f"Error reading tasks from {source}: {e}")
        
        return tasks
    
    def generate_gentle_reminders(self, user_mood: str = "neutral", 
                                 current_context: str = "") -> List[GentleReminder]:
        """
        Generate gentle, emotionally-aware reminders
        """
        reminders = []
        current_time = datetime.now()
        
        # Check upcoming calendar events
        for event in self.calendar_events:
            if not event.reminder_sent:
                time_until = (event.start_time - current_time).total_seconds()
                
                # Generate reminders based on timing and emotional context
                if 0 < time_until <= 3600:  # Next hour
                    reminder = self._create_event_reminder(event, user_mood, time_until)
                    if reminder:
                        reminders.append(reminder)
                        event.reminder_sent = True
        
        # Check overdue or important tasks
        for task in self.tasks.values():
            if task.status == "pending":
                reminder = self._create_task_reminder(task, user_mood, current_context)
                if reminder:
                    reminders.append(reminder)
        
        # Store reminder history
        self.reminder_history.extend(reminders)
        self.save_data()
        
        return reminders
    
    def check_time_focus(self, user_mood: str, time_since_last_activity: float) -> Optional[GentleReminder]:
        """
        Check if user might be distracted and offer gentle guidance
        """
        if user_mood == "distracted" and time_since_last_activity > 1800:  # 30 minutes
            # Find the most relevant current task
            active_task = self._find_current_context_task()
            
            if active_task:
                return self._create_focus_reminder(active_task, time_since_last_activity)
        
        return None
    
    def _read_calendar_file(self, filepath: str) -> List[CalendarEvent]:
        """Read simple calendar file format"""
        events = []
        
        if not os.path.exists(filepath):
            return events
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Simple format: "YYYY-MM-DD HH:MM | Title | Description"
                    parts = line.split('|')
                    if len(parts) >= 2:
                        date_time = datetime.strptime(parts[0].strip(), '%Y-%m-%d %H:%M')
                        title = parts[1].strip()
                        description = parts[2].strip() if len(parts) > 2 else ""
                        
                        event = CalendarEvent(
                            title=title,
                            start_time=date_time,
                            end_time=date_time + timedelta(hours=1),  # Default 1 hour
                            description=description
                        )
                        events.append(event)
                        
                except ValueError as e:
                    self.logger.warning(f"Could not parse calendar line {line_num}: {line}")
        
        return events
    
    def _read_json_calendar(self, filepath: str) -> List[CalendarEvent]:
        """Read JSON calendar format"""
        events = []
        
        if not os.path.exists(filepath):
            return events
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for event_data in data:
                    start_time = datetime.fromisoformat(event_data.get('start_time', datetime.now().isoformat()))
                    end_time = datetime.fromisoformat(event_data.get('end_time', (start_time + timedelta(hours=1)).isoformat()))
                    
                    event = CalendarEvent(
                        title=event_data.get('title', 'Untitled Event'),
                        start_time=start_time,
                        end_time=end_time,
                        description=event_data.get('description', ''),
                        location=event_data.get('location', ''),
                        priority=event_data.get('priority', 'normal')
                    )
                    events.append(event)
        
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error reading JSON calendar {filepath}: {e}")
        
        return events
    
    def _read_txt_tasks(self, filepath: str) -> List[Task]:
        """Read simple text task format"""
        tasks = []
        
        if not os.path.exists(filepath):
            return tasks
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse priority indicators
                priority = "normal"
                if line.startswith("!!!"):
                    priority = "urgent"
                    line = line[3:].strip()
                elif line.startswith("!!"):
                    priority = "high"
                    line = line[2:].strip()
                elif line.startswith("!"):
                    priority = "high"
                    line = line[1:].strip()
                
                # Parse completion status
                status = "pending"
                if line.startswith("[x]") or line.startswith("[X]"):
                    status = "completed"
                    line = line[3:].strip()
                elif line.startswith("[ ]"):
                    line = line[3:].strip()
                
                # Create task
                task_id = f"txt_{line_num}_{hash(line) % 10000}"
                task = Task(
                    id=task_id,
                    title=line,
                    priority=priority,
                    status=status
                )
                
                tasks.append(task)
        
        return tasks
    
    def _read_markdown_tasks(self, filepath: str) -> List[Task]:
        """Read markdown-style tasks"""
        tasks = []
        
        if not os.path.exists(filepath):
            return tasks
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find task patterns in markdown
        task_patterns = [
            r'- \[ \] (.+)',  # - [ ] Task
            r'- \[x\] (.+)',  # - [x] Completed task
            r'\* \[ \] (.+)', # * [ ] Task
            r'\* \[x\] (.+)', # * [x] Completed task
        ]
        
        for pattern in task_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                task_text = match.group(1).strip()
                status = "completed" if "[x]" in match.group(0).lower() else "pending"
                
                task_id = f"md_{hash(task_text) % 10000}"
                task = Task(
                    id=task_id,
                    title=task_text,
                    status=status
                )
                
                tasks.append(task)
        
        return tasks
    
    def _read_json_tasks(self, filepath: str) -> List[Task]:
        """Read JSON task format"""
        tasks = []
        
        if not os.path.exists(filepath):
            return tasks
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for task_data in data:
                    task = Task(
                        id=task_data.get('id', f"json_{hash(str(task_data)) % 10000}"),
                        title=task_data.get('title', 'Untitled Task'),
                        description=task_data.get('description', ''),
                        priority=task_data.get('priority', 'normal'),
                        status=task_data.get('status', 'pending'),
                        context_tags=task_data.get('tags', [])
                    )
                    
                    # Parse due date if present
                    if 'due_date' in task_data:
                        try:
                            task.due_date = datetime.fromisoformat(task_data['due_date'])
                        except ValueError:
                            pass
                    
                    tasks.append(task)
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filepath}: {e}")
        
        return tasks
    
    def _analyze_event_emotion(self, event: CalendarEvent) -> str:
        """Analyze emotional tone of calendar event"""
        title_lower = event.title.lower()
        description_lower = event.description.lower()
        
        # Keywords for different emotional tones
        excited_keywords = ['party', 'celebration', 'vacation', 'date', 'fun', 'exciting']
        anxious_keywords = ['exam', 'interview', 'presentation', 'deadline', 'urgent', 'important']
        relaxed_keywords = ['massage', 'spa', 'meditation', 'yoga', 'walk', 'leisure']
        
        text = title_lower + " " + description_lower
        
        if any(keyword in text for keyword in excited_keywords):
            return "excited"
        elif any(keyword in text for keyword in anxious_keywords):
            return "anxious"
        elif any(keyword in text for keyword in relaxed_keywords):
            return "relaxed"
        else:
            return "neutral"
    
    def _analyze_task_emotion(self, task: Task) -> float:
        """Analyze emotional weight of a task (0.0 to 1.0)"""
        weight = 0.0
        
        # Priority affects emotional weight
        priority_weights = {
            "low": 0.1,
            "normal": 0.3,
            "high": 0.6,
            "urgent": 0.9
        }
        weight += priority_weights.get(task.priority, 0.3)
        
        # Keywords that increase emotional weight
        stress_keywords = ['deadline', 'urgent', 'important', 'critical', 'asap']
        positive_keywords = ['fun', 'easy', 'simple', 'enjoy']
        
        text = (task.title + " " + task.description).lower()
        
        if any(keyword in text for keyword in stress_keywords):
            weight += 0.3
        elif any(keyword in text for keyword in positive_keywords):
            weight -= 0.2
        
        return max(0.0, min(1.0, weight))
    
    def _create_event_reminder(self, event: CalendarEvent, user_mood: str, 
                              time_until: float) -> Optional[GentleReminder]:
        """Create emotionally-aware event reminder"""
        if time_until <= 0:
            return None
        
        minutes_until = int(time_until / 60)
        
        # Tone-matched phrasing based on user mood and event emotion
        if user_mood == "stressed" or event.emotional_tone == "anxious":
            tone = "supportive"
            content = f"I wanted to gently remind you that '{event.title}' is coming up in {minutes_until} minutes. You've got this."
        elif user_mood == "relaxed" or event.emotional_tone == "relaxed":
            tone = "gentle"
            content = f"Just a soft reminder that '{event.title}' is in {minutes_until} minutes. No rush."
        elif event.emotional_tone == "excited":
            tone = "encouraging"
            content = f"Something lovely is coming up—'{event.title}' in {minutes_until} minutes!"
        else:
            tone = "gentle"
            content = f"'{event.title}' is in {minutes_until} minutes, when you're ready."
        
        return GentleReminder(
            content=content,
            related_item=event.title,
            urgency=min(1.0, 60.0 / max(1, minutes_until)),  # More urgent as time approaches
            tone=tone,
            created_at=time.time()
        )
    
    def _create_task_reminder(self, task: Task, user_mood: str, 
                             current_context: str) -> Optional[GentleReminder]:
        """Create gentle task reminder if appropriate"""
        # Don't overwhelm with reminders
        recent_reminders = [r for r in self.reminder_history 
                          if r.related_item == task.id and (time.time() - r.created_at) < 3600]
        if recent_reminders:
            return None
        
        # Check if task is overdue or high priority
        should_remind = False
        urgency = 0.0
        
        if task.due_date and task.due_date < datetime.now():
            should_remind = True
            urgency = 0.8
        elif task.priority in ["high", "urgent"]:
            should_remind = True
            urgency = 0.6
        elif task.emotional_weight > 0.7:
            should_remind = True
            urgency = task.emotional_weight
        
        if not should_remind:
            return None
        
        # Tone-matched reminder
        if user_mood == "distracted":
            tone = "gentle"
            content = f"When you have a moment, '{task.title}' is waiting for your attention."
        elif user_mood == "stressed":
            tone = "supportive"
            content = f"I notice '{task.title}' is on your list. Would it help to break it into smaller steps?"
        elif task.priority == "urgent":
            tone = "urgent"
            content = f"'{task.title}' seems time-sensitive. Would you like to tackle it now?"
        else:
            tone = "encouraging"
            content = f"'{task.title}' might be a good one to work on when you're ready."
        
        return GentleReminder(
            content=content,
            related_item=task.id,
            urgency=urgency,
            tone=tone,
            suggested_response="Would you like help breaking this down?",
            created_at=time.time()
        )
    
    def _find_current_context_task(self) -> Optional[Task]:
        """Find the most relevant task for current context"""
        pending_tasks = [t for t in self.tasks.values() if t.status == "pending"]
        
        if not pending_tasks:
            return None
        
        # Sort by priority and emotional weight
        sorted_tasks = sorted(pending_tasks, 
                            key=lambda t: (t.emotional_weight, 
                                         {"urgent": 4, "high": 3, "normal": 2, "low": 1}.get(t.priority, 1)), 
                            reverse=True)
        
        return sorted_tasks[0]
    
    def _create_focus_reminder(self, task: Task, time_lost: float) -> GentleReminder:
        """Create a gentle focus reminder"""
        minutes_lost = int(time_lost / 60)
        
        content = f"I noticed you've been away for {minutes_lost} minutes. Would you like to come back to '{task.title}'?"
        
        return GentleReminder(
            content=content,
            related_item=task.id,
            urgency=0.4,  # Gentle, not urgent
            tone="gentle",
            suggested_response="I'm here if you need help refocusing.",
            created_at=time.time()
        )
    
    def load_data(self):
        """Load utility data from files"""
        try:
            # Load tasks
            tasks_file = os.path.join(self.data_path, "tasks.json")
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data:
                        task = Task(**task_data)
                        self.tasks[task.id] = task
            
            # Load preferences
            prefs_file = os.path.join(self.data_path, "preferences.json")
            if os.path.exists(prefs_file):
                with open(prefs_file, 'r') as f:
                    self.personal_preferences = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading utility data: {e}")
    
    def save_data(self):
        """Save utility data to files"""
        try:
            # Save tasks
            tasks_file = os.path.join(self.data_path, "tasks.json")
            tasks_data = []
            for task in self.tasks.values():
                task_dict = {
                    'id': task.id,
                    'title': task.title,
                    'description': task.description,
                    'priority': task.priority,
                    'status': task.status,
                    'created_at': task.created_at,
                    'emotional_weight': task.emotional_weight,
                    'difficulty_level': task.difficulty_level,
                    'estimated_duration': task.estimated_duration,
                    'context_tags': task.context_tags
                }
                if task.due_date:
                    task_dict['due_date'] = task.due_date.isoformat()
                if task.completed_at:
                    task_dict['completed_at'] = task.completed_at
                    
                tasks_data.append(task_dict)
            
            with open(tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            # Save preferences
            prefs_file = os.path.join(self.data_path, "preferences.json")
            with open(prefs_file, 'w') as f:
                json.dump(self.personal_preferences, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving utility data: {e}")
    
    def get_utility_analytics(self) -> Dict[str, Any]:
        """Get analytics about utility usage"""
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
        pending_tasks = len([t for t in self.tasks.values() if t.status == "pending"])
        
        high_priority_tasks = len([t for t in self.tasks.values() 
                                 if t.priority in ["high", "urgent"] and t.status == "pending"])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "completion_rate": completed_tasks / max(1, total_tasks),
            "high_priority_pending": high_priority_tasks,
            "calendar_events": len(self.calendar_events),
            "reminders_sent": len(self.reminder_history),
            "average_task_emotional_weight": sum(t.emotional_weight for t in self.tasks.values()) / max(1, total_tasks)
        }

# Global utility assistant instance
utility_assistant = UtilityAssistant()

# Example usage and testing
if __name__ == "__main__":
    def test_utility_assistant():
        """Test the utility assistant system"""
        print("=== Testing Utility Assistant ===")
        
        assistant = UtilityAssistant()
        
        # Test adding some sample tasks
        print("\n1. Adding Sample Tasks:")
        sample_tasks = [
            Task(id="test1", title="Finish project report", priority="high", 
                 description="Complete the quarterly analysis"),
            Task(id="test2", title="Call dentist", priority="normal"),
            Task(id="test3", title="!!! Pay rent", priority="urgent")
        ]
        
        for task in sample_tasks:
            assistant.tasks[task.id] = task
            print(f"  Added: {task.title} (priority: {task.priority})")
        
        # Test reminder generation
        print("\n2. Testing Reminder Generation:")
        reminders = assistant.generate_gentle_reminders(user_mood="stressed")
        print(f"  Generated {len(reminders)} reminders")
        
        for reminder in reminders:
            print(f"    {reminder.tone}: {reminder.content}")
        
        # Test focus check
        print("\n3. Testing Focus Check:")
        focus_reminder = assistant.check_time_focus("distracted", 2000)  # 33+ minutes
        if focus_reminder:
            print(f"  Focus reminder: {focus_reminder.content}")
        else:
            print("  No focus reminder needed")
        
        # Test analytics
        print("\n4. Testing Analytics:")
        analytics = assistant.get_utility_analytics()
        print(f"  Total tasks: {analytics['total_tasks']}")
        print(f"  Pending tasks: {analytics['pending_tasks']}")
        print(f"  High priority pending: {analytics['high_priority_pending']}")
        print(f"  Average emotional weight: {analytics['average_task_emotional_weight']:.2f}")
        
        print("\n=== Utility Assistant Test Complete ===")
    
    test_utility_assistant()
