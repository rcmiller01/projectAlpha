# relationship_ai.py
# Phase 3: Advanced relationship AI for intelligent advice and conflict resolution

import json
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class RelationshipIssueType(Enum):
    COMMUNICATION = "communication"
    TRUST = "trust"
    INTIMACY = "intimacy"
    CONFLICT = "conflict"
    EMOTIONAL_DISTANCE = "emotional_distance"
    EXPECTATIONS = "expectations"
    TIME_TOGETHER = "time_together"
    PHYSICAL_AFFECTION = "physical_affection"
    FUTURE_PLANS = "future_plans"
    PERSONAL_GROWTH = "personal_growth"

class AdviceType(Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PREVENTIVE = "preventive"
    EMERGENCY = "emergency"

@dataclass
class RelationshipIssue:
    issue_type: RelationshipIssueType
    severity: float  # 0.0 to 1.0
    description: str
    detected_at: datetime
    context: Dict
    resolved: bool = False
    resolution_date: Optional[datetime] = None

@dataclass
class RelationshipAdvice:
    advice_type: AdviceType
    title: str
    description: str
    steps: List[str]
    expected_outcome: str
    time_frame: str
    priority: int  # 1-5, 5 being highest

class RelationshipAI:
    def __init__(self):
        self.issues_history = []
        self.advice_given = []
        self.relationship_metrics = {}
        self.conflict_patterns = {}
        self.emotional_trends = []
        
        # Load relationship knowledge base
        self.knowledge_base = self._load_knowledge_base()
        self.advice_templates = self._load_advice_templates()
        self.conflict_resolution_strategies = self._load_conflict_strategies()
        
        # Relationship health indicators
        self.health_indicators = {
            "communication_quality": 0.8,
            "emotional_intimacy": 0.7,
            "trust_level": 0.9,
            "conflict_resolution": 0.6,
            "shared_goals": 0.8,
            "physical_affection": 0.7,
            "quality_time": 0.6,
            "future_alignment": 0.8
        }
    
    def _load_knowledge_base(self) -> Dict:
        """Load relationship psychology knowledge base"""
        return {
            "communication_patterns": {
                "healthy": ["active_listening", "i_statements", "validation", "compromise"],
                "unhealthy": ["stonewalling", "criticism", "defensiveness", "contempt"]
            },
            "emotional_needs": {
                "security": ["reassurance", "consistency", "reliability"],
                "connection": ["quality_time", "physical_touch", "shared_activities"],
                "appreciation": ["recognition", "gratitude", "acknowledgment"],
                "growth": ["support", "encouragement", "space_for_development"]
            },
            "conflict_styles": {
                "collaborative": "Working together to find mutually beneficial solutions",
                "compromising": "Finding middle ground through give and take",
                "accommodating": "Prioritizing partner's needs over own",
                "avoiding": "Stepping back from conflict to prevent escalation",
                "competing": "Asserting own position strongly"
            },
            "relationship_stages": {
                "honeymoon": "Initial intense attraction and idealization",
                "power_struggle": "Differences emerge, conflicts arise",
                "stability": "Finding balance and routine",
                "commitment": "Deep commitment and long-term planning",
                "co_creation": "Building life together with shared purpose"
            }
        }
    
    def _load_advice_templates(self) -> Dict[RelationshipIssueType, List[Dict]]:
        """Load advice templates for different relationship issues"""
        return {
            RelationshipIssueType.COMMUNICATION: [
                {
                    "title": "Improve Active Listening",
                    "description": "Practice truly hearing and understanding your partner",
                    "steps": [
                        "Maintain eye contact and open body language",
                        "Listen without interrupting or planning your response",
                        "Reflect back what you heard to confirm understanding",
                        "Ask clarifying questions when needed",
                        "Validate your partner's feelings"
                    ],
                    "expected_outcome": "Better understanding and reduced misunderstandings",
                    "time_frame": "2-4 weeks of consistent practice"
                },
                {
                    "title": "Use 'I' Statements",
                    "description": "Express your feelings without blaming your partner",
                    "steps": [
                        "Start sentences with 'I feel' instead of 'You make me'",
                        "Describe the specific behavior that affects you",
                        "Explain the impact on your feelings",
                        "Request specific changes or actions",
                        "Be open to your partner's perspective"
                    ],
                    "expected_outcome": "Reduced defensiveness and more productive conversations",
                    "time_frame": "Immediate implementation"
                }
            ],
            RelationshipIssueType.TRUST: [
                {
                    "title": "Rebuild Trust Through Consistency",
                    "description": "Demonstrate reliability through consistent actions",
                    "steps": [
                        "Keep all promises, no matter how small",
                        "Be transparent about your actions and decisions",
                        "Follow through on commitments consistently",
                        "Communicate openly about concerns or changes",
                        "Give your partner space to express their feelings"
                    ],
                    "expected_outcome": "Gradual restoration of trust and security",
                    "time_frame": "3-6 months of consistent behavior"
                }
            ],
            RelationshipIssueType.INTIMACY: [
                {
                    "title": "Deepen Emotional Connection",
                    "description": "Create opportunities for deeper emotional sharing",
                    "steps": [
                        "Schedule regular 'check-in' conversations",
                        "Share your thoughts, dreams, and fears",
                        "Ask meaningful questions about your partner's inner world",
                        "Create shared experiences and memories",
                        "Practice vulnerability and emotional openness"
                    ],
                    "expected_outcome": "Stronger emotional bond and deeper understanding",
                    "time_frame": "Ongoing practice with noticeable results in 1-2 months"
                }
            ],
            RelationshipIssueType.CONFLICT: [
                {
                    "title": "Implement Time-Out Strategy",
                    "description": "Take breaks during heated arguments to prevent escalation",
                    "steps": [
                        "Agree on a signal for when either partner needs a break",
                        "Set a specific time to resume the conversation (15-30 minutes)",
                        "Use the break to calm down and reflect",
                        "Return to the conversation with a calmer perspective",
                        "Focus on finding solutions rather than assigning blame"
                    ],
                    "expected_outcome": "More productive conflict resolution and reduced hurt feelings",
                    "time_frame": "Immediate implementation"
                }
            ]
        }
    
    def _load_conflict_strategies(self) -> Dict[str, Dict]:
        """Load conflict resolution strategies"""
        return {
            "de_escalation": {
                "name": "De-escalation Techniques",
                "techniques": [
                    "Take deep breaths together",
                    "Use calming language and tone",
                    "Acknowledge each other's feelings",
                    "Find common ground or shared concerns",
                    "Agree to pause and return later"
                ]
            },
            "problem_solving": {
                "name": "Collaborative Problem Solving",
                "techniques": [
                    "Define the problem clearly together",
                    "Brainstorm multiple solutions",
                    "Evaluate options based on both partners' needs",
                    "Choose a solution that works for both",
                    "Implement and review the solution"
                ]
            },
            "emotional_validation": {
                "name": "Emotional Validation",
                "techniques": [
                    "Acknowledge your partner's feelings as valid",
                    "Show empathy and understanding",
                    "Avoid minimizing or dismissing emotions",
                    "Express care and concern",
                    "Offer comfort and support"
                ]
            }
        }
    
    def analyze_relationship_health(self) -> Dict:
        """Analyze overall relationship health"""
        # Calculate overall health score
        total_score = sum(self.health_indicators.values())
        average_score = total_score / len(self.health_indicators)
        
        # Identify areas needing attention
        areas_of_concern = [
            area for area, score in self.health_indicators.items()
            if score < 0.7
        ]
        
        # Generate health assessment
        if average_score >= 0.8:
            health_status = "excellent"
            summary = "Your relationship is in excellent health with strong foundations."
        elif average_score >= 0.7:
            health_status = "good"
            summary = "Your relationship is generally healthy with some areas for improvement."
        elif average_score >= 0.6:
            health_status = "fair"
            summary = "Your relationship needs attention in several areas."
        else:
            health_status = "concerning"
            summary = "Your relationship requires immediate attention and support."
        
        return {
            "overall_score": average_score,
            "health_status": health_status,
            "summary": summary,
            "indicators": self.health_indicators,
            "areas_of_concern": areas_of_concern,
            "recommendations": self._generate_health_recommendations(areas_of_concern)
        }
    
    def _generate_health_recommendations(self, areas_of_concern: List[str]) -> List[str]:
        """Generate recommendations for improving relationship health"""
        recommendations = []
        
        for area in areas_of_concern:
            if area == "communication_quality":
                recommendations.append("Practice active listening and use 'I' statements")
            elif area == "emotional_intimacy":
                recommendations.append("Schedule regular deep conversations and share feelings")
            elif area == "conflict_resolution":
                recommendations.append("Learn and practice de-escalation techniques")
            elif area == "physical_affection":
                recommendations.append("Increase non-sexual touch and physical closeness")
            elif area == "quality_time":
                recommendations.append("Plan regular date nights and shared activities")
        
        return recommendations
    
    def detect_relationship_issues(self, conversation_data: Dict, emotional_data: Dict) -> List[RelationshipIssue]:
        """Detect potential relationship issues from conversation and emotional data"""
        issues = []
        
        # Analyze communication patterns
        if self._detect_communication_issues(conversation_data):
            issues.append(RelationshipIssue(
                issue_type=RelationshipIssueType.COMMUNICATION,
                severity=0.6,
                description="Communication patterns suggest potential misunderstandings",
                detected_at=datetime.now(),
                context=conversation_data
            ))
        
        # Analyze emotional distance
        if self._detect_emotional_distance(emotional_data):
            issues.append(RelationshipIssue(
                issue_type=RelationshipIssueType.EMOTIONAL_DISTANCE,
                severity=0.5,
                description="Signs of emotional distance detected",
                detected_at=datetime.now(),
                context=emotional_data
            ))
        
        # Analyze conflict patterns
        if self._detect_conflict_patterns(conversation_data):
            issues.append(RelationshipIssue(
                issue_type=RelationshipIssueType.CONFLICT,
                severity=0.7,
                description="Recurring conflict patterns identified",
                detected_at=datetime.now(),
                context=conversation_data
            ))
        
        # Store issues
        self.issues_history.extend(issues)
        
        return issues
    
    def _detect_communication_issues(self, conversation_data: Dict) -> bool:
        """Detect communication issues in conversation data"""
        # Analyze for unhealthy communication patterns
        text = conversation_data.get("text", "").lower()
        
        unhealthy_patterns = [
            "you always", "you never", "you make me",
            "it's your fault", "you're wrong", "you don't understand"
        ]
        
        return any(pattern in text for pattern in unhealthy_patterns)
    
    def _detect_emotional_distance(self, emotional_data: Dict) -> bool:
        """Detect emotional distance from emotional data"""
        # Check for signs of emotional withdrawal
        emotions = emotional_data.get("emotions", {})
        
        withdrawal_indicators = [
            emotions.get("withdrawn", 0) > 0.5,
            emotions.get("distant", 0) > 0.5,
            emotions.get("disconnected", 0) > 0.5
        ]
        
        return any(withdrawal_indicators)
    
    def _detect_conflict_patterns(self, conversation_data: Dict) -> bool:
        """Detect recurring conflict patterns"""
        # Analyze for conflict indicators
        text = conversation_data.get("text", "").lower()
        
        conflict_indicators = [
            "argument", "fight", "disagree", "upset", "angry",
            "frustrated", "annoyed", "problem", "issue"
        ]
        
        return any(indicator in text for indicator in conflict_indicators)
    
    def generate_relationship_advice(self, issue_type: Optional[RelationshipIssueType] = None) -> RelationshipAdvice:
        """Generate personalized relationship advice"""
        if issue_type is None:
            # Analyze current relationship state and suggest most relevant advice
            health_analysis = self.analyze_relationship_health()
            if health_analysis["areas_of_concern"]:
                # Focus on the most concerning area
                most_concerning = min(
                    health_analysis["areas_of_concern"],
                    key=lambda x: self.health_indicators.get(x, 1.0)
                )
                
                # Map health indicators to issue types
                issue_mapping = {
                    "communication_quality": RelationshipIssueType.COMMUNICATION,
                    "emotional_intimacy": RelationshipIssueType.INTIMACY,
                    "conflict_resolution": RelationshipIssueType.CONFLICT,
                    "trust_level": RelationshipIssueType.TRUST
                }
                
                issue_type = issue_mapping.get(most_concerning, RelationshipIssueType.COMMUNICATION)
        
        # Get advice template
        templates = self.advice_templates.get(issue_type, [])
        if not templates:
            # Fallback to general communication advice
            templates = self.advice_templates[RelationshipIssueType.COMMUNICATION]
        
        # Select appropriate template
        template = random.choice(templates)
        
        # Create advice object
        advice = RelationshipAdvice(
            advice_type=AdviceType.SHORT_TERM,
            title=template["title"],
            description=template["description"],
            steps=template["steps"],
            expected_outcome=template["expected_outcome"],
            time_frame=template["time_frame"],
            priority=3
        )
        
        # Store advice given
        self.advice_given.append({
            "advice": advice,
            "given_at": datetime.now(),
            "issue_type": issue_type.value if issue_type else "general"
        })
        
        return advice
    
    def provide_conflict_resolution(self, conflict_type: str, severity: float) -> Dict:
        """Provide specific conflict resolution guidance"""
        strategies = []
        
        if severity > 0.8:
            # High conflict - focus on de-escalation
            strategies.append(self.conflict_resolution_strategies["de_escalation"])
            advice_type = "emergency"
        elif severity > 0.5:
            # Medium conflict - focus on problem solving
            strategies.append(self.conflict_resolution_strategies["problem_solving"])
            advice_type = "immediate"
        else:
            # Low conflict - focus on emotional validation
            strategies.append(self.conflict_resolution_strategies["emotional_validation"])
            advice_type = "preventive"
        
        return {
            "conflict_type": conflict_type,
            "severity": severity,
            "advice_type": advice_type,
            "strategies": strategies,
            "immediate_actions": self._get_immediate_actions(severity),
            "long_term_approaches": self._get_long_term_approaches(conflict_type)
        }
    
    def _get_immediate_actions(self, severity: float) -> List[str]:
        """Get immediate actions for conflict resolution"""
        if severity > 0.8:
            return [
                "Take a 20-minute break to calm down",
                "Use deep breathing exercises",
                "Write down your feelings before speaking",
                "Agree to return to the conversation when both are calm"
            ]
        elif severity > 0.5:
            return [
                "Acknowledge each other's feelings",
                "Take turns speaking without interruption",
                "Focus on the specific issue, not past conflicts",
                "Look for areas of agreement"
            ]
        else:
            return [
                "Express appreciation for your partner",
                "Share positive feelings and experiences",
                "Plan a pleasant activity together",
                "Practice active listening"
            ]
    
    def _get_long_term_approaches(self, conflict_type: str) -> List[str]:
        """Get long-term approaches for preventing similar conflicts"""
        approaches = [
            "Schedule regular relationship check-ins",
            "Learn and practice communication skills together",
            "Establish clear boundaries and expectations",
            "Develop shared goals and values"
        ]
        
        if conflict_type == "communication":
            approaches.extend([
                "Consider couples counseling or relationship coaching",
                "Read relationship books together",
                "Practice active listening exercises"
            ])
        elif conflict_type == "trust":
            approaches.extend([
                "Build trust through consistent actions",
                "Be transparent about important decisions",
                "Follow through on all commitments"
            ])
        
        return approaches
    
    def track_relationship_progress(self, metric: str, value: float):
        """Track relationship progress over time"""
        if metric not in self.relationship_metrics:
            self.relationship_metrics[metric] = []
        
        self.relationship_metrics[metric].append({
            "value": value,
            "timestamp": datetime.now()
        })
        
        # Update health indicators if applicable
        if metric in self.health_indicators:
            self.health_indicators[metric] = value
    
    def get_relationship_insights(self) -> Dict:
        """Get comprehensive relationship insights"""
        health_analysis = self.analyze_relationship_health()
        
        # Analyze trends
        trends = self._analyze_trends()
        
        # Get recent advice
        recent_advice = self.advice_given[-5:] if self.advice_given else []
        
        # Get unresolved issues
        unresolved_issues = [
            issue for issue in self.issues_history
            if not issue.resolved
        ]
        
        return {
            "health_analysis": health_analysis,
            "trends": trends,
            "recent_advice": recent_advice,
            "unresolved_issues": unresolved_issues,
            "relationship_strengths": self._identify_strengths(),
            "growth_opportunities": self._identify_growth_opportunities(),
            "recommendations": self._generate_comprehensive_recommendations()
        }
    
    def _analyze_trends(self) -> Dict:
        """Analyze relationship trends over time"""
        trends = {}
        
        for metric, values in self.relationship_metrics.items():
            if len(values) >= 2:
                recent_values = values[-5:]  # Last 5 measurements
                if len(recent_values) >= 2:
                    first_value = recent_values[0]["value"]
                    last_value = recent_values[-1]["value"]
                    change = last_value - first_value
                    
                    trends[metric] = {
                        "change": change,
                        "direction": "improving" if change > 0 else "declining" if change < 0 else "stable",
                        "magnitude": abs(change)
                    }
        
        return trends
    
    def _identify_strengths(self) -> List[str]:
        """Identify relationship strengths"""
        strengths = []
        
        for indicator, value in self.health_indicators.items():
            if value >= 0.8:
                if indicator == "communication_quality":
                    strengths.append("Strong communication skills")
                elif indicator == "trust_level":
                    strengths.append("High level of trust")
                elif indicator == "shared_goals":
                    strengths.append("Aligned future goals")
                elif indicator == "future_alignment":
                    strengths.append("Shared vision for the future")
        
        return strengths
    
    def _identify_growth_opportunities(self) -> List[str]:
        """Identify areas for relationship growth"""
        opportunities = []
        
        for indicator, value in self.health_indicators.items():
            if value < 0.7:
                if indicator == "conflict_resolution":
                    opportunities.append("Improve conflict resolution skills")
                elif indicator == "quality_time":
                    opportunities.append("Increase quality time together")
                elif indicator == "physical_affection":
                    opportunities.append("Enhance physical affection")
                elif indicator == "emotional_intimacy":
                    opportunities.append("Deepen emotional connection")
        
        return opportunities
    
    def _generate_comprehensive_recommendations(self) -> List[Dict]:
        """Generate comprehensive relationship recommendations"""
        recommendations = []
        
        # Immediate actions
        health_analysis = self.analyze_relationship_health()
        if health_analysis["areas_of_concern"]:
            recommendations.append({
                "type": "immediate",
                "title": "Address Areas of Concern",
                "description": "Focus on improving the identified areas of concern",
                "actions": health_analysis["recommendations"]
            })
        
        # Long-term growth
        growth_opportunities = self._identify_growth_opportunities()
        if growth_opportunities:
            recommendations.append({
                "type": "long_term",
                "title": "Relationship Growth Plan",
                "description": "Develop a plan for ongoing relationship improvement",
                "actions": growth_opportunities
            })
        
        # Preventive measures
        recommendations.append({
            "type": "preventive",
            "title": "Maintain Relationship Health",
            "description": "Continue practices that strengthen your relationship",
            "actions": [
                "Regular relationship check-ins",
                "Quality time together",
                "Open and honest communication",
                "Physical affection and intimacy",
                "Shared activities and experiences"
            ]
        })
        
        return recommendations

# Global relationship AI instance
relationship_ai = RelationshipAI()

def get_relationship_ai() -> RelationshipAI:
    """Get the global relationship AI instance"""
    return relationship_ai

def analyze_relationship_health() -> Dict:
    """Analyze relationship health"""
    return relationship_ai.analyze_relationship_health()

def generate_relationship_advice(issue_type: Optional[str] = None) -> Dict:
    """Generate relationship advice"""
    if issue_type:
        try:
            issue_enum = RelationshipIssueType(issue_type)
            advice = relationship_ai.generate_relationship_advice(issue_enum)
        except:
            advice = relationship_ai.generate_relationship_advice()
    else:
        advice = relationship_ai.generate_relationship_advice()
    
    return {
        "title": advice.title,
        "description": advice.description,
        "steps": advice.steps,
        "expected_outcome": advice.expected_outcome,
        "time_frame": advice.time_frame,
        "priority": advice.priority
    }

def provide_conflict_resolution(conflict_type: str, severity: float) -> Dict:
    """Provide conflict resolution guidance"""
    return relationship_ai.provide_conflict_resolution(conflict_type, severity)

def get_relationship_insights() -> Dict:
    """Get comprehensive relationship insights"""
    return relationship_ai.get_relationship_insights() 