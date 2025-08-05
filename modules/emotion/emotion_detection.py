"""
Emotion Detection for Unified Companion
Analyzes text for emotional content and user state
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class EmotionDetector:
    """
    Detects emotions and emotional patterns in user messages
    Provides context for companion responses
    """
    
    def __init__(self):
        self.emotion_keywords = {
            "happy": [
                "happy", "joy", "excited", "great", "wonderful", "amazing", 
                "fantastic", "excellent", "love", "awesome", "thrilled",
                "delighted", "cheerful", "glad", "pleased", "content"
            ],
            "sad": [
                "sad", "depressed", "down", "unhappy", "miserable", "upset",
                "disappointed", "heartbroken", "grief", "sorrow", "blue",
                "melancholy", "gloomy", "dejected", "despondent"
            ],
            "angry": [
                "angry", "mad", "furious", "irritated", "annoyed", "frustrated",
                "rage", "outraged", "livid", "irate", "pissed", "enraged",
                "aggravated", "incensed", "indignant"
            ],
            "anxious": [
                "anxious", "worried", "nervous", "stressed", "panic", "fear",
                "scared", "terrified", "overwhelmed", "uneasy", "tense",
                "apprehensive", "restless", "agitated", "concerned"
            ],
            "confused": [
                "confused", "lost", "puzzled", "bewildered", "perplexed",
                "uncertain", "unclear", "don't understand", "mixed up",
                "baffled", "mystified", "stumped"
            ],
            "curious": [
                "curious", "interested", "intrigued", "wondering", "fascinated",
                "eager", "keen", "inquisitive", "want to know", "how does",
                "why", "what if", "tell me more"
            ],
            "grateful": [
                "grateful", "thankful", "appreciate", "thanks", "blessed",
                "fortunate", "lucky", "indebted", "obliged"
            ],
            "lonely": [
                "lonely", "alone", "isolated", "solitary", "abandoned",
                "friendless", "disconnected", "empty", "hollow"
            ],
            "excited": [
                "excited", "pumped", "energetic", "enthusiastic", "eager",
                "thrilled", "ecstatic", "elated", "hyper", "buzzing"
            ],
            "calm": [
                "calm", "peaceful", "relaxed", "serene", "tranquil",
                "composed", "centered", "balanced", "zen", "chill"
            ]
        }
        
        self.intensity_modifiers = {
            "very": 1.5,
            "extremely": 2.0,
            "incredibly": 2.0,
            "really": 1.3,
            "so": 1.4,
            "quite": 1.2,
            "pretty": 1.1,
            "somewhat": 0.8,
            "a bit": 0.7,
            "slightly": 0.6,
            "kind of": 0.8,
            "sort of": 0.8
        }
        
        # Positive/negative sentiment indicators
        self.positive_indicators = [
            "!", "😊", "😄", "😃", "🙂", "❤️", "💕", "✨", "🎉",
            "amazing", "great", "wonderful", "perfect", "love"
        ]
        
        self.negative_indicators = [
            "😞", "😢", "😭", "💔", "😔", "terrible", "awful", 
            "horrible", "hate", "worst", "sucks"
        ]
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for emotional content
        Returns emotion classification with confidence and intensity
        """
        
        if not text or not text.strip():
            return {
                "primary_emotion": "neutral",
                "confidence": 0.5,
                "intensity": 0.5,
                "emotions": {"neutral": 0.5},
                "sentiment": "neutral"
            }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        # Score each emotion category
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            
            for keyword in keywords:
                if keyword in text_lower:
                    base_score = 1.0
                    
                    # Check for intensity modifiers before the keyword
                    words = text_lower.split()
                    for i, word in enumerate(words):
                        if keyword in word:
                            # Look for modifiers in previous words
                            for j in range(max(0, i-2), i):
                                if words[j] in self.intensity_modifiers:
                                    base_score *= self.intensity_modifiers[words[j]]
                    
                    score += base_score
            
            if score > 0:
                emotion_scores[emotion] = min(score, 1.0)  # Cap at 1.0
        
        # If no emotions detected, analyze sentiment
        if not emotion_scores:
            sentiment_score = self._analyze_sentiment(text)
            if sentiment_score > 0.6:
                emotion_scores["happy"] = sentiment_score
            elif sentiment_score < 0.4:
                emotion_scores["sad"] = 1 - sentiment_score
            else:
                emotion_scores["neutral"] = 0.5
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            primary_name = primary_emotion[0]
            primary_score = primary_emotion[1]
        else:
            primary_name = "neutral"
            primary_score = 0.5
            emotion_scores["neutral"] = 0.5
        
        # Calculate overall sentiment
        sentiment = self._calculate_sentiment(emotion_scores)
        
        # Calculate intensity based on text features
        intensity = self._calculate_intensity(text, primary_score)
        
        return {
            "primary_emotion": primary_name,
            "confidence": primary_score,
            "intensity": intensity,
            "emotions": emotion_scores,
            "sentiment": sentiment,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive_count = sum(1 for indicator in self.positive_indicators if indicator in text.lower())
        negative_count = sum(1 for indicator in self.negative_indicators if indicator in text.lower())
        
        total_indicators = positive_count + negative_count
        if total_indicators == 0:
            return 0.5  # Neutral
        
        return positive_count / total_indicators
    
    def _calculate_sentiment(self, emotion_scores: Dict[str, float]) -> str:
        """Calculate overall sentiment from emotion scores"""
        positive_emotions = ["happy", "excited", "grateful", "curious", "calm"]
        negative_emotions = ["sad", "angry", "anxious", "lonely", "confused"]
        
        positive_score = sum(emotion_scores.get(emotion, 0) for emotion in positive_emotions)
        negative_score = sum(emotion_scores.get(emotion, 0) for emotion in negative_emotions)
        
        if positive_score > negative_score * 1.2:
            return "positive"
        elif negative_score > positive_score * 1.2:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_intensity(self, text: str, emotion_score: float) -> float:
        """Calculate emotional intensity based on text features"""
        base_intensity = emotion_score
        
        # Adjust based on text features
        if "!" in text:
            base_intensity += 0.1 * text.count("!")
        
        if text.isupper():
            base_intensity += 0.2
        
        # Multiple question marks indicate higher intensity
        if "??" in text:
            base_intensity += 0.1
        
        # Repetition indicates intensity
        words = text.lower().split()
        repeated_words = len(words) - len(set(words))
        if repeated_words > 0:
            base_intensity += 0.1
        
        return min(base_intensity, 1.0)  # Cap at 1.0
    
    async def analyze_conversation_pattern(self, recent_messages: List[str]) -> Dict[str, Any]:
        """Analyze emotional patterns across recent conversation"""
        
        if not recent_messages:
            return {"pattern": "no_data", "trend": "stable"}
        
        emotions_over_time = []
        
        for message in recent_messages:
            emotion_data = await self.analyze_text(message)
            emotions_over_time.append({
                "emotion": emotion_data["primary_emotion"],
                "sentiment": emotion_data["sentiment"],
                "intensity": emotion_data["intensity"]
            })
        
        # Analyze trends
        sentiments = [e["sentiment"] for e in emotions_over_time]
        intensities = [e["intensity"] for e in emotions_over_time]
        
        # Simple trend analysis
        if len(sentiments) >= 3:
            recent_sentiment = sentiments[-1]
            earlier_sentiment = sentiments[0]
            
            if recent_sentiment == "positive" and earlier_sentiment != "positive":
                trend = "improving"
            elif recent_sentiment == "negative" and earlier_sentiment != "negative":
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        avg_intensity = sum(intensities) / len(intensities) if intensities else 0.5
        
        # Detect conversation patterns
        emotion_names = [e["emotion"] for e in emotions_over_time]
        most_common_emotion = max(set(emotion_names), key=emotion_names.count) if emotion_names else "neutral"
        
        pattern_analysis = {
            "dominant_emotion": most_common_emotion,
            "trend": trend,
            "average_intensity": avg_intensity,
            "conversation_length": len(recent_messages),
            "emotional_stability": self._calculate_stability(emotions_over_time)
        }
        
        return pattern_analysis
    
    def _calculate_stability(self, emotions_over_time: List[Dict[str, Any]]) -> str:
        """Calculate emotional stability in conversation"""
        if len(emotions_over_time) < 3:
            return "insufficient_data"
        
        sentiments = [e["sentiment"] for e in emotions_over_time]
        unique_sentiments = len(set(sentiments))
        
        if unique_sentiments == 1:
            return "very_stable"
        elif unique_sentiments == 2:
            return "stable"
        else:
            return "variable"
    
    async def get_supportive_response_suggestions(self, emotion_data: Dict[str, Any]) -> List[str]:
        """Get suggestions for supportive responses based on detected emotion"""
        
        primary_emotion = emotion_data["primary_emotion"]
        intensity = emotion_data["intensity"]
        
        suggestions = {
            "sad": [
                "I can hear that you're going through something difficult. I'm here to listen.",
                "It sounds like you're feeling down. Would you like to talk about what's happening?",
                "I'm sorry you're feeling this way. Sometimes it helps to share what's on your mind."
            ],
            "anxious": [
                "I notice you seem worried about something. Would it help to talk through what's concerning you?",
                "Anxiety can be really tough. I'm here to listen and support you through this.",
                "It sounds like you're feeling stressed. Let's take this one step at a time."
            ],
            "angry": [
                "I can sense your frustration. That sounds really difficult to deal with.",
                "It sounds like something has really upset you. I'm here to listen.",
                "Your feelings are completely valid. Would you like to talk about what happened?"
            ],
            "confused": [
                "It sounds like you're trying to work through something complex. I'm happy to help you think it through.",
                "I can help you break this down into smaller pieces if that would be helpful.",
                "Sometimes talking through confusion can help clarify things. What's on your mind?"
            ],
            "happy": [
                "I love hearing the joy in your message! What's making you so happy?",
                "Your excitement is contagious! I'd love to hear more about this.",
                "It's wonderful that you're feeling so positive! Tell me more!"
            ],
            "excited": [
                "Your enthusiasm is amazing! I'm excited to hear about this too!",
                "I can feel your energy! What's got you so excited?",
                "This sounds like something really special! I'd love to know more."
            ]
        }
        
        return suggestions.get(primary_emotion, [
            "I'm here to listen and support you.",
            "Thank you for sharing that with me.",
            "How are you feeling about everything right now?"
        ])
