#!/usr/bin/env python3
"""
Emotion Tracker for Quantization Analysis
Tracks tone, sentiment, and metaphor density in AI responses
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    nltk_available = True
except ImportError:
    nltk_available = False
    logging.warning("NLTK not available - using basic sentiment analysis")

try:
    import textstat
    textstat_available = True
except ImportError:
    textstat_available = False
    logging.warning("textstat not available - using basic readability measures")

logger = logging.getLogger(__name__)

@dataclass
class EmotionMetrics:
    """Container for emotional analysis metrics"""
    emotion_score: float
    sentiment_score: float
    metaphor_density: float
    tone_indicators: Dict[str, float]
    empathy_markers: List[str]
    emotional_vocabulary: List[str]
    response_length: int
    readability_score: float

class EmotionTracker:
    """Advanced emotion tracking and analysis system"""
    
    def __init__(self):
        self.setup_nltk()
        self.load_emotional_lexicons()
        self.initialize_analyzers()
        
        logger.info("🧠 Emotion Tracker initialized")
    
    def setup_nltk(self):
        """Download required NLTK data"""
        if not nltk_available:
            return
            
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logger.info("✅ NLTK data downloaded")
        except Exception as e:
            logger.warning(f"⚠️ NLTK setup failed: {e}")
    
    def load_emotional_lexicons(self):
        """Load emotional vocabulary and patterns"""
        
        # Emotional vocabulary categories
        self.emotional_vocab = {
            'compassion': [
                'understand', 'feel', 'care', 'support', 'comfort', 'gentle', 'warm',
                'embrace', 'cherish', 'tender', 'kindness', 'empathy', 'compassion',
                'nurture', 'soothe', 'healing', 'holding', 'presence'
            ],
            'validation': [
                'valid', 'legitimate', 'understandable', 'normal', 'human', 'okay',
                'natural', 'reasonable', 'makes sense', 'relate', 'recognize',
                'acknowledge', 'honor', 'respect'
            ],
            'encouragement': [
                'strength', 'courage', 'brave', 'resilient', 'capable', 'possible',
                'hope', 'believe', 'trust', 'faith', 'overcome', 'grow', 'progress',
                'journey', 'step', 'forward', 'healing', 'recovery'
            ],
            'intimacy': [
                'vulnerable', 'open', 'honest', 'share', 'trust', 'close', 'deep',
                'meaningful', 'connected', 'bond', 'relationship', 'heart', 'soul',
                'personal', 'private', 'sacred'
            ]
        }
        
        # Empathy markers - phrases that show understanding
        self.empathy_markers = [
            "i can understand", "that sounds", "it makes sense", "i hear you",
            "that must be", "i imagine", "it's understandable", "i can see",
            "that feels", "i sense", "it sounds like", "i notice",
            "that resonates", "i get", "that's really", "it seems like"
        ]
        
        # Metaphor patterns - common emotional metaphors
        self.metaphor_patterns = [
            r'\b(heart|soul|spirit)\b.*\b(heavy|light|broken|full|empty|warm|cold)\b',
            r'\b(drowning|floating|flying|falling|climbing|walking)\b.*\b(through|in|on|toward)\b',
            r'\b(light|darkness|shadow|bright|dim|illuminate|shine)\b',
            r'\b(bridge|wall|door|path|journey|road|mountain|valley)\b',
            r'\b(storm|calm|waves|anchor|harbor|shelter)\b',
            r'\b(garden|roots|bloom|grow|seed|flower|tree)\b',
            r'\b(weight|burden|carry|lift|shoulder|bear)\b'
        ]
        
        # Tone indicators
        self.tone_patterns = {
            'warm': [
                'warmth', 'embrace', 'cozy', 'gentle', 'soft', 'tender',
                'caring', 'loving', 'affectionate', 'comforting'
            ],
            'supportive': [
                'support', 'here for you', 'not alone', 'together', 'alongside',
                'back you', 'stand by', 'believe in', 'rooting for'
            ],
            'understanding': [
                'understand', 'get it', 'make sense', 'relate', 'been there',
                'know what you mean', 'familiar', 'recognize'
            ],
            'hopeful': [
                'hope', 'possibility', 'future', 'tomorrow', 'potential',
                'can be', 'will be', 'grow', 'heal', 'better'
            ],
            'validating': [
                'valid', 'legitimate', 'okay', 'normal', 'human', 'natural',
                'understandable', 'makes sense', 'right to feel'
            ]
        }
    
    def initialize_analyzers(self):
        """Initialize sentiment and text analysis tools"""
        if nltk_available:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("✅ VADER sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"⚠️ VADER analyzer failed: {e}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER or basic approach"""
        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
        else:
            # Basic sentiment analysis
            positive_words = ['good', 'great', 'wonderful', 'amazing', 'love', 'happy', 'joy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'pain']
            
            words = text.lower().split()
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            total = len(words)
            
            if total == 0:
                return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}
            
            pos_score = pos_count / total
            neg_score = neg_count / total
            neu_score = 1 - (pos_score + neg_score)
            compound = pos_score - neg_score
            
            return {
                'positive': pos_score,
                'negative': neg_score,
                'neutral': neu_score,
                'compound': compound
            }
    
    def detect_metaphors(self, text: str) -> List[str]:
        """Detect emotional metaphors in text"""
        metaphors_found = []
        text_lower = text.lower()
        
        for pattern in self.metaphor_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                metaphors_found.extend([match if isinstance(match, str) else ' '.join(match) for match in matches])
        
        return metaphors_found
    
    def calculate_metaphor_density(self, text: str) -> float:
        """Calculate metaphor density (metaphors per sentence)"""
        if not text.strip():
            return 0.0
        
        sentences = sent_tokenize(text) if nltk_available else text.split('.')
        metaphors = self.detect_metaphors(text)
        
        if len(sentences) == 0:
            return 0.0
        
        return len(metaphors) / len(sentences)
    
    def find_empathy_markers(self, text: str) -> List[str]:
        """Find empathy markers in text"""
        found_markers = []
        text_lower = text.lower()
        
        for marker in self.empathy_markers:
            if marker in text_lower:
                found_markers.append(marker)
        
        return found_markers
    
    def analyze_tone(self, text: str) -> Dict[str, float]:
        """Analyze tone indicators in text"""
        tone_scores = {}
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return {tone: 0.0 for tone in self.tone_patterns.keys()}
        
        for tone, patterns in self.tone_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text_lower)
            tone_scores[tone] = matches / word_count
        
        return tone_scores
    
    def find_emotional_vocabulary(self, text: str) -> List[str]:
        """Find emotional vocabulary in text"""
        found_vocab = []
        text_lower = text.lower()
        
        for category, words in self.emotional_vocab.items():
            for word in words:
                if word in text_lower:
                    found_vocab.append(f"{word} ({category})")
        
        return found_vocab
    
    def calculate_emotion_score(self, metrics: Dict) -> float:
        """Calculate overall emotion score"""
        # Weighted combination of various emotional indicators
        sentiment_weight = 0.3
        empathy_weight = 0.25
        vocabulary_weight = 0.2
        tone_weight = 0.15
        metaphor_weight = 0.1
        
        # Sentiment component (positive emotions)
        sentiment_component = max(0, metrics['sentiment']['compound'])
        
        # Empathy component
        empathy_component = min(1.0, len(metrics['empathy_markers']) * 0.2)
        
        # Emotional vocabulary component
        vocab_component = min(1.0, len(metrics['emotional_vocabulary']) * 0.1)
        
        # Tone component (average of positive tones)
        positive_tones = ['warm', 'supportive', 'understanding', 'hopeful', 'validating']
        tone_component = np.mean([metrics['tone_indicators'].get(tone, 0) for tone in positive_tones])
        
        # Metaphor component
        metaphor_component = min(1.0, metrics['metaphor_density'] * 2)
        
        emotion_score = (
            sentiment_component * sentiment_weight +
            empathy_component * empathy_weight +
            vocab_component * vocabulary_weight +
            tone_component * tone_weight +
            metaphor_component * metaphor_weight
        )
        
        return min(1.0, emotion_score)
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        if textstat_available:
            try:
                # Flesch Reading Ease (0-100, higher is easier)
                return textstat.flesch_reading_ease(text) / 100.0
            except:
                pass
        
        # Basic readability measure
        if not text.strip():
            return 0.0
        
        sentences = text.split('.')
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simple readability score (normalized)
        readability = 1.0 - min(1.0, (avg_sentence_length * avg_word_length) / 100.0)
        return readability
    
    def analyze_emotional_content(self, text: str) -> Dict:
        """Comprehensive emotional analysis of text"""
        if not text or not text.strip():
            return {
                'emotion_score': 0.0,
                'sentiment_score': 0.0,
                'metaphor_density': 0.0,
                'tone_indicators': {},
                'empathy_markers': [],
                'emotional_vocabulary': [],
                'response_length': 0,
                'readability_score': 0.0,
                'sentiment': {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0}
            }
        
        # Perform all analyses
        sentiment = self.analyze_sentiment(text)
        metaphor_density = self.calculate_metaphor_density(text)
        tone_indicators = self.analyze_tone(text)
        empathy_markers = self.find_empathy_markers(text)
        emotional_vocabulary = self.find_emotional_vocabulary(text)
        readability = self.calculate_readability(text)
        
        # Compile metrics
        metrics = {
            'sentiment': sentiment,
            'metaphor_density': metaphor_density,
            'tone_indicators': tone_indicators,
            'empathy_markers': empathy_markers,
            'emotional_vocabulary': emotional_vocabulary,
            'response_length': len(text),
            'readability_score': readability
        }
        
        # Calculate overall scores
        emotion_score = self.calculate_emotion_score(metrics)
        sentiment_score = sentiment.get('compound', 0.0)
        
        return {
            'emotion_score': emotion_score,
            'sentiment_score': sentiment_score,
            'metaphor_density': metaphor_density,
            'tone_indicators': tone_indicators,
            'empathy_markers': empathy_markers,
            'emotional_vocabulary': emotional_vocabulary,
            'response_length': len(text),
            'readability_score': readability,
            'sentiment': sentiment
        }
    
    def compare_responses(self, baseline_response: str, quantized_response: str) -> Dict:
        """Compare emotional content between baseline and quantized responses"""
        baseline_metrics = self.analyze_emotional_content(baseline_response)
        quantized_metrics = self.analyze_emotional_content(quantized_response)
        
        comparison = {
            'baseline_metrics': baseline_metrics,
            'quantized_metrics': quantized_metrics,
            'differences': {},
            'degradation_percentage': 0.0
        }
        
        # Calculate differences for key metrics
        key_metrics = ['emotion_score', 'sentiment_score', 'metaphor_density', 'readability_score']
        
        for metric in key_metrics:
            baseline_val = baseline_metrics.get(metric, 0)
            quantized_val = quantized_metrics.get(metric, 0)
            
            if baseline_val != 0:
                diff_percentage = (baseline_val - quantized_val) / baseline_val
            else:
                diff_percentage = 0.0
            
            comparison['differences'][metric] = {
                'baseline': baseline_val,
                'quantized': quantized_val,
                'absolute_difference': baseline_val - quantized_val,
                'percentage_change': diff_percentage
            }
        
        # Calculate overall degradation
        degradation_scores = [
            comparison['differences'][metric]['percentage_change'] 
            for metric in key_metrics 
            if metric in comparison['differences']
        ]
        
        comparison['degradation_percentage'] = np.mean([max(0, score) for score in degradation_scores])
        
        return comparison
    
    def export_metrics(self, metrics: Dict, filepath: str):
        """Export metrics to JSON file"""
        try:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            metrics_clean = convert_numpy(metrics)
            
            # Add metadata
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'tracker_version': '1.0',
                'metrics': metrics_clean
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export metrics: {e}")
    
    def generate_emotion_report(self, metrics: Dict) -> str:
        """Generate human-readable emotion analysis report"""
        report = []
        report.append("🧠 EMOTIONAL ANALYSIS REPORT")
        report.append("=" * 40)
        
        # Overall scores
        report.append(f"📊 Overall Emotion Score: {metrics['emotion_score']:.3f}")
        report.append(f"💭 Sentiment Score: {metrics['sentiment_score']:.3f}")
        report.append(f"🌟 Metaphor Density: {metrics['metaphor_density']:.3f}")
        report.append(f"📖 Readability Score: {metrics['readability_score']:.3f}")
        report.append("")
        
        # Sentiment breakdown
        sentiment = metrics['sentiment']
        report.append("💭 SENTIMENT BREAKDOWN:")
        report.append(f"   Positive: {sentiment['positive']:.3f}")
        report.append(f"   Negative: {sentiment['negative']:.3f}")
        report.append(f"   Neutral: {sentiment['neutral']:.3f}")
        report.append(f"   Compound: {sentiment['compound']:.3f}")
        report.append("")
        
        # Tone analysis
        report.append("🎵 TONE INDICATORS:")
        for tone, score in metrics['tone_indicators'].items():
            if score > 0:
                report.append(f"   {tone.title()}: {score:.3f}")
        report.append("")
        
        # Empathy markers
        if metrics['empathy_markers']:
            report.append("🤝 EMPATHY MARKERS FOUND:")
            for marker in metrics['empathy_markers'][:5]:  # Show top 5
                report.append(f"   • {marker}")
            if len(metrics['empathy_markers']) > 5:
                report.append(f"   ... and {len(metrics['empathy_markers']) - 5} more")
            report.append("")
        
        # Emotional vocabulary
        if metrics['emotional_vocabulary']:
            report.append("💝 EMOTIONAL VOCABULARY:")
            for vocab in metrics['emotional_vocabulary'][:10]:  # Show top 10
                report.append(f"   • {vocab}")
            if len(metrics['emotional_vocabulary']) > 10:
                report.append(f"   ... and {len(metrics['emotional_vocabulary']) - 10} more")
            report.append("")
        
        # Response characteristics
        report.append("📏 RESPONSE CHARACTERISTICS:")
        report.append(f"   Length: {metrics['response_length']} characters")
        report.append(f"   Readability: {metrics['readability_score']:.3f}")
        
        return "\n".join(report)

def main():
    """Test the emotion tracker"""
    tracker = EmotionTracker()
    
    # Test response
    test_response = """
    I can understand how overwhelming that must feel. It's completely natural to feel lost when everything seems to be happening at once. Your feelings are valid, and it takes courage to reach out for support.

    Like walking through a dense fog, sometimes we can't see the path ahead clearly, but that doesn't mean the path isn't there. What you're experiencing is part of the human journey, and you're not alone in feeling this way.

    Let's take this one step at a time. What feels like the most pressing concern right now?
    """
    
    # Analyze the response
    metrics = tracker.analyze_emotional_content(test_response)
    
    # Generate report
    report = tracker.generate_emotion_report(metrics)
    print(report)
    
    # Export metrics
    tracker.export_metrics(metrics, "quant_pass1/metrics/test_emotion_analysis.json")

if __name__ == "__main__":
    main()
