#!/usr/bin/env python3
"""
Mirror Loop - Background reflection and truth-checking for The Mirror

This script runs as a background task for The Mirror council member,
performing emotional reflection, contradiction detection, and truth validation
during idle periods.

Author: Dolphin AI System
Date: August 1, 2025
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MirrorLoop:
    """Background reflection process for The Mirror"""
    
    def __init__(self, reflection_log_path: str = "emotion_logs/mirror_reflections.jsonl"):
        self.reflection_log_path = reflection_log_path
        self.reflection_prompts = [
            "What contradictions exist in recent interactions?",
            "What truths are being avoided or overlooked?",
            "How do emotions and logic align or conflict?",
            "What patterns emerge from recent responses?",
            "What deeper meanings lie beneath surface communications?"
        ]
        self.is_running = False
        
    def scan_recent_logs(self, hours_back: int = 24) -> List[Dict]:
        """Scan recent logs for reflection material"""
        entries = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Look for various log files to reflect upon
        log_dirs = ["emotion_logs", "reflection_logs", "logs"]
        
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                for filename in os.listdir(log_dir):
                    if filename.endswith('.jsonl'):
                        filepath = os.path.join(log_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                for line in f:
                                    try:
                                        entry = json.loads(line.strip())
                                        
                                        # Check if entry has timestamp and is recent
                                        if 'timestamp' in entry:
                                            entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                                            if entry_time > cutoff_time:
                                                entries.append(entry)
                                    except (json.JSONDecodeError, ValueError):
                                        continue
                        except Exception as e:
                            logger.debug(f"[Mirror Loop] Could not read {filepath}: {e}")
                            continue
        
        logger.info(f"[Mirror Loop] Found {len(entries)} recent entries for reflection")
        return entries
    
    def analyze_contradictions(self, entries: List[Dict]) -> List[Dict]:
        """Analyze entries for contradictions and inconsistencies"""
        contradictions = []
        
        # Simple contradiction detection patterns
        contradiction_keywords = [
            ("positive", "negative"), ("good", "bad"), ("yes", "no"),
            ("agree", "disagree"), ("love", "hate"), ("trust", "doubt"),
            ("hope", "despair"), ("certain", "uncertain")
        ]
        
        for i, entry in enumerate(entries):
            content = str(entry).lower()
            
            for pos_word, neg_word in contradiction_keywords:
                if pos_word in content and neg_word in content:
                    contradictions.append({
                        "type": "semantic_contradiction",
                        "positive_term": pos_word,
                        "negative_term": neg_word,
                        "entry_index": i,
                        "content_sample": str(entry)[:200],
                        "detected_at": datetime.now().isoformat()
                    })
        
        return contradictions
    
    def generate_reflection(self, entries: List[Dict], contradictions: List[Dict]) -> Dict:
        """Generate a reflection based on analyzed entries"""
        import random
        
        prompt = random.choice(self.reflection_prompts)
        
        reflection = {
            "timestamp": datetime.now().isoformat(),
            "reflector": "The Mirror",
            "emotional_signature": "DeepResonance-03",
            "reflection_prompt": prompt,
            "analysis": {
                "entries_analyzed": len(entries),
                "contradictions_found": len(contradictions),
                "dominant_themes": self._extract_themes(entries),
                "emotional_consistency": self._assess_emotional_consistency(entries),
                "truth_assessment": self._assess_truth_indicators(entries)
            },
            "contradictions": contradictions[:5],  # Limit for readability
            "insights": self._generate_insights(entries, contradictions),
            "metadata": {
                "generation_context": "background_reflection",
                "council_member": "The Mirror",
                "loop_iteration": True
            }
        }
        
        return reflection
    
    def _extract_themes(self, entries: List[Dict]) -> List[str]:
        """Extract dominant themes from entries"""
        # Simple keyword frequency analysis
        theme_keywords = {
            "emotional": ["emotion", "feeling", "heart", "love", "sadness", "joy"],
            "analytical": ["analyze", "logic", "data", "pattern", "reasoning"],
            "creative": ["create", "imagine", "dream", "artistic", "inspiration"],
            "relational": ["relationship", "connection", "communication", "understanding"],
            "spiritual": ["spirit", "faith", "purpose", "meaning", "sacred"]
        }
        
        theme_scores = {}
        all_text = " ".join(str(entry).lower() for entry in entries)
        
        for theme, keywords in theme_keywords.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            if score > 0:
                theme_scores[theme] = score
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes[:3]]
    
    def _assess_emotional_consistency(self, entries: List[Dict]) -> float:
        """Assess emotional consistency across entries"""
        # Simple heuristic based on emotional scoring if available
        scores = []
        for entry in entries:
            if 'emotional_resonance_score' in entry:
                scores.append(entry['emotional_resonance_score'])
            elif 'score' in entry:
                scores.append(entry['score'])
        
        if not scores:
            return 0.5  # Neutral if no scores available
        
        # Calculate variance (lower variance = higher consistency)
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        consistency = max(0, 1 - variance)  # Convert variance to consistency measure
        
        return round(consistency, 3)
    
    def _assess_truth_indicators(self, entries: List[Dict]) -> Dict:
        """Assess truth indicators in entries"""
        truth_keywords = ["true", "truth", "honest", "authentic", "genuine", "real"]
        doubt_keywords = ["false", "lie", "deception", "fake", "artificial", "pretend"]
        
        all_text = " ".join(str(entry).lower() for entry in entries)
        
        truth_count = sum(all_text.count(keyword) for keyword in truth_keywords)
        doubt_count = sum(all_text.count(keyword) for keyword in doubt_keywords)
        
        total_indicators = truth_count + doubt_count
        truth_ratio = truth_count / total_indicators if total_indicators > 0 else 0.5
        
        return {
            "truth_indicators": truth_count,
            "doubt_indicators": doubt_count,
            "truth_ratio": round(truth_ratio, 3),
            "assessment": "high_truth" if truth_ratio > 0.7 else "mixed" if truth_ratio > 0.3 else "high_doubt"
        }
    
    def _generate_insights(self, entries: List[Dict], contradictions: List[Dict]) -> List[str]:
        """Generate insights based on analysis"""
        insights = []
        
        if len(contradictions) > 3:
            insights.append(f"High contradiction density detected ({len(contradictions)} contradictions)")
        elif len(contradictions) == 0:
            insights.append("Remarkable consistency observed in recent communications")
        
        if len(entries) > 20:
            insights.append("High activity level - increased vigilance recommended")
        elif len(entries) < 5:
            insights.append("Low activity level - system may be in quiet contemplation")
        
        # Add domain-specific insights
        insights.append("Emotional patterns require deeper examination")
        insights.append("Truth-seeking protocols remain active")
        
        return insights
    
    def log_reflection(self, reflection: Dict):
        """Log reflection to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.reflection_log_path), exist_ok=True)
            
            with open(self.reflection_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(reflection, ensure_ascii=False) + '\n')
                
            logger.info(f"[Mirror Loop] Logged reflection with {reflection['analysis']['contradictions_found']} contradictions")
            
        except Exception as e:
            logger.error(f"[Mirror Loop] Error logging reflection: {e}")
    
    def run_reflection_cycle(self):
        """Run a complete reflection cycle"""
        logger.info("[Mirror Loop] Starting reflection cycle")
        
        # Scan recent logs
        entries = self.scan_recent_logs()
        
        # Analyze for contradictions
        contradictions = self.analyze_contradictions(entries)
        
        # Generate reflection
        reflection = self.generate_reflection(entries, contradictions)
        
        # Log the reflection
        self.log_reflection(reflection)
        
        logger.info("[Mirror Loop] Reflection cycle completed")
        return reflection
    
    def start_background_reflection(self, interval_minutes: int = 60):
        """Start continuous background reflection"""
        logger.info(f"[Mirror Loop] Starting background reflection (interval: {interval_minutes}m)")
        self.is_running = True
        
        while self.is_running:
            try:
                self.run_reflection_cycle()
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("[Mirror Loop] Background reflection stopped by user")
                break
            except Exception as e:
                logger.error(f"[Mirror Loop] Error in background reflection: {e}")
                time.sleep(60)  # Brief recovery pause
    
    def stop_reflection(self):
        """Stop background reflection"""
        self.is_running = False
        logger.info("[Mirror Loop] Reflection stopped")

def main():
    """Main entry point for mirror loop"""
    mirror_loop = MirrorLoop()
    
    print("🪞 Mirror Loop - The Mirror's Background Process")
    print("=" * 46)
    
    # Run a test reflection cycle
    print("Running test reflection cycle...")
    reflection = mirror_loop.run_reflection_cycle()
    
    print(f"\nReflection completed:")
    print(f"  Entries analyzed: {reflection['analysis']['entries_analyzed']}")
    print(f"  Contradictions found: {reflection['analysis']['contradictions_found']}")
    print(f"  Dominant themes: {', '.join(reflection['analysis']['dominant_themes'])}")
    print(f"  Truth assessment: {reflection['analysis']['truth_assessment']['assessment']}")
    
    print("\nTo start continuous background reflection, call start_background_reflection()")

if __name__ == "__main__":
    main()
