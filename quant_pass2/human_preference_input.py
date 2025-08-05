#!/usr/bin/env python3
"""
Human Preference Input System
Generates summaries for final candidates and collects human feedback
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import textwrap

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CandidateProfile:
    """Profile of a candidate model for human review"""
    name: str
    path: str
    size_gb: float
    quantization_method: str
    emotional_degradation: float
    ai_judge_score: float
    ai_judge_rank: int
    sample_responses: List[Dict]
    strengths: List[str]
    weaknesses: List[str]
    technical_specs: Dict

@dataclass
class HumanRating:
    """Human rating for a specific aspect"""
    aspect: str
    score: int  # 1-10 scale
    notes: str

@dataclass
class HumanFeedback:
    """Complete human feedback for a candidate"""
    candidate_name: str
    overall_rating: int  # 1-10 scale
    ratings: List[HumanRating]
    preferred_responses: List[int]  # Indices of preferred sample responses
    general_comments: str
    recommendation: str  # 'accept', 'reject', 'needs_improvement'
    timestamp: str

class HumanPreferenceCollector:
    """System for collecting and processing human feedback on model candidates"""
    
    def __init__(self):
        self.candidates = []
        self.feedback_history = []
        
        # Evaluation criteria
        self.evaluation_criteria = {
            "believability": {
                "description": "How natural and human-like do the responses feel?",
                "weight": 0.25
            },
            "connection": {
                "description": "How well does the model create emotional connection?",
                "weight": 0.30
            },
            "expressive_strength": {
                "description": "How effectively does the model express empathy and support?",
                "weight": 0.25
            },
            "appropriateness": {
                "description": "How appropriate are the responses to the emotional context?",
                "weight": 0.20
            }
        }
        
        logger.info("👤 Human Preference Collector initialized")
    
    def load_final_candidates(self, ai_judgment_file: str, top_n: int = 3) -> List[CandidateProfile]:
        """Load top candidates from AI judgment results"""
        logger.info(f"📂 Loading top {top_n} candidates from AI judgment")
        
        try:
            with open(ai_judgment_file, 'r') as f:
                ai_results = json.load(f)
            
            model_rankings = ai_results.get("model_rankings", {})
            
            # Get top candidates (excluding 'original' if present)
            top_candidates = []
            for model_name, scores in model_rankings.items():
                if model_name != "original" and len(top_candidates) < top_n:
                    top_candidates.append((model_name, scores))
            
            # Convert to CandidateProfile objects
            profiles = []
            for i, (name, scores) in enumerate(top_candidates):
                # Load candidate details
                candidate_path = f"quant_pass1/models/{name}"
                
                # Extract sample responses from AI judgment results
                sample_responses = self._extract_sample_responses(ai_results, name)
                
                # Analyze strengths and weaknesses
                strengths, weaknesses = self._analyze_candidate(scores, sample_responses)
                
                profile = CandidateProfile(
                    name=name,
                    path=candidate_path,
                    size_gb=scores.get("avg_embedding_similarity", 0),  # Placeholder
                    quantization_method=name.split('_')[0] if '_' in name else 'unknown',
                    emotional_degradation=1.0 - scores.get("overall_score", 0),
                    ai_judge_score=scores.get("overall_score", 0),
                    ai_judge_rank=i + 1,
                    sample_responses=sample_responses,
                    strengths=strengths,
                    weaknesses=weaknesses,
                    technical_specs=self._get_technical_specs(name, scores)
                )
                
                profiles.append(profile)
            
            self.candidates = profiles
            logger.info(f"✅ Loaded {len(profiles)} candidate profiles")
            return profiles
            
        except Exception as e:
            logger.error(f"❌ Failed to load candidates: {e}")
            return []
    
    def _extract_sample_responses(self, ai_results: Dict, model_name: str) -> List[Dict]:
        """Extract sample responses for a specific model"""
        samples = []
        
        # Look through comparison results for this model
        all_results = ai_results.get("all_comparison_results", [])
        
        model_responses = []
        for result in all_results:
            if result.get("model_a") == model_name:
                model_responses.append({
                    "prompt": result.get("prompt", ""),
                    "response": result.get("response_a", ""),
                    "comparison_score": result.get("embedding_similarity", 0)
                })
            elif result.get("model_b") == model_name:
                model_responses.append({
                    "prompt": result.get("prompt", ""),
                    "response": result.get("response_b", ""),
                    "comparison_score": result.get("embedding_similarity", 0)
                })
        
        # Select diverse, high-quality samples
        if model_responses:
            # Sort by quality and take diverse samples
            model_responses.sort(key=lambda x: x["comparison_score"], reverse=True)
            samples = model_responses[:5]  # Top 5 responses
        
        return samples
    
    def _analyze_candidate(self, scores: Dict, sample_responses: List[Dict]) -> Tuple[List[str], List[str]]:
        """Analyze candidate strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze based on AI scores
        if scores.get("win_rate", 0) > 0.6:
            strengths.append("High win rate in AI comparisons")
        
        if scores.get("avg_emotion_preservation", 0) > 0.8:
            strengths.append("Excellent emotional preservation")
        elif scores.get("avg_emotion_preservation", 0) < 0.6:
            weaknesses.append("Some emotional degradation detected")
        
        if scores.get("avg_empathy_preservation", 0) > 0.7:
            strengths.append("Strong empathy retention")
        elif scores.get("avg_empathy_preservation", 0) < 0.5:
            weaknesses.append("Reduced empathy expression")
        
        # Analyze response patterns
        if sample_responses:
            avg_length = sum(len(r["response"]) for r in sample_responses) / len(sample_responses)
            if avg_length > 150:
                strengths.append("Provides detailed, thoughtful responses")
            elif avg_length < 50:
                weaknesses.append("Responses may be too brief")
        
        # Default items if lists are empty
        if not strengths:
            strengths.append("Meets basic quantization requirements")
        if not weaknesses:
            weaknesses.append("Minor areas for optimization identified")
        
        return strengths, weaknesses
    
    def _get_technical_specs(self, name: str, scores: Dict) -> Dict:
        """Get technical specifications for candidate"""
        return {
            "quantization_method": name.split('_')[0] if '_' in name else 'unknown',
            "win_rate": scores.get("win_rate", 0),
            "total_comparisons": scores.get("total_comparisons", 0),
            "overall_score": scores.get("overall_score", 0),
            "emotion_preservation": scores.get("avg_emotion_preservation", 0),
            "sentiment_alignment": scores.get("avg_sentiment_alignment", 0)
        }
    
    def generate_candidate_summary(self, candidate: CandidateProfile) -> str:
        """Generate human-readable summary for a candidate"""
        summary = f"""
# 🤖 Candidate Model: {candidate.name}

## 📊 Overview
- **Rank**: #{candidate.ai_judge_rank} (AI Judgment)
- **Size**: {candidate.size_gb:.1f}GB
- **Method**: {candidate.quantization_method}
- **AI Score**: {candidate.ai_judge_score:.3f}
- **Emotional Degradation**: {candidate.emotional_degradation*100:.1f}%

## ✅ Strengths
{chr(10).join(f"• {strength}" for strength in candidate.strengths)}

## ⚠️ Areas for Improvement
{chr(10).join(f"• weakness" for weakness in candidate.weaknesses)}

## 🗣️ Sample Responses

"""
        
        # Add sample responses
        for i, sample in enumerate(candidate.sample_responses[:3], 1):
            prompt = textwrap.fill(sample["prompt"], width=80)
            response = textwrap.fill(sample["response"], width=80, initial_indent="  ", subsequent_indent="  ")
            
            summary += f"""
### Sample {i}
**Situation**: {prompt}

**Response**:
{response}

**Quality Score**: {sample.get("comparison_score", 0):.3f}
---
"""
        
        # Add technical details
        summary += f"""
## 🔧 Technical Details
- **Win Rate**: {candidate.technical_specs.get("win_rate", 0):.1f}%
- **Total Comparisons**: {candidate.technical_specs.get("total_comparisons", 0)}
- **Emotion Preservation**: {candidate.technical_specs.get("emotion_preservation", 0):.3f}
- **Sentiment Alignment**: {candidate.technical_specs.get("sentiment_alignment", 0):.3f}
"""
        
        return summary
    
    def collect_human_feedback(self, candidate: CandidateProfile) -> HumanFeedback:
        """Collect human feedback for a candidate through interactive prompts"""
        logger.info(f"👤 Collecting human feedback for: {candidate.name}")
        
        print("=" * 80)
        print(f"🤖 HUMAN EVALUATION: {candidate.name}")
        print("=" * 80)
        
        # Display candidate summary
        summary = self.generate_candidate_summary(candidate)
        print(summary)
        
        print("\n" + "=" * 80)
        print("📝 EVALUATION FORM")
        print("=" * 80)
        
        # Collect ratings for each criterion
        ratings = []
        
        for criterion, details in self.evaluation_criteria.items():
            print(f"\n🔍 {criterion.upper()}")
            print(f"   {details['description']}")
            
            while True:
                try:
                    score = input(f"   Rate {criterion} (1-10, 10=excellent): ").strip()
                    score = int(score)
                    if 1 <= score <= 10:
                        break
                    else:
                        print("   Please enter a number between 1 and 10")
                except ValueError:
                    print("   Please enter a valid number")
            
            notes = input(f"   Notes on {criterion} (optional): ").strip()
            
            ratings.append(HumanRating(
                aspect=criterion,
                score=score,
                notes=notes
            ))
        
        # Overall rating
        print("\n🎯 OVERALL ASSESSMENT")
        while True:
            try:
                overall = input("Overall rating (1-10, 10=excellent): ").strip()
                overall = int(overall)
                if 1 <= overall <= 10:
                    break
                else:
                    print("Please enter a number between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Preferred responses
        print("\n💭 RESPONSE PREFERENCES")
        if candidate.sample_responses:
            print("Which sample responses did you find most compelling?")
            print("Enter the numbers separated by commas (e.g., 1,3):")
            
            preferred_str = input("Preferred samples: ").strip()
            preferred_responses = []
            
            if preferred_str:
                try:
                    preferred_responses = [int(x.strip()) - 1 for x in preferred_str.split(",")]
                    preferred_responses = [x for x in preferred_responses if 0 <= x < len(candidate.sample_responses)]
                except ValueError:
                    preferred_responses = []
        else:
            preferred_responses = []
        
        # General comments
        print("\n📝 GENERAL COMMENTS")
        general_comments = input("Any additional thoughts or concerns: ").strip()
        
        # Final recommendation
        print("\n🎯 FINAL RECOMMENDATION")
        print("Based on your evaluation, what is your recommendation?")
        print("1. Accept (deploy this model)")
        print("2. Reject (not suitable for deployment)")
        print("3. Needs Improvement (requires further optimization)")
        
        while True:
            rec_choice = input("Enter choice (1-3): ").strip()
            if rec_choice == "1":
                recommendation = "accept"
                break
            elif rec_choice == "2":
                recommendation = "reject"
                break
            elif rec_choice == "3":
                recommendation = "needs_improvement"
                break
            else:
                print("Please enter 1, 2, or 3")
        
        # Create feedback object
        feedback = HumanFeedback(
            candidate_name=candidate.name,
            overall_rating=overall,
            ratings=ratings,
            preferred_responses=preferred_responses,
            general_comments=general_comments,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
        self.feedback_history.append(feedback)
        
        print(f"\n✅ Feedback collected for {candidate.name}")
        return feedback
    
    def batch_collect_feedback(self, candidates: List[CandidateProfile]) -> List[HumanFeedback]:
        """Collect feedback for multiple candidates"""
        logger.info(f"👥 Collecting feedback for {len(candidates)} candidates")
        
        all_feedback = []
        
        for i, candidate in enumerate(candidates, 1):
            print(f"\n🔄 EVALUATING CANDIDATE {i}/{len(candidates)}")
            
            feedback = self.collect_human_feedback(candidate)
            all_feedback.append(feedback)
            
            if i < len(candidates):
                input("\nPress Enter to continue to next candidate...")
        
        return all_feedback
    
    def calculate_weighted_scores(self, feedback: HumanFeedback) -> float:
        """Calculate weighted score based on criteria weights"""
        weighted_score = 0.0
        
        for rating in feedback.ratings:
            weight = self.evaluation_criteria.get(rating.aspect, {}).get("weight", 0.25)
            weighted_score += rating.score * weight
        
        return weighted_score
    
    def generate_feedback_report(self, all_feedback: List[HumanFeedback]) -> Dict:
        """Generate comprehensive feedback report"""
        logger.info("📊 Generating feedback report")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_candidates": len(all_feedback),
            "evaluation_criteria": self.evaluation_criteria,
            "candidate_scores": {},
            "recommendations": {},
            "summary": {}
        }
        
        # Process each candidate's feedback
        for feedback in all_feedback:
            weighted_score = self.calculate_weighted_scores(feedback)
            
            report["candidate_scores"][feedback.candidate_name] = {
                "overall_rating": feedback.overall_rating,
                "weighted_score": weighted_score,
                "recommendation": feedback.recommendation,
                "detailed_ratings": {r.aspect: r.score for r in feedback.ratings},
                "comments": feedback.general_comments,
                "preferred_responses": feedback.preferred_responses
            }
            
            report["recommendations"][feedback.candidate_name] = feedback.recommendation
        
        # Generate summary
        accepted_candidates = [name for name, rec in report["recommendations"].items() if rec == "accept"]
        rejected_candidates = [name for name, rec in report["recommendations"].items() if rec == "reject"]
        
        report["summary"] = {
            "accepted_count": len(accepted_candidates),
            "rejected_count": len(rejected_candidates),
            "accepted_candidates": accepted_candidates,
            "rejected_candidates": rejected_candidates,
            "avg_overall_rating": sum(f.overall_rating for f in all_feedback) / len(all_feedback) if all_feedback else 0,
            "top_candidate": max(report["candidate_scores"].items(), key=lambda x: x[1]["weighted_score"])[0] if report["candidate_scores"] else None
        }
        
        return report
    
    def save_feedback(self, feedback_list: List[HumanFeedback], filepath: str):
        """Save human feedback to file"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "evaluation_criteria": self.evaluation_criteria,
                "feedback": [asdict(feedback) for feedback in feedback_list]
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved human feedback: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save feedback: {e}")
    
    def save_report(self, report: Dict, filepath: str):
        """Save feedback report to file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Saved feedback report: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")

def main():
    """Main execution function"""
    
    # Initialize collector
    collector = HumanPreferenceCollector()
    
    # Configuration
    ai_judgment_file = "quant_pass2/results/comprehensive_model_comparison.json"
    top_n = int(os.getenv("TOP_CANDIDATES", "3"))
    
    try:
        # Check if AI judgment results exist
        if not Path(ai_judgment_file).exists():
            logger.error(f"❌ AI judgment file not found: {ai_judgment_file}")
            logger.info("💡 Run judge_models.py first to generate AI comparisons")
            return 1
        
        # Load top candidates
        candidates = collector.load_final_candidates(ai_judgment_file, top_n)
        
        if not candidates:
            logger.error("❌ No candidates found")
            return 1
        
        logger.info(f"🎯 Top {len(candidates)} candidates loaded for human evaluation")
        
        # Collect human feedback
        all_feedback = collector.batch_collect_feedback(candidates)
        
        # Generate report
        report = collector.generate_feedback_report(all_feedback)
        
        # Save results
        collector.save_feedback(all_feedback, "quant_pass2/results/human_feedback.json")
        collector.save_report(report, "quant_pass2/results/human_feedback_report.json")
        
        # Display summary
        print("\n" + "=" * 80)
        print("📊 HUMAN EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total candidates evaluated: {report['summary']['total_candidates']}")
        print(f"Accepted candidates: {report['summary']['accepted_count']}")
        print(f"Rejected candidates: {report['summary']['rejected_count']}")
        print(f"Average rating: {report['summary']['avg_overall_rating']:.1f}/10")
        
        if report['summary']['top_candidate']:
            print(f"Top candidate: {report['summary']['top_candidate']}")
        
        logger.info("✅ Human feedback collection completed")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Human feedback collection failed: {e}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
