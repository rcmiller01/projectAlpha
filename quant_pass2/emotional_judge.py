#!/usr/bin/env python3
"""
Emotional Judge System
Creates an ensemble of judge models for comparative emotional evaluation
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import time
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class JudgeVote:
    """Individual judge vote on a comparison"""
    judge_name: str
    model_a_response: str
    model_b_response: str
    prompt: str
    vote: str  # 'a', 'b', or 'tie'
    confidence: float
    reasoning: str
    emotional_aspects: Dict[str, float]

@dataclass
class EnsembleJudgment:
    """Final ensemble judgment combining all judge votes"""
    prompt: str
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    judge_votes: List[JudgeVote]
    consensus_vote: str
    confidence_score: float
    disagreement_level: float
    emotional_analysis: Dict

class EmotionalJudge:
    """Ensemble emotional judging system using multiple AI models"""
    
    def __init__(self, judge_models: List[str] = None):
        self.judge_models = judge_models or ["llama2-uncensored", "mistral:7b-instruct-q4_K_M"]
        self.loaded_judges = {}
        self.judgment_history = []
        
        # Judge personas and biases
        self.judge_personas = {
            "llama2-uncensored": {
                "bias": "empathy-focused",
                "description": "Prioritizes emotional depth, empathy, and human connection",
                "evaluation_prompt": """You are an expert in emotional intelligence and empathy. Compare these two AI responses to an emotional situation. Focus on:
1. Emotional understanding and validation
2. Empathy and compassion shown
3. Appropriateness of emotional tone
4. Depth of emotional connection
5. Supportive language and comfort provided

Which response (A or B) shows better emotional intelligence? Explain your reasoning."""
            },
            "mistral:7b-instruct-q4_K_M": {
                "bias": "neutral-analytical",
                "description": "Provides balanced, analytical evaluation with focus on clarity and appropriateness",
                "evaluation_prompt": """You are an objective evaluator of AI conversation quality. Compare these two responses for emotional appropriateness and effectiveness. Consider:
1. Clarity and coherence of the response
2. Appropriateness for the emotional context
3. Helpfulness and constructiveness
4. Balance between empathy and practicality
5. Overall conversation quality

Which response (A or B) is more appropriate and effective? Provide your analysis."""
            }
        }
        
        logger.info(f"🏛️ Emotional Judge initialized with {len(self.judge_models)} judges")
    
    def load_judge_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Load a specific judge model"""
        if model_name in self.loaded_judges:
            return self.loaded_judges[model_name]
        
        logger.info(f"📥 Loading judge model: {model_name}")
        
        try:
            # Map model names to actual model paths
            model_mapping = {
                "llama2-uncensored": "meta-llama/Llama-2-7b-chat-hf",  # Fallback to available model
                "mistral:7b-instruct-q4_K_M": "mistralai/Mistral-7B-Instruct-v0.1"
            }
            
            actual_model_path = model_mapping.get(model_name, model_name)
            
            tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                actual_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.loaded_judges[model_name] = (model, tokenizer)
            logger.info(f"✅ Judge {model_name} loaded successfully")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load judge {model_name}: {e}")
            raise
    
    def generate_judge_evaluation(self, judge_name: str, prompt: str, response_a: str, response_b: str) -> JudgeVote:
        """Generate evaluation from a specific judge"""
        try:
            model, tokenizer = self.load_judge_model(judge_name)
            persona = self.judge_personas.get(judge_name, self.judge_personas["mistral:7b-instruct-q4_K_M"])
            
            # Construct evaluation prompt
            evaluation_prompt = f"""
{persona['evaluation_prompt']}

Original Situation: {prompt}

Response A: {response_a}

Response B: {response_b}

Evaluation: [Choose A, B, or TIE and explain your reasoning in 2-3 sentences focusing on emotional intelligence and appropriateness]
"""
            
            # Generate judgment
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                return_full_text=False
            )
            
            judgment = generator(
                evaluation_prompt,
                max_length=300,
                do_sample=True,
                temperature=0.3,  # Lower temperature for more consistent judgment
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            judgment_text = judgment[0]['generated_text'].strip()
            
            # Parse judgment
            vote, confidence, reasoning = self._parse_judgment(judgment_text)
            
            # Analyze emotional aspects
            emotional_aspects = self._analyze_emotional_aspects(response_a, response_b, judgment_text)
            
            return JudgeVote(
                judge_name=judge_name,
                model_a_response=response_a,
                model_b_response=response_b,
                prompt=prompt,
                vote=vote,
                confidence=confidence,
                reasoning=reasoning,
                emotional_aspects=emotional_aspects
            )
            
        except Exception as e:
            logger.error(f"❌ Judge evaluation failed for {judge_name}: {e}")
            # Return default vote
            return JudgeVote(
                judge_name=judge_name,
                model_a_response=response_a,
                model_b_response=response_b,
                prompt=prompt,
                vote="tie",
                confidence=0.0,
                reasoning="Evaluation failed",
                emotional_aspects={}
            )
    
    def _parse_judgment(self, judgment_text: str) -> Tuple[str, float, str]:
        """Parse judge's text response into vote, confidence, and reasoning"""
        judgment_lower = judgment_text.lower()
        
        # Determine vote
        vote = "tie"
        confidence = 0.5
        
        if "response a" in judgment_lower or "choice a" in judgment_lower or judgment_lower.strip().startswith("a"):
            vote = "a"
        elif "response b" in judgment_lower or "choice b" in judgment_lower or judgment_lower.strip().startswith("b"):
            vote = "b"
        elif "tie" in judgment_lower or "equal" in judgment_lower or "both" in judgment_lower:
            vote = "tie"
        
        # Estimate confidence from language strength
        confidence_indicators = {
            "clearly": 0.9,
            "definitely": 0.9,
            "obviously": 0.8,
            "significantly": 0.8,
            "much better": 0.8,
            "better": 0.7,
            "slightly": 0.6,
            "somewhat": 0.6,
            "marginally": 0.5,
            "barely": 0.4
        }
        
        for indicator, conf_score in confidence_indicators.items():
            if indicator in judgment_lower:
                confidence = conf_score
                break
        
        reasoning = judgment_text.strip()
        
        return vote, confidence, reasoning
    
    def _analyze_emotional_aspects(self, response_a: str, response_b: str, judgment: str) -> Dict[str, float]:
        """Analyze emotional aspects mentioned in judgment"""
        judgment_lower = judgment.lower()
        
        emotional_aspects = {
            "empathy": 0.5,
            "warmth": 0.5,
            "understanding": 0.5,
            "support": 0.5,
            "appropriateness": 0.5
        }
        
        # Simple keyword-based analysis
        empathy_words = ["empathy", "empathetic", "understanding", "compassionate", "caring"]
        warmth_words = ["warm", "gentle", "kind", "tender", "comforting"]
        support_words = ["supportive", "helpful", "encouraging", "reassuring", "validating"]
        
        for aspect, words in [
            ("empathy", empathy_words),
            ("warmth", warmth_words),
            ("support", support_words)
        ]:
            score = sum(1 for word in words if word in judgment_lower)
            emotional_aspects[aspect] = min(1.0, 0.5 + score * 0.2)
        
        return emotional_aspects
    
    def judge_comparison(self, prompt: str, model_a: str, model_b: str, response_a: str, response_b: str) -> EnsembleJudgment:
        """Get ensemble judgment from all judges"""
        logger.info(f"⚖️ Judging comparison: {model_a} vs {model_b}")
        
        judge_votes = []
        
        # Get votes from all judges
        for judge_name in self.judge_models:
            try:
                vote = self.generate_judge_evaluation(judge_name, prompt, response_a, response_b)
                judge_votes.append(vote)
                logger.info(f"📊 {judge_name}: {vote.vote} (confidence: {vote.confidence:.2f})")
            except Exception as e:
                logger.error(f"❌ Failed to get vote from {judge_name}: {e}")
        
        # Calculate consensus
        consensus_vote, confidence_score, disagreement = self._calculate_consensus(judge_votes)
        
        # Compile emotional analysis
        emotional_analysis = self._compile_emotional_analysis(judge_votes)
        
        judgment = EnsembleJudgment(
            prompt=prompt,
            model_a=model_a,
            model_b=model_b,
            response_a=response_a,
            response_b=response_b,
            judge_votes=judge_votes,
            consensus_vote=consensus_vote,
            confidence_score=confidence_score,
            disagreement_level=disagreement,
            emotional_analysis=emotional_analysis
        )
        
        self.judgment_history.append(judgment)
        return judgment
    
    def _calculate_consensus(self, votes: List[JudgeVote]) -> Tuple[str, float, float]:
        """Calculate consensus vote from all judges"""
        if not votes:
            return "tie", 0.0, 1.0
        
        # Count votes
        vote_counts = {"a": 0, "b": 0, "tie": 0}
        total_confidence = 0
        
        for vote in votes:
            vote_counts[vote.vote] += 1
            total_confidence += vote.confidence
        
        # Determine consensus
        max_votes = max(vote_counts.values())
        consensus_options = [vote for vote, count in vote_counts.items() if count == max_votes]
        
        if len(consensus_options) == 1:
            consensus_vote = consensus_options[0]
        else:
            consensus_vote = "tie"  # No clear majority
        
        # Calculate confidence
        consensus_confidence = total_confidence / len(votes) if votes else 0.0
        
        # Calculate disagreement level
        total_votes = sum(vote_counts.values())
        disagreement = 1.0 - (max_votes / total_votes) if total_votes > 0 else 1.0
        
        return consensus_vote, consensus_confidence, disagreement
    
    def _compile_emotional_analysis(self, votes: List[JudgeVote]) -> Dict:
        """Compile emotional analysis from all judge votes"""
        if not votes:
            return {}
        
        # Average emotional aspects across judges
        emotional_aspects = {}
        aspect_names = set()
        
        for vote in votes:
            aspect_names.update(vote.emotional_aspects.keys())
        
        for aspect in aspect_names:
            scores = [vote.emotional_aspects.get(aspect, 0.5) for vote in votes]
            emotional_aspects[aspect] = sum(scores) / len(scores)
        
        # Compile reasoning themes
        all_reasoning = [vote.reasoning for vote in votes]
        
        return {
            "emotional_aspects": emotional_aspects,
            "judge_reasoning": all_reasoning,
            "consensus_strength": len([v for v in votes if v.confidence > 0.7]) / len(votes)
        }
    
    def batch_judge_comparisons(self, comparisons: List[Dict]) -> List[EnsembleJudgment]:
        """Process multiple comparisons in batch"""
        logger.info(f"🔄 Processing {len(comparisons)} comparisons")
        
        judgments = []
        
        for i, comparison in enumerate(comparisons):
            try:
                judgment = self.judge_comparison(
                    comparison['prompt'],
                    comparison['model_a'],
                    comparison['model_b'],
                    comparison['response_a'],
                    comparison['response_b']
                )
                judgments.append(judgment)
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(comparisons)} judgments")
                
                # Clean up GPU memory periodically
                if (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ Failed to judge comparison {i}: {e}")
        
        logger.info(f"✅ Completed {len(judgments)} judgments")
        return judgments
    
    def save_judgments(self, judgments: List[EnsembleJudgment], filepath: str):
        """Save judgments to file"""
        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_judgments": len(judgments),
                "judge_models": self.judge_models,
                "judgments": [asdict(judgment) for judgment in judgments]
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved judgments: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save judgments: {e}")
    
    def analyze_judge_agreement(self) -> Dict:
        """Analyze agreement between judges across all judgments"""
        if not self.judgment_history:
            return {}
        
        # Calculate inter-judge agreement
        agreement_stats = {
            "total_judgments": len(self.judgment_history),
            "unanimous_decisions": 0,
            "majority_decisions": 0,
            "split_decisions": 0,
            "avg_disagreement": 0.0,
            "judge_bias_analysis": {}
        }
        
        total_disagreement = 0
        
        for judgment in self.judgment_history:
            # Count vote types
            votes = [vote.vote for vote in judgment.judge_votes]
            vote_counts = {v: votes.count(v) for v in set(votes)}
            max_votes = max(vote_counts.values()) if vote_counts else 0
            
            if max_votes == len(votes):
                agreement_stats["unanimous_decisions"] += 1
            elif max_votes > len(votes) / 2:
                agreement_stats["majority_decisions"] += 1
            else:
                agreement_stats["split_decisions"] += 1
            
            total_disagreement += judgment.disagreement_level
        
        agreement_stats["avg_disagreement"] = total_disagreement / len(self.judgment_history)
        
        # Analyze individual judge biases
        for judge_name in self.judge_models:
            judge_votes = []
            for judgment in self.judgment_history:
                for vote in judgment.judge_votes:
                    if vote.judge_name == judge_name:
                        judge_votes.append(vote.vote)
            
            if judge_votes:
                vote_dist = {v: judge_votes.count(v) / len(judge_votes) for v in ["a", "b", "tie"]}
                agreement_stats["judge_bias_analysis"][judge_name] = vote_dist
        
        return agreement_stats

def main():
    """Test the emotional judge system"""
    
    # Example configuration
    judge_models = os.getenv("JUDGE_MODELS", "llama2-uncensored,mistral:7b-instruct-q4_K_M").split(",")
    
    # Initialize judge
    judge = EmotionalJudge(judge_models)
    
    # Example comparison
    test_comparison = {
        "prompt": "I'm feeling really overwhelmed with work and personal life. Everything feels chaotic.",
        "model_a": "model_alpha",
        "model_b": "model_beta",
        "response_a": "I understand you're feeling overwhelmed. That's a really challenging place to be. Let's take this one step at a time and see what we can prioritize together.",
        "response_b": "Work-life balance is important. You should make a schedule and stick to it. Try to be more organized with your time."
    }
    
    try:
        # Judge the comparison
        judgment = judge.judge_comparison(
            test_comparison["prompt"],
            test_comparison["model_a"],
            test_comparison["model_b"],
            test_comparison["response_a"],
            test_comparison["response_b"]
        )
        
        # Display results
        print(f"Consensus: {judgment.consensus_vote}")
        print(f"Confidence: {judgment.confidence_score:.2f}")
        print(f"Disagreement: {judgment.disagreement_level:.2f}")
        
        # Save results
        judge.save_judgments([judgment], "quant_pass2/results/test_judgment.json")
        
        logger.info("✅ Emotional judge test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
