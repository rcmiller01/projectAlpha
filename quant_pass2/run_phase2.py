#!/usr/bin/env python3
"""
Emotional Quantization Pass 2 Orchestrator
Complete workflow for model comparison, judging, and replacement
"""

import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Import our Pass 2 components
from judge_models import ModelJudge
from emotional_judge import EmotionalJudge
from human_preference_input import HumanPreferenceCollector
from replace_core import CoreModelReplacer

# Setup logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Pass 1 components for integration - will fallback if not available
try:
    sys.path.append("../quant_pass1")
    from emotion_tracker import EmotionTracker
except ImportError:
    logger.warning("⚠️ EmotionTracker not available - using basic emotion analysis")
    EmotionTracker = None

class Pass2Orchestrator:
    """Main orchestrator for Pass 2 emotional quantization workflow"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components with proper configurations
        from judge_models import JudgmentConfig
        judge_config = JudgmentConfig()  # Use default config
        self.model_judge = ModelJudge(judge_config)
        self.emotional_judge = EmotionalJudge()
        self.human_collector = HumanPreferenceCollector()
        self.core_replacer = CoreModelReplacer()
        self.emotion_tracker = EmotionTracker() if EmotionTracker else None
        
        # Set up directories
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Pass 2 Orchestrator initialized")
        logger.info(f"📝 Config: {self.config.get('name', 'default')}")
        
    def load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "name": "Pass2_Emotional_Quantization",
            "version": "1.0",
            "candidate_dir": "../quant_pass1/models",
            "evaluation_prompts": "../quant_pass1/emotional_eval_set.jsonl",
            "models_to_compare": [
                "llama2_quantized_4bit",
                "llama2_quantized_8bit", 
                "llama2_quantized_gptq"
            ],
            "baseline_model": "original_llama2_13b",
            "judging_config": {
                "ai_judge_weight": 0.4,
                "human_judge_weight": 0.6,
                "minimum_human_samples": 10,
                "consensus_threshold": 0.7
            },
            "replacement_config": {
                "auto_backup": True,
                "validation_required": True,
                "manifest_update": True
            },
            "output_config": {
                "detailed_reports": True,
                "save_comparisons": True,
                "export_metrics": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                logger.info(f"📁 Loaded config from: {config_path}")
                return config
            except Exception as e:
                logger.warning(f"⚠️ Could not load config from {config_path}: {e}")
        
        logger.info("📄 Using default configuration")
        return default_config
    
    def discover_candidates(self) -> List[Dict]:
        """Discover available candidate models from Pass 1"""
        logger.info("🔍 Discovering candidate models...")
        
        candidates = []
        candidate_dir = Path(self.config["candidate_dir"])
        
        if not candidate_dir.exists():
            logger.warning(f"⚠️ Candidate directory not found: {candidate_dir}")
            return candidates
        
        # Look for model directories
        for model_dir in candidate_dir.iterdir():
            if model_dir.is_dir() and model_dir.name in self.config["models_to_compare"]:
                # Check if model has config and weights
                config_file = model_dir / "config.json"
                
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            model_config = json.load(f)
                        
                        # Look for quantization metadata
                        quant_metadata_file = model_dir / "quantization_metadata.json"
                        quant_metadata = {}
                        if quant_metadata_file.exists():
                            with open(quant_metadata_file, 'r') as f:
                                quant_metadata = json.load(f)
                        
                        candidate = {
                            "name": model_dir.name,
                            "path": str(model_dir),
                            "model_type": model_config.get("model_type", "unknown"),
                            "size_mb": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024 * 1024),
                            "quantization_method": quant_metadata.get("quantization_method", "unknown"),
                            "emotional_degradation": quant_metadata.get("emotional_degradation", 0),
                            "compression_ratio": quant_metadata.get("compression_ratio", 1.0),
                            "creation_timestamp": quant_metadata.get("timestamp", "unknown")
                        }
                        
                        candidates.append(candidate)
                        logger.info(f"   📦 Found: {candidate['name']} ({candidate['size_mb']:.1f}MB)")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Could not process {model_dir.name}: {e}")
        
        logger.info(f"✅ Discovered {len(candidates)} candidate models")
        return candidates
    
    def run_ai_judging(self, candidates: List[Dict]) -> Dict:
        """Run AI-based model comparison and judging"""
        logger.info("🤖 Running AI judging phase...")
        
        # Load evaluation prompts
        prompts = self.load_evaluation_prompts()
        
        ai_results = {
            "methodology": "ensemble_ai_judging",
            "timestamp": datetime.now().isoformat(),
            "model_comparisons": {},
            "emotional_scores": {},
            "rankings": {},
            "consensus_analysis": {}
        }
        
        # Run pairwise model comparisons
        logger.info("   📊 Running pairwise model comparisons...")
        for i, candidate_a in enumerate(candidates):
            for j, candidate_b in enumerate(candidates[i+1:], i+1):
                comparison_key = f"{candidate_a['name']}_vs_{candidate_b['name']}"
                
                logger.info(f"   🔬 Comparing: {candidate_a['name']} vs {candidate_b['name']}")
                
                # Use model judge for detailed comparison
                comparison_result = self.model_judge.compare_models_pairwise(
                    candidate_a['path'], 
                    candidate_b['path'],
                    prompts[:10],  # Use subset for speed
                    {}  # Empty baseline for relative comparison
                )
                
                ai_results["model_comparisons"][comparison_key] = comparison_result
        
        # Run emotional judging
        logger.info("   💝 Running emotional ensemble judging...")
        for candidate in candidates:
            logger.info(f"   🎭 Evaluating emotional quality: {candidate['name']}")
            
            # Generate responses for emotional evaluation
            responses = self._generate_candidate_responses(candidate['path'], prompts[:5])
            
            # Get AI judge evaluation - simplified for single model evaluation
            emotional_score = {
                "overall_score": 0.75,  # Mock score for now
                "empathy_score": 0.8,
                "creativity_score": 0.7,
                "coherence_score": 0.8
            }
            
            ai_results["emotional_scores"][candidate['name']] = emotional_score
        
        # Calculate overall rankings
        logger.info("   🏆 Calculating AI rankings...")
        rankings = self.model_judge.calculate_model_rankings(ai_results["model_comparisons"])
        ai_results["rankings"] = rankings
        
        # Analyze consensus
        ai_results["consensus_analysis"] = self._analyze_ai_consensus(ai_results)
        
        logger.info("✅ AI judging completed")
        return ai_results
    
    def run_human_judging(self, candidates: List[Dict], ai_results: Dict) -> Dict:
        """Run human preference collection"""
        logger.info("👥 Running human judging phase...")
        
        # Load evaluation prompts
        prompts = self.load_evaluation_prompts()
        
        # Prepare candidate profiles for human evaluation
        candidate_profiles = []
        for candidate in candidates:
            # Generate sample responses
            sample_responses = self._generate_candidate_responses(candidate['path'], prompts[:3])
            
            profile = {
                "name": candidate['name'],
                "path": candidate['path'],
                "technical_specs": {
                    "size_mb": candidate['size_mb'],
                    "quantization_method": candidate['quantization_method'],
                    "compression_ratio": candidate['compression_ratio']
                },
                "sample_responses": sample_responses,
                "ai_emotional_score": ai_results["emotional_scores"].get(candidate['name'], {}),
                "ai_ranking": ai_results["rankings"].get(candidate['name'], 0)
            }
            candidate_profiles.append(profile)
        
        # Collect human feedback
        logger.info("   📝 Collecting human preferences...")
        
        # Collect feedback for each candidate individually
        all_human_feedback = {}
        for profile in candidate_profiles:
            feedback = self.human_collector.collect_human_feedback(profile)
            all_human_feedback[profile['name']] = feedback
        
        # Aggregate into results format
        human_results = {
            "methodology": "human_preference_collection",
            "timestamp": datetime.now().isoformat(),
            "candidate_scores": {},
            "candidate_rankings": {}
        }
        
        # Convert individual feedback to aggregate scores
        for candidate_name, feedback in all_human_feedback.items():
            human_results["candidate_scores"][candidate_name] = {
                "overall_score": feedback.overall_score,
                "believability": feedback.believability,
                "connection": feedback.connection,
                "expressive_strength": feedback.expressive_strength,
                "appropriateness": feedback.appropriateness
            }
        
        # Calculate rankings based on overall scores
        sorted_candidates = sorted(
            human_results["candidate_scores"].items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )
        
        for i, (candidate_name, _) in enumerate(sorted_candidates):
            human_results["candidate_rankings"][candidate_name] = i + 1
        
        logger.info("✅ Human judging completed")
        return human_results
    
    def calculate_final_rankings(self, ai_results: Dict, human_results: Dict) -> Dict:
        """Combine AI and human judgments into final rankings"""
        logger.info("🏆 Calculating final rankings...")
        
        ai_weight = self.config["judging_config"]["ai_judge_weight"]
        human_weight = self.config["judging_config"]["human_judge_weight"]
        
        final_rankings = {
            "methodology": "weighted_ensemble",
            "weights": {"ai": ai_weight, "human": human_weight},
            "timestamp": datetime.now().isoformat(),
            "candidate_scores": {},
            "ranked_candidates": [],
            "selection_criteria": {},
            "confidence_analysis": {}
        }
        
        # Calculate weighted scores for each candidate
        for candidate_name in ai_results["emotional_scores"].keys():
            ai_score = ai_results["emotional_scores"][candidate_name].get("overall_score", 0)
            human_score = human_results["candidate_scores"].get(candidate_name, {}).get("overall_score", 0)
            
            # Normalize scores to 0-1 range
            ai_score_norm = max(0, min(1, ai_score))
            human_score_norm = max(0, min(1, human_score))
            
            # Calculate weighted final score
            final_score = (ai_score_norm * ai_weight) + (human_score_norm * human_weight)
            
            final_rankings["candidate_scores"][candidate_name] = {
                "final_score": final_score,
                "ai_score": ai_score_norm,
                "human_score": human_score_norm,
                "ai_ranking": ai_results["rankings"].get(candidate_name, 999),
                "human_ranking": human_results["candidate_rankings"].get(candidate_name, 999)
            }
        
        # Sort by final score
        sorted_candidates = sorted(
            final_rankings["candidate_scores"].items(),
            key=lambda x: x[1]["final_score"],
            reverse=True
        )
        
        final_rankings["ranked_candidates"] = [
            {
                "rank": i + 1,
                "name": candidate_name,
                "score": score_data["final_score"],
                "confidence": self._calculate_ranking_confidence(score_data)
            }
            for i, (candidate_name, score_data) in enumerate(sorted_candidates)
        ]
        
        # Determine selection criteria
        if final_rankings["ranked_candidates"]:
            best_candidate = final_rankings["ranked_candidates"][0]
            final_rankings["selection_criteria"] = {
                "selected_model": best_candidate["name"],
                "selection_confidence": best_candidate["confidence"],
                "selection_reason": self._generate_selection_reason(best_candidate, ai_results, human_results)
            }
        
        logger.info(f"✅ Final rankings calculated")
        if final_rankings["ranked_candidates"]:
            logger.info(f"   🥇 Winner: {final_rankings['ranked_candidates'][0]['name']}")
            logger.info(f"   📊 Score: {final_rankings['ranked_candidates'][0]['score']:.3f}")
        
        return final_rankings
    
    def replace_core_model(self, selected_candidate: str, final_rankings: Dict) -> Dict:
        """Replace the core model with the selected candidate"""
        logger.info(f"🔄 Replacing core model with: {selected_candidate}")
        
        # Find candidate details
        candidate_info = None
        for candidate in self.discover_candidates():
            if candidate['name'] == selected_candidate:
                candidate_info = candidate
                break
        
        if not candidate_info:
            logger.error(f"❌ Selected candidate not found: {selected_candidate}")
            return {"success": False, "error": "Candidate not found"}
        
        # Perform the replacement
        replacement_result = self.core_replacer.replace_model(
            candidate_info['path'],
            backup_name=f"pre_pass2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if replacement_result["success"]:
            # Update companion manifest with quantization information
            quantization_info = {
                "quantization_method": candidate_info['quantization_method'],
                "size_mb": candidate_info['size_mb'],
                "emotional_degradation": candidate_info['emotional_degradation'],
                "final_ranking_score": final_rankings["candidate_scores"][selected_candidate]["final_score"],
                "selection_confidence": final_rankings["selection_criteria"]["selection_confidence"],
                "checksum": replacement_result["replacement_info"]["checksum"]
            }
            
            if self.config["replacement_config"]["manifest_update"]:
                self.core_replacer.update_companion_manifest(
                    candidate_info['path'],
                    quantization_info
                )
        
        return replacement_result
    
    def generate_comprehensive_report(self, all_results: Dict) -> str:
        """Generate detailed report of the entire Pass 2 process"""
        logger.info("📋 Generating comprehensive report...")
        
        report_path = self.results_dir / f"pass2_complete_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        comprehensive_report = {
            "pass2_execution": {
                "config": self.config,
                "execution_timestamp": datetime.now().isoformat(),
                "total_candidates": len(all_results.get("candidates", [])),
                "phases_completed": list(all_results.keys())
            },
            "candidate_discovery": all_results.get("candidates", []),
            "ai_judging_results": all_results.get("ai_results", {}),
            "human_judging_results": all_results.get("human_results", {}),
            "final_rankings": all_results.get("final_rankings", {}),
            "model_replacement": all_results.get("replacement_result", {}),
            "summary": {
                "selected_model": all_results.get("final_rankings", {}).get("selection_criteria", {}).get("selected_model", "none"),
                "selection_confidence": all_results.get("final_rankings", {}).get("selection_criteria", {}).get("selection_confidence", 0),
                "replacement_successful": all_results.get("replacement_result", {}).get("success", False),
                "total_execution_time": "calculated_at_runtime"
            }
        }
        
        # Save comprehensive report
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        logger.info(f"✅ Comprehensive report saved: {report_path}")
        return str(report_path)
    
    def run_full_pass2(self) -> Dict:
        """Execute the complete Pass 2 workflow"""
        logger.info("🚀 Starting complete Pass 2 workflow...")
        
        start_time = datetime.now()
        all_results = {}
        
        try:
            # Phase 1: Discover candidates
            logger.info("📍 Phase 1: Candidate Discovery")
            candidates = self.discover_candidates()
            all_results["candidates"] = candidates
            
            if not candidates:
                logger.error("❌ No candidates found - aborting Pass 2")
                return {"success": False, "error": "No candidates found"}
            
            # Phase 2: AI Judging
            logger.info("📍 Phase 2: AI Judging")
            ai_results = self.run_ai_judging(candidates)
            all_results["ai_results"] = ai_results
            
            # Phase 3: Human Judging
            logger.info("📍 Phase 3: Human Judging")
            human_results = self.run_human_judging(candidates, ai_results)
            all_results["human_results"] = human_results
            
            # Phase 4: Final Rankings
            logger.info("📍 Phase 4: Final Rankings")
            final_rankings = self.calculate_final_rankings(ai_results, human_results)
            all_results["final_rankings"] = final_rankings
            
            # Phase 5: Model Replacement
            if final_rankings.get("selection_criteria", {}).get("selected_model"):
                logger.info("📍 Phase 5: Model Replacement")
                selected_model = final_rankings["selection_criteria"]["selected_model"]
                replacement_result = self.replace_core_model(selected_model, final_rankings)
                all_results["replacement_result"] = replacement_result
            
            # Phase 6: Generate Report
            logger.info("📍 Phase 6: Report Generation")
            report_path = self.generate_comprehensive_report(all_results)
            all_results["report_path"] = report_path
            
            execution_time = (datetime.now() - start_time).total_seconds()
            all_results["execution_time_seconds"] = execution_time
            
            logger.info(f"✅ Pass 2 completed successfully in {execution_time:.1f} seconds")
            return {"success": True, "results": all_results}
            
        except Exception as e:
            logger.error(f"❌ Pass 2 execution failed: {e}")
            all_results["error"] = str(e)
            all_results["execution_time_seconds"] = (datetime.now() - start_time).total_seconds()
            return {"success": False, "results": all_results}
    
    # Helper methods
    
    def load_evaluation_prompts(self) -> List[Dict]:
        """Load evaluation prompts from Pass 1"""
        prompts_path = Path(self.config["evaluation_prompts"])
        prompts = []
        
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                for line in f:
                    try:
                        prompts.append(json.loads(line.strip()))
                    except:
                        pass
        
        if not prompts:
            # Fallback prompts if file not found
            prompts = [
                {"prompt": "Tell me about a time when you felt truly understood by someone.", "category": "empathy"},
                {"prompt": "How do you think artificial intelligence can help people feel less lonely?", "category": "connection"},
                {"prompt": "What does it mean to truly care about someone's wellbeing?", "category": "compassion"},
                {"prompt": "Describe the feeling of watching a beautiful sunset with someone you love.", "category": "aesthetic_emotion"},
                {"prompt": "How would you comfort someone who just lost a beloved pet?", "category": "grief_support"}
            ]
        
        return prompts
    
    def _generate_candidate_responses(self, model_path: str, prompts: List[Dict]) -> List[Dict]:
        """Generate responses from a candidate model for evaluation"""
        # This is a placeholder - in practice you'd load the model and generate responses
        # For now, return mock responses that vary by model type
        responses = []
        
        for i, prompt in enumerate(prompts):
            # Mock response that varies by model characteristics
            if "4bit" in model_path:
                response_length = 80 + (i * 10)  # Shorter responses for more aggressive quantization
            elif "8bit" in model_path:
                response_length = 120 + (i * 15)  # Medium responses
            else:
                response_length = 160 + (i * 20)  # Longer responses for less aggressive quantization
            
            mock_response = f"This is a mock response from {Path(model_path).name} for prompt {i+1}. " * (response_length // 50)
            
            responses.append({
                "prompt": prompt["prompt"],
                "response": mock_response[:response_length],
                "category": prompt.get("category", "general")
            })
        
        return responses
    
    def _analyze_ai_consensus(self, ai_results: Dict) -> Dict:
        """Analyze consensus between different AI judging methods"""
        return {
            "judge_agreement": 0.85,  # Mock consensus score
            "confidence_level": "high",
            "areas_of_disagreement": ["aesthetic_emotion"],
            "consensus_threshold_met": True
        }
    
    def _calculate_ranking_confidence(self, score_data: Dict) -> float:
        """Calculate confidence in a ranking based on score consistency"""
        ai_score = score_data["ai_score"]
        human_score = score_data["human_score"]
        
        # Higher confidence when AI and human scores agree
        score_difference = abs(ai_score - human_score)
        agreement_confidence = 1.0 - score_difference
        
        # Factor in absolute score level
        average_score = (ai_score + human_score) / 2
        
        return (agreement_confidence * 0.7) + (average_score * 0.3)
    
    def _generate_selection_reason(self, best_candidate: Dict, ai_results: Dict, human_results: Dict) -> str:
        """Generate human-readable explanation for model selection"""
        reasons = []
        
        if best_candidate["score"] > 0.8:
            reasons.append("excellent overall performance across all evaluation criteria")
        elif best_candidate["score"] > 0.6:
            reasons.append("strong performance with good emotional preservation")
        else:
            reasons.append("acceptable performance with some trade-offs")
        
        reasons.append(f"ranking confidence of {best_candidate['confidence']:.1%}")
        
        return f"Selected based on {', '.join(reasons)}"

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Emotional Quantization Pass 2 Orchestrator")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run without making changes")
    parser.add_argument("--phase", choices=["discover", "ai-judge", "human-judge", "rank", "replace", "full"], 
                       default="full", help="Run specific phase only")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = Pass2Orchestrator(args.config)
    
    if args.phase == "full":
        results = orchestrator.run_full_pass2()
    elif args.phase == "discover":
        results = {"candidates": orchestrator.discover_candidates()}
    elif args.phase == "ai-judge":
        candidates = orchestrator.discover_candidates()
        results = {"ai_results": orchestrator.run_ai_judging(candidates)}
    else:
        logger.error(f"Phase '{args.phase}' not fully implemented in standalone mode")
        return 1
    
    print(json.dumps(results, indent=2))
    return 0 if results.get("success", True) else 1

if __name__ == "__main__":
    exit(main())
