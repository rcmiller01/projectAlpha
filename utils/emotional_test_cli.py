#!/usr/bin/env python3
"""
Emotional Test CLI - Advanced testing suite for quantized model emotional evaluation.

This script provides comprehensive emotional testing capabilities for the
Unified AI Companion's quantization evaluation system.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from emotion_loop_core import (
        EmotionLoopManager, 
        QuantizationCandidate, 
        run_emotional_test, 
        run_emotional_test_suite
    )
except ImportError as e:
    print(f"❌ Failed to import emotion_loop_core: {e}")
    print("   Make sure you're running from the project root directory")
    sys.exit(1)

def create_test_candidates(names: Optional[List[str]] = None) -> List[QuantizationCandidate]:
    """Create test candidates for emotional evaluation."""
    if names is None:
        names = ["model_q8", "model_q6", "model_q5", "model_q4", "model_q3"]
    
    candidates = []
    base_sizes = {"q8": 15.2, "q6": 12.5, "q5": 10.2, "q4": 8.8, "q3": 6.5}
    
    for name in names:
        # Extract quantization level
        q_level = None
        for q in base_sizes.keys():
            if q in name.lower():
                q_level = q
                break
        
        size = base_sizes.get(q_level, 10.0)
        candidates.append(QuantizationCandidate(
            name=name,
            size_gb=size,
            file_path=f"models/{name}.bin"
        ))
    
    return candidates

def load_custom_prompts(file_path: str) -> List[str]:
    """Load custom prompts from a text or JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'prompts' in data:
                    return data['prompts']
                else:
                    print(f"⚠️  JSON file should contain a list or {'prompts': [...]} object")
                    return []
            else:
                # Text file - one prompt per line
                return [line.strip() for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Failed to load prompts from {file_path}: {e}")
        return []

def save_test_results(results: dict, output_file: str) -> None:
    """Save test results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Test results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

def run_comparative_analysis(results: dict) -> None:
    """Run comparative analysis of emotional test results."""
    print(f"\n📊 Comparative Emotional Analysis")
    print("=" * 60)
    
    # Find best and worst performers
    candidates = list(results.keys())
    if not candidates:
        print("No results to analyze")
        return
    
    averages = {name: data.get('average', 0.0) for name, data in results.items()}
    best_performer = max(averages, key=averages.get)
    worst_performer = min(averages, key=averages.get)
    
    print(f"🏆 Best emotional performer: {best_performer} (avg: {averages[best_performer]:.3f})")
    print(f"⚠️  Needs improvement: {worst_performer} (avg: {averages[worst_performer]:.3f})")
    
    # Calculate score distribution
    all_scores = []
    for data in results.values():
        all_scores.extend([v for k, v in data.items() if k != 'average'])
    
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        min_score = min(all_scores)
        max_score = max(all_scores)
        
        print(f"\n📈 Score Distribution:")
        print(f"   Average: {avg_score:.3f}")
        print(f"   Range: {min_score:.3f} - {max_score:.3f}")
        print(f"   Total evaluations: {len(all_scores)}")

def main():
    parser = argparse.ArgumentParser(description="Emotional Test CLI for Quantized Models")
    parser.add_argument('--candidates', nargs='+', help='Model candidate names to test')
    parser.add_argument('--prompts-file', help='File containing custom prompts (JSON or text)')
    parser.add_argument('--output', default='emotional_test_results.json', help='Output file for results')
    parser.add_argument('--single-prompt', help='Test with a single custom prompt')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--anchor-config', help='Path to anchor settings config file')
    parser.add_argument('--analysis-only', help='Run analysis on existing results file')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    print("🎭 Emotional Test CLI - Quantized Model Evaluation")
    print("=" * 60)
    
    # Analysis-only mode
    if args.analysis_only:
        try:
            with open(args.analysis_only, 'r') as f:
                results = json.load(f)
            run_comparative_analysis(results)
            return
        except Exception as e:
            print(f"❌ Failed to load results file: {e}")
            return
    
    # Create candidates
    candidates = create_test_candidates(args.candidates)
    print(f"📋 Testing {len(candidates)} candidates: {[c.name for c in candidates]}")
    
    # Load custom prompts if specified
    custom_prompts = None
    if args.prompts_file:
        custom_prompts = load_custom_prompts(args.prompts_file)
        if custom_prompts:
            print(f"📝 Loaded {len(custom_prompts)} custom prompts")
        else:
            print("⚠️  No custom prompts loaded, using defaults")
    elif args.single_prompt:
        custom_prompts = [args.single_prompt]
        print(f"📝 Testing single prompt: {args.single_prompt}")
    
    # Initialize emotion loop manager
    config_path = args.anchor_config or 'config/anchor_settings.json'
    manager = EmotionLoopManager(config_path=config_path)
    print(f"🧭 Using anchor weights: {manager.anchor_weights}")
    
    # Run emotional test suite
    if args.single_prompt:
        print(f"\n🎯 Single Prompt Test")
        print("-" * 40)
        results = {}
        for candidate in candidates:
            score = run_emotional_test(candidate, args.single_prompt)
            results[candidate.name] = {"single_prompt": score, "average": score}
    else:
        results = run_emotional_test_suite(candidates, custom_prompts)
    
    # Run comparative analysis
    run_comparative_analysis(results)
    
    # Save results
    full_results = {
        "metadata": {
            "timestamp": str(Path().resolve()),
            "anchor_config": config_path,
            "anchor_weights": manager.anchor_weights,
            "candidates_tested": len(candidates),
            "prompts_used": len(custom_prompts) if custom_prompts else 8
        },
        "results": results
    }
    
    save_test_results(full_results, args.output)
    
    print(f"\n🎯 Emotional testing complete!")
    print(f"   Results saved to: {args.output}")
    print(f"   Run with --analysis-only {args.output} to re-analyze")

if __name__ == "__main__":
    main()
