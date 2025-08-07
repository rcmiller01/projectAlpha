#!/usr/bin/env python3
"""
Emotion loop core script for the Emotional Presence Engine.

This script processes emotional states, evaluates context, generates responses, 
and logs the results for further analysis. Includes thread safety via copy-on-write, 
logging for loop ticks, damping/cross-emotion inhibition models, memory pruning, 
and synthetic introspection foundations. Now includes a revival mechanic for phantom memories.
"""
import json
import time
import logging
from pathlib import Path
from copy import deepcopy
from collections import Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionState:
    """Explicit struct for representing an emotion."""
    def __init__(self, name, intensity, context, priority=1.0, original_valence=None):
        self.name = name
        self.intensity = intensity
        self.context = context
        self.priority = priority
        self.original_valence = original_valence or intensity  # Default to initial intensity

    def apply_damping(self, factor=0.9):
        """Apply damping to reduce intensity."""
        self.intensity *= factor

    def inhibit(self, other_emotion, inhibition_factor=0.2):
        """Apply cross-emotion inhibition."""
        if self.name != other_emotion.name:
            self.intensity -= other_emotion.intensity * inhibition_factor
            self.intensity = max(0, self.intensity)  # Ensure intensity is non-negative

    def decay_priority(self, decay_rate=0.05):
        """Reduce priority over time."""
        self.priority -= decay_rate
        self.priority = max(0, self.priority)  # Ensure priority is non-negative

    def is_painful(self, pain_threshold=0.8):
        """Determine if the emotion is painful based on intensity."""
        return self.intensity > pain_threshold

    def has_drifted(self, drift_threshold=0.3):
        """Check if the emotion has drifted too far from its original valence."""
        drift = abs(self.intensity - self.original_valence)
        return drift > drift_threshold

    def revive_as_phantom(self):
        """Revive the memory as a phantom with reduced clarity and increased priority."""
        self.intensity *= 0.7  # Reduce clarity
        self.priority += 0.5  # Increase emotional weight
        logger.info(f"Memory revived as phantom: {self.__dict__}")

def revive_forgotten_memories(emotions, forgotten_memories, trigger_context):
    """
    Revive forgotten memories if they are emotionally re-triggered.

    Args:
        emotions (list): List of current EmotionState objects.
        forgotten_memories (list): List of forgotten EmotionState objects.
        trigger_context (str): Context that may trigger revival.

    Returns:
        list: Updated list of EmotionState objects with revived memories.
    """
    for memory in forgotten_memories:
        if trigger_context.lower() in memory.context.lower():
            memory.revive_as_phantom()
            emotions.append(memory)
    return emotions

# Emotion queue for throttling
emotion_queue = []

def queue_emotion_for_later(emotion):
    """
    Queue an emotion for later processing during high affective load.
    
    Args:
        emotion (EmotionState): The emotion to queue for later processing.
    """
    global emotion_queue
    emotion_queue.append(emotion)
    logger.info(f"Queued emotion '{emotion.name}' for later processing")

def process_queued_emotions(emotions, max_process=3):
    """
    Process queued emotions gradually to avoid overload.
    
    Args:
        emotions (list): Current list of EmotionState objects.
        max_process (int): Maximum number of queued emotions to process per cycle.
        
    Returns:
        list: Updated list of EmotionState objects.
    """
    global emotion_queue
    if emotion_queue:
        to_process = emotion_queue[:max_process]
        emotion_queue = emotion_queue[max_process:]
        emotions.extend(to_process)
        logger.info(f"Processed {len(to_process)} queued emotions")
    return emotions

def throttle_emotions(emotions, affective_score_delta, threshold=0.5):
    """
    Throttle emotions by queuing them if the affective score delta exceeds the threshold.
    
    Args:
        emotions (list): List of EmotionState objects.
        affective_score_delta (float): Change in affective score.
        threshold (float): Threshold for queuing emotions.
        
    Returns:
        list: Updated list of EmotionState objects.
    """
    if affective_score_delta > threshold:
        logger.warning(f"Affective score delta {affective_score_delta} exceeds threshold {threshold}. Queuing high-intensity emotions.")
        # Queue emotions with intensity above threshold
        high_intensity_emotions = [e for e in emotions if e.intensity > threshold]
        for emotion in high_intensity_emotions:
            queue_emotion_for_later(emotion)
        # Return emotions below threshold
        return [e for e in emotions if e.intensity <= threshold]
    return emotions

def load_state(config_path):
    """
    Load the initial state from the configuration file.

    Args:
        config_path (Path): Path to the configuration file containing seed emotions.

    Returns:
        list: List of EmotionState objects.
    """
    with open(config_path) as f:
        raw_emotions = json.load(f)["emotions"]
    return [EmotionState(e["name"], e["intensity"], e["context"]) for e in raw_emotions]

def validate_and_prune_emotions(emotions, max_emotions=100):
    """
    Validate and prune the list of emotions to prevent unbounded growth.

    Args:
        emotions (list): List of EmotionState objects.
        max_emotions (int): Maximum number of emotions to retain.

    Returns:
        tuple: Pruned list of EmotionState objects and forgotten memories.
    """
    # Sort emotions by priority (highest first) and prune excess
    emotions = sorted(emotions, key=lambda e: e.priority, reverse=True)
    pruned_emotions = emotions[:max_emotions]
    forgotten_memories = emotions[max_emotions:]

    # Remove painful memories based on heuristics
    pruned_emotions = [e for e in pruned_emotions if not e.is_painful()]

    return pruned_emotions, forgotten_memories

def evaluate_context(emotions):
    """
    Process the loaded emotions to extract relevant data.

    Args:
        emotions (list): List of EmotionState objects.

    Returns:
        list: Processed list of emotions.
    """
    for emotion in emotions:
        emotion.apply_damping()
        emotion.decay_priority()
        for other_emotion in emotions:
            emotion.inhibit(other_emotion)
    return emotions

def apply_response(processed_emotions):
    """
    Generate a response or summary based on processed emotions.

    Args:
        processed_emotions (list): List of processed EmotionState objects.

    Returns:
        dict: Summary of the emotional processing cycle.
    """
    return {
        "timestamp": time.time(),
        "processed_emotions": [
            {"name": e.name, "intensity": e.intensity, "context": e.context, "priority": e.priority}
            for e in processed_emotions
        ],
        "summary": "Cycle complete"
    }

def record_trace(result, log_path, insights_path):
    """
    Log the results and insights to specified files.

    Args:
        result (dict): Summary of the emotional processing cycle.
        log_path (Path): Path to the log file for storing results.
        insights_path (Path): Path to the insights file for storing summaries.
    """
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    with open(insights_path, "w") as f:
        f.write(json.dumps({"last_summary": result["summary"]}, indent=2))

def process_emotional_loop(emotions, forgotten_memories, trigger_context):
    """
    Process the emotional loop using a copy-on-write mechanism.

    Args:
        emotions (list): List of EmotionState objects.
        forgotten_memories (list): List of forgotten EmotionState objects.
        trigger_context (str): Context that may trigger revival.

    Returns:
        dict: Updated emotional state data after processing.
    """
    # Create a local copy of the emotional state
    local_emotions = deepcopy(emotions)

    # Log the initial state
    logger.info("Starting emotional loop with state: %s", [e.__dict__ for e in local_emotions])

    # Process the emotions (simulate updates)
    processed_emotions = evaluate_context(local_emotions)

    # Revive forgotten memories if triggered
    processed_emotions = revive_forgotten_memories(processed_emotions, forgotten_memories, trigger_context)

    # Validate and prune emotions
    pruned_emotions, forgotten_memories = validate_and_prune_emotions(processed_emotions)

    # Throttle emotions if affective score delta is high
    affective_score_delta = sum(e.intensity for e in pruned_emotions) - sum(e.intensity for e in emotions)
    pruned_emotions = throttle_emotions(pruned_emotions, affective_score_delta)

    result = apply_response(pruned_emotions)

    # Log the processed state
    logger.info("Processed emotional state: %s", result["processed_emotions"])

    return result, forgotten_memories

def main():
    """
    Main function to execute the emotional processing loop.

    This function loads the initial state, processes emotions, generates a response, 
    and logs the results and insights.
    """
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "seed_emotions.json"
    log_path = base_dir / "logs" / "loop_results.jsonl"
    insights_path = base_dir / "logs" / "anchor_insights.json"

    emotions = load_state(config_path)
    forgotten_memories = []
    trigger_context = "example trigger"  # Replace with dynamic context

    result, forgotten_memories = process_emotional_loop(emotions, forgotten_memories, trigger_context)
    record_trace(result, log_path, insights_path)

if __name__ == "__main__":
    main()
