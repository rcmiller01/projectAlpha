#!/usr/bin/env python3
"""
Emotion loop core script for the Emotional Presence Engine.

This script processes emotional states, evaluates context, generates responses,
and logs the results for further analysis. Includes thread safety via copy-on-write,
logging for loop ticks, damping/cross-emotion inhibition models, memory pruning,
and synthetic introspection foundations. Now includes a revival mechanic for phantom memories.

Enhanced with security features:
- Affective delta threshold monitoring
- Batch emotion updates for performance
- Comprehensive loop execution logging
- Input validation and sanitization
"""
import json
import time
import logging
from pathlib import Path
from copy import deepcopy
from collections import Counter, deque
import math
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
MAX_EMOTION_INTENSITY = 1.0
MIN_EMOTION_INTENSITY = 0.0
AFFECTIVE_DELTA_THRESHOLD = 0.8  # Alert if single emotion change exceeds this
BATCH_UPDATE_SIZE = 10  # Maximum emotions to update in single batch
MAX_CONTEXT_LENGTH = 1000  # Maximum length for emotion context strings
VALID_EMOTION_NAMES = {
    'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation',
    'contemplative', 'vibrant', 'serene', 'tender', 'passionate', 'mystical',
    'melancholic', 'euphoric', 'anxious', 'content', 'nostalgic', 'hopeful'
}

# Thread lock for emotion state modifications
emotion_lock = threading.Lock()

# Loop execution logging
loop_execution_log = deque(maxlen=1000)  # Keep last 1000 loop executions

def validate_emotion_input(emotion_data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate emotion input data for security and integrity"""
    try:
        # Check if input is dictionary
        if not isinstance(emotion_data, dict):
            return False, "Emotion data must be a dictionary"

        # Validate name
        if 'name' not in emotion_data:
            return False, "Missing required field: name"

        name = emotion_data['name']
        if not isinstance(name, str):
            return False, "Emotion name must be a string"

        # Sanitize name - only allow alphanumeric and valid emotion names
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            return False, "Emotion name contains invalid characters"

        if name.lower() not in VALID_EMOTION_NAMES:
            logger.warning(f"Unknown emotion name: {name}")

        # Validate intensity
        if 'intensity' in emotion_data:
            intensity = emotion_data['intensity']
            if not isinstance(intensity, (int, float)):
                return False, "Intensity must be a number"

            if intensity < MIN_EMOTION_INTENSITY or intensity > MAX_EMOTION_INTENSITY:
                return False, f"Intensity must be between {MIN_EMOTION_INTENSITY} and {MAX_EMOTION_INTENSITY}"

        # Validate context length
        if 'context' in emotion_data:
            context = emotion_data['context']
            if not isinstance(context, str):
                return False, "Context must be a string"

            if len(context) > MAX_CONTEXT_LENGTH:
                return False, f"Context length exceeds maximum of {MAX_CONTEXT_LENGTH} characters"

        # Validate priority
        if 'priority' in emotion_data:
            priority = emotion_data['priority']
            if not isinstance(priority, (int, float)):
                return False, "Priority must be a number"

            if priority < 0 or priority > 10:
                return False, "Priority must be between 0 and 10"

        return True, "Valid"

    except Exception as e:
        logger.error(f"Error validating emotion input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def log_loop_execution(loop_type: str, duration: float, emotions_processed: int,
                      affective_deltas: List[float], status: str = "success"):
    """Log emotion loop execution details"""
    try:
        execution_entry = {
            'timestamp': datetime.now().isoformat(),
            'loop_type': loop_type,
            'duration_ms': round(duration * 1000, 2),
            'emotions_processed': emotions_processed,
            'max_affective_delta': max(affective_deltas) if affective_deltas else 0,
            'avg_affective_delta': sum(affective_deltas) / len(affective_deltas) if affective_deltas else 0,
            'status': status,
            'thread_id': threading.get_ident()
        }

        loop_execution_log.append(execution_entry)

        # Log warning for high affective deltas
        max_delta = max(affective_deltas) if affective_deltas else 0
        if max_delta > AFFECTIVE_DELTA_THRESHOLD:
            logger.warning(f"High affective delta detected: {max_delta:.3f} in {loop_type} loop")

        logger.info(f"Loop execution logged: {loop_type} - {emotions_processed} emotions in {duration*1000:.1f}ms")

    except Exception as e:
        logger.error(f"Error logging loop execution: {str(e)}")

def sanitize_context_string(context: str) -> str:
    """Sanitize emotion context strings"""
    if not isinstance(context, str):
        return ""

    # Remove potential injection patterns
    context = re.sub(r'[<>"\']', '', context)

    # Limit length
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "..."

    return context.strip()

class EmotionState:
    """Explicit struct for representing an emotion with security enhancements."""
    def __init__(self, name, intensity, context, priority=1.0, original_valence=None):
        # Validate and sanitize inputs
        validation_result, validation_message = validate_emotion_input({
            'name': name,
            'intensity': intensity,
            'context': context,
            'priority': priority
        })

        if not validation_result:
            raise ValueError(f"Invalid emotion state: {validation_message}")

        self.name = str(name).lower()
        self.intensity = max(MIN_EMOTION_INTENSITY, min(MAX_EMOTION_INTENSITY, float(intensity)))
        self.context = sanitize_context_string(str(context))
        self.priority = max(0, min(10, float(priority)))
        self.original_valence = original_valence or self.intensity

        # Security tracking
        self.creation_time = datetime.now()
        self.last_modified = datetime.now()
        self.modification_count = 0
        self.affective_deltas = []

        logger.debug(f"Created emotion state: {self.name} (intensity: {self.intensity})")

    def apply_damping(self, factor=0.9):
        """Apply damping to reduce intensity with delta tracking."""
        with emotion_lock:
            old_intensity = self.intensity
            self.intensity *= factor
            self.intensity = max(MIN_EMOTION_INTENSITY, min(MAX_EMOTION_INTENSITY, self.intensity))

            # Track affective delta
            delta = abs(self.intensity - old_intensity)
            self.affective_deltas.append(delta)

            # Keep only recent deltas
            if len(self.affective_deltas) > 20:
                self.affective_deltas = self.affective_deltas[-20:]

            self._update_modification_tracking()

            if delta > AFFECTIVE_DELTA_THRESHOLD:
                logger.warning(f"Large damping delta for {self.name}: {delta:.3f}")

    def inhibit(self, other_emotion, inhibition_factor=0.2):
        """Apply cross-emotion inhibition with delta tracking."""
        if not isinstance(other_emotion, EmotionState):
            logger.error("Invalid emotion type for inhibition")
            return

        with emotion_lock:
            if self.name != other_emotion.name:
                old_intensity = self.intensity
                inhibition_amount = other_emotion.intensity * inhibition_factor
                self.intensity -= inhibition_amount
                self.intensity = max(MIN_EMOTION_INTENSITY, self.intensity)

                # Track affective delta
                delta = abs(self.intensity - old_intensity)
                self.affective_deltas.append(delta)

                self._update_modification_tracking()

                if delta > AFFECTIVE_DELTA_THRESHOLD:
                    logger.warning(f"Large inhibition delta for {self.name}: {delta:.3f}")

    def decay_priority(self, decay_rate=0.05):
        """Reduce priority over time with bounds checking."""
        with emotion_lock:
            old_priority = self.priority
            self.priority -= decay_rate
            self.priority = max(0, self.priority)

            self._update_modification_tracking()

            if old_priority > 0 and self.priority == 0:
                logger.debug(f"Emotion {self.name} priority decayed to zero")

    def is_painful(self, pain_threshold=0.8):
        """Determine if the emotion is painful based on intensity."""
        return self.intensity > pain_threshold

    def has_drifted(self, drift_threshold=0.3):
        """Check if emotion has drifted significantly from original valence."""
        if self.original_valence is None:
            return False
        return abs(self.intensity - self.original_valence) > drift_threshold

    def get_recent_deltas(self) -> List[float]:
        """Get recent affective deltas for monitoring."""
        return self.affective_deltas.copy()

    def get_modification_stats(self) -> Dict[str, Any]:
        """Get modification statistics for security monitoring."""
        return {
            'creation_time': self.creation_time.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'modification_count': self.modification_count,
            'recent_deltas': self.get_recent_deltas(),
            'max_recent_delta': max(self.affective_deltas) if self.affective_deltas else 0
        }

    def _update_modification_tracking(self):
        """Update modification tracking for security monitoring."""
        self.last_modified = datetime.now()
        self.modification_count += 1

class PhantomMemory(EmotionState):
    """A revived emotion with reduced clarity and increased priority."""

    def revive_as_phantom(self):
        """Revive the memory as a phantom with reduced clarity and increased priority."""
        with emotion_lock:
            old_intensity = self.intensity
            self.intensity *= 0.7  # Reduce clarity
            self.priority += 0.5  # Increase emotional weight

            # Track the revival delta
            delta = abs(self.intensity - old_intensity)
            self.affective_deltas.append(delta)

            self._update_modification_tracking()
            logger.info(f"Memory revived as phantom: {self.name} (new intensity: {self.intensity})")

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
