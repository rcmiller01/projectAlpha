import json
import os
from datetime import datetime

# SECTION 1: Load or Initialize User Signature
DEFAULT_SIGNATURE = {
    "playfulness": 0.5,
    "vulnerability": 0.4,
    "intimacy": 0.5,
    "responsiveness": 0.6,
    "restraint": 0.5,
    "sensuality": 0.4,
    "preference_mode": "neutral",  # "romantic", "creative", "supportive", etc.
}

USER_SIGNATURE_FILE = "memory/user_signature.json"


def load_user_signature():
    if os.path.exists(USER_SIGNATURE_FILE):
        with open(USER_SIGNATURE_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_SIGNATURE.copy()


def save_user_signature(signature):
    os.makedirs(os.path.dirname(USER_SIGNATURE_FILE), exist_ok=True)
    with open(USER_SIGNATURE_FILE, "w") as f:
        json.dump(signature, f, indent=2)

# SECTION 2: Preference Update Logic

def update_signature(expression_dial, feedback_score):
    """Adjust emotional axes based on user feedback."""
    signature = load_user_signature()

    learning_rate = 0.05 if feedback_score == 1 else -0.03
    for axis in signature:
        if axis in expression_dial:
            delta = learning_rate * (expression_dial[axis] - signature[axis])
            signature[axis] = max(0.0, min(1.0, signature[axis] + delta))

    save_user_signature(signature)
    log_personalization_event(signature, feedback_score)

# SECTION 3: Apply Signature to Dial

def personalize_dial(expression_dial):
    signature = load_user_signature()
    for axis in expression_dial:
        if axis in signature:
            expression_dial[axis] = (expression_dial[axis] + signature[axis]) / 2
    return expression_dial

# SECTION 4: Logging

def log_personalization_event(updated_signature, score):
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "score": score,
        "updated_signature": updated_signature,
    }
    with open("logs/personalization_log.jsonl", "a") as log:
        log.write(json.dumps(log_entry) + "\n")

