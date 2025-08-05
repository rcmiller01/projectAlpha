import json
import os
from datetime import datetime, timedelta

LOG_PATH = os.path.join('logs', 'crisis_events.jsonl')


def record_crisis_event(user_id: str, level: str, details: dict | None = None) -> dict:
    """Record a crisis event for later reference"""
    event = {
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'level': level,
        'details': details or {}
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(event) + '\n')
    return event


def get_recent_crisis_events(user_id: str, days: int = 7) -> list[dict]:
    """Retrieve recent crisis events for a user"""
    events: list[dict] = []
    cutoff = datetime.now() - timedelta(days=days)
    try:
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get('user_id') != user_id:
                    continue
                ts = datetime.fromisoformat(data.get('timestamp'))
                if ts >= cutoff:
                    events.append(data)
    except FileNotFoundError:
        pass
    return events
