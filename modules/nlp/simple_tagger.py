import json
import os
import re
from collections import Counter
from datetime import datetime

TAG_LOG_PATH = os.path.join('logs', 'symbol_tags.jsonl')


def extract_tags(text: str) -> list[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    return [w for w, c in counts.items() if c > 1 and len(w) > 3]


def tag_text(text: str) -> list[str]:
    tags = extract_tags(text)
    if not tags:
        return []
    record = {
        'timestamp': datetime.now().isoformat(),
        'tags': tags,
        'text': text
    }
    os.makedirs(os.path.dirname(TAG_LOG_PATH), exist_ok=True)
    with open(TAG_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    return tags


def get_tag_history(limit: int = 50) -> list[dict]:
    history: list[dict] = []
    try:
        with open(TAG_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()[-limit:]
        history = [json.loads(line) for line in lines]
    except FileNotFoundError:
        pass
    return history
