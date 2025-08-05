# time_helpers.py
# Utility functions for time formatting and scheduling

from datetime import datetime, timedelta

def get_current_time_iso():
    return datetime.now().isoformat()

def time_since(timestamp_iso):
    try:
        past = datetime.fromisoformat(timestamp_iso)
        return (datetime.now() - past).total_seconds()
    except Exception:
        return None

def add_hours_to_now(hours):
    return (datetime.now() + timedelta(hours=hours)).isoformat()
