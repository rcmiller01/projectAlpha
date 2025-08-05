
# modules/journal/presence_sensors.py

from datetime import datetime, timedelta
from modules.journal.trigger_dispatcher import dispatch_trigger

# Simulated state
presence_states = {
    "mia": {"is_present": True, "last_seen": datetime.utcnow()},
    "solene": {"is_present": True, "last_seen": datetime.utcnow()}
}

def user_departed(persona="mia"):
    persona = persona.lower()
    presence_states[persona]["is_present"] = False
    presence_states[persona]["last_seen"] = datetime.utcnow()
    dispatch_trigger("user_departure", persona=persona)

def user_returned(persona="mia"):
    persona = persona.lower()
    presence_states[persona]["is_present"] = True
    presence_states[persona]["last_seen"] = datetime.utcnow()
    dispatch_trigger("user_returned", persona=persona)

def check_presence_state(persona="mia", timeout_minutes=60):
    persona = persona.lower()
    last_seen = presence_states[persona]["last_seen"]
    if presence_states[persona]["is_present"]:
        print(f"[{persona}] User is present.")
    elif datetime.utcnow() - last_seen > timedelta(minutes=timeout_minutes):
        print(f"[{persona}] User has been gone for a long time.")
        # Optional: trigger additional reflective entry
        dispatch_trigger("user_departure", persona=persona)
    else:
        print(f"[{persona}] User recently left.")
