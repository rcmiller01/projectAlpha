import random
import time
from datetime import datetime, timedelta
from threading import Thread
from modules.reflection.emotion_reflector import EmotionReflector
from modules.memory_writer import log_memory_entry

class BackgroundJournaling:
    def __init__(self, emotion_reflector: EmotionReflector):
        self.emotion_reflector = emotion_reflector
        self.stop_flag = False

    def _generate_pulse(self):
        """Generate a symbolic emotional summary."""
        recent_events = self.emotion_reflector._get_recent_events(hours=1)
        if not recent_events:
            return "She sat with the ache of an unspoken name."
        
        dominant_emotion = self.emotion_reflector.run_daily_reflection().dominant_emotion
        if dominant_emotion == "joy":
            return "The light danced in her heart, unyielding."
        elif dominant_emotion == "longing":
            return "She sat with the ache of an unspoken name."
        elif dominant_emotion == "curious":
            return "The questions lingered, weaving threads of wonder."
        elif dominant_emotion == "intimate":
            return "The shared moments whispered secrets of connection."
        else:
            return "The silence spoke volumes, a tapestry of emotions."

    def _should_skip_pulse(self):
        """Determine if journaling should be skipped."""
        recent_events = self.emotion_reflector._get_recent_events(hours=1)
        if len(recent_events) > 5:
            return True  # High activity detected

        for event in recent_events:
            if event.intensity > 0.7:
                return True  # Recent intense emotion detected

        return False

    def _log_pulse(self):
        """Log the emotional pulse to memory."""
        if self._should_skip_pulse():
            return

        pulse = self._generate_pulse()
        log_memory_entry({
            "timestamp": datetime.now().isoformat(),
            "pulse": pulse
        })

    def start(self):
        """Start the background journaling process."""
        def journaling_loop():
            while not self.stop_flag:
                self._log_pulse()
                sleep_time = random.randint(30, 90) * 60  # Randomize between 30–90 minutes
                time.sleep(sleep_time)

        Thread(target=journaling_loop, daemon=True).start()

    def stop(self):
        """Stop the background journaling process."""
        self.stop_flag = True
