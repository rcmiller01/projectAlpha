#!/bin/bash
BASE_DIR="$(dirname "$(dirname "$0")")"
EMOTION_LOG_DIR="$BASE_DIR/emotion_logs"
mkdir -p "$EMOTION_LOG_DIR"
CRON_LINE="0 * * * * python3 $BASE_DIR/core/reflection_agent.py >> $EMOTION_LOG_DIR/reflection_cron.log 2>&1"
if command -v crontab >/dev/null 2>&1; then
  (crontab -l 2>/dev/null | grep -v -F "$BASE_DIR/core/reflection_agent.py" || true; echo "$CRON_LINE") | crontab -
  echo "Reflection agent scheduled to run every hour."
  echo "Cron entry: $CRON_LINE"
else
  echo "Note: 'crontab' not found. Please schedule manually."
fi
