#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 "$DIR/../core/emotion_loop_core.py"
python3 "$DIR/../core/reflection_agent.py"
