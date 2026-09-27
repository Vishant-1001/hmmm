#!/usr/bin/env bash
# Start the API (+ built UI) in the background; PID in logs/api.pid. Usage: scripts/serve.sh [artifact_dir]
cd "$(dirname "$0")/.."
mkdir -p logs
if [ -f logs/api.pid ] && kill -0 "$(cat logs/api.pid)" 2>/dev/null; then kill "$(cat logs/api.pid)"; sleep 1; fi
FBS_ARTIFACT_DIR="${1:-artifacts}" nohup .venv/bin/python -m uvicorn forecast_bust.api.app:app --host 127.0.0.1 --port "${FBS_API_PORT:-8000}" > logs/api.log 2>&1 &
echo $! > logs/api.pid
echo "API on http://127.0.0.1:${FBS_API_PORT:-8000} (pid $(cat logs/api.pid))"
