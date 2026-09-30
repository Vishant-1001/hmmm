#!/usr/bin/env bash
# Single-instance supervisor for the resumable wind/MSLP download (scripts/download_extras.sh).
#
#   setsid nohup scripts/download_supervisor.sh >/dev/null 2>&1 &
#
# - flock guarantees one supervisor (and so one downloader) at a time.
# - Restarts the downloader through its own resume path (completed blocks are skipped; blocks are
#   written to .tmp and renamed, so a kill never leaves a partial block) when it exits early or
#   when no new block has appeared for STALL_MIN minutes (normal pace: ~2 min per block).
# - Never deletes data. Exits by itself when every geopotential block has its extras block.
# Log: logs/download_supervisor.log (downloader output: logs/download_extras.log). Stop: kill the PID in
# logs/download_supervisor.pid (its downloader child is stopped with it).
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
exec 9>logs/download_supervisor.lock
flock -n 9 || { echo "$(date -Is) another supervisor holds the lock; exiting" >> logs/download_supervisor.log; exit 0; }
echo $$ > logs/download_supervisor.pid

STALL_MIN=${STALL_MIN:-30}
MAX_RESTARTS=${MAX_RESTARTS:-40}
log() { echo "$(date -Is) $*" >> logs/download_supervisor.log; }
expected() { ls data/cache/ens/block_*.nc 2>/dev/null | wc -l; }
done_n() { ls data/cache/ens_extra/block_*.nc 2>/dev/null | wc -l; }

child=""
cleanup() { [ -n "$child" ] && kill "$child" 2>/dev/null; pkill -P "$child" 2>/dev/null; log "supervisor stopped"; }
trap 'cleanup; exit 0' TERM INT

restarts=0
log "supervisor start: $(done_n)/$(expected) blocks present"
while :; do
  if [ "$(done_n)" -ge "$(expected)" ]; then
    log "COMPLETE: $(done_n)/$(expected) blocks present; validate with python -m forecast_bust.data.validate_extras"
    exit 0
  fi
  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    log "GAVE UP after $restarts restarts at $(done_n)/$(expected); inspect logs/download_extras.log"
    exit 1
  fi
  setsid scripts/download_extras.sh >> logs/download_extras.log 2>&1 &
  child=$!
  log "downloader started pid=$child (restart #$restarts) at $(done_n)/$(expected)"
  last=$(done_n); last_t=$(date +%s)
  while kill -0 "$child" 2>/dev/null; do
    sleep 60
    n=$(done_n)
    if [ "$n" -gt "$last" ]; then last=$n; last_t=$(date +%s); fi
    if [ $(( $(date +%s) - last_t )) -ge $(( STALL_MIN * 60 )) ]; then
      log "STALL: no new block for ${STALL_MIN} min at $n/$(expected); last log: $(tail -1 logs/download_extras.log)"
      kill -- -"$child" 2>/dev/null || kill "$child" 2>/dev/null
      sleep 5
      break
    fi
  done
  wait "$child" 2>/dev/null
  log "downloader pid=$child ended at $(done_n)/$(expected)"
  child=""
  restarts=$((restarts + 1))
  [ "$(done_n)" -lt "$(expected)" ] && sleep 60
done
