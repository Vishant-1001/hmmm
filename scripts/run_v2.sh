#!/usr/bin/env bash
# v2 research run in its own namespace (artifacts/v2, models/v2, data/interim/v2); v1 artifacts untouched.
#   scripts/run_v2.sh dev     development split only (train 2018-19, val 2020, dev-test 2021); never 2022
#   scripts/run_v2.sh final   train 2018-20, val 2021, single scoring of 2022 (see config/model_v2.yaml test_history)
# Requires the geopotential cache (scripts/download_all.sh) and extras (scripts/download_extras.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export FBS_RUN=v2
case "${1:-}" in
  dev) export FBS_SPLIT=dev
       $PY -m forecast_bust.train --no-reassemble
       $PY -m forecast_bust.evaluate ;;
  final) unset FBS_SPLIT
       $PY -m forecast_bust.train --no-reassemble
       $PY -m forecast_bust.evaluate
       $PY -m forecast_bust.replay
       $PY scripts/write_reports.py artifacts/v2
       (cd frontend && npm run build) ;;
  *) echo "usage: $0 dev|final" >&2; exit 2 ;;
esac
echo RUN_V2_${1}_DONE
