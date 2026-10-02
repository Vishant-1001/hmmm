#!/usr/bin/env bash
# v3 (quantile gradient boosting) in its own namespace (artifacts/v3, models/v3, data/interim/v3); v1/v2 untouched.
#   scripts/run_v3.sh dev     development split (train 2018-19, val 2020, dev-test 2021): search, fit, go/no-go gate
#   scripts/run_v3.sh final   train 2018-20, val 2021 with qgb.locked_params; ONE evaluation of 2022; demo bundle
# Requires the geopotential cache (scripts/download_all.sh) and extras (scripts/download_extras.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export FBS_RUN=v3
case "${1:-}" in
  dev) export FBS_SPLIT=dev
       $PY -m forecast_bust.train_qgb --no-reassemble
       $PY -m forecast_bust.evaluate_qgb ;;
  final) unset FBS_SPLIT
       $PY -m forecast_bust.train_qgb --no-reassemble
       $PY -m forecast_bust.evaluate_qgb
       $PY -m forecast_bust.demo.build
       (cd frontend && npm run build) ;;
  *) echo "usage: $0 dev|final" >&2; exit 2 ;;
esac
echo RUN_V3_${1}_DONE
