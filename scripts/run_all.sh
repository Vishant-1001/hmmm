#!/usr/bin/env bash
# Full reproducible research run on the FINAL split (train 2018-2020, validation 2021, test 2022).
# The test year is scored exactly once by `evaluate`, after all models/calibrators are frozen.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
unset FBS_SPLIT
$PY -m forecast_bust.train
$PY -m forecast_bust.evaluate
$PY -m forecast_bust.replay
$PY scripts/write_reports.py
(cd frontend && npm run build)
echo RUN_ALL_DONE
