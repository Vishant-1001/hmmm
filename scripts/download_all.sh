#!/usr/bin/env bash
# Resumable download of all real data used by the experiment (safe to re-run).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY -m forecast_bust.data.wb2 forecasts --max-blocks 1
$PY -m forecast_bust.data.wb2 reference
$PY -m forecast_bust.data.wb2 climatology
$PY -m forecast_bust.data.wb2 forecasts
echo DOWNLOAD_DONE
