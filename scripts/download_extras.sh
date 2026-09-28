#!/usr/bin/env bash
# Resumable download of the v2 extra variables (u, v at 500/700/850 hPa and MSLP; ensemble mean/std)
# for exactly the initialisations already cached for geopotential. ~2 min per block at ~1 MB/s.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY -m forecast_bust.data.wb2 extra_climatology
$PY -m forecast_bust.data.wb2 extras
echo EXTRAS_DONE
