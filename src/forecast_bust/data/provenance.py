"""Provider provenance constants (dependency-free, so the API can label sources without importing xarray).

`forecast_bust.data.providers` implements the providers; these constants are the single source of their labels.
"""
from __future__ import annotations

PROVIDER_INFO = {
    "ecmwf_research": {"dataset": "ECMWF IFS ENS via WeatherBench 2 (64x32 conservative)",
                       "source_label": "ECMWF IFS ENS / ERA5 (historical research archive)",
                       "synthetic": False, "demo_only": False},
    "ncmrwf_tigge": {"dataset": "TIGGE (ECMWF ECDS tigge-forecasts), origin NCMRWF (dems)",
                     "source_label": "NCMRWF NEPS via TIGGE/ECDS", "synthetic": False, "demo_only": False},
    "synthetic": {"dataset": "synthetic demonstration scenario", "source_label": "Synthetic demonstration scenario",
                  "synthetic": True, "demo_only": True},
}
# Every cached ECMWF block holds members 1..50 (states.nc n_members min = max = 50)
ECMWF_MEMBER_COUNT = 50

# Model/provider modes. b2_ecmwf is the served benchmark model; b2_ncmrwf is the same frozen B2 applied to NCMRWF
# inputs (not validated for NCMRWF); synthetic_demo is never a scientific result.
B2_MODES = {
    "b2_ecmwf": {"provider": "ecmwf_research", "artifact_dir": "ecmwf"},
    "b2_ncmrwf": {"provider": "ncmrwf_tigge", "artifact_dir": "ncmrwf"},
    "synthetic_demo": {"provider": "synthetic", "artifact_dir": "../demo/synthetic"},
}
