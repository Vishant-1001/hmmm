"""End-to-end checks on REAL cached data and real pipeline artifacts.

Skipped automatically when the real-data cache / artifacts are absent (e.g. CI without
the ~4.5 GB download). Nothing here uses synthetic data.
"""
import json

import numpy as np
import pandas as pd
import pytest

from forecast_bust.config import REPO_ROOT

STATES = REPO_ROOT / "data" / "interim" / "states.nc"
TABLE = REPO_ROOT / "data" / "interim" / "table.parquet"
ART = REPO_ROOT / "artifacts"


@pytest.mark.skipif(not STATES.exists(), reason="real-data cache not present")
def test_real_states_alignment_and_units():
    import xarray as xr
    ds = xr.open_dataset(STATES)
    z = ds["ens_mean"].sel(level=500)
    assert 4800 < float(z.min()) and float(z.max()) < 6100  # Z500 in metres
    assert set(ds.lead.values.tolist()) == set(range(24, 241, 24))
    assert int(ds["n_members"].min()) == 50
    # forecast-minus-ERA5 grows with lead on average (sanity of valid-time alignment)
    err = np.abs(ds["ens_mean"].sel(level=500) - ds["era5_z500"]).mean(("init", "longitude", "latitude")).values
    assert err[-1] > 2 * err[0]


@pytest.mark.skipif(not TABLE.exists(), reason="final pipeline table not present")
def test_real_labels_training_only():
    df = pd.read_parquet(TABLE, columns=["split", "bust", "init_time", "valid_time", "an_n_eligible"])
    tr = df[df["split"] == "train"]
    assert 0.07 < tr["bust"].mean() < 0.13  # ~10% by construction of TRAIN Q90
    # purge: no training row verifies after the training period
    assert tr["valid_time"].max() <= pd.Timestamp("2020-12-31T23:59")


@pytest.mark.skipif(not (ART / "replay" / "index.json").exists(), reason="replay artifacts not present")
def test_real_replay_blind_file_has_no_verification():
    idx = json.loads((ART / "replay" / "index.json").read_text())
    cid = idx["cases"][0]["case_id"]
    text = (ART / "replay" / cid / "forecast.json").read_text()
    # the blind file may contain past (already verified) analogue errors, never this case's verification
    for key in ("era5_z500", '"bust_q95"', '"hidden_bust"', '"threshold_q90"', '"signature_label"'):
        assert key not in text
    v = json.loads((ART / "replay" / cid / "verification.json").read_text())
    assert len(v["regions"]) == 64 and all(len(r["days"]) == 10 for r in v["regions"])
