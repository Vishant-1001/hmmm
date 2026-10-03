"""Persist the frozen TRAIN-only B2 spread reference of the ECMWF benchmark (artifacts/b2/ecmwf/) and
check that the provider path reproduces the benchmark's own B2 features and probabilities.

    .venv/bin/python scripts/build_b2_references.py

Reads data/interim/v2/table.parquet (TRAIN rows only for the reference). The equivalence check runs
the ECMWFResearchProvider -> b2_provider path on one fixed VALIDATION initialisation (the first one in the
cache, 2021-01-01 00 UTC) and compares it with the v2 pipeline's stored features and p_B2.
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from forecast_bust.config import REPO_ROOT, clean_json

OUT = REPO_ROOT / "artifacts" / "b2" / "ecmwf"
TABLE = REPO_ROOT / "data" / "interim" / "v2" / "table.parquet"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "init_time", "region_id", "lead_day", "season", "split", "spread_m", "spread_pct", "spread_thr_ratio"]
    t = pd.read_parquet(TABLE, columns=cols)
    train = t[t["split"] == "train"]
    ref = train[["region_id", "lead_day", "season", "spread_m"]].astype({"spread_m": "float64"})
    ref.to_parquet(OUT / "spread_reference.parquet", index=False)
    from forecast_bust.b2_provider import reference, run_b2
    from forecast_bust.data.providers import ECMWFResearchProvider
    ref_obj = reference()
    val_init = pd.Timestamp(t.loc[t["split"] == "validation", "init_time"].min())
    b = ECMWFResearchProvider().retrieve(val_init, list(range(24, 241, 24)))
    out = run_b2(b, "b2_ecmwf")
    got = out["rows"].set_index("case_id")
    exp = t.set_index("case_id").loc[got.index]
    pred = pd.read_parquet(REPO_ROOT / "data" / "interim" / "v2" / "predictions.parquet", columns=["case_id", "p_B2"]).set_index("case_id")
    diffs = {c: float(np.nanmax(np.abs(got[c].to_numpy(float) - exp[c].to_numpy(float))))
             for c in ("spread_m", "spread_pct", "spread_thr_ratio")}
    diffs["p_B2"] = float(np.nanmax(np.abs(got["b2_probability"].to_numpy() - pred.loc[got.index, "p_B2"].to_numpy())))
    ok = diffs["spread_m"] < 1e-3 and diffs["spread_pct"] < 1e-9 and diffs["spread_thr_ratio"] < 1e-4 and diffs["p_B2"] < 1e-5
    manifest = {
        "mode": "b2_ecmwf", "provider": "ecmwf_research", "synthetic": False,
        "model": "B2 v2 (locked 8b196ee), exported booster artifacts/v2/demo/model/b2_booster.json",
        "calibration": "ECMWF_IFS_v2_validation2021_isotonic (artifacts/v2/demo/model/models.json B2.isotonic)",
        "spread_reference": {"file": "spread_reference.parquet", "fitted_on": "train",
                             "period": ref_obj.train_period, "rows": int(len(ref)),
                             "sha256": hashlib.sha256((OUT / "spread_reference.parquet").read_bytes()).hexdigest()},
        "thresholds": "artifacts/v2/thresholds.json (TRAIN Q90)", "scales": "artifacts/v2/preprocessing.json (TRAIN)",
        "benchmark_results": "artifacts/v2/metrics.json - UNCHANGED by the provider work",
        "provider_path_equivalence": {"init": str(val_init), "split": "validation", "rows": int(len(got)),
                                      "max_abs_diff": diffs, "passed": bool(ok),
                                      "timings_ms": out["timings_ms"]},
    }
    (OUT / "b2_manifest.json").write_text(json.dumps(clean_json(manifest), indent=1))
    print(json.dumps(manifest["provider_path_equivalence"], indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
