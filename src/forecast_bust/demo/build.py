"""Build the demo bundle used by the live inference engine (python -m forecast_bust.demo.build).

Nothing here predicts anything: it only packages REAL, already-computed forecast-state inputs so
that the API can run the frozen models at request time without the multi-GB research cache.

  artifacts/demo/registry.json                 case registry (fixed selection rule, below)
  artifacts/demo/model_meta.json               model version, training/calibration windows, features
  artifacts/demo/model/                        the FROZEN final models exported from models/models.joblib
                                               (XGBoost JSON boosters, isotonic calibrators, B0 table,
                                               support model, analogue scaler/radius); parity-tested
  artifacts/demo/train_reference.json          TRAIN quantile grids (percentiles in explanations)
  artifacts/demo/cases/<id>/forecast_state.parquet   forecast-time features, 64 regions x Day 1-10;
                                               contains NO verification column (tested)
  artifacts/demo/cases/<id>/fields.json        ensemble-mean / spread Z500 fields (display)
  artifacts/demo/cases/<id>/verification.json  ERA5 verification - served only on reveal
  artifacts/demo/memory.parquet                historical forecast-state memory (analogue space +
                                               verified outcome + valid time; zstd, ~25 MB, committed so
                                               a fresh clone runs the demo without the research cache)

Case selection rule (fixed before any demo output was viewed): from the existing 2022 replay set,
the earliest seed-drawn RANDOM case in each season (JF, MAM, JJAS, OND), plus the earliest
existing hidden-bust STRESS case (selected by verification in the v1 replay build; labelled
"not representative").
"""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd

from forecast_bust.analogues.memory import analogue_space
from forecast_bust.config import (ARTIFACT_DIR, INTERIM_DIR, MODEL_DIR, REPO_ROOT, SERVED_ARTIFACT_DIR, clean_json,
                                  data_config)
from forecast_bust.features.build import FORBIDDEN_INPUTS, GROUPS
from forecast_bust.verification.alignment import season_of

DEMO_DIR = ARTIFACT_DIR / "demo"          # where `build` writes (this run's namespace)
MEMORY_PATH = DEMO_DIR / "memory.parquet"
SERVED_DEMO_DIR = SERVED_ARTIFACT_DIR / "demo"   # what the engine/API serve
ID_COLS = ["case_id", "init_time", "valid_time", "lead_hours", "lead_day", "region_id", "row", "col", "lat", "lon",
           "season", "season_code", "region_code", "init_hour"]
# MEM features are recomputed live by the engine from the memory store, so they are not packaged.
STATE_FEATURES = sorted({f for g, fs in GROUPS.items() if g != "MEM" for f in fs} - set(ID_COLS))
MEMORY_COLS = ["case_id", "init_time", "valid_time", "region_id", "lead_day", "split", "bust", "norm_error",
               "sig_class"]


def select_cases(index: list[dict]) -> list[dict]:
    rand = sorted((c for c in index if c["selection"] == "random"), key=lambda c: c["init_time"])
    stress = sorted((c for c in index if c["selection"] == "stress"), key=lambda c: c["init_time"])
    out, seen = [], set()
    for c in rand:
        s = season_of(np.array([np.datetime64(c["init_time"])]))[0]
        if s not in seen:
            seen.add(s)
            out.append({**c, "season": s, "demo_role": f"Representative case ({s}, random draw)"})
    if stress:
        c = stress[0]
        out.append({**c, "season": season_of(np.array([np.datetime64(c["init_time"])]))[0],
                    "demo_role": "Hidden-bust stress case - selected by verification, NOT representative"})
    return sorted(out, key=lambda c: c["init_time"])


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def export_models() -> dict:
    """Export the frozen final B0 / B2 / Sentinel(FULL) models to portable files (no retraining).

    Boosters: XGBoost native JSON, evaluated with iteration_range (0, best_iteration + 1), exactly as
    the sklearn wrapper does. Isotonic calibrators: their fitted (X_thresholds_, y_thresholds_) knots;
    IsotonicRegression(out_of_bounds="clip").predict is linear interpolation between them."""
    import joblib
    models = joblib.load(MODEL_DIR / "models.joblib")
    out = DEMO_DIR / "model"
    out.mkdir(parents=True, exist_ok=True)
    spec = {}
    for name in ("B2", "FULL"):
        m = models[name]
        m.model.get_booster().save_model(out / f"{name.lower()}_booster.json")
        cal = m.calibrator
        spec[name] = {"booster": f"model/{name.lower()}_booster.json", "features": m.features,
                      "best_iteration": int(m.best_iteration), "learner": m.learner,
                      "isotonic": {"x": np.asarray(cal.X_thresholds_, float).tolist(),
                                   "y": np.asarray(cal.y_thresholds_, float).tolist()}}
    b0 = models["B0"]
    spec["B0"] = {"group": ["region_id", "lead_day", "season"], "global_rate": b0.global_rate,
                  "table": b0.table.to_dict(orient="records")}
    (out / "models.json").write_text(json.dumps(clean_json(spec)))
    shutil.copy(ARTIFACT_DIR / "support_diagnostics.json", out / "support.json")
    shutil.copy(ARTIFACT_DIR / "analogue_memory_causal_online.json", out / "analogue_memory.json")
    return spec


def build() -> None:
    table = pd.read_parquet(INTERIM_DIR / "table.parquet")
    for f in STATE_FEATURES:
        if any(f.startswith(x) for x in FORBIDDEN_INPUTS):
            raise ValueError(f"verification quantity {f} would be packaged as a forecast-state input")
    index = json.loads((ARTIFACT_DIR / "replay" / "index.json").read_text())["cases"]
    cases = select_cases(index)
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    (DEMO_DIR / "cases").mkdir(parents=True)
    reg = []
    for c in cases:
        t = pd.Timestamp(c["init_time"])
        rows = table[table["init_time"] == t].sort_values(["region_id", "lead_day"])
        if len(rows) == 0:
            raise ValueError(f"no forecast state for {c['case_id']}")
        d = DEMO_DIR / "cases" / c["case_id"]
        d.mkdir()
        rows[ID_COLS + STATE_FEATURES].to_parquet(d / "forecast_state.parquet", index=False)
        src = ARTIFACT_DIR / "replay" / c["case_id"]
        fc = json.loads((src / "forecast.json").read_text())
        (d / "fields.json").write_text(json.dumps(fc["fields"]))
        shutil.copy(src / "verification.json", d / "verification.json")
        reg.append({"case_id": c["case_id"], "init_time": str(t), "season": c["season"], "selection": c["selection"],
                    "demo_role": c["demo_role"], "selection_note": c["selection_note"],
                    "valid_range": [str(rows["valid_time"].min()), str(rows["valid_time"].max())],
                    "n_regions": int(rows["region_id"].nunique()), "lead_days": sorted(int(x) for x in rows["lead_day"].unique()),
                    "split": str(rows["split"].iloc[0]) if "split" in rows else "test",
                    "model_input": f"artifacts/demo/cases/{c['case_id']}/forecast_state.parquet",
                    "verification": f"artifacts/demo/cases/{c['case_id']}/verification.json"})
    (DEMO_DIR / "registry.json").write_text(json.dumps({
        "selection_rule": __doc__.split("Case selection rule")[1].strip(), "cases": reg}, indent=1))

    # historical memory store: every verified case (all splits); eligibility is decided at query time
    mem_cols = MEMORY_COLS + [c for c in analogue_space(table) if c not in MEMORY_COLS]
    mem = table[mem_cols].copy()
    for c in analogue_space(table):
        mem[c] = mem[c].astype(np.float32)
    mem["lead_day"] = mem["lead_day"].astype(np.int16)
    mem.to_parquet(MEMORY_PATH, index=False, compression="zstd")

    tr = table[table["split"] == "train"]
    ref_cols = sorted(set(STATE_FEATURES) | {"abs_anom500", "rev_nbhd"} | set(GROUPS["MEM"]))
    qs = np.linspace(0, 1, 1001)
    ref = {c: np.nanquantile(tr[c].to_numpy(dtype=float), qs).round(5).tolist() for c in ref_cols
           if c in tr and tr[c].notna().any()}
    (DEMO_DIR / "train_reference.json").write_text(json.dumps(clean_json(
        {"note": "1001-point TRAIN quantile grids; used only to express feature values as training percentiles",
         "base_rate": float(tr["bust"].mean()), "quantiles": ref})))

    export_models()
    em = json.loads((ARTIFACT_DIR / "experiment_manifest.json").read_text())
    met = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    meta = {
        "model_artifact": str((MODEL_DIR / "models.joblib").relative_to(REPO_ROOT)),
        "model_version": f"{em.get('run', 'v1')}-final ({em['git_rev']})", "experiment_created": em["created"],
        "retrained_for_demo": False,
        "training_window": em["splits"]["train"]["first_init"][:10] + " .. " + em["splits"]["train"]["last_init"][:10],
        "calibration_window": em["splits"]["validation"]["first_init"][:10] + " .. "
                              + em["splits"]["validation"]["last_init"][:10],
        "calibration": "isotonic regression fitted on validation predictions only, then frozen",
        "learner": "XGBoost CPU hist (one shared model: region, lead day, season + forecast-state features)",
        "sentinel_features": em["feature_sets"]["FULL"], "b2_features": em["feature_sets"]["B2"],
        "selected_groups": em["selected_groups"],
        "selection_note": (f"Sentinel = B2 inputs + {', '.join(em['selected_groups'])} (groups locked before the test "
                           "year was scored). Measured test result vs B2: see /api/metrics full_vs_b2."
                           if em["selected_groups"] else
                           "No candidate feature group passed validation, so the Sentinel uses B2's inputs and its "
                           "probabilities equal B2's. The UI shows this as measured."),
        "pca": "models/pca.joblib (TRAIN-only fit, frozen); PCs are packaged in the forecast state",
        "features_available": {g: fs for g, fs in GROUPS.items()},
        "not_in_sentinel": sorted(set(em["feature_groups"]) - set(em["selected_groups"]) - {"SPREAD"}),
        "hyperparameters": em.get("hyperparameters", {}).get("sentinel") or em["model_config"]["xgboost"],
        "seed": em["seed"],
        "alert_threshold": met["models"]["FULL"]["operating_point"]["threshold"],
        "alert_threshold_definition": "Sentinel probability giving a 10% false-alarm rate on VALIDATION",
        "confidence_definition": "reliability confidence = 1 - P(bust)",
        "exported_from": f"{(MODEL_DIR / 'models.joblib').relative_to(REPO_ROOT)} -> "
                         f"{(DEMO_DIR / 'model').relative_to(REPO_ROOT)}/ (identical predictions, parity-tested)",
        "bundle_git_rev": _git_rev(), "target": data_config()["variable"] + " 500 hPa (Z500)",
    }
    (DEMO_DIR / "model_meta.json").write_text(json.dumps(clean_json(meta), indent=1))
    print(f"demo bundle: {len(reg)} cases -> {DEMO_DIR}; memory {len(mem)} rows -> {MEMORY_PATH}")


if __name__ == "__main__":
    build()
