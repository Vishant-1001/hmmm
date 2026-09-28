"""Build precomputed REAL historical replay cases + the post-verification fingerprint store.

For each selected test initialisation two files are written:
  artifacts/replay/<case>/forecast.json      information available at initialisation (+ model output)
  artifacts/replay/<case>/verification.json  ERA5-based verification, served only on "reveal"

Case selection (documented; uses verification only, never model output):
  * "stress" cases: the test initialisations with the most hidden-bust region-days at Day 3-7
    (NOT representative of overall performance);
  * "random" cases: initialisations drawn uniformly at random (fixed seed) from the test split.
"""
from __future__ import annotations

import json
import logging
import shutil

import joblib
import numpy as np
import pandas as pd

from forecast_bust.analogues.memory import compute_memory_features
from forecast_bust.config import clean_json, ARTIFACT_DIR, MODEL_DIR, data_config, model_config
from forecast_bust.data.assemble import load_states
from forecast_bust.data.regions import build_regions
from forecast_bust.explainability.explain import explain_row
from forecast_bust.explainability.priority import EVIDENCE_NAMES, formula, priority_score
from forecast_bust.features.build import GROUPS
from forecast_bust.labels.signature import CLASSES, LABELS
from forecast_bust.pipeline import PRED, TABLE
from forecast_bust.support.ood import SUPPORT_LEVELS

log = logging.getLogger(__name__)
N_STRESS, N_RANDOM = 4, 8


def _r(x, n=4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, n) if np.isfinite(x) else None


def select_cases(te: pd.DataFrame, seed: int) -> list[dict]:
    mid = te[te["lead_day"].between(3, 7)]
    score = mid.groupby("init_time")["hidden_bust"].sum().sort_values(ascending=False, kind="stable")
    stress = list(score.index[:N_STRESS])
    rng = np.random.default_rng(seed)
    pool = sorted(set(te["init_time"].unique()) - set(stress))
    rand = list(rng.choice(pool, size=min(N_RANDOM, len(pool)), replace=False))
    out = [{"init_time": pd.Timestamp(t), "selection": "stress",
            "selection_note": f"Hidden-bust stress case: {int(score[t])} hidden-bust region-days at Day 3-7 "
                              f"(selected by verification only; not representative)"} for t in stress]
    out += [{"init_time": pd.Timestamp(t), "selection": "random",
             "selection_note": "Randomly sampled test initialisation (fixed seed)"} for t in sorted(rand)]
    return out


def expected_signature(nb_idx: np.ndarray, table: pd.DataFrame) -> dict:
    if nb_idx is None or len(nb_idx) == 0:
        return {"basis": "none", "n": 0, "distribution": None}
    sub = table.iloc[nb_idx]
    busts = sub[sub["bust"] == 1]
    basis, use = ("analogue busts", busts) if len(busts) >= 3 else ("all analogues", sub)
    counts = use["sig_class"].value_counts()
    return {"basis": basis, "n": int(len(use)),
            "distribution": {c: float(counts.get(c, 0) / len(use)) for c in CLASSES}}


def build() -> None:
    mcfg = model_config()
    table = pd.read_parquet(TABLE)
    preds = pd.read_parquet(PRED)
    table = table.merge(preds, on="case_id", how="left")
    models = joblib.load(MODEL_DIR / "models.joblib")
    full = models["FULL"]
    ds = load_states()
    tr = table[table["split"] == "train"]
    te_mask = table["split"] == "test"
    te = table[te_mask]
    ref = {f: np.sort(tr[f].dropna().to_numpy()) for f in full.features + ["abs_anom500", "rev_nbhd"]}
    base_rate = float(tr["bust"].mean())
    cases = select_cases(te, mcfg["seed"])
    sel_inits = {c["init_time"] for c in cases}
    sel_rows = np.where(te_mask & table["init_time"].isin(sel_inits))[0]
    fp_rows = np.where(te_mask & (table["bust"] == 1))[0]
    _, _, neigh = compute_memory_features(table.drop(columns=GROUPS["MEM"]),
                                          neighbour_rows=np.union1d(sel_rows, fp_rows))

    # ---- post-verification failure fingerprint store (all test busts) ----
    fps = []
    train_sig = tr[tr["bust"] == 1]["sig_class"].value_counts(normalize=True)
    for i in fp_rows:
        row = table.iloc[i]
        nb = neigh.get(int(i))
        exp = expected_signature(nb[0] if nb else None, table)
        dist = exp["distribution"]
        fps.append({"forecast_state_id": row["case_id"], "init_time": row["init_time"], "region_id": row["region_id"],
                    "lead_day": int(row["lead_day"]), "actual_bust": 1, "normalized_error": float(row["norm_error"]),
                    "error_m": float(row["error_m"]), "actual_signature": row["sig_class"],
                    "phase_share": float(row["sig_phase_share"]), "bias_m": float(row["sig_bias_m"]),
                    "pattern_corr": float(row["sig_corr"]), "verification_time": row["valid_time"],
                    "expected_top1": max(dist, key=dist.get) if dist else None,
                    "p_actual_signature": dist.get(row["sig_class"]) if dist else None,
                    "expected_basis": exp["basis"]})
    fp = pd.DataFrame(fps)
    fp.to_parquet(ARTIFACT_DIR / "fingerprints_test.parquet", index=False)
    has = fp["expected_top1"].notna()
    clim_top1 = train_sig.idxmax()
    fpm = {
        "n_test_busts": int(len(fp)), "n_with_analogue_expectation": int(has.sum()),
        "top1_agreement": float((fp.loc[has, "expected_top1"] == fp.loc[has, "actual_signature"]).mean()),
        "mean_prob_assigned_to_actual": float(fp.loc[has, "p_actual_signature"].mean()),
        "reference_climatological_top1": clim_top1,
        "reference_top1_agreement": float((fp.loc[has, "actual_signature"] == clim_top1).mean()),
        "reference_mean_prob_assigned": float(fp.loc[has, "actual_signature"].map(train_sig).fillna(0).mean()),
        "training_bust_signature_frequencies": train_sig.to_dict(),
        "actual_test_signature_frequencies": fp["actual_signature"].value_counts(normalize=True).to_dict(),
        "note": "Reference = always predicting the most common TRAIN bust signature / TRAIN frequencies.",
    }
    (ARTIFACT_DIR / "fingerprint_metrics.json").write_text(json.dumps(clean_json(fpm), indent=1))

    # ---- replay cases ----
    out_dir = ARTIFACT_DIR / "replay"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    regions = build_regions(ds.latitude.values, ds.longitude.values, data_config()["target_domain"])
    lvl = int(np.where(ds.level.values == data_config()["target_level"])[0][0])
    index = []
    thr = json.loads((ARTIFACT_DIR / "metrics.json").read_text())["models"]["FULL"]["operating_point"]["threshold"]
    for c in cases:
        t = c["init_time"]
        cid = t.strftime("%Y%m%d%H")
        rows = table[(table["init_time"] == t)].sort_values(["region_id", "lead_day"])
        pos = rows.index.to_numpy()
        contrib = full.contributions(rows)
        base_m = full.base_margin_of(rows) if hasattr(full, "base_margin_of") else None
        pr = priority_score(rows["p_FULL"].values, rows["p_B2"].values, rows["lead_day"].values,
                            rows["evidence_level"].values)
        per_region = {}
        queue = []
        for k, (ix, row) in enumerate(rows.iterrows()):
            nb = neigh.get(int(ix))
            analogues = []
            if nb:
                for j, d in list(zip(*nb))[:5]:
                    a = table.iloc[j]
                    analogues.append({"case_id": a["case_id"], "init_time": str(a["init_time"]), "distance": _r(d, 3),
                                      "bust": int(a["bust"]), "normalized_error": _r(a["norm_error"], 3),
                                      "signature": a["sig_class"] if int(a["bust"]) else None})
            exp = expected_signature(nb[0] if nb else None, table)
            ex = explain_row(row, contrib[k], full.features, ref, float(row["p_FULL"]), float(row["p_B2"]), base_rate,
                             baseline_logodds=None if base_m is None else float(base_m[k]))
            attribution_note = ex.pop("attribution_note")
            for dr in ex["drivers"]:
                dr["value"] = _r(dr["value"], 3)
                dr["train_percentile"] = _r(dr["train_percentile"], 1)
                dr["contribution_logodds"] = _r(dr["contribution_logodds"], 3)
            ex["groups"] = {g: _r(v["contribution_logodds"], 3) for g, v in ex["groups"].items()}
            ev = EVIDENCE_NAMES[int(row["evidence_level"])]
            day = {
                "lead_day": int(row["lead_day"]), "valid_time": str(row["valid_time"]),
                "p_bust": _r(row["p_FULL"]), "confidence": _r(1 - row["p_FULL"]), "p_spread_baseline": _r(row["p_B2"]),
                "p_climatology": _r(row["p_B0"]), "disagreement_pp": _r(100 * (row["p_FULL"] - row["p_B2"]), 1),
                "spread_m": _r(row["spread_m"], 2), "spread_pct": _r(row["spread_pct"], 3),
                "alert": bool(row["p_FULL"] >= thr),
                "support": SUPPORT_LEVELS[int(row["support_level"])], "support_distance": _r(row["support_distance"], 3),
                "evidence": ev, "analogues_within_radius": _r(row["an_n_within"], 0),
                "verified_cases_available": _r(row["an_n_eligible"], 0), "analogue_bust_rate": _r(row["an_bust_rate"], 3),
                "priority_score": _r(pr[k], 4), "explanation": ex, "analogues": analogues,
                "historical_failure_signature": exp,
            }
            per_region.setdefault(row["region_id"], []).append(day)
            queue.append({"region_id": row["region_id"], "lead_day": int(row["lead_day"]), "score": float(pr[k]),
                          "p_bust": _r(row["p_FULL"]), "p_spread_baseline": _r(row["p_B2"]),
                          "disagreement_pp": day["disagreement_pp"], "evidence": ev,
                          "analogues_within_radius": day["analogues_within_radius"], "support": day["support"]})
        reg_out = []
        for r in regions:
            traj = per_region[r.region_id]
            ps = np.array([d["p_bust"] for d in traj])
            above = [d["lead_day"] for d in traj if d["alert"]]
            reg_out.append({**r.to_dict(), "trajectory": traj, "peak_risk_day": int(traj[int(np.argmax(ps))]["lead_day"]),
                            "peak_p_bust": _r(ps.max()),
                            "first_alert_day": above[0] if above else None,
                            "alert_days": above})
        queue = sorted(queue, key=lambda q: -q["score"])[:20]
        ii = int(np.where(ds.init.values == np.datetime64(t))[0][0])
        fields = {"lead_days": [int(h) // 24 for h in ds.lead.values],
                  "lats": ds.latitude.values.tolist(), "lons": ds.longitude.values.tolist(),
                  "ens_mean_z500": np.round(ds["ens_mean"].values[ii, :, lvl], 1).tolist(),
                  "ens_spread_z500": np.round(ds["ens_std"].values[ii, :, lvl], 2).tolist(),
                  "layout": "[lead][lon][lat], metres"}
        forecast = {
            "case_id": cid, "init_time": str(t), "mode": "Historical research replay (precomputed, not live)",
            "data_source": "ECMWF IFS ENS 50 members via WeatherBench 2 (64x32, 5.625 deg), Z500",
            "split": "test", "selection": c["selection"], "selection_note": c["selection_note"],
            "alert_threshold": {"p_bust": thr, "definition": "FULL model probability giving 10% false-alarm rate "
                                                              "on VALIDATION (product threshold)"},
            "priority_formula": formula(), "priority_queue": queue, "attribution_note": attribution_note, "regions": reg_out, "fields": fields,
            "blind": True,
        }
        verif_regions = []
        for r in regions:
            vr = rows[rows["region_id"] == r.region_id]
            verif_regions.append({"region_id": r.region_id, "days": [{
                "lead_day": int(v["lead_day"]), "error_m": _r(v["error_m"], 2), "normalized_error": _r(v["norm_error"], 3),
                "threshold_q90": _r(v["q_primary"], 3), "bust": int(v["bust"]), "bust_q95": int(v["bust_q95"]),
                "hidden_bust": int(v["hidden_bust"]), "signature": v["sig_class"],
                "signature_label": LABELS[v["sig_class"]], "phase_share": _r(v["sig_phase_share"], 3),
                "bias_m": _r(v["sig_bias_m"], 2), "pattern_corr": _r(v["sig_corr"], 3)} for _, v in vr.iterrows()]})
        y = rows["bust"].values
        a = rows["p_FULL"].values >= thr
        b2a = rows["p_B2"].values >= json.loads((ARTIFACT_DIR / "metrics.json").read_text())[
            "models"]["B2"]["operating_point"]["threshold"]
        verification = {
            "case_id": cid, "reference": "ERA5 (verification reference analysis; not perfect truth)",
            "era5_z500": np.round(ds["era5_z500"].values[ii], 1).tolist(),
            "regions": verif_regions,
            "summary": {"n_bust_region_days": int(y.sum()), "n_hidden_bust_region_days": int(rows["hidden_bust"].sum()),
                        "sentinel_alerts": int(a.sum()), "sentinel_hits": int((a & (y == 1)).sum()),
                        "b2_alerts": int(b2a.sum()), "b2_hits": int((b2a & (y == 1)).sum()),
                        "jaccard_sentinel": float((a & (y == 1)).sum() / max((a | (y == 1)).sum(), 1)),
                        "jaccard_b2": float((b2a & (y == 1)).sum() / max((b2a | (y == 1)).sum(), 1))},
        }
        d = out_dir / cid
        d.mkdir()
        (d / "forecast.json").write_text(json.dumps(clean_json(forecast), default=str))
        (d / "verification.json").write_text(json.dumps(clean_json(verification), default=str))
        index.append({"case_id": cid, "init_time": str(t), "selection": c["selection"],
                      "selection_note": c["selection_note"], "n_alerts": int(a.sum())})
        log.info("replay case %s written", cid)
    (out_dir / "index.json").write_text(json.dumps(clean_json({"cases": index,
                                                    "selection_rule": __doc__.split("Case selection")[1].strip()}), indent=1))
