"""Live inference engine for the interactive demo (historical replay, real model execution).

On every case run the engine:
  1. loads the stored forecast-state inputs of one real initialisation (64 regions x Day 1-10),
  2. runs the frozen B0 climatology, the B2 spread-only booster and the Sentinel (FULL) booster,
     then their validation-fitted isotonic calibrators,
  3. queries the historical forecast-state memory (analogues verified no later than the init time),
  4. computes support / OOD distance, evidence strength, review priority,
  5. computes TreeSHAP attributions of the Sentinel booster for explanations.

Nothing here is a lookup of precomputed probabilities: predictions come from the exported
boosters (artifacts/demo/model/) evaluated on the stored inputs. Verification (ERA5) is read
from a separate file, and only by `reveal()`.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from forecast_bust.config import model_config
from forecast_bust.demo.build import DEMO_DIR, MEMORY_PATH
from forecast_bust.explainability.explain import FEATURE_LABELS, explain_row
from forecast_bust.explainability.priority import EVIDENCE_NAMES, formula, priority_score
from forecast_bust.labels.signature import CLASSES, LABELS
from forecast_bust.support.ood import SUPPORT_LEVELS, evidence_strength, support_distance


class CaseNotFound(KeyError):
    pass


class Calibrated:
    """Exported XGBoost booster + isotonic calibrator (identical to CalibratedGBM.predict)."""

    def __init__(self, spec: dict, root: Path):
        self.features = spec["features"]
        self.best_iteration = spec["best_iteration"]
        self.booster = xgb.Booster()
        self.booster.load_model(root / spec["booster"])
        self.iso_x = np.asarray(spec["isotonic"]["x"])
        self.iso_y = np.asarray(spec["isotonic"]["y"])

    def _dm(self, df: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(df[self.features])

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dm(df), iteration_range=(0, self.best_iteration + 1))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self.iso_x, self.iso_y)

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dm(df), pred_contribs=True, iteration_range=(0, self.best_iteration + 1))


@dataclass
class CaseRun:
    case_id: str
    meta: dict
    rows: pd.DataFrame                   # 640 rows, forecast state + model outputs (no verification)
    contrib: np.ndarray                  # Sentinel TreeSHAP (n_rows, n_features + 1)
    neighbours: list                     # per row: (memory indices, distances)
    timings_ms: dict = field(default_factory=dict)


class Engine:
    def __init__(self, demo_dir: Path = DEMO_DIR, memory_path: Path = MEMORY_PATH):
        t0 = time.perf_counter()
        self.dir = demo_dir
        self.registry = json.loads((demo_dir / "registry.json").read_text())
        self.meta = json.loads((demo_dir / "model_meta.json").read_text())
        spec = json.loads((demo_dir / "model" / "models.json").read_text())
        self.sentinel = Calibrated(spec["FULL"], demo_dir)
        self.b2 = Calibrated(spec["B2"], demo_dir)
        b0 = spec["B0"]
        self.b0_table = pd.DataFrame(b0["table"])
        self.b0_group = b0["group"]
        self.b0_rate = b0["global_rate"]
        self.support = json.loads((demo_dir / "model" / "support.json").read_text())
        am = json.loads((demo_dir / "model" / "analogue_memory.json").read_text())
        self.space, self.radius = am["space"], am["radius"]
        self.mu = np.array([am["scaler"]["mean"][c] for c in self.space])
        self.sd = np.array([am["scaler"]["std"][c] for c in self.space])
        self.k, self.kmin = am["k"], am["min_analogues"]
        tr = json.loads((demo_dir / "train_reference.json").read_text())
        self.ref = {k: np.asarray(v, float) for k, v in tr["quantiles"].items()}
        self.base_rate = tr["base_rate"]
        self.threshold = self.meta["alert_threshold"]
        self._load_memory(memory_path)
        self._lock = threading.Lock()
        self._cache: dict[str, CaseRun] = {}
        self.startup_ms = round(1000 * (time.perf_counter() - t0), 1)

    # ---------------- historical forecast-state memory ----------------
    def _load_memory(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"historical memory store {path} missing: run python -m forecast_bust.demo.build")
        mem = pd.read_parquet(path)
        mem = mem.sort_values(["region_id", "lead_day", "valid_time"], kind="stable").reset_index(drop=True)
        X = ((mem[self.space].to_numpy(dtype=float) - self.mu) / self.sd).astype(np.float32)
        self.mem_X = np.nan_to_num(X)
        self.mem = mem[["case_id", "init_time", "valid_time", "region_id", "lead_day", "split", "bust", "norm_error",
                        "sig_class"]]
        self.mem_valid = mem["valid_time"].values.astype("datetime64[ns]")
        self.mem_groups = {k: (int(v.min()), int(v.max()) + 1) for k, v in mem.groupby(["region_id", "lead_day"]).indices.items()}
        self.mem_bust = mem["bust"].to_numpy(float)
        self.mem_err = mem["norm_error"].to_numpy(float)

    def memory_lookup(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """MEM features exactly as forecast_bust.analogues.memory (causal_online): candidates of the same
        region and lead day with valid_time <= query init_time."""
        Xq = np.nan_to_num(((rows[self.space].to_numpy(dtype=float) - self.mu) / self.sd).astype(np.float32))
        init = rows["init_time"].values.astype("datetime64[ns]")
        out = {c: np.full(len(rows), np.nan) for c in
               ["an_n_within", "an_dist1", "an_dist_mean", "an_bust_rate", "an_err_med", "an_err_q90", "an_n_eligible"]}
        neigh = []
        for i, (rid, ld) in enumerate(zip(rows["region_id"].to_numpy(), rows["lead_day"].to_numpy())):
            lo, hi = self.mem_groups[(rid, int(ld))]
            hi = lo + int(np.searchsorted(self.mem_valid[lo:hi], init[i], side="right"))
            idx = np.arange(lo, hi)
            d = np.sqrt(((self.mem_X[lo:hi].astype(np.float64) - Xq[i].astype(np.float64)) ** 2).sum(1)).astype(np.float32)
            order = np.argsort(d, kind="stable")[: self.k]
            dk = d[order]
            cand = idx[order]
            n = len(order)
            out["an_n_eligible"][i] = len(idx)
            out["an_n_within"][i] = int((d <= self.radius).sum())
            out["an_dist1"][i] = dk[0] if n else np.nan
            if n >= self.kmin:
                out["an_dist_mean"][i] = dk.mean()
                out["an_bust_rate"][i] = self.mem_bust[cand].mean()
                out["an_err_med"][i] = np.median(self.mem_err[cand])
                out["an_err_q90"][i] = np.quantile(self.mem_err[cand], 0.9)
            neigh.append((cand, dk))
        res = rows.copy()
        for c, v in out.items():
            res[c] = v.astype(np.float32)
        return res, neigh

    # ---------------- inference ----------------
    def case_entry(self, case_id: str) -> dict:
        for c in self.registry["cases"]:
            if c["case_id"] == case_id:
                return c
        raise CaseNotFound(case_id)

    def b0_predict(self, df: pd.DataFrame) -> np.ndarray:
        m = df[self.b0_group].merge(self.b0_table, on=self.b0_group, how="left")["p"].to_numpy()
        return np.where(np.isnan(m), self.b0_rate, m)

    def run(self, case_id: str, fresh: bool = False) -> CaseRun:
        with self._lock:
            if not fresh and case_id in self._cache:
                return self._cache[case_id]
            entry = self.case_entry(case_id)
            t = {}
            t0 = time.perf_counter()
            rows = pd.read_parquet(self.dir / "cases" / case_id / "forecast_state.parquet")
            rows = rows.sort_values(["region_id", "lead_day"]).reset_index(drop=True)
            t["load_inputs"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            rows["p_sentinel"] = self.sentinel.calibrate(self.sentinel.predict_raw(rows))
            rows["p_b2"] = self.b2.calibrate(self.b2.predict_raw(rows))
            rows["p_b0"] = self.b0_predict(rows)
            t["model_inference"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            rows, neigh = self.memory_lookup(rows)
            t["memory_lookup"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            dist, lvl = support_distance(rows, self.support)
            rows["support_distance"], rows["support_level"] = dist, lvl
            rows["evidence_level"] = evidence_strength(lvl, rows["an_n_within"].to_numpy(), rows["an_bust_rate"].to_numpy(),
                                                       rows["p_sentinel"].to_numpy(), self.kmin)
            rows["priority"] = priority_score(rows["p_sentinel"].to_numpy(), rows["p_b2"].to_numpy(),
                                              rows["lead_day"].to_numpy(), rows["evidence_level"].to_numpy())
            t["support_evidence_priority"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            contrib = self.sentinel.contributions(rows)
            t["attribution"] = time.perf_counter() - t0
            run = CaseRun(case_id, entry, rows, contrib, neigh,
                          {k: round(1000 * v, 1) for k, v in t.items()} | {"total": round(1000 * sum(t.values()), 1)})
            self._cache[case_id] = run
            return run

    # ---------------- views ----------------
    def _cell(self, r: pd.Series) -> dict:
        p, b2 = float(r["p_sentinel"]), float(r["p_b2"])
        return {"lead_day": int(r["lead_day"]), "valid_time": str(r["valid_time"]), "bust_probability": p,
                "reliability_confidence": 1 - p, "b2_probability": b2, "b0_probability": float(r["p_b0"]),
                "disagreement_pp": 100 * (p - b2), "alert": bool(p >= self.threshold),
                "spread_m": float(r["spread_m"]), "spread_pct": _f(r["spread_pct"]),
                "support_level": SUPPORT_LEVELS[int(r["support_level"])], "support_distance": _f(r["support_distance"]),
                "evidence_quality": EVIDENCE_NAMES[int(r["evidence_level"])],
                "analogues_within_radius": _i(r["an_n_within"]), "verified_cases_available": _i(r["an_n_eligible"]),
                "analogue_bust_rate": _f(r["an_bust_rate"]), "priority_score": float(r["priority"])}

    def overview(self, case_id: str) -> dict:
        run = self.run(case_id)
        regions = []
        for rid, g in run.rows.groupby("region_id", sort=True):
            g = g.sort_values("lead_day")
            cells = [self._cell(r) for _, r in g.iterrows()]
            ps = np.array([c["bust_probability"] for c in cells])
            first = g.iloc[0]
            regions.append({"region_id": rid, "lat": float(first["lat"]), "lon": float(first["lon"]),
                            "row": int(first["row"]), "col": int(first["col"]),
                            "peak_lead_day": int(np.argmax(ps)) + 1, "peak_bust_probability": float(ps.max()),
                            "alert_days": [c["lead_day"] for c in cells if c["alert"]], "days": cells})
        queue = run.rows.sort_values("priority", ascending=False).head(20)
        return {"case": run.meta, "model": self.model_summary(), "timings_ms": run.timings_ms,
                "alert_threshold": self.threshold, "priority_formula": formula(),
                "regions": regions,
                "priority_queue": [{"region_id": r["region_id"], **self._cell(r)} for _, r in queue.iterrows()],
                "lead_summary": [{"lead_day": d, "mean_bust_probability": float(g["p_sentinel"].mean()),
                                  "max_bust_probability": float(g["p_sentinel"].max()),
                                  "n_alerts": int((g["p_sentinel"] >= self.threshold).sum()),
                                  "mean_disagreement_pp": float(100 * (g["p_sentinel"] - g["p_b2"]).mean())}
                                 for d, g in run.rows.groupby("lead_day")],
                "blind": True}

    def _row(self, run: CaseRun, region_id: str, lead_day: int) -> tuple[int, pd.Series]:
        m = np.where((run.rows["region_id"] == region_id) & (run.rows["lead_day"] == lead_day))[0]
        if len(m) != 1:
            raise CaseNotFound(f"{region_id} day {lead_day}")
        return int(m[0]), run.rows.iloc[int(m[0])]

    def region(self, case_id: str, region_id: str) -> dict:
        run = self.run(case_id)
        g = run.rows[run.rows["region_id"] == region_id].sort_values("lead_day")
        if g.empty:
            raise CaseNotFound(region_id)
        return {"case_id": case_id, "region_id": region_id, "lat": float(g.iloc[0]["lat"]),
                "lon": float(g.iloc[0]["lon"]), "alert_threshold": self.threshold,
                "trajectory": [self._cell(r) for _, r in g.iterrows()]}

    def expected_signature(self, cand: np.ndarray) -> dict:
        if len(cand) == 0:
            return {"basis": "none", "n": 0, "distribution": None, "top": None}
        sub = self.mem.iloc[cand]
        busts = sub[sub["bust"] == 1]
        basis, use = ("analogue busts", busts) if len(busts) >= 3 else ("all analogues", sub)
        counts = use["sig_class"].value_counts()
        dist = {c: float(counts.get(c, 0) / len(use)) for c in CLASSES}
        top = max(dist, key=dist.get)
        return {"basis": basis, "n": int(len(use)), "distribution": dist, "top": top, "top_label": LABELS[top]}

    def explain(self, case_id: str, region_id: str, lead_day: int) -> dict:
        run = self.run(case_id)
        i, r = self._row(run, region_id, lead_day)
        cell = self._cell(r)
        ex = explain_row(r, run.contrib[i], self.sentinel.features, self.ref, cell["bust_probability"],
                         cell["b2_probability"], self.base_rate)
        cand, dk = run.neighbours[i]
        analogues = []
        for j, d in list(zip(cand, dk))[:8]:
            a = self.mem.iloc[int(j)]
            analogues.append({"case_id": a["case_id"], "init_time": str(a["init_time"]), "distance": float(d),
                              "bust": int(a["bust"]), "normalized_error": float(a["norm_error"]),
                              "signature": a["sig_class"] if int(a["bust"]) else None,
                              "split": a["split"]})
        context = []  # forecast-state values of features the validated model does NOT use (context only)
        for f in ["anom500", "grad_mag", "m_sign_agree", "m_p10p90", "spread_nbhd", "spread_growth", "pc_norm",
                  "rev_nbhd", "rec_err"]:
            v = _f(r.get(f))
            ref = self.ref.get("abs_anom500" if f == "anom500" else f)
            pct = None
            if v is not None and ref is not None:
                pct = float(np.searchsorted(ref, abs(v) if f == "anom500" else v, side="right") / len(ref) * 100)
            context.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "value": v, "train_percentile": pct})
        bias = float(run.contrib[i][-1])
        return {"case_id": case_id, "region_id": region_id, "lead_day": lead_day, **cell,
                "attribution": {"method": "TreeSHAP on the Sentinel booster (log-odds, before isotonic calibration)",
                                "bias_logodds": bias, "drivers": ex["drivers"],
                                "groups": {g: v["contribution_logodds"] for g, v in ex["groups"].items()},
                                "note": ex["attribution_note"]},
                "evidence": ex["evidence"],
                "analogue_summary": {"k": self.k, "radius": self.radius, "rule": "same region and lead day; "
                                     "verified (valid_time) no later than this initialisation",
                                     "verified_cases_available": cell["verified_cases_available"],
                                     "within_radius": cell["analogues_within_radius"],
                                     "bust_rate": cell["analogue_bust_rate"],
                                     "climatological_bust_rate": self.base_rate,
                                     "median_normalized_error": _f(r["an_err_med"]),
                                     "q90_normalized_error": _f(r["an_err_q90"]),
                                     "nearest": analogues},
                "failure_signature": self.expected_signature(cand),
                "context_features": context,
                "model_inputs": {f: _f(r[f]) for f in self.sentinel.features}}

    # ---------------- verification (only on explicit reveal) ----------------
    @lru_cache(maxsize=16)
    def _verification(self, case_id: str) -> dict:
        self.case_entry(case_id)
        return json.loads((self.dir / "cases" / case_id / "verification.json").read_text())

    def reveal(self, case_id: str, region_id: str, lead_day: int) -> dict:
        run = self.run(case_id)
        v = self._verification(case_id)
        i, r = self._row(run, region_id, lead_day)
        reg = next(x for x in v["regions"] if x["region_id"] == region_id)
        day = next(d for d in reg["days"] if d["lead_day"] == lead_day)
        exp = self.expected_signature(run.neighbours[i][0])
        cell = self._cell(r)
        # memory update: this verified region/lead joins the store; it becomes eligible for any later
        # initialisation whose init_time >= its valid_time.
        lo, hi = self.mem_groups[(region_id, lead_day)]
        n_before = int(cell["verified_cases_available"] or 0)
        alerts = run.rows["p_sentinel"].to_numpy() >= self.threshold
        order = run.rows[["region_id", "lead_day"]].apply(tuple, axis=1).tolist()
        vmap = {(x["region_id"], d["lead_day"]): d["bust"] for x in v["regions"] for d in x["days"]}
        busts = np.array([vmap[k] for k in order])
        return {"case_id": case_id, "region_id": region_id, "lead_day": lead_day,
                "reference": v["reference"],
                "forecast": {"bust_probability": cell["bust_probability"], "b2_probability": cell["b2_probability"],
                             "alert": cell["alert"], "expected_signature": exp},
                "verification": {"actual_bust": bool(day["bust"]), "normalized_error": day["normalized_error"],
                                 "threshold_q90": day["threshold_q90"], "error_m": day["error_m"],
                                 "hidden_bust": bool(day["hidden_bust"]),
                                 "failure_fingerprint": {"signature": day["signature"],
                                                         "label": day["signature_label"],
                                                         "phase_share": day["phase_share"], "bias_m": day["bias_m"],
                                                         "pattern_corr": day["pattern_corr"]}},
                "comparison": {"expected_top": exp.get("top"),
                               "p_expected_for_actual": (exp["distribution"] or {}).get(day["signature"]),
                               "match": exp.get("top") == day["signature"] if day["bust"] else None,
                               "note": "The expected signature is evaluated against the actual fingerprint only "
                                       "when the region-day verified as a bust."},
                "trajectory": [{"lead_day": d["lead_day"], "actual_bust": bool(d["bust"]),
                                "normalized_error": d["normalized_error"], "threshold_q90": d["threshold_q90"],
                                "signature": d["signature"]} for d in sorted(reg["days"], key=lambda d: d["lead_day"])],
                "case_summary": {"region_days": int(len(busts)), "verified_busts": int(busts.sum()),
                                 "sentinel_alerts": int(alerts.sum()), "sentinel_hits": int((alerts & (busts == 1)).sum()),
                                 "hidden_busts": v["summary"]["n_hidden_bust_region_days"]},
                "bust_map": [{"region_id": x["region_id"], "lead_day": d["lead_day"], "bust": d["bust"],
                              "signature": d["signature"]} for x in v["regions"] for d in x["days"]],
                "memory_update": {"entries_added": int(len(busts)),
                                  "entry": {"region_id": region_id, "lead_day": lead_day,
                                            "valid_time": cell["valid_time"], "bust": bool(day["bust"]),
                                            "normalized_error": day["normalized_error"],
                                            "signature": day["signature"]},
                                  "verified_cases_before": n_before,
                                  "verified_cases_after": n_before + 1,
                                  "available_from": cell["valid_time"],
                                  "store_size_region_lead": int(hi - lo),
                                  "note": "Verified outcomes become analogue candidates for initialisations at or "
                                          "after their valid time (causal memory). The persisted research store "
                                          "already contains them; nothing is re-trained."}}

    @lru_cache(maxsize=16)
    def fields(self, case_id: str) -> dict:
        self.case_entry(case_id)
        return json.loads((self.dir / "cases" / case_id / "fields.json").read_text())

    def model_summary(self) -> dict:
        m = self.meta
        return {k: m.get(k) for k in ("model_version", "model_artifact", "exported_from", "retrained_for_demo",
                                      "training_window", "calibration_window", "calibration", "learner",
                                      "sentinel_features", "b2_features", "selected_groups", "selection_note",
                                      "pending", "alert_threshold", "alert_threshold_definition",
                                      "confidence_definition", "target")}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _i(x):
    x = _f(x)
    return None if x is None else int(round(x))


_ENGINE: Engine | None = None


def engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
