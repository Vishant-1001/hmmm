# Demo guide (≈3 minutes)

Start: `scripts/serve.sh` → open http://127.0.0.1:8000 . Everything shown is a **historical research
replay** of real 2022 ECMWF IFS ENS forecasts (the held-out test year). There is no live feed.

## Case choice (documented, not cherry-picked for model performance)

`artifacts/replay/index.json` lists 12 cases:
* 4 **stress** cases — the 2022 initialisations with the most *hidden-bust* region-days at Day 3–7,
  ranked by verification only (the model output plays no role in the ranking). They are chosen because they
  are hard for a spread-only system, and they are **not representative** of average performance.
* 8 **random** cases (fixed seed) from the 2022 test initialisations.

For the pitch use the first stress case and then show one random case, saying so explicitly. Whatever the
model does on the chosen case, the overall benchmark (Analytics tab) is the evidence.

## Script

| Time | Screen | Say / show |
|---|---|---|
| 0:00–0:20 | header + banner | SIH26079 problem: occasional large medium-range errors. Sentinel is a reliability layer over existing NWP, not a new forecast. Research replay, not live. |
| 0:20–0:50 | Map + day buttons | Bust-risk map D1…D10 for the chosen initialisation. Risk = probability the *existing* forecast exceeds the TRAIN-Q90 error threshold. Alerts use the validation-chosen 10%-FAR threshold. |
| 0:50–1:20 | Priority queue → region | Click the top entry: bust risk, reliability confidence, spread baseline B2, confidence disagreement, peak-risk day, evidence strength, historical support. |
| 1:20–1:45 | WHY panel | Numeric evidence A–E, attributions with training percentiles, analogues, historical failure-signature distribution ("similar forecast states previously failed through…", evidence not proof). |
| 1:45–2:00 | still blind | "Verification has not been revealed." (mode flag says BLIND MODE) |
| 2:00–2:20 | REVEAL VERIFICATION | Verified busts outlined on the map (dashed = hidden bust); summary box: Sentinel vs B2 alerts/hits and overlap. |
| 2:20–2:40 | Trajectory + error bars + fingerprint | Predicted risk trajectory vs actual error/Q90 bars; expected signature vs actual fingerprint; Forecast − ERA5 field. |
| 2:40–3:00 | Analytics tab | B2 vs Sentinel: AUPRC with bootstrap CI and the pre-registered verdict, calibration diagram, hidden-bust recall, warning lead. Only measured values; if improvement is not established, say so. |

## Likely jury questions

* *Doesn't the ensemble already give uncertainty?* Yes — spread is the baseline (B2, calibrated, same learner).
  We test whether anything beyond spread helps; the Analytics tab shows the measured answer.
* *Is this just forecast-error prediction?* That concept exists. Our contribution is the connected, verified
  workflow: regional Day 1–10 bust probability, spread-baseline disagreement, historical evidence with support
  quality, blind replay and failure fingerprints.
* *Does it improve the forecast / predict weather?* No. It estimates the probability that the existing forecast
  exceeds a defined error threshold.
* *What if it doesn't beat spread?* Then we report that — see `docs/evaluation.md`.
* *NCMRWF?* The architecture supports NCMRWF-compatible ensemble ingestion (adapter contract), but the validated
  research prototype uses real historical ECMWF IFS ENS data.
* *Why Z500?* A coherent large-scale circulation target for the first validated experiment.
* *Why not a Transformer/GNN?* The MVP prioritises validation against a strong baseline; complexity must earn
  its place with measured gains.
