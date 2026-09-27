# Forecast Bust Sentinel

Research prototype for **SIH26079 — AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts**
(MoES / NCMRWF). The official problem statement is reproduced verbatim in
[`docs/sih_source.md`](docs/sih_source.md).

> NWP tells you what it predicts. Forecast Bust Sentinel estimates when that prediction is likely to fail.

Forecast Bust Sentinel is an ML reliability layer over an *existing* medium-range NWP ensemble. It does
**not** forecast the weather. It estimates, per 5°×5° region and per lead day (Day 1–10), the probability
that the existing forecast will exceed a project-defined large-error ("bust") threshold.

**Status:** under construction — see [`BUILD_PROGRESS.md`](BUILD_PROGRESS.md). No metrics exist yet.
