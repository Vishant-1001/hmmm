# SIH26079 — Source-Locked Material

This file reproduces source material **verbatim**. It is kept separate from all
project-specific implementation material. Nothing in this file is our wording;
nothing in our architecture documents is official SIH wording.

## Problem statement (source-locked, verbatim)

SIH26079 - AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts

• Problem Statement Medium-range weather forecasts sometimes show large errors during rapidly evolving systems such as monsoon depressions, heavy rainfall events, western disturbances, cyclones, heat waves and break/active monsoon phases. Such forecast failures, or 'forecast busts', can affect operational decision-making.

• Challenge The challenge is to develop an AI/ML-based system that can identify regions and lead times where the forecast is likely to have high uncertainty or large error. The system should compare current NWP forecast patterns with historical forecast error behaviour and provide a forecast confidence indicator.

Expected Outcome - Description Forecast confidence map - Region-wise confidence for Day 1 to Day 10 forecasts Forecast bust probability - Probability of large forecast error over different regions Error-prone area detection - Identification of areas where model forecast may be unreliable Explainable output - Key meteorological reasons for low confidence Prototype dashboard/API - Simple interface for operational use

## Metadata (as observed in a public SIH26079 record — verify on the portal before submission)

| Field | Value |
|---|---|
| S.No. | 79 |
| PS Number | SIH26079 |
| Title | AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts |
| Organization | Ministry of Earth Sciences (MoES) |
| Department | National Centre for Medium Range Weather Forecasting (NCMRWF) |
| Category | Software |
| Theme | Smart Automation (**public mirrors have shown inconsistent Theme metadata — verify on the official portal**) |
| Deadline for Idea Submission | 30 September 2026 |
| Dataset Link | N/A |
| Contact Info | N/A |
| Youtube Link | N/A |

PS number, title, organization and department are treated as stable identifiers.
We do not invent dataset or contact information that the portal does not provide.

## Evaluation rubric (source-locked, verbatim)

The SAH 2026 internal rubric supplied for this project is:

| Criterion | Marks |
|---|---:|
| Novelty & Innovation | 10 |
| Technical Approach & Complexity | 10 |
| Feasibility & Viability | 10 |
| Impact, Scale & Sustainability | 10 |
| Prototype & Demonstration Readiness | 5 |
| Presentation & Format Compliance | 5 |
| **TOTAL** | **50** |

The rubric specifically values:

### Novelty & Innovation
- originality relative to existing/known approaches;
- clear differentiation from earlier SIH submissions and off-the-shelf products.

### Technical Approach & Complexity
- sound architecture;
- methodology;
- engineering depth;
- justification of technology choices;
- non-trivial implementation.

### Feasibility & Viability
- buildability;
- credible assumptions;
- realistic risks;
- practical data/resource assumptions.

### Impact, Scale & Sustainability
- benefit to end user and sponsor;
- scale;
- economic/social/environmental relevance;
- future scope.

### Prototype & Demonstration Readiness
- evidence of a working module or validated PoC;
- quality of live demonstration;
- ability to explain measured results.

### Presentation & Format Compliance
- clarity;
- six-slide format;
- quality of response to jury questions.

The bottom-line product requirement is therefore not:

> "A sophisticated idea."

It is:

> **A functioning system whose technical decisions, assumptions, measurements, and limitations can survive jury questioning.**
