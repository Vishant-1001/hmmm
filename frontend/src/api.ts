// Thin client for the FastAPI backend. Errors are surfaced verbatim; the UI never
// substitutes placeholder numbers when a request fails.
import type { CaseIndexItem, ForecastCase, Verification } from "./types";

export class ApiError extends Error {}

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(`DATA UNAVAILABLE (${detail})`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => get<{ status: string; mode: string }>("/api/health"),
  cases: () => get<{ mode: string; cases: CaseIndexItem[]; selection_rule: string }>("/api/forecast/cases"),
  // full blind case (forecast-time information + model output only)
  forecastCase: (id: string) => get<ForecastCase>(`/api/replay/${id}/full`),
  verification: (id: string) => get<Verification>(`/api/replay/${id}/verification`),
  metrics: () => get<any>("/api/metrics"),
  provenance: () => get<any>("/api/provenance"),
};
