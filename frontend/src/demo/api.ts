// Client for the live-inference demo API. Errors are surfaced verbatim; no placeholder numbers.
import type { CaseList, Explanation, Fields, RegionDetail, Reveal, RunResult } from "./types";

export class ApiError extends Error {}

async function call<T>(path: string, method: "GET" | "POST" = "GET"): Promise<T> {
  const r = await fetch(path, { method });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { detail = (await r.json()).detail ?? detail; } catch { /* non-JSON body */ }
    throw new ApiError(`DATA UNAVAILABLE (${detail})`);
  }
  return r.json() as Promise<T>;
}

export const demoApi = {
  cases: () => call<CaseList>("/api/demo/cases"),
  run: (id: string) => call<RunResult>(`/api/demo/cases/${id}/run`, "POST"),
  region: (id: string, rid: string) => call<RegionDetail>(`/api/demo/cases/${id}/regions/${rid}`),
  explain: (id: string, rid: string, day: number) => call<Explanation>(`/api/demo/cases/${id}/regions/${rid}/explain?lead_day=${day}`),
  reveal: (id: string, rid: string, day: number) => call<Reveal>(`/api/demo/cases/${id}/regions/${rid}/reveal?lead_day=${day}`, "POST"),
  fields: (id: string) => call<Fields>(`/api/demo/cases/${id}/fields`),
};
