// Demo app flow on REAL API responses (demo_fixture.json is dumped from the FastAPI app by
// FastAPI TestClient; nothing in it is hand-written). Checks that displayed values equal API values
// and that verification is never requested before the reveal click.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DemoApp from "../demo/DemoApp";
import { num, pct } from "../demo/ui";
import fx from "./demo_fixture.json";

const { case_id: cid, region_id: rid, lead_day: day } = fx;
let calls: string[] = [];

function mockApi() {
  calls = [];
  globalThis.fetch = vi.fn(async (url: string) => {
    calls.push(url);
    const u = url.split("?")[0];
    const body =
      u === "/api/demo/cases" ? fx.cases
      : u === `/api/demo/cases/${cid}/run` ? fx.run
      : u === `/api/demo/cases/${cid}/regions/${rid}` ? fx.region
      : u === `/api/demo/cases/${cid}/regions/${rid}/explain` ? fx.explain
      : u === `/api/demo/cases/${cid}/regions/${rid}/reveal` ? fx.reveal
      : null;
    if (body == null) return { ok: false, status: 404, json: async () => ({ detail: "not in fixture" }) } as Response;
    return { ok: true, status: 200, json: async () => body } as Response;
  }) as any;
}

beforeEach(() => { window.location.hash = "#/overview"; mockApi(); });
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("demo app (live-inference API)", () => {
  it("runs the case and shows the model's overview values", async () => {
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId("n-alerts")).toBeTruthy());
    expect(calls).toContain(`/api/demo/cases/${cid}/run`);
    expect(screen.getByTestId("n-alerts").textContent).toBe(String(fx.run.lead_summary[day - 1].n_alerts));
    expect(screen.getByTestId("replay-badge").textContent).toBe("HISTORICAL REPLAY");
    expect(screen.getByTestId("source-badge").textContent).toBe("SOURCE · ECMWF IFS / ERA5");
    expect(screen.getByTestId("selection-pill").textContent).toContain(rid);
  });

  it("region -> reliability -> why flagged -> blind verification -> reveal", async () => {
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId(`prio-${rid}`)).toBeTruthy());

    fireEvent.click(screen.getByTestId(`prio-${rid}`));
    await waitFor(() => expect(screen.getByTestId("rel-p")).toBeTruthy());
    const c = fx.region.trajectory[day - 1];
    expect(screen.getByTestId("rel-p").textContent).toBe(pct(c.bust_probability));
    expect(screen.getByTestId("rel-b0").textContent).toBe(pct(c.b0_probability));
    expect(screen.getByTestId("rel-risk").textContent).toBe(c.risk_level);
    expect(screen.getByTestId("rel-analogue").textContent).toBe(c.analogue_evidence);
    expect(screen.getByTestId("rel-evidence").textContent).toBe(c.evidence_quality);
    expect(screen.queryByText(/confidence/i)).toBeNull();

    fireEvent.click(screen.getByTestId("go-evidence"));
    await waitFor(() => expect(screen.getByTestId("why-p")).toBeTruthy());
    const x = fx.explain;
    expect(screen.getByTestId("why-p").textContent).toBe(pct(x.bust_probability));
    expect(screen.getByTestId("why-b0").textContent).toBe(pct(x.b0_probability));
    expect(screen.getByTestId("why-risk").textContent).toBe(x.risk_level);
    expect(screen.getByTestId("why-analogue").textContent).toBe(x.analogue_evidence);
    for (const t of x.interpretation) expect(screen.getByTestId("why-list").textContent).toContain(t);
    expect(screen.getByTestId("why-support").textContent).toBe(x.support_level);
    expect(screen.getByTestId("why-evidence").textContent).toBe(x.evidence_quality);
    expect(screen.getByTestId("an-within").textContent).toBe(String(x.analogue_summary.within_radius ?? "—"));
    for (const d of x.attribution.drivers) expect(screen.getByTestId(`driver-${d.feature}`)).toBeTruthy();
    if (x.failure_signature.top_label) expect(screen.getByTestId("exp-sig").textContent).toBe(x.failure_signature.top_label);

    fireEvent.click(screen.getByTestId("go-verification"));
    await waitFor(() => expect(screen.getByTestId("reveal")).toBeTruthy());
    expect(screen.getByTestId("verif-mode").textContent).toContain("BLIND");
    expect(calls.some((u) => u.includes("/reveal"))).toBe(false);
    expect(screen.queryByTestId("actual-bust")).toBeNull();
    expect(screen.queryByTestId("actual-err")).toBeNull();
    expect(screen.queryByTestId("fingerprint")).toBeNull();

    fireEvent.click(screen.getByTestId("reveal"));
    await waitFor(() => expect(screen.getByTestId("memory-update")).toBeTruthy());
    const v = fx.reveal;
    expect(calls.filter((u) => u.includes("/reveal")).length).toBeGreaterThan(0);
    expect(screen.getByTestId("actual-bust").textContent).toBe(v.verification.actual_bust ? "BUST" : "NO BUST");
    expect(screen.getByTestId("actual-err").textContent).toBe(v.verification.normalized_error.toFixed(3));
    expect(screen.getByTestId("actual-rmse").textContent).toBe(`${num(v.verification.error_m, 1)} m`);
    expect(screen.getByTestId("verif-mode").textContent).toContain("POST-VERIFICATION");
    expect(screen.getByTestId("fingerprint").textContent).toContain(v.verification.failure_fingerprint.label);
    expect(screen.getByTestId("mem-before").textContent).toBe(String(v.memory_update.verified_cases_before));
    expect(screen.getByTestId("mem-after").textContent).toBe(String(v.memory_update.verified_cases_after));

    // reset returns to blind mode: truth hidden again, reveal available again
    fireEvent.click(screen.getByTestId("reset"));
    await waitFor(() => expect(screen.getByTestId("reveal")).toBeTruthy());
    expect(screen.getByTestId("verif-mode").textContent).toContain("BLIND");
    expect(screen.queryByTestId("actual-bust")).toBeNull();
    expect(screen.queryByTestId("fingerprint")).toBeNull();
    const before = calls.filter((u) => u.includes("/reveal")).length;
    fireEvent.click(screen.getByTestId("reveal"));
    await waitFor(() => expect(screen.getByTestId("actual-bust")).toBeTruthy());
    expect(calls.filter((u) => u.includes("/reveal")).length).toBe(before + 1);
  });

  it("Day 1-10: switching lead day updates the reliability values without off-by-one", async () => {
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId(`prio-${rid}`)).toBeTruthy());
    fireEvent.click(screen.getByTestId(`prio-${rid}`));
    await waitFor(() => expect(screen.getByTestId("rel-p")).toBeTruthy());
    for (const d of [1, 5, 10, 2]) {
      fireEvent.click(screen.getByTestId(`day-${d}`));
      const c = fx.region.trajectory[d - 1];
      expect(c.lead_day).toBe(d);
      await waitFor(() => expect(screen.getByTestId("rel-p").textContent).toBe(pct(c.bust_probability)));
      expect(screen.getByTestId("selection-pill").textContent).toContain(`Day ${d}`);
    }
  });

  it("shows the served model id and real-data provenance", async () => {
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId("n-alerts")).toBeTruthy());
    expect(screen.getByTestId("served-model").textContent).toContain(fx.run.model.model_id);
    expect(screen.getByTestId("source-badge").textContent).toBe("SOURCE · ECMWF IFS / ERA5");
    expect(screen.getByTestId("model-note").textContent).toContain("b2_spread_calibrated");
  });

  it("shows API errors instead of placeholder numbers", async () => {
    globalThis.fetch = vi.fn(async () => ({ ok: false, status: 503, json: async () => ({ detail: "NOT YET COMPUTED" }) }) as Response) as any;
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId("error").textContent).toContain("NOT YET COMPUTED"));
    expect(screen.queryByTestId("n-alerts")).toBeNull();
  });

  it("Retry re-issues the failed request and recovers", async () => {
    const real = globalThis.fetch;
    let failed = false;
    globalThis.fetch = vi.fn(async (url: string, init?: RequestInit) => {
      if (!failed) { failed = true; return { ok: false, status: 503, json: async () => ({ detail: "waking up" }) } as Response; }
      return (real as any)(url, init);
    }) as any;
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId("error").textContent).toContain("waking up"));
    fireEvent.click(screen.getByTestId("retry"));
    await waitFor(() => expect(screen.getByTestId("n-alerts")).toBeTruthy());
    expect(screen.queryByTestId("error")).toBeNull();
  });

  it("re-selecting the current case keeps the loaded run (no endless loading)", async () => {
    render(<DemoApp />);
    await waitFor(() => expect(screen.getByTestId("n-alerts")).toBeTruthy());
    const runs = calls.filter((u) => u.endsWith("/run")).length;
    fireEvent.change(screen.getByTestId("case-select"), { target: { value: cid } });
    expect(screen.queryByTestId("loading")).toBeNull();
    expect(screen.getByTestId("n-alerts")).toBeTruthy();
    expect(calls.filter((u) => u.endsWith("/run")).length).toBe(runs);
  });
});
