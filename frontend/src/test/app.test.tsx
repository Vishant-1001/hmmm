// Frontend behaviour tests with TEST FIXTURES (mocked API).
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Analytics } from "../components/Analytics";
import { CaseView } from "../components/CaseView";
import { fixtureCase, fixtureVerification } from "./fixtures";

function mockFetch(routes: Record<string, unknown | number>) {
  const calls: string[] = [];
  globalThis.fetch = vi.fn(async (url: string) => {
    calls.push(url);
    const hit = Object.keys(routes).find((k) => url.endsWith(k));
    const v = hit ? routes[hit] : 404;
    if (typeof v === "number") return { ok: false, status: v, json: async () => ({ detail: "NOT YET COMPUTED" }) } as Response;
    return { ok: true, status: 200, json: async () => v } as Response;
  }) as any;
  return calls;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

const cases = [{ case_id: "2022010100", init_time: "2022-01-01T00:00:00", selection: "random", selection_note: "", n_alerts: 1 }];

describe("blind replay", () => {
  it("renders risk without requesting verification, then reveals", async () => {
    const calls = mockFetch({ "/full": fixtureCase, "/verification": fixtureVerification });
    render(<CaseView cases={cases} />);
    await waitFor(() => expect(screen.getByTestId("p-bust")).toBeTruthy());
    expect(screen.getByTestId("p-bust").textContent).toBe("60%");
    expect(screen.getByTestId("mode-flag").textContent).toContain("BLIND MODE");
    expect(calls.some((c) => c.includes("verification"))).toBe(false);
    expect(screen.queryByTestId("fingerprint")).toBeNull();
    await act(async () => { fireEvent.click(screen.getByTestId("reveal")); });
    await waitFor(() => expect(screen.getByTestId("verif-summary")).toBeTruthy());
    expect(screen.getByTestId("mode-flag").textContent).toContain("REVEALED");
    expect(screen.getByTestId("fingerprint").textContent).toContain("Position / phase");
  });

  it("shows an explicit error instead of numbers when the case is unavailable", async () => {
    mockFetch({});
    render(<CaseView cases={cases} />);
    await waitFor(() => expect(screen.getByText(/DATA UNAVAILABLE/)).toBeTruthy());
    expect(screen.queryByTestId("p-bust")).toBeNull();
  });
});

describe("analytics", () => {
  it("never shows fallback metrics when metrics are not computed", async () => {
    mockFetch({ "/api/metrics": 503 });
    render(<Analytics />);
    await waitFor(() => expect(screen.getByTestId("metrics-error")).toBeTruthy());
    expect(screen.getByTestId("metrics-error").textContent).toContain("NOT YET COMPUTED");
    expect(screen.queryByTestId("model-table")).toBeNull();
  });
});
