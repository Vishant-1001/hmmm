// Provenance badge: real sources are named, synthetic scenarios are always marked DEMO MODE.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { SourceBadge } from "../demo/ui";
import type { ForecastSource } from "../demo/types";
import fx from "./demo_fixture.json";

const base = fx.run.source as ForecastSource;
const days = Array.from({ length: 10 }, (_, i) => i + 1);
afterEach(cleanup);

describe("SourceBadge", () => {
  it("renders the real ECMWF source from the API response", () => {
    render(<SourceBadge source={base} />);
    expect(screen.getByTestId("source-badge").textContent).toBe("SOURCE · ECMWF IFS / ERA5");
    expect(base.synthetic).toBe(false);
  });

  it("renders NCMRWF TIGGE for the NCMRWF provider", () => {
    render(<SourceBadge source={{ ...base, provider: "ncmrwf_tigge", ensemble_member_count: 22 }} />);
    expect(screen.getByTestId("source-badge").textContent).toBe("SOURCE · NCMRWF TIGGE");
  });

  it("marks synthetic data as demo and never as a real source", () => {
    render(<SourceBadge source={{ ...base, provider: "synthetic", synthetic: true, demo_only: true }} />);
    const t = screen.getByTestId("source-badge").textContent ?? "";
    expect(t).toBe("DEMO MODE · SYNTHETIC SCENARIO");
    expect(t).not.toMatch(/ECMWF|NCMRWF|ERA5|observed|validated/i);
  });

  it("a synthetic flag wins even if the provider name looks real", () => {
    render(<SourceBadge source={{ ...base, synthetic: true }} />);
    expect(screen.getByTestId("source-badge").textContent).toBe("DEMO MODE · SYNTHETIC SCENARIO");
  });

  it("switching provider keeps Day 1-10", () => {
    for (const provider of ["ecmwf_research", "ncmrwf_tigge", "synthetic"]) {
      const s = { ...base, provider, synthetic: provider === "synthetic", demo_only: provider === "synthetic" };
      render(<SourceBadge source={s} />);
      expect(s.lead_days).toEqual(days);
      cleanup();
    }
  });
});
