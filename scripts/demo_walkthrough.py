"""Drive the REAL running application (FastAPI + built React UI) in a headless browser.

    python scripts/demo_walkthrough.py [--url http://127.0.0.1:8000] [--shots artifacts/screenshots]
                                       [--video artifacts/demo_video] [--pace 1.0]

Flow: select case -> model run -> region selection -> Reliability -> Why Flagged -> Verification
(blind) -> Reveal -> Priority -> Trust. At each step the value shown in the UI is compared with the
value returned by the API for the same request (the E2E test in tests/test_e2e_demo.py calls this).
Case / region choice is a fixed rule, not a curated pick: the first registry case drawn at random
(earliest), and the top entry of the model's own priority queue.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import urllib.request
from pathlib import Path


def chromium_path() -> str | None:
    env = os.environ.get("FBS_CHROMIUM")
    if env:
        return env
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")))
    return hits[-1] if hits else None


def _api(url: str, path: str, method: str = "GET"):
    req = urllib.request.Request(url + path, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def walkthrough(url: str, shots: Path | None = None, video: Path | None = None, pace: float = 0.0) -> dict:
    from playwright.sync_api import sync_playwright

    timings, checks, requests = {}, [], []

    def check(name, ui, api):
        ok = ui.strip() == api.strip()
        checks.append({"check": name, "ui": ui.strip(), "api": api.strip(), "ok": ok})
        if not ok:
            raise AssertionError(f"{name}: UI {ui!r} != API {api!r}")

    def wait(s):
        if pace:
            time.sleep(s * pace)

    def shot(page, name):
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shots / name), full_page=False)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chromium_path(), headless=True)
        ctx_kw = {"viewport": {"width": 1600, "height": 1000}, "device_scale_factor": 1}
        if video:
            ctx_kw |= {"record_video_dir": str(video), "record_video_size": {"width": 1600, "height": 1000}}
        ctx = browser.new_context(**ctx_kw)
        page = ctx.new_page()
        page.on("request", lambda r: requests.append(r.url))

        t0 = time.perf_counter()
        page.goto(url + "/#/overview")
        page.wait_for_selector("[data-testid=n-alerts]")
        timings["open_app_and_first_case_ms"] = round(1000 * (time.perf_counter() - t0))
        cases = _api(url, "/api/demo/cases")["cases"]
        case = next(c for c in cases if c["selection"] == "random")
        assert page.input_value("[data-testid=case-select]") == case["case_id"]
        run = _api(url, f"/api/demo/cases/{case['case_id']}/run", "POST")
        # after a run the UI selects the top-priority region and ITS lead day (DemoApp.tsx)
        day = run["priority_queue"][0]["lead_day"] if run["priority_queue"] else 3
        check(f"overview alerts Day {day}", page.inner_text("[data-testid=n-alerts]"),
              str(run["lead_summary"][day - 1]["n_alerts"]))
        wait(4)
        shot(page, "01_overview.png")

        # switch case (model executes for the new initialisation), then back
        other = next(c for c in cases if c["case_id"] != case["case_id"] and c["selection"] == "random")
        t0 = time.perf_counter()
        page.select_option("[data-testid=case-select]", other["case_id"])
        page.wait_for_selector("[data-testid=n-alerts]")
        timings["case_switch_ms"] = round(1000 * (time.perf_counter() - t0))
        wait(3)
        page.select_option("[data-testid=case-select]", case["case_id"])
        page.wait_for_selector("[data-testid=n-alerts]")
        wait(2)

        # region selection via the model's priority queue (table for the selected day)
        top = sorted(run["regions"], key=lambda r: -r["days"][day - 1]["priority_score"])[0]
        rid = top["region_id"]
        t0 = time.perf_counter()
        page.click(f"[data-testid=prio-{rid}]")
        page.wait_for_selector("[data-testid=rel-p]")
        timings["reliability_screen_ms"] = round(1000 * (time.perf_counter() - t0))
        reg = _api(url, f"/api/demo/cases/{case['case_id']}/regions/{rid}")
        c = reg["trajectory"][day - 1]
        check("reliability P(bust)", page.inner_text("[data-testid=rel-p]"), pct(c["bust_probability"]))
        check("reliability confidence", page.inner_text("[data-testid=rel-conf]"), pct(c["reliability_confidence"]))
        check("reliability B2", page.inner_text("[data-testid=rel-b2]"), pct(c["b2_probability"]))
        wait(5)
        shot(page, "02_reliability.png")

        t0 = time.perf_counter()
        page.click("[data-testid=go-evidence]")
        page.wait_for_selector("[data-testid=attribution]")
        page.wait_for_selector("[data-testid=an-within]")
        timings["why_flagged_ms"] = round(1000 * (time.perf_counter() - t0))
        ex = _api(url, f"/api/demo/cases/{case['case_id']}/regions/{rid}/explain?lead_day={day}")
        check("why P(bust)", page.inner_text("[data-testid=why-p]"), pct(ex["bust_probability"]))
        check("why B2", page.inner_text("[data-testid=why-b2]"), pct(ex["b2_probability"]))
        check("why support", page.inner_text("[data-testid=why-support]"), ex["support_level"])
        check("why evidence quality", page.inner_text("[data-testid=why-evidence]"), ex["evidence_quality"])
        check("why analogues within radius", page.inner_text("[data-testid=an-within]"),
              str(ex["analogue_summary"]["within_radius"]))
        if ex["failure_signature"]["top_label"]:
            check("why expected signature", page.inner_text("[data-testid=exp-sig]"), ex["failure_signature"]["top_label"])
        page.wait_for_timeout(600)  # field map
        wait(6)
        shot(page, "03_why_flagged.png")
        page.mouse.wheel(0, 700)
        wait(4)
        page.mouse.wheel(0, -700)

        page.click("[data-testid=go-verification]")
        page.wait_for_selector("[data-testid=reveal]")
        assert "BLIND" in page.inner_text("[data-testid=verif-mode]")
        assert not any("/reveal" in u for u in requests), "verification requested before reveal"
        wait(4)
        shot(page, "04a_verification_blind.png")
        t0 = time.perf_counter()
        page.click("[data-testid=reveal]")
        page.wait_for_selector("[data-testid=memory-update]")
        timings["verification_reveal_ms"] = round(1000 * (time.perf_counter() - t0))
        rv = _api(url, f"/api/demo/cases/{case['case_id']}/regions/{rid}/reveal?lead_day={day}", "POST")
        check("actual bust", page.inner_text("[data-testid=actual-bust]"),
              "BUST" if rv["verification"]["actual_bust"] else "NO BUST")
        check("normalized error", page.inner_text("[data-testid=actual-err]"), f"{rv['verification']['normalized_error']:.3f}")
        assert rv["verification"]["failure_fingerprint"]["label"] in page.inner_text("[data-testid=fingerprint]")
        check("memory before", page.inner_text("[data-testid=mem-before]"), str(rv["memory_update"]["verified_cases_before"]))
        wait(4)
        shot(page, "04_verification.png")
        page.mouse.wheel(0, 900)
        wait(5)
        page.wait_for_timeout(300)
        shot(page, "04b_verification_memory.png")

        page.click("[data-testid=nav-priority]")
        page.wait_for_selector("[data-testid=priority-table]")
        wait(4)
        shot(page, "05_priority.png")
        page.click("[data-testid=nav-trust]")
        page.wait_for_selector("[data-testid=model-table]")
        wait(5)
        shot(page, "06_model_trust.png")
        page.click("[data-testid=nav-overview]")
        page.wait_for_selector("[data-testid=n-alerts]")
        wait(3)
        ctx.close()
        browser.close()
    return {"case_id": case["case_id"], "region_id": rid, "lead_day": day, "timings_ms": timings, "checks": checks,
            "reveal_requested_before_click": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--shots", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pace", type=float, default=0.0)
    a = ap.parse_args()
    print(json.dumps(walkthrough(a.url, a.shots, a.video, a.pace), indent=1))
