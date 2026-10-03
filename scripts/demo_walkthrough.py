"""Drive the REAL running application (FastAPI + built React UI) in a headless browser.

    python scripts/demo_walkthrough.py [--url http://127.0.0.1:8000] [--api-url URL] [--shots artifacts/screenshots]
                                       [--video artifacts/demo_video] [--pace 1.0]

Also the public-deployment smoke test: --url = the public UI, --api-url = the public API (when separate).

Flow: select case -> model run -> region selection -> Day 5 -> Reliability -> Why Flagged -> blind replay
(truth hidden) -> Reveal (ERA5 error, fingerprint) -> Reset -> Priority -> Trust -> direct navigation,
refresh and back/forward. Browser console errors and page errors fail the run. At each step the value shown in the UI is compared with the
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


def walkthrough(url: str, shots: Path | None = None, video: Path | None = None, pace: float = 0.0,
                api_url: str | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    timings, checks, requests, console_errors = {}, [], [], []
    api = api_url or url

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
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        t0 = time.perf_counter()
        page.goto(url + "/#/overview")
        page.wait_for_selector("[data-testid=n-alerts]")
        timings["open_app_and_first_case_ms"] = round(1000 * (time.perf_counter() - t0))
        cases = _api(api, "/api/demo/cases")["cases"]
        case = next(c for c in cases if c["selection"] == "random")
        assert page.input_value("[data-testid=case-select]") == case["case_id"]
        run = _api(api, f"/api/demo/cases/{case['case_id']}/run", "POST")
        # after a run the UI selects the top-priority region and ITS lead day (DemoApp.tsx)
        day = run["priority_queue"][0]["lead_day"] if run["priority_queue"] else 3
        check(f"overview alerts Day {day}", page.inner_text("[data-testid=n-alerts]"),
              str(run["lead_summary"][day - 1]["n_alerts"]))
        check("source badge", page.inner_text("[data-testid=source-badge]"), "SOURCE · ECMWF IFS / ERA5")
        check("served model", page.inner_text("[data-testid=served-model]").split(" ")[1], run["model"]["model_id"])
        assert page.locator("[data-testid^=cell-]").count() == len(run["regions"]), "map cells missing"
        # Day 5: overview alert count must follow the selected lead day
        page.click("[data-testid=day-5]")
        check("overview alerts Day 5", page.inner_text("[data-testid=n-alerts]"), str(run["lead_summary"][4]["n_alerts"]))
        page.click(f"[data-testid=day-{day}]")
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
        reg = _api(api, f"/api/demo/cases/{case['case_id']}/regions/{rid}")
        c = reg["trajectory"][day - 1]
        check("reliability P(bust)", page.inner_text("[data-testid=rel-p]"), pct(c["bust_probability"]))
        check("reliability B0 baseline", page.inner_text("[data-testid=rel-b0]"), pct(c["b0_probability"]))
        check("reliability risk level", page.inner_text("[data-testid=rel-risk]"), c["risk_level"])
        check("reliability evidence quality", page.inner_text("[data-testid=rel-evidence]"), c["evidence_quality"])
        page.click("[data-testid=day-5]")
        check("reliability P(bust) Day 5", page.inner_text("[data-testid=rel-p]"), pct(reg["trajectory"][4]["bust_probability"]))
        page.click(f"[data-testid=day-{day}]")
        check("reliability P(bust) back", page.inner_text("[data-testid=rel-p]"), pct(c["bust_probability"]))
        wait(5)
        shot(page, "02_reliability.png")

        t0 = time.perf_counter()
        page.click("[data-testid=go-evidence]")
        page.wait_for_selector("[data-testid=attribution]")
        page.wait_for_selector("[data-testid=an-within]")
        timings["why_flagged_ms"] = round(1000 * (time.perf_counter() - t0))
        ex = _api(api, f"/api/demo/cases/{case['case_id']}/regions/{rid}/explain?lead_day={day}")
        check("why P(bust)", page.inner_text("[data-testid=why-p]"), pct(ex["bust_probability"]))
        check("why B0 baseline", page.inner_text("[data-testid=why-b0]"), pct(ex["b0_probability"]))
        check("why risk level", page.inner_text("[data-testid=why-risk]"), ex["risk_level"])
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
        assert page.locator("[data-testid=actual-bust]").count() == 0 and page.locator("[data-testid=fingerprint]").count() == 0
        wait(4)
        shot(page, "04a_verification_blind.png")
        t0 = time.perf_counter()
        page.click("[data-testid=reveal]")
        page.wait_for_selector("[data-testid=memory-update]")
        timings["verification_reveal_ms"] = round(1000 * (time.perf_counter() - t0))
        rv = _api(api, f"/api/demo/cases/{case['case_id']}/regions/{rid}/reveal?lead_day={day}", "POST")
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
        page.click("[data-testid=reset]")
        page.wait_for_selector("[data-testid=reveal]")
        assert "BLIND" in page.inner_text("[data-testid=verif-mode]")
        assert page.locator("[data-testid=actual-bust]").count() == 0, "truth still visible after reset"
        checks.append({"check": "reset hides verification", "ui": "BLIND", "api": "BLIND", "ok": True})

        page.click("[data-testid=nav-priority]")
        page.wait_for_selector("[data-testid=priority-table]")
        wait(4)
        shot(page, "05_priority.png")
        page.click("[data-testid=nav-trust]")
        page.wait_for_selector("[data-testid=headline-table]")
        check("served model type", page.inner_text("[data-testid=model-type]").split(" ")[0], run["model"]["model_type"])
        check("served model id", page.inner_text("[data-testid=model-id]"), run["model"]["model_id"])
        assert "authenticated retrieval pending" in page.inner_text("[data-testid=ncmrwf-status]")
        page.wait_for_selector("[data-testid=headline-b2]")
        wait(5)
        shot(page, "06_model_trust.png")
        page.click("[data-testid=nav-overview]")
        page.wait_for_selector("[data-testid=n-alerts]")
        # direct navigation + hard refresh on every route, then browser back/forward
        for route, sel_ in [("overview", "n-alerts"), ("reliability", "rel-p"), ("evidence", "why-p"),
                            ("priority", "priority-table"), ("verification", "reveal"), ("trust", "model-id")]:
            page.goto(f"{url}/#/{route}")
            page.reload()
            page.wait_for_selector(f"[data-testid={sel_}]", timeout=60000)
        page.go_back()   # trust -> verification (the previous history entry)
        page.wait_for_selector("[data-testid=reveal]")
        page.go_forward()
        page.wait_for_selector("[data-testid=model-id]")
        checks.append({"check": "direct navigation, refresh, back/forward", "ui": "ok", "api": "ok", "ok": True})
        wait(3)
        ctx.close()
        browser.close()
    if console_errors:
        raise AssertionError(f"browser console errors: {console_errors[:5]}")
    return {"case_id": case["case_id"], "region_id": rid, "lead_day": day, "timings_ms": timings, "checks": checks,
            "console_errors": console_errors, "api_hosts": sorted({u.split('/api/')[0] for u in requests if '/api/' in u}),
            "reveal_requested_before_click": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--shots", type=Path)
    ap.add_argument("--video", type=Path)
    ap.add_argument("--pace", type=float, default=0.0)
    a = ap.parse_args()
    print(json.dumps(walkthrough(a.url, a.shots, a.video, a.pace, a.api_url), indent=1))
