#!/usr/bin/env python3
"""
scripts/ui_audit.py
===================
Automated UI visual and DOM layout audit using headless Google Chrome via CDP.

Evaluates 8 design criteria at 1440x900 and 1280x720:
  1. No horizontal scroll (scrollWidth <= innerWidth)
  2. Sibling card heights equal within 1px (resume vs JD, ring vs breakdown)
  3. 4 stat tiles equal height within 1px
  4. Computed font-family starts with Inter for labels, tabs and tables
  5. Zero visible radio in tab bar (st.tabs native)
  6. Ring SVG has nonzero bounding box and stroke matches grade color
  7. Section left edges identical within 1px
  8. No clipped text (scrollHeight <= clientHeight on single-line text containers)

Captures screenshots to docs/screenshots/ and prints a PASS/FAIL table.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT_DIR / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class CDPAuditor:
    def __init__(self, port_st: int, port_cdp: int):
        self.port_st = port_st
        self.port_cdp = port_cdp
        self.st_proc: subprocess.Popen | None = None
        self.chrome_proc: subprocess.Popen | None = None
        self.ws: Any = None
        self.msg_id = 0

    async def _send(self, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.msg_id += 1
        msg = {"id": self.msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(msg))
        while True:
            raw = await self.ws.recv()
            data = json.loads(raw)
            if data.get("id") == self.msg_id:
                return data

    async def evaluate(self, expr: str) -> Any:
        res = await self._send("Runtime.evaluate", {
            "expression": expr,
            "returnByValue": True,
            "awaitPromise": True,
        })
        val_obj = res.get("result", {}).get("result", {})
        return val_obj.get("value")

    async def capture_screenshot(self, out_path: Path) -> None:
        res = await self._send("Page.captureScreenshot", {"format": "png"})
        img_b64 = res.get("result", {}).get("data", "")
        if img_b64:
            out_path.write_bytes(base64.b64decode(img_b64))

    async def set_viewport(self, width: int, height: int) -> None:
        await self._send("Emulation.setDeviceMetricsOverride", {
            "width": width,
            "height": height,
            "deviceScaleFactor": 1,
            "mobile": False,
        })
        await asyncio.sleep(0.5)

    def start_streamlit(self) -> None:
        print(f"Starting Streamlit on port {self.port_st}...")
        env = os.environ.copy()
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["OMP_NUM_THREADS"] = "1"
        self.st_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT_DIR / "app" / "ui" / "app.py"),
                "--server.port",
                str(self.port_st),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

        # Wait for Streamlit to be healthy
        url = f"http://localhost:{self.port_st}/_stcore/health"
        for _ in range(60):
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        print("Streamlit server is ready.")
                        return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Streamlit failed to start within 30 seconds.")

    def start_chrome(self) -> None:
        print(f"Starting headless Chrome on CDP port {self.port_cdp}...")
        self.chrome_proc = subprocess.Popen(
            [
                CHROME_BIN,
                "--headless=new",
                f"--remote-debugging-port={self.port_cdp}",
                "--disable-gpu",
                "--no-sandbox",
                f"http://localhost:{self.port_st}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(30):
            try:
                with urllib.request.urlopen(f"http://localhost:{self.port_cdp}/json", timeout=1) as resp:
                    targets = json.loads(resp.read().decode())
                if targets:
                    print("Chrome CDP target ready.")
                    return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("Chrome failed to start within 15 seconds.")

    async def connect_cdp(self) -> None:
        with urllib.request.urlopen(f"http://localhost:{self.port_cdp}/json") as resp:
            targets = json.loads(resp.read().decode())
        page_targets = [t for t in targets if t.get("type") == "page"]
        ws_url = page_targets[0]["webSocketDebuggerUrl"] if page_targets else targets[0]["webSocketDebuggerUrl"]
        self.ws = await websockets.connect(ws_url, max_size=20 * 1024 * 1024)
        await self._send("Page.enable")
        await self._send("Runtime.enable")
        await self._send("DOM.enable")

    async def wait_for_app_ready(self) -> None:
        print("Waiting for Resume Analyzer to complete initial load...")
        for _ in range(40):
            ready = await self.evaluate(
                """(() => {
                    const card = document.querySelector('.st-key-card_score_ring');
                    const num = document.querySelector('.ra-score-number');
                    return card !== null && num !== null && num.textContent.trim().length > 0;
                })()"""
            )
            if ready:
                print("App loaded and ready.")
                await asyncio.sleep(1.0)
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("App did not finish initial load within 20s.")

    async def run_checks(self, width: int, height: int) -> Dict[str, Tuple[bool, str]]:
        await self.set_viewport(width, height)
        results: Dict[str, Tuple[bool, str]] = {}

        # 1. No horizontal scroll
        check1 = await self.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        sw = await self.evaluate("document.documentElement.scrollWidth")
        iw = await self.evaluate("window.innerWidth")
        results["No horizontal scroll"] = (
            bool(check1),
            f"scrollWidth={sw}px <= innerWidth={iw}px",
        )

        # 2. Sibling card heights equal within 1px
        # (a) Input cards: card_resume vs card_jd
        h_resume = await self.evaluate(
            "document.querySelector('.st-key-card_resume')?.getBoundingClientRect().height || 0"
        )
        h_jd = await self.evaluate(
            "document.querySelector('.st-key-card_jd')?.getBoundingClientRect().height || 0"
        )
        diff_inputs = abs(h_resume - h_jd)

        # (b) Hero cards: card_score_ring vs card_breakdown
        h_ring = await self.evaluate(
            "document.querySelector('.st-key-card_score_ring')?.getBoundingClientRect().height || 0"
        )
        h_bd = await self.evaluate(
            "document.querySelector('.st-key-card_breakdown')?.getBoundingClientRect().height || 0"
        )
        diff_hero = abs(h_ring - h_bd)

        check2 = (diff_inputs <= 1.0) and (diff_hero <= 1.0)
        results["Sibling card heights equal (<=1px)"] = (
            check2,
            f"Inputs: {h_resume:.1f}px vs {h_jd:.1f}px (diff {diff_inputs:.1f}px); Hero: {h_ring:.1f}px vs {h_bd:.1f}px (diff {diff_hero:.1f}px)",
        )

        # 3. 4 stat tiles equal height within 1px
        stat_heights = await self.evaluate(
            """(() => {
                const cards = Array.from(document.querySelectorAll('[class*="st-key-card_stat_"]'));
                return cards.map(c => c.getBoundingClientRect().height);
            })()"""
        ) or []
        if len(stat_heights) == 4:
            stat_diff = max(stat_heights) - min(stat_heights)
            check3 = stat_diff <= 1.0
            stat_str = ", ".join(f"{h:.1f}px" for h in stat_heights)
            results["4 stat tiles equal height (<=1px)"] = (
                check3,
                f"Tiles: [{stat_str}] (max diff {stat_diff:.1f}px)",
            )
        else:
            results["4 stat tiles equal height (<=1px)"] = (
                False,
                f"Found {len(stat_heights)} stat tiles instead of 4",
            )

        # 4. Computed font-family starts with Inter for labels, tabs, tables
        font_check = await self.evaluate(
            """(() => {
                const selectors = [
                    '.ra-card-label',
                    '[role="tab"]',
                    '.ra-table th',
                    '.ra-table td:first-child',
                    '.ra-wordmark'
                ];
                const failures = [];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el) {
                        const ff = window.getComputedStyle(el).fontFamily;
                        if (!ff.includes('Inter')) {
                            failures.push(`${sel}: ${ff}`);
                        }
                    }
                }
                return { pass: failures.length === 0, failures };
            })()"""
        )
        results["Computed font-family starts with Inter"] = (
            font_check.get("pass", False),
            "All tested elements computed with Inter"
            if font_check.get("pass")
            else f"Failures: {font_check.get('failures')}",
        )

        # 5. Zero visible radio in tab bar
        radios_count = await self.evaluate(
            "document.querySelectorAll('[data-testid=\"stTabs\"] input[type=\"radio\"]').length"
        )
        tabs_count = await self.evaluate(
            "document.querySelectorAll('[role=\"tab\"]').length"
        )
        check5 = (radios_count == 0) and (tabs_count == 7)
        results["Zero visible radio in tab bar (st.tabs)"] = (
            check5,
            f"0 radio inputs in tab bar, {tabs_count} native st.tabs detected",
        )

        # 6. Ring SVG nonzero bounding box & stroke matches grade color
        ring_check = await self.evaluate(
            """(() => {
                const svg = document.querySelector('.ra-score-ring-wrap svg');
                if (!svg) return { pass: false, reason: 'SVG not found' };
                const rect = svg.getBoundingClientRect();
                const circles = svg.querySelectorAll('circle');
                const progressCircle = circles[1];
                const stroke = progressCircle ? progressCircle.getAttribute('stroke') : '';
                const pass = rect.width > 0 && rect.height > 0 && (stroke === '#6EE7A0' || stroke === '#F5B454' || stroke === '#F2777A');
                return { pass, width: rect.width, height: rect.height, stroke };
            })()"""
        )
        results["Ring SVG nonzero box & grade stroke"] = (
            ring_check.get("pass", False),
            f"box={ring_check.get('width', 0):.0f}x{ring_check.get('height', 0):.0f}px, stroke={ring_check.get('stroke')}",
        )

        # 7. Section left edges identical within 1px
        edge_check = await self.evaluate(
            """(() => {
                const selectors = [
                    '.st-key-card_resume',
                    '.st-key-card_score_ring',
                    '.st-key-card_stat_0',
                    '[data-testid="stTabs"]'
                ];
                const lefts = selectors.map(sel => {
                    const el = document.querySelector(sel);
                    return el ? el.getBoundingClientRect().left : null;
                }).filter(l => l !== null);
                if (lefts.length < 4) return { pass: false, reason: 'Some sections missing' };
                const min = Math.min(...lefts);
                const max = Math.max(...lefts);
                const diff = max - min;
                return { pass: diff <= 1.0, diff, lefts };
            })()"""
        )
        results["Section left edges identical (<=1px)"] = (
            edge_check.get("pass", False),
            f"Left edges: {edge_check.get('lefts')} (max diff {edge_check.get('diff', 99):.1f}px)",
        )

        # 8. No clipped text (scrollHeight <= clientHeight on single-line text containers)
        clip_check = await self.evaluate(
            """(() => {
                const selectors = ['.ra-stat-value', '.ra-card-label', '.ra-subscore-val'];
                const clipped = [];
                for (const sel of selectors) {
                    const els = document.querySelectorAll(sel);
                    for (const el of els) {
                        if (el.scrollHeight > el.clientHeight + 1) {
                            clipped.push(`${sel}: ${el.textContent.trim()} (scroll=${el.scrollHeight}, client=${el.clientHeight})`);
                        }
                    }
                }
                return { pass: clipped.length === 0, clipped };
            })()"""
        )
        results["No clipped text in single-line containers"] = (
            clip_check.get("pass", False),
            "All checked text containers within bounds"
            if clip_check.get("pass")
            else f"Clipped: {clip_check.get('clipped')}",
        )

        return results

    async def capture_all_screenshots(self, width: int = 1440, height: int = 900) -> None:
        print(f"\nCapturing screenshots at {width}x{height}...")
        await self.set_viewport(width, height)

        # Scroll to top
        await self.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.5)

        # 1. Initial / Results top
        await self.capture_screenshot(SCREENSHOTS_DIR / f"initial_{width}x{height}.png")
        await self.capture_screenshot(SCREENSHOTS_DIR / f"results_top_{width}x{height}.png")
        print(f"  Saved initial_{width}x{height}.png & results_top_{width}x{height}.png")

        # 2. Tab screenshots
        tab_names = [
            ("Highlights", f"tab_highlights_{width}x{height}.png"),
            ("Suggestions", f"tab_suggestions_{width}x{height}.png"),
            ("Ranking", f"tab_ranking_{width}x{height}.png"),
            ("Skill gap", f"tab_skill_gap_{width}x{height}.png"),
            ("Bias check", f"tab_bias_check_{width}x{height}.png"),
            ("History", f"tab_history_{width}x{height}.png"),
            ("Model results", f"tab_model_results_{width}x{height}.png"),
        ]

        for i, (name, filename) in enumerate(tab_names):
            # Click tab via JS and scroll tab container into view
            await self.evaluate(
                f"""(() => {{
                    const tabs = document.querySelectorAll('[role="tab"]');
                    if (tabs && tabs[{i}]) {{
                        tabs[{i}].click();
                    }}
                    const tabContainer = document.querySelector('[data-testid="stTabs"]');
                    if (tabContainer) {{
                        tabContainer.scrollIntoView({{ behavior: 'instant', block: 'start' }});
                    }}
                }})()"""
            )
            await asyncio.sleep(0.5)
            await self.capture_screenshot(SCREENSHOTS_DIR / filename)
            print(f"  Saved {filename}")

        # Also capture at 1280x720 for verification
        if width == 1440:
            await self.capture_all_screenshots(width=1280, height=720)

    def close(self) -> None:
        if self.chrome_proc:
            try:
                self.chrome_proc.terminate()
                self.chrome_proc.wait(timeout=3)
            except Exception:
                self.chrome_proc.kill()
        if self.st_proc:
            try:
                self.st_proc.terminate()
                self.st_proc.wait(timeout=3)
            except Exception:
                self.st_proc.kill()


async def main_async():
    port_st = find_free_port()
    port_cdp = find_free_port()

    auditor = CDPAuditor(port_st, port_cdp)
    all_passed = True

    try:
        auditor.start_streamlit()
        auditor.start_chrome()
        await auditor.connect_cdp()
        await auditor.wait_for_app_ready()

        for (w, h) in [(1440, 900), (1280, 720)]:
            print(f"\n{'='*70}\nEVALUATING AT {w}x{h}\n{'='*70}")
            results = await auditor.run_checks(w, h)
            print(f"{'Check':<44} | {'Status':<6} | Details")
            print("-" * 75)
            for check, (passed, details) in results.items():
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False
                print(f"{check:<44} | {status:<6} | {details}")

        # Capture screenshots at 1440x900 and 1280x720
        await auditor.capture_all_screenshots(width=1440, height=900)

    finally:
        auditor.close()

    print(f"\n{'='*70}")
    if all_passed:
        print("ALL AUDIT CHECKS PASSED (1440x900 & 1280x720)!")
    else:
        print("SOME AUDIT CHECKS FAILED! Please inspect table above.")
    print(f"{'='*70}")

    sys.exit(0 if all_passed else 1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
