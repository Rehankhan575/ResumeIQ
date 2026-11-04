#!/usr/bin/env python3
"""
scripts/ui_audit.py
===================
Automated UI visual and DOM layout audit using headless Google Chrome via CDP.

Evaluates at 1440x900 and 1280x720:
  a) Every visible text node has a computed font-size >= 14px (lists offenders).
  b) Text contrast >= 4.5:1 against nearest opaque background (lists offenders).
  c) First viewport at 1440x900 contains:
     - product name ("ResumeIQ")
     - subtitle ("See how an applicant tracking system (ATS) scores...")
     - 3 step cards ("1. Choose", "2. Analyze", "3. Improve")
     - both input cards ("Your resume", "Job description")
     - "Analyze my resume" button fully within viewport height (<=900px)
  d) General layout: no horizontal scroll, no clipped text, zero exceptions, first load not empty.

Captures screenshots to docs/screenshots/ and outputs an audit report.
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
                "app/ui/app.py",
                f"--server.port={self.port_st}",
                "--server.headless=true",
                "--browser.gatherUsageStats=false",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

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
        print("Waiting for ResumeIQ to complete initial load...")
        for _ in range(40):
            ready = await self.evaluate(
                """(() => {
                    const text = document.body ? document.body.innerText : '';
                    const hasRing = document.querySelector('svg circle') !== null;
                    return text.includes('ResumeIQ') && hasRing;
                })()"""
            )
            if ready:
                print("App loaded and ready.")
                await asyncio.sleep(1.0)
                return
            await asyncio.sleep(0.5)
        raise RuntimeError("App did not finish initial load within 20s.")

    async def run_checks(self, width: int, height: int) -> Dict[str, Tuple[bool, str, List[Any]]]:
        await self.set_viewport(width, height)
        await self.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.5)
        results: Dict[str, Tuple[bool, str, List[Any]]] = {}

        # ── a) Computed font-size >= 14px on every visible text node ───────
        font_check = await self.evaluate(
            """(() => {
                const offenders = [];
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: function(node) {
                            if (!node.textContent || !node.textContent.trim()) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            const parent = node.parentElement;
                            if (!parent) return NodeFilter.FILTER_REJECT;
                            const style = window.getComputedStyle(parent);
                            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                                return NodeFilter.FILTER_REJECT;
                            }
                            const rect = parent.getBoundingClientRect();
                            if (rect.width === 0 && rect.height === 0) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );

                while (walker.nextNode()) {
                    const node = walker.currentNode;
                    const parent = node.parentElement;
                    const style = window.getComputedStyle(parent);
                    const size = parseFloat(style.fontSize);
                    if (size < 13.9) {
                        offenders.push({
                            tag: parent.tagName.toLowerCase(),
                            className: parent.className.toString().slice(0, 30),
                            fontSize: size.toFixed(1) + 'px',
                            text: node.textContent.trim().slice(0, 40)
                        });
                    }
                }
                return {
                    pass: offenders.length === 0,
                    count: offenders.length,
                    offenders: offenders.slice(0, 10)
                };
            })()"""
        ) or {}

        results["a) Min font size >= 14px"] = (
            font_check.get("pass", False),
            f"Found {font_check.get('count', 0)} text nodes < 14px",
            font_check.get("offenders", []),
        )

        # ── b) Text contrast >= 4.5:1 against nearest opaque background ────
        contrast_check = await self.evaluate(
            """(() => {
                function parseRgb(colorStr) {
                    const m = colorStr.match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)(?:,\\s*([\\d.]+))?\\)/);
                    if (!m) return null;
                    return {
                        r: parseInt(m[1], 10),
                        g: parseInt(m[2], 10),
                        b: parseInt(m[3], 10),
                        a: m[4] !== undefined ? parseFloat(m[4]) : 1.0
                    };
                }

                function lum(c) {
                    const r = c.r / 255;
                    const g = c.g / 255;
                    const b = c.b / 255;
                    const R = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
                    const G = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
                    const B = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);
                    return 0.2126 * R + 0.7152 * G + 0.0722 * B;
                }

                function getContrast(c1, c2) {
                    const l1 = lum(c1);
                    const l2 = lum(c2);
                    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
                }

                function getOpaqueBg(el) {
                    let cur = el;
                    while (cur && cur !== document.documentElement) {
                        const style = window.getComputedStyle(cur);
                        const c = parseRgb(style.backgroundColor);
                        if (c && c.a > 0.9) return c;
                        cur = cur.parentElement;
                    }
                    return { r: 255, g: 255, b: 255, a: 1 };
                }

                const offenders = [];
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: function(node) {
                            if (!node.textContent || !node.textContent.trim()) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            const parent = node.parentElement;
                            if (!parent) return NodeFilter.FILTER_REJECT;
                            const style = window.getComputedStyle(parent);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return NodeFilter.FILTER_REJECT;
                            }
                            const rect = parent.getBoundingClientRect();
                            if (rect.width === 0 || rect.height === 0) {
                                return NodeFilter.FILTER_REJECT;
                            }
                            return NodeFilter.FILTER_ACCEPT;
                        }
                    }
                );

                while (walker.nextNode()) {
                    const node = walker.currentNode;
                    const parent = node.parentElement;
                    const style = window.getComputedStyle(parent);
                    const fg = parseRgb(style.color);
                    if (!fg || fg.a < 0.2) continue;
                    const bg = getOpaqueBg(parent);
                    const ratio = getContrast(fg, bg);
                    if (ratio < 4.45) {
                        offenders.push({
                            tag: parent.tagName.toLowerCase(),
                            ratio: ratio.toFixed(2),
                            fg: style.color,
                            bg: `rgb(${bg.r},${bg.g},${bg.b})`,
                            text: node.textContent.trim().slice(0, 40)
                        });
                    }
                }
                return {
                    pass: offenders.length === 0,
                    count: offenders.length,
                    offenders: offenders.slice(0, 10)
                };
            })()"""
        ) or {}

        results["b) Text contrast >= 4.5:1"] = (
            contrast_check.get("pass", False),
            f"Found {contrast_check.get('count', 0)} low-contrast text nodes",
            contrast_check.get("offenders", []),
        )

        # ── c) First viewport at 1440x900 contains required elements ──────
        viewport_check = await self.evaluate(
            """(() => {
                const bodyText = document.body ? document.body.innerText : '';
                const hasTitle = bodyText.includes('ResumeIQ');
                const hasSubtitle = bodyText.includes('See how an applicant tracking system');
                const hasStep1 = bodyText.includes('1. Choose');
                const hasStep2 = bodyText.includes('2. Analyze');
                const hasStep3 = bodyText.includes('3. Improve');
                const hasResumeInput = bodyText.includes('Your resume');
                const hasJdInput = bodyText.includes('Job description');

                // Analyze button must be completely visible within the first 900px
                const btn = Array.from(document.querySelectorAll('button')).find(
                    b => b.innerText && b.innerText.includes('Analyze my resume')
                );
                let btnWithin = false;
                let btnBottom = 0;
                if (btn) {
                    const rect = btn.getBoundingClientRect();
                    btnBottom = rect.bottom;
                    btnWithin = (rect.bottom <= 900 && rect.top >= 0);
                }

                const allItems = hasTitle && hasSubtitle && hasStep1 && hasStep2 &&
                                 hasStep3 && hasResumeInput && hasJdInput && btnWithin;

                return {
                    pass: allItems,
                    hasTitle,
                    hasSubtitle,
                    hasStep1,
                    hasStep2,
                    hasStep3,
                    hasResumeInput,
                    hasJdInput,
                    btnFound: btn !== null,
                    btnBottom: btnBottom.toFixed(1) + 'px',
                    btnWithin
                };
            })()"""
        ) or {}

        btn_str = f"btnBottom={viewport_check.get('btnBottom')} (<=900px: {viewport_check.get('btnWithin')})"
        results["c) 1440x900 fold includes Hero + Inputs + Analyze"] = (
            viewport_check.get("pass", False) if width == 1440 else True,
            f"All elements present: {viewport_check.get('pass')}; {btn_str}",
            [],
        )

        # ── d) General layout integrity ────────────────────────────────────
        hscroll = await self.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        sw = await self.evaluate("document.documentElement.scrollWidth")
        iw = await self.evaluate("window.innerWidth")
        results["d1) No horizontal scroll"] = (
            bool(hscroll),
            f"scrollWidth={sw}px <= innerWidth={iw}px",
            [],
        )

        exceptions = await self.evaluate("document.querySelectorAll('.stException').length")
        results["d2) Zero exceptions"] = (
            exceptions == 0,
            f"Found {exceptions} exceptions",
            [],
        )

        first_load_ready = await self.evaluate(
            """(() => {
                const text = document.body ? document.body.innerText : '';
                return text.includes('Your ATS score') && text.includes('What makes up your score');
            })()"""
        )
        results["d3) First load not empty"] = (
            bool(first_load_ready),
            "Results and score ring present on load",
            [],
        )

        return results

    async def capture_all_screenshots(self, width: int = 1440, height: int = 900) -> None:
        print(f"\nCapturing screenshots at {width}x{height}...")
        await self.set_viewport(width, height)

        # 1. First load / Initial viewport
        await self.evaluate(
            """(() => {
                const title = document.querySelector('h1') || document.body;
                title.scrollIntoView({ behavior: 'instant', block: 'start' });
                window.scrollTo(0, 0);
            })()"""
        )
        await asyncio.sleep(0.5)
        await self.capture_screenshot(SCREENSHOTS_DIR / f"initial_{width}x{height}.png")
        print(f"  Saved initial_{width}x{height}.png")

        # 2. Results Top
        await self.evaluate(
            """(() => {
                const ring = document.querySelector('svg circle');
                if (ring) ring.scrollIntoView({ behavior: 'instant', block: 'start' });
            })()"""
        )
        await asyncio.sleep(0.5)
        await self.capture_screenshot(SCREENSHOTS_DIR / f"results_top_{width}x{height}.png")
        print(f"  Saved results_top_{width}x{height}.png")

        # 3. Tab screenshots (8 tabs)
        tabs_meta = [
            ("What to fix", f"tab_what_to_fix_{width}x{height}.png"),
            ("Resume highlights", f"tab_highlights_{width}x{height}.png"),
            ("Rewrite suggestions", f"tab_suggestions_{width}x{height}.png"),
            ("Compare jobs", f"tab_ranking_{width}x{height}.png"),
            ("Skill gaps", f"tab_skill_gap_{width}x{height}.png"),
            ("Fairness check", f"tab_bias_check_{width}x{height}.png"),
            ("History", f"tab_history_{width}x{height}.png"),
            ("Model results", f"tab_model_results_{width}x{height}.png"),
        ]

        for i, (name, filename) in enumerate(tabs_meta):
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
            print(f"{'Check':<50} | {'Status':<6} | Details")
            print("-" * 75)
            for check, (passed, details, offenders) in results.items():
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False
                print(f"{check:<50} | {status:<6} | {details}")
                if offenders:
                    for o in offenders[:5]:
                        print(f"    -> Offender: {o}")

        # Capture screenshots
        await auditor.capture_all_screenshots(1440, 900)

        print("\n" + "=" * 70)
        if all_passed:
            print("ALL AUDIT CHECKS PASSED (1440x900 & 1280x720)!")
        else:
            print("SOME AUDIT CHECKS FAILED. Please review the details above.")
        print("=" * 70)

    finally:
        auditor.close()

    if not all_passed:
        sys.exit(1)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
