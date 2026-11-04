#!/usr/bin/env python3
"""
scripts/capture_react.py
========================
Automated capture of the React frontend at 1440px and 390px using headless Google Chrome via CDP.
Saves:
  - docs/screenshots/react_analyze_1440.png
  - docs/screenshots/react_results_1440.png
  - docs/screenshots/react_rank_1440.png
  - docs/screenshots/react_mobile_390.png
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
from typing import Any, Dict

import websockets

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
SCREENSHOTS_DIR = ROOT_DIR / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def run_capture():
    port_vite = find_free_port()
    port_cdp = find_free_port()

    print(f"Starting Vite dev server on port {port_vite}...")
    vite_proc = subprocess.Popen(
        ["npx", "vite", "--port", str(port_vite), "--host", "localhost"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = f"http://localhost:{port_vite}"
    for _ in range(30):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print("Vite dev server is ready.")
                    break
        except Exception:
            time.sleep(0.5)

    print(f"Starting headless Chrome on CDP port {port_cdp}...")
    chrome_proc = subprocess.Popen(
        [
            CHROME_BIN,
            "--headless=new",
            f"--remote-debugging-port={port_cdp}",
            "--disable-gpu",
            "--no-sandbox",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    ws_url = None
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://localhost:{port_cdp}/json", timeout=1) as resp:
                tabs = json.loads(resp.read().decode())
                for tab in tabs:
                    if tab.get("type") == "page" and "webSocketDebuggerUrl" in tab:
                        ws_url = tab["webSocketDebuggerUrl"]
                        break
            if ws_url:
                break
        except Exception:
            time.sleep(0.3)

    if not ws_url:
        vite_proc.terminate()
        chrome_proc.terminate()
        raise RuntimeError("Failed to connect to Chrome CDP.")

    print(f"Connected to Chrome CDP: {ws_url}")
    msg_id = 0
    async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
        async def send(method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
            nonlocal msg_id
            msg_id += 1
            msg = {"id": msg_id, "method": method, "params": params or {}}
            await ws.send(json.dumps(msg))
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("id") == msg_id:
                    return data

        await send("Page.enable")
        await send("DOM.enable")

        # 1. Capture 1440px Desktop: Analyze Screen
        print("Capturing desktop 1440x900 (Analyze)...")
        await send("Emulation.setDeviceMetricsOverride", {
            "width": 1440,
            "height": 900,
            "deviceScaleFactor": 1,
            "mobile": False,
        })
        await asyncio.sleep(1.5)

        res_1440 = await send("Page.captureScreenshot", {"format": "png"})
        b64_1440 = res_1440.get("result", {}).get("data", "")
        out_1440 = SCREENSHOTS_DIR / "react_analyze_1440.png"
        out_1440.write_bytes(base64.b64decode(b64_1440))
        print(f"Saved {out_1440} ({out_1440.stat().st_size} bytes)")

        # 2. Switch to Results View and Capture 1440px Desktop
        print("Switching to Results view and capturing 1440x900...")
        await send("Runtime.evaluate", {
            "expression": """(() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const resultsBtn = btns.find(b => b.innerText && b.innerText.includes('Results'));
                if (resultsBtn) resultsBtn.click();
            })()"""
        })
        await asyncio.sleep(1.0)
        res_res = await send("Page.captureScreenshot", {"format": "png"})
        b64_res = res_res.get("result", {}).get("data", "")
        out_res = SCREENSHOTS_DIR / "react_results_1440.png"
        out_res.write_bytes(base64.b64decode(b64_res))
        print(f"Saved {out_res} ({out_res.stat().st_size} bytes)")

        # 3. Switch to Rank View and Capture 1440px Desktop
        print("Switching to Rank view and capturing 1440x900...")
        await send("Runtime.evaluate", {
            "expression": """(() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const rankBtn = btns.find(b => b.innerText && b.innerText.includes('Rank jobs'));
                if (rankBtn) rankBtn.click();
            })()"""
        })
        await asyncio.sleep(1.0)
        res_rank = await send("Page.captureScreenshot", {"format": "png"})
        b64_rank = res_rank.get("result", {}).get("data", "")
        out_rank = SCREENSHOTS_DIR / "react_rank_1440.png"
        out_rank.write_bytes(base64.b64decode(b64_rank))
        print(f"Saved {out_rank} ({out_rank.stat().st_size} bytes)")

        # 4. Switch back to Analyze and Capture 390px Mobile
        print("Capturing mobile 390x844 (Analyze)...")
        await send("Runtime.evaluate", {
            "expression": """(() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const analyzeBtn = btns.find(b => b.innerText && b.innerText.includes('Analyze'));
                if (analyzeBtn) analyzeBtn.click();
            })()"""
        })
        await send("Emulation.setDeviceMetricsOverride", {
            "width": 390,
            "height": 844,
            "deviceScaleFactor": 2,
            "mobile": True,
        })
        await asyncio.sleep(1.0)

        res_390 = await send("Page.captureScreenshot", {"format": "png"})
        b64_390 = res_390.get("result", {}).get("data", "")
        out_390 = SCREENSHOTS_DIR / "react_mobile_390.png"
        out_390.write_bytes(base64.b64decode(b64_390))
        print(f"Saved {out_390} ({out_390.stat().st_size} bytes)")

    chrome_proc.terminate()
    vite_proc.terminate()
    try:
        chrome_proc.wait(timeout=2)
        vite_proc.wait(timeout=2)
    except Exception:
        chrome_proc.kill()
        vite_proc.kill()


if __name__ == "__main__":
    asyncio.run(run_capture())
