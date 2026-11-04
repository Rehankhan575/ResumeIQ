#!/usr/bin/env python3
"""
scripts/capture_initial.py
==========================
One-shot capture of the initial load at 1440x900 via headless Chrome CDP.
Saves to docs/screenshots/initial_1440x900.png.
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
SCREENSHOTS_DIR = ROOT_DIR / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def run_capture():
    port_st = find_free_port()
    port_cdp = find_free_port()

    print(f"Starting Streamlit on port {port_st}...")
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    st_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/ui/app.py",
            f"--server.port={port_st}",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = f"http://localhost:{port_st}/_stcore/health"
    for _ in range(60):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    print("Streamlit ready.")
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
            f"http://localhost:{port_st}",
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
        st_proc.terminate()
        chrome_proc.terminate()
        raise RuntimeError("Failed to connect to Chrome CDP.")

    print(f"Connecting to Chrome CDP: {ws_url}")
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

        # Set 1440x900 viewport
        await send("Emulation.setDeviceMetricsOverride", {
            "width": 1440,
            "height": 900,
            "deviceScaleFactor": 1,
            "mobile": False,
        })

        # Wait for Streamlit to render initial view
        print("Waiting for Streamlit page to stabilize...")
        for i in range(60):
            res = await send("Runtime.evaluate", {
                "expression": """(() => {
                    const text = document.body ? document.body.innerText : '';
                    const btn = Array.from(document.querySelectorAll('button')).some(b => b.innerText && b.innerText.includes('Analyze my resume'));
                    return text.includes('ResumeIQ') && btn;
                })()""",
                "returnByValue": True,
            })
            val = res.get("result", {}).get("result", {}).get("value")
            if val:
                print(f"App rendered successfully at iteration {i}.")
                break
            await asyncio.sleep(0.5)

        # Extra settle time for fonts and animations
        await asyncio.sleep(2.0)

        # Scroll to top
        await send("Runtime.evaluate", {
            "expression": """(() => {
                window.scrollTo(0, 0);
                const c1 = document.querySelector('[data-testid="stAppViewContainer"]');
                if (c1) c1.scrollTo(0, 0);
                const c2 = document.querySelector('.main');
                if (c2) c2.scrollTo(0, 0);
            })()"""
        })
        await asyncio.sleep(0.5)

        # Capture screenshot
        out_path = SCREENSHOTS_DIR / "initial_1440x900.png"
        res = await send("Page.captureScreenshot", {"format": "png"})
        img_b64 = res.get("result", {}).get("data", "")
        if img_b64:
            out_path.write_bytes(base64.b64decode(img_b64))
            print(f"Captured: {out_path} ({out_path.stat().st_size} bytes)")
        else:
            print("Failed to capture screenshot data.")

    chrome_proc.terminate()
    st_proc.terminate()
    try:
        chrome_proc.wait(timeout=3)
        st_proc.wait(timeout=3)
    except Exception:
        chrome_proc.kill()
        st_proc.kill()


if __name__ == "__main__":
    asyncio.run(run_capture())
