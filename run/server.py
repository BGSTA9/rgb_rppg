"""
rPPG WebSocket bridge — streams run4.py metrics to the Next.js dashboard.

Run:  python server.py
      (in a separate terminal from the dashboard)
"""
from __future__ import annotations

import asyncio
import csv
import json
import threading
import time
from datetime import datetime
from typing import Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from run4 import MODEL_ZOO, RPPGMonitor

# ── Shared state ──────────────────────────────────────────────────────────────

monitor = RPPGMonitor()

# Latest BVP snapshot written by the camera thread, read by asyncio.
# List assignment is atomic under the GIL.
_bvp_snapshot: list[float] = []

clients: Set[WebSocket] = set()

# ── Camera thread (no OpenCV window) ─────────────────────────────────────────

def _camera_thread() -> None:
    global _bvp_snapshot
    with monitor.model.video_capture(monitor.camera):
        for frame, box in monitor.model.preview:
            if frame is None:
                continue
            monitor.frame_count += 1
            if box is not None:
                monitor.last_face_seen = time.time()
            if not monitor.paused:
                now = time.time()
                if now - monitor.last_hr_time > 1.0:
                    monitor.update_metrics()
                    monitor.last_hr_time = now
            bvp = monitor.bvp_window()
            if bvp is not None and len(bvp):
                _bvp_snapshot = [float(v) for v in bvp[-300:]]

# ── WebSocket broadcast loop (10 Hz) ─────────────────────────────────────────

async def _broadcast_loop() -> None:
    while True:
        await asyncio.sleep(0.1)
        if not clients:
            continue
        elapsed = time.time() - monitor.start_time
        msg = json.dumps({
            "hr":             monitor.current_hr,
            "hrv":            monitor.current_hrv,
            "snr":            monitor.current_snr,
            "fps":            monitor.frame_count / max(elapsed, 0.001),
            "face_detected":  bool(
                monitor.last_face_seen
                and time.time() - monitor.last_face_seen < 1.5
            ),
            "signal_quality": (
                "live"    if monitor.current_hr else
                "warming" if elapsed < 10       else
                "stale"
            ),
            "model_name":  monitor.model_name,
            "hr_history":  list(monitor.hr_history),
            "bvp":         _bvp_snapshot,
        })
        dead: Set[WebSocket] = set()
        for ws in list(clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        clients -= dead

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="rPPG Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    threading.Thread(target=_camera_thread, daemon=True).start()
    asyncio.create_task(_broadcast_loop())


@app.websocket("/ws")
async def _ws(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; client sends nothing
    except WebSocketDisconnect:
        clients.discard(ws)


@app.post("/control/{action}")
async def _control(action: str, name: str = "") -> dict:
    if action == "pause":
        monitor.paused = True
    elif action == "resume":
        monitor.paused = False
    elif action == "reset":
        monitor.reset()
    elif action == "model" and name in MODEL_ZOO:
        monitor.model_name = name
        monitor.model = monitor._load_model(name)
        monitor.reset(quiet=True)
    elif action == "save":
        _save_csv()
    return {"ok": True}


def _save_csv() -> None:
    try:
        bvp, t = monitor.model.bvp()
        path = monitor.log_dir / f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_seconds", "bvp"])
            for ti, bi in zip(t, bvp):
                w.writerow([f"{ti:.4f}", f"{bi:.6f}"])
        print(f"[rPPG] saved → {path}")
    except Exception as exc:
        print(f"[rPPG] save failed: {exc}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
