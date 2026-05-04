"""
═══════════════════════════════════════════════════════════════════════
  rPPG Dashboard Server
  ─────────────────────
  Integrates the run4.py rPPG monitor with the HTML dashboard.

  • Runs the rppg model in a background thread (same pipeline as run4.py)
  • Streams live HR / HRV / SNR / BVP via Server-Sent Events  → /events
  • Streams the camera feed (with face box overlay) as MJPEG → /video_feed
  • Serves the dashboard                                     → /

  Usage:
    pip install flask rppg opencv-python numpy
    python server.py                       # uses ME-flow, camera 0
    python server.py --model PhysFormer    # pick a different model
    python server.py --port 8080           # different port

  Then open http://127.0.0.1:5000/ in your browser.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rppg
from flask import Flask, Response, send_file


# ─────────────────────────────── Config ────────────────────────────────
DEFAULT_MODEL    = "ME-flow"
HR_UPDATE_PERIOD = 1.0   # seconds between HR recomputations
HR_WINDOW        = 10    # seconds of signal used for HR
BVP_WINDOW       = 8     # seconds of BVP shown in the waveform
HR_HISTORY_LEN   = 120   # samples kept for min/avg/max stats


# ─────────────────────────── Shared state ──────────────────────────────
class State:
    """Thread-safe state shared between the rPPG worker and Flask routes."""
    def __init__(self):
        self.lock           = threading.Lock()
        self.hr             = None
        self.hrv            = None
        self.snr            = None
        self.bvp            = []           # last BVP_WINDOW seconds, downsampled
        self.hr_history     = deque(maxlen=HR_HISTORY_LEN)
        self.face_ok        = False
        self.last_face_time = 0.0
        self.last_hr_update = None
        self.fps            = 0.0
        self.model_name     = DEFAULT_MODEL
        self.latest_jpeg    = None
        self.running        = True
        self.error          = None

state = State()


# ─────────────────── Helpers (lifted from run4.py) ─────────────────────
def to_float(v, prefer_keys=("value", "rmssd", "sdnn", "mean", "median",
                             "db", "snr", "hr")):
    """Coerce float / int / numeric-string / dict-with-numeric → float."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in prefer_keys:
            if k in v:
                out = to_float(v[k])
                if out is not None:
                    return out
        for sub in v.values():
            out = to_float(sub)
            if out is not None:
                return out
        return None
    if isinstance(v, (list, tuple)) and v:
        return to_float(v[-1])
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ───────────────────────── rPPG worker thread ──────────────────────────
def rppg_worker(camera_idx: int, model_name: str):
    """Pulls frames from rppg, encodes JPEGs for MJPEG, computes metrics."""
    print(f"[rPPG] loading model '{model_name}'…")
    try:
        model = rppg.Model(model_name)
        state.model_name = model_name
    except Exception as exc:
        print(f"[rPPG] '{model_name}' unavailable ({exc}); using default.")
        model = rppg.Model()
        state.model_name = "default"

    frame_count   = 0
    start_time    = time.time()
    last_hr_time  = 0.0
    # Coral red in BGR for the face box overlay
    BOX_COLOR = (77, 77, 255)

    try:
        with model.video_capture(camera_idx):
            for frame, box in model.preview:
                if not state.running:
                    break
                if frame is None:
                    continue

                frame_count += 1
                now = time.time()

                # rppg yields RGB; cv2 wants BGR
                try:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except cv2.error:
                    continue

                # draw the face box (matches the dashboard's coral accent)
                if box is not None:
                    try:
                        (y1, y2), (x1, x2) = box
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), BOX_COLOR, 2)
                        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                            cv2.circle(bgr, (cx, cy), 4, BOX_COLOR, -1)
                    except Exception:
                        pass

                ok, buf = cv2.imencode(".jpg", bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 72])

                # commit frame + face/fps state
                with state.lock:
                    if ok:
                        state.latest_jpeg = buf.tobytes()
                    if box is not None:
                        state.face_ok = True
                        state.last_face_time = now
                    elif now - state.last_face_time > 1.5:
                        state.face_ok = False
                    elapsed = now - start_time
                    state.fps = frame_count / elapsed if elapsed > 0 else 0.0

                # ── metrics every HR_UPDATE_PERIOD seconds ───────────
                if now - last_hr_time > HR_UPDATE_PERIOD:
                    last_hr_time = now

                    # HR / HRV / SNR
                    try:
                        res = model.hr(start=-HR_WINDOW)
                        if res:
                            hr  = to_float(res.get("hr"))
                            hrv = to_float(res.get("hrv") or res.get("rmssd"))
                            snr = to_float(res.get("snr") or res.get("snr_db"))
                            with state.lock:
                                if hr is not None and hr > 0:
                                    state.hr = hr
                                    state.hr_history.append(hr)
                                    state.last_hr_update = now
                                if hrv is not None:
                                    state.hrv = hrv
                                if snr is not None:
                                    state.snr = snr
                    except Exception as exc:
                        with state.lock:
                            state.error = f"hr(): {exc!r}"

                    # BVP waveform (downsampled for the wire)
                    try:
                        bvp, _ = model.bvp(start=-BVP_WINDOW)
                        if bvp is not None and len(bvp) > 0:
                            arr = np.asarray(bvp, dtype=np.float32)
                            if len(arr) > 300:
                                idx = np.linspace(0, len(arr) - 1, 300).astype(int)
                                arr = arr[idx]
                            with state.lock:
                                state.bvp = arr.tolist()
                    except Exception:
                        pass

                    if state.hr is not None:
                        snr_part = (f"  SNR {state.snr:5.1f} dB"
                                    if state.snr is not None else "")
                        print(f"[rPPG] HR {state.hr:6.1f} BPM{snr_part}")

    except Exception as exc:
        traceback.print_exc()
        with state.lock:
            state.error = f"worker: {exc!r}"
    finally:
        print("[rPPG] worker stopped.")


# ─────────────────────────────── Flask ─────────────────────────────────
app = Flask(__name__)
HTML_PATH = Path(__file__).parent / "rppg_dashboard.html"


@app.route("/")
def index():
    return send_file(HTML_PATH)


@app.route("/events")
def events():
    """Server-Sent Events: live metrics @ 5 Hz."""
    def gen():
        while True:
            with state.lock:
                hist = list(state.hr_history)
                payload = {
                    "hr":      state.hr,
                    "hrv":     state.hrv,
                    "snr":     state.snr,
                    "bvp":     state.bvp,
                    "hr_min":  min(hist) if hist else None,
                    "hr_avg":  (sum(hist) / len(hist)) if hist else None,
                    "hr_max":  max(hist) if hist else None,
                    "face_ok": state.face_ok,
                    "fps":     round(state.fps, 1),
                    "model":   state.model_name,
                    "error":   state.error,
                }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.2)
    return Response(gen(), mimetype="text/event-stream")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the camera with face box overlay."""
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            with state.lock:
                jpeg = state.latest_jpeg
            if jpeg is not None:
                yield boundary + jpeg + b"\r\n"
            time.sleep(0.033)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ─────────────────────────────── CLI ───────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="rPPG dashboard server")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"rppg model name (default: {DEFAULT_MODEL})")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    t = threading.Thread(target=rppg_worker,
                         args=(args.camera, args.model),
                         daemon=True)
    t.start()

    print(f"[server] open http://{args.host}:{args.port}/")
    try:
        # threaded=True so SSE + MJPEG + index can serve concurrently
        app.run(host=args.host, port=args.port,
                threaded=True, debug=False, use_reloader=False)
    finally:
        state.running = False


if __name__ == "__main__":
    main()