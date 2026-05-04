"""
═══════════════════════════════════════════════════════════════════════
  rPPG Ultimate — Browser Heart Rate Monitor
  ──────────────────────────────────────────
  Run:  python run4.py
  Open: http://localhost:5050

  Browser captures the camera, sends frames to this server.
  This server runs the rPPG algorithm and returns live metrics.
  All display happens in the browser via dash.html.
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import socketserver
import time
import threading
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import rppg


# ─────────────────────────────── Config ────────────────────────────────

MODEL_ZOO = [
    "ME-flow",       # arXiv 2025 — low-latency state-space  ★ default
    "ME-chunk",      # arXiv 2025 — chunked state-space (offline)
    "RhythmMamba",   # AAAI  2025 — frequency-constrained Mamba
    "PhysMamba",     # CCBR  2024 — dual-branch Mamba
    "FacePhys",      #             — optimised state-space
    "EfficientPhys", # WACV  2023 — self-attention TSCAN variant
    "PhysFormer",    # CVPR  2022 — temporal-difference transformer
    "TSCAN",         # NeurIPS 2020
    "PhysNet",       # BMVC  2019
]
DEFAULT_MODEL    = "ME-flow"
HR_UPDATE_PERIOD = 1.0     # seconds between HR recomputations
HR_WINDOW        = 10      # seconds of signal used for HR
BVP_WINDOW       = 8       # seconds shown in browser waveform
HR_HIST_LEN      = 120     # HR history samples
PORT             = 5050

_DASH_HTML = Path(__file__).with_name("dash.html")


# ─────────────────────── Shared state (thread-safe) ────────────────────

# Rolling buffer of RGB frames received from the browser
_frame_buffer: deque[np.ndarray] = deque(maxlen=1800)   # 60 s @ 30 fps
_frame_times:  deque[float]      = deque(maxlen=1800)
_frame_lock  = threading.Lock()

_metrics: dict = {
    "hr":              None,
    "hrv":             None,
    "snr":             None,
    "bvp":             [],
    "hr_history":      [],
    "fps":             0.0,
    "model":           DEFAULT_MODEL,
    "frames_received": 0,
    "status":          "waiting",    # waiting | warming_up | live
}
_metrics_lock = threading.Lock()
_current_model_name = DEFAULT_MODEL


# ─────────────────────────────── Helpers ───────────────────────────────

def _to_float(v, _pref=("value", "rmssd", "sdnn", "mean", "median",
                         "db", "snr", "hr")):
    """Coerce a possibly nested rPPG result value to float."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for k in _pref:
            if k in v:
                out = _to_float(v[k])
                if out is not None:
                    return out
        for sub in v.values():
            out = _to_float(sub)
            if out is not None:
                return out
        return None
    if isinstance(v, (list, tuple)) and v:
        return _to_float(v[-1])
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _bvp_to_hr(signal: np.ndarray, fps: float) -> float | None:
    """FFT-based HR from a BVP signal (fallback when model does not return HR)."""
    if len(signal) < fps * 3:
        return None
    sig   = signal - signal.mean()
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fps)
    mag   = np.abs(np.fft.rfft(sig))
    mask  = (freqs >= 0.667) & (freqs <= 3.0)   # 40–180 BPM
    if not mask.any():
        return None
    return round(float(freqs[mask][np.argmax(mag[mask])]) * 60.0, 1)


# ─────────────────────────── Processing thread ─────────────────────────

def _processing_loop(model: rppg.Model) -> None:
    """Continuously process accumulated browser frames with rPPG."""
    hr_history: deque[float] = deque(maxlen=HR_HIST_LEN)
    last_run = 0.0

    while True:
        time.sleep(0.2)
        now = time.time()
        if now - last_run < HR_UPDATE_PERIOD:
            continue

        # Snapshot the frame buffer
        with _frame_lock:
            n_buf  = len(_frame_buffer)
            frames = list(_frame_buffer)
            times  = list(_frame_times)

        if n_buf < 15:
            with _metrics_lock:
                _metrics["status"] = "waiting"
            continue

        # Estimate actual FPS from timestamps
        fps = 30.0
        if len(times) >= 2 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        fps = float(np.clip(fps, 5.0, 60.0))

        try:
            # Use at most HR_WINDOW seconds of frames
            n_use  = min(len(frames), max(15, int(fps * HR_WINDOW)))
            tensor = np.array(frames[-n_use:], dtype=np.uint8)   # (T, H, W, 3) RGB

            result = model.process_video_tensor(tensor, fps=fps)

            hr = hrv = snr = None
            bvp_raw = None

            if isinstance(result, dict):
                hr      = _to_float(result.get("hr"))
                hrv     = _to_float(result.get("hrv") or result.get("rmssd"))
                snr     = _to_float(result.get("snr") or result.get("snr_db"))
                bvp_raw = (result.get("bvp") or result.get("signal")
                           or result.get("pred") or result.get("output"))
            elif isinstance(result, np.ndarray):
                bvp_raw = result.flatten()
            elif hasattr(result, "__iter__"):
                try:
                    bvp_raw = np.array(list(result), dtype=np.float32)
                except Exception:
                    pass

            bvp_list: list[float] = []
            if bvp_raw is not None:
                sig = np.asarray(bvp_raw, dtype=np.float32).flatten()
                if hr is None:
                    hr = _bvp_to_hr(sig, fps)
                keep = int(fps * BVP_WINDOW)
                bvp_list = [float(x) for x in sig[-keep:]]

            status = "warming_up"
            if hr is not None and 30.0 < hr < 220.0:
                hr_history.append(hr)
                status = "live"
                print(f"[rPPG] HR {hr:6.1f} BPM"
                      + (f"   HRV {hrv:.0f} ms" if hrv else "")
                      + (f"   SNR {snr:.1f} dB"  if snr else "")
                      + f"   ({fps:.1f} fps, {n_use} frames)")

            with _metrics_lock:
                _metrics.update({
                    "hr":         round(hr,  1) if hr  else None,
                    "hrv":        round(hrv, 1) if hrv else None,
                    "snr":        round(snr, 1) if snr else None,
                    "bvp":        bvp_list,
                    "hr_history": list(hr_history),
                    "fps":        round(fps, 1),
                    "model":      _current_model_name,
                    "status":     status,
                })

        except Exception as exc:
            traceback.print_exc()
            print(f"[rPPG] processing error: {exc}")

        last_run = now


# ─────────────────────────── HTTP server ───────────────────────────────

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle each request in a separate daemon thread."""
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):

    # ── routing ───────────────────────────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/metrics":
            self._serve_metrics()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/frame":
            self._receive_frame()
        else:
            self.send_error(404)

    # ── handlers ──────────────────────────────────────────────────────
    def _serve_html(self):
        try:
            body = _DASH_HTML.read_bytes()
        except FileNotFoundError:
            self.send_error(404, "dash.html not found — keep it next to run4.py")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_metrics(self):
        with _metrics_lock:
            data = dict(_metrics)
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _receive_frame(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            self.send_response(400)
            self.end_headers()
            return

        raw   = self.rfile.read(length)
        arr   = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # → BGR

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with _frame_lock:
                _frame_buffer.append(rgb)
                _frame_times.append(time.time())
            with _metrics_lock:
                _metrics["frames_received"] += 1

        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── helpers ───────────────────────────────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, *_):
        pass   # silence the default access log


# ─────────────────────────────── CLI ───────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="rPPG Ultimate — browser heart-rate monitor")
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help=f"rPPG model name (default: {DEFAULT_MODEL}). "
                        f"Available: {', '.join(MODEL_ZOO)}")
    p.add_argument("--port",    type=int, default=PORT,
                   help=f"HTTP port (default: {PORT})")
    p.add_argument("--log-dir", default="rppg_logs",
                   help="directory for saved snapshots / CSV sessions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _current_model_name = args.model

    print(f"[rPPG] Loading model '{args.model}'…")
    try:
        model = rppg.Model(args.model)
    except Exception as exc:
        print(f"[rPPG] '{args.model}' unavailable ({exc}); falling back to default.")
        model = rppg.Model()

    with _metrics_lock:
        _metrics["model"] = _current_model_name

    # Background rPPG processing thread
    proc = threading.Thread(target=_processing_loop, args=(model,), daemon=True)
    proc.start()

    # HTTP server — blocks here until Ctrl+C
    server = _ThreadedHTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"\n[rPPG]  Dashboard  →  http://localhost:{args.port}/")
    print(f"[rPPG]  Open the URL in your browser and allow camera access.")
    print(f"[rPPG]  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rPPG] Stopped.")