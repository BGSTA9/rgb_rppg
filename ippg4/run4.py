from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import rppg
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ─────────────────────────────── Config ────────────────────────────────

MODEL_ZOO = [
    "ME-flow", "ME-chunk", "RhythmMamba", "PhysMamba",
    "FacePhys", "EfficientPhys", "PhysFormer", "TSCAN", "PhysNet",
]
DEFAULT_MODEL    = "ME-flow"
HR_UPDATE_PERIOD = 1.0
HR_WINDOW        = 10
BVP_WINDOW       = 8
HR_HIST_LEN      = 120

# BGR color palette
C_PANEL  = (38, 38, 44)
C_ACCENT = (96, 232, 96)
C_TEXT   = (235, 235, 235)
C_DIM    = (140, 140, 150)
C_WARN   = (60, 200, 255)
C_BAD    = (60, 60, 240)
C_BVP    = (255, 180, 80)


# ──────────────────────────── Streaming ─────────────────────────────────
#
# Patches over the original streaming layer:
#   1. ThreadingHTTPServer instead of HTTPServer — the long-lived
#      /video_feed connection no longer blocks every other request,
#      so reloads of /, /health, etc. work even with the stream open.
#   2. no-store cache header on /, so the browser never serves a stale
#      dash.html from a previous run when run4.py is offline.
#   3. CORS allow-origin: * — lets the dashboard work even if dash.html
#      is opened from disk (file:// origin) instead of localhost:5050.
#   4. /health JSON endpoint — fast alive-check that doesn't depend on
#      the camera or model loading.
#   5. Placeholder JPEG ("warming up…") served whenever no real frame
#      has been encoded yet, so the MJPEG stream is never empty (some
#      browsers fire onerror on an empty multipart response).

_latest_jpeg: bytes | None = None
_frame_lock   = threading.Lock()
_encode_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
_DASH_HTML = Path(__file__).with_name("dash.html")

# Process-wide stats exposed at /health
_state = {
    "started_at":       time.time(),
    "frames_pushed":    0,
    "frames_encoded":   0,
    "first_frame_at":   None,
    "last_frame_at":    None,
}
_state_lock = threading.Lock()


def _make_placeholder_jpeg() -> bytes:
    """A 'warming up' frame served while the camera/model spin up."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 48)
    cv2.putText(img, "rPPG Ultimate", (170, 165),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, C_ACCENT, 2, cv2.LINE_AA)
    cv2.putText(img, "warming up...", (240, 205),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, "loading camera + model", (185, 235),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, C_DIM, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


_PLACEHOLDER_JPEG = _make_placeholder_jpeg()


class _EncoderWorker(threading.Thread):
    """JPEG encoding off the main thread (preserves original optimization 4)."""
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        global _latest_jpeg
        while True:
            frame = _encode_queue.get()
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            if not ok:
                continue
            jpeg = buf.tobytes()
            with _frame_lock:
                _latest_jpeg = jpeg
            with _state_lock:
                now = time.time()
                _state["frames_encoded"] += 1
                _state["last_frame_at"]   = now
                if _state["first_frame_at"] is None:
                    _state["first_frame_at"] = now


def _push_frame(bgr_frame: np.ndarray) -> None:
    """Non-blocking enqueue (preserves original optimization 4)."""
    with _state_lock:
        _state["frames_pushed"] += 1
    try:
        _encode_queue.put_nowait(bgr_frame)
    except queue.Full:
        try:
            _encode_queue.get_nowait()   # discard stale frame
        except queue.Empty:
            pass
        _encode_queue.put_nowait(bgr_frame)


_CORS = (
    ("Access-Control-Allow-Origin",  "*"),
    ("Access-Control-Allow-Methods", "GET, OPTIONS"),
    ("Access-Control-Allow-Headers", "*"),
)


class _StreamHandler(BaseHTTPRequestHandler):
    def _cors(self):
        for k, v in _CORS:
            self.send_header(k, v)

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/video_feed":
            self._serve_mjpeg()
        elif self.path == "/health":
            self._serve_health()
        elif self.path == "/snapshot.jpg":
            self._serve_snapshot()
        else:
            self.send_response(404); self._cors(); self.end_headers()

    def _serve_html(self):
        try:
            body = _DASH_HTML.read_bytes()
        except FileNotFoundError:
            self.send_response(404); self._cors(); self.end_headers()
            self.wfile.write(b"dash.html not found next to run4.py")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_health(self):
        with _state_lock, _frame_lock:
            payload = {
                **_state,
                "uptime_s":    round(time.time() - _state["started_at"], 1),
                "has_jpeg":    _latest_jpeg is not None,
                "queue_depth": _encode_queue.qsize(),
            }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_snapshot(self):
        with _frame_lock:
            jpeg = _latest_jpeg or _PLACEHOLDER_JPEG
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(jpeg)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=rpframe"
        )
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma",  "no-cache")
        self.send_header("Expires", "0")
        self._cors()
        self.end_headers()

        try:
            while True:
                with _frame_lock:
                    jpeg = _latest_jpeg or _PLACEHOLDER_JPEG
                try:
                    self.wfile.write(
                        b"--rpframe\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
                time.sleep(0.033)
        except Exception:
            pass

    def log_message(self, *_):
        pass


def _start_stream_server(port: int = 5050) -> None:
    _EncoderWorker().start()
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), _StreamHandler)
    except OSError as e:
        print(f"[rPPG] could not bind port {port}: {e}")
        print(f"[rPPG] another process is using port {port}. "
              f"Close it or pass --port to use a different port.")
        raise
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[rPPG] Dashboard  →  http://localhost:{port}/")
    print(f"[rPPG] Video feed →  http://localhost:{port}/video_feed")
    print(f"[rPPG] Health     →  http://localhost:{port}/health")


# ──────────────────────── Optimization 1: panel batching ────────────────

class _PanelBatch:
    """
    Collect all panel rectangles during a HUD draw, then apply a SINGLE
    weighted overlay at the end.

    Original cost: N × O(W·H)  (one clone + addWeighted per panel)
    Optimised cost: 1 × O(W·H)  (one clone + addWeighted total)
    """
    __slots__ = ("_rects", "_color", "_alpha")

    def __init__(self, color=C_PANEL, alpha=0.55):
        self._rects: list[tuple] = []
        self._color = color
        self._alpha = alpha

    def add(self, p1: tuple[int, int], p2: tuple[int, int]) -> None:
        self._rects.append((p1, p2))

    def flush(self, img: np.ndarray) -> None:
        if not self._rects:
            return
        overlay = img.copy()                     # one clone for all panels
        for p1, p2 in self._rects:
            cv2.rectangle(overlay, p1, p2, self._color, -1)
        cv2.addWeighted(overlay, self._alpha, img, 1 - self._alpha, 0, img)
        self._rects.clear()


# ──────────────── Optimization 5: polyline / x-axis caching ─────────────

class _XAxisCache:
    def __init__(self):
        self._cache: dict[tuple, np.ndarray] = {}

    def get(self, n: int, x0: int, w: int) -> np.ndarray:
        key = (n, x0, w)
        arr = self._cache.get(key)
        if arr is None:
            arr = np.linspace(x0 + 4, x0 + w - 4, n).astype(np.int32)
            self._cache[key] = arr
        return arr


_xaxis = _XAxisCache()


class _PolylineCache:
    def __init__(self):
        self._key:  int | None        = None
        self._pts:  np.ndarray | None = None

    @staticmethod
    def _fast_hash(arr: np.ndarray, geom: tuple) -> int:
        return hash(arr.tobytes()) ^ hash(geom)

    def get_or_compute(
        self,
        signal: np.ndarray,
        x0: int, y0: int, w: int, h: int
    ) -> np.ndarray:
        geom = (x0, y0, w, h)
        key  = self._fast_hash(signal, geom)
        if key == self._key and self._pts is not None:
            return self._pts
        s   = signal.astype(np.float32)
        s  -= s.mean()
        rng = float(np.max(np.abs(s))) or 1.0
        s  /= rng
        n   = len(s)
        xs  = _xaxis.get(n, x0, w)
        ys  = (y0 + h / 2 - s * (h / 2 - 8)).astype(np.int32)
        pts = np.column_stack((xs, ys))
        self._key = key
        self._pts = pts
        return pts


_bvp_poly  = _PolylineCache()
_hist_poly = _PolylineCache()


# ──────────────── Optimization 7: hr_to_color cached ─────────────────────

@lru_cache(maxsize=512)
def _hr_color_cached(hr_int: int | None) -> tuple:
    if hr_int is None:
        return C_DIM
    if 50 <= hr_int <= 110:
        return C_ACCENT
    if 40 <= hr_int < 50 or 110 < hr_int <= 140:
        return C_WARN
    return C_BAD


def hr_to_color(hr: float | None) -> tuple:
    return _hr_color_cached(None if hr is None else int(hr))


# ─────────────────────────── Drawing helpers ───────────────────────────

def put_label(img, text, org, scale=0.55, color=C_TEXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thick, cv2.LINE_AA)


def draw_bvp(img, signal, origin, size, batch: _PanelBatch, color=C_BVP):
    x0, y0 = origin
    w, h   = size
    batch.add((x0 - 6, y0 - 6), (x0 + w + 6, y0 + h + 6))
    put_label(img, "BVP (filtered)", (x0 + 8, y0 + 16), 0.5, C_DIM)
    if signal is None or len(signal) < 2:
        put_label(img, "waiting for signal…", (x0 + 8, y0 + h // 2), 0.5, C_DIM)
        return
    pts = _bvp_poly.get_or_compute(signal, x0, y0, w, h)
    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


def draw_hr_history(img, history: list, origin, size, batch: _PanelBatch):
    x0, y0 = origin
    w, h   = size
    batch.add((x0 - 6, y0 - 6), (x0 + w + 6, y0 + h + 6))
    put_label(img, "HR history (BPM)", (x0 + 8, y0 + 16), 0.5, C_DIM)
    if len(history) < 2:
        return
    hr   = np.asarray(history, dtype=np.float32)
    lo   = max(40.0,  float(hr.min()) - 5.0)
    hi   = min(180.0, float(hr.max()) + 5.0)
    span = max(hi - lo, 1.0)
    ys_raw = (y0 + h - (hr - lo) / span * (h - 24) - 4).astype(np.int32)
    n  = len(hr)
    xs = _xaxis.get(n, x0, w)
    pts = np.column_stack((xs, ys_raw))
    cv2.polylines(img, [pts], False, hr_to_color(float(hr[-1])), 2, cv2.LINE_AA)
    put_label(img, f"{hi:.0f}", (x0 + w - 30, y0 + 28), 0.4, C_DIM)
    put_label(img, f"{lo:.0f}", (x0 + w - 30, y0 + h - 6), 0.4, C_DIM)


# ─────────────── Optimization 3: background metrics worker ───────────────

class _MetricsWorker(threading.Thread):
    def __init__(self, model, period: float, bvp_window_secs: int):
        super().__init__(daemon=True)
        self._model      = model
        self._period     = period
        self._bvp_secs   = bvp_window_secs
        self._lock       = threading.Lock()

        self.hr:       float | None      = None
        self.hrv:      float | None      = None
        self.snr:      float | None      = None
        self.bvp:      np.ndarray | None = None
        self.updated:  float | None      = None
        self.last_err: str | None        = None

    @staticmethod
    def _to_float(root) -> float | None:
        PREFER = ("value", "rmssd", "sdnn", "mean", "median",
                  "db", "snr", "hr")
        worklist = deque([root])
        while worklist:
            v = worklist.popleft()
            if v is None or isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    continue
            if isinstance(v, dict):
                for k in PREFER:
                    if k in v:
                        worklist.appendleft(v[k])
                        break
                else:
                    worklist.extend(v.values())
                continue
            if isinstance(v, (list, tuple)) and v:
                worklist.append(v[-1])
        return None

    def run(self):
        while True:
            t0 = time.monotonic()
            self._tick()
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._period - elapsed))

    def _tick(self):
        try:
            res = self._model.hr(start=-HR_WINDOW)
        except Exception as exc:
            with self._lock:
                self.last_err = f"hr(): {exc!r}"
            return

        if res:
            hr  = self._to_float(res.get("hr"))
            hrv = self._to_float(res.get("hrv") or res.get("rmssd"))
            snr = self._to_float(res.get("snr") or res.get("snr_db"))
            with self._lock:
                if hr and hr > 0:
                    self.hr      = hr
                    self.updated = time.time()
                if hrv is not None:
                    self.hrv = hrv
                if snr is not None:
                    self.snr = snr

        try:
            bvp, _ = self._model.bvp(start=-self._bvp_secs)
            with self._lock:
                self.bvp = np.asarray(bvp, dtype=np.float32) if bvp is not None else None
        except Exception:
            pass

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "hr": self.hr, "hrv": self.hrv, "snr": self.snr,
                "bvp": self.bvp, "updated": self.updated,
                "last_err": self.last_err,
            }

    def reset(self):
        with self._lock:
            self.hr = self.hrv = self.snr = self.bvp = self.updated = None


# ─────────────────────────── Main monitor ──────────────────────────────

class RPPGMonitor:
    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 camera: int = 0,
                 log_dir: str = "rppg_logs",
                 port: int = 5050,
                 headless: bool = False):
        self.model_name = model_name
        self.model      = self._load_model(model_name)
        self.camera     = camera
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.port       = port
        self.headless   = headless

        self.hr_history     = deque(maxlen=HR_HIST_LEN)
        self.last_face_seen: float | None = None
        self.frame_count    = 0
        self.start_time     = time.time()

        self.show_bvp  = True
        self.show_hist = True
        self.show_box  = True
        self.paused    = False

        self._metrics = _MetricsWorker(self.model, HR_UPDATE_PERIOD, BVP_WINDOW)
        self._metrics.start()

        self._snap: dict = {}

    @staticmethod
    def _load_model(name: str):
        print(f"[rPPG] loading model '{name}'…")
        try:
            return rppg.Model(name)
        except Exception as exc:
            print(f"[rPPG] '{name}' unavailable ({exc}); using default.")
            return rppg.Model()

    def cycle_model(self):
        idx = (MODEL_ZOO.index(self.model_name) + 1) % len(MODEL_ZOO) \
            if self.model_name in MODEL_ZOO else 0
        self.model_name = MODEL_ZOO[idx]
        self.model = self._load_model(self.model_name)
        self._metrics._model = self.model
        self.reset(quiet=True)
        print(f"[rPPG] switched to {self.model_name}")

    def process_video_tensor(self, tensor: np.ndarray, fps: float = 30.0):
        return self.model.process_video_tensor(tensor, fps=fps)

    def process_faces_tensor(self, tensor: np.ndarray, fps: float = 30.0):
        return self.model.process_faces_tensor(tensor, fps=fps)

    def draw_hud(self, frame: np.ndarray, box) -> None:
        h, w = frame.shape[:2]
        now  = time.time()
        snap = self._snap

        if box is not None:
            self.last_face_seen = now

        face_age = (now - self.last_face_seen) if self.last_face_seen else None
        hr_age   = (now - snap["updated"])    if snap.get("updated")    else None
        face_ok  = face_age is not None and face_age < 1.5
        hr_fresh = hr_age   is not None and hr_age   < HR_UPDATE_PERIOD * 3

        batch = _PanelBatch()

        try:
            batch.add((10, 10), (300, 122))
            color    = hr_to_color(snap.get("hr")) if hr_fresh else C_DIM
            hr_text  = f"{snap['hr']:.1f}" if snap.get("hr")  else "--"
            snr_text = f"{snap['snr']:.1f}" if snap.get("snr") is not None else "--"
            hrv_text = f"{snap['hrv']:.0f}" if snap.get("hrv") else "--"
            put_label(frame, "Heart Rate", (24, 36), 0.55, C_DIM)
            put_label(frame, hr_text, (24, 82), 1.5, color, 2)
            put_label(frame, "BPM", (160, 82), 0.7, C_DIM)
            sub = f"SNR {snr_text} dB    HRV {hrv_text} ms"
            if hr_age is not None and not hr_fresh:
                sub += f"   (stale {hr_age:.0f}s)"
            put_label(frame, sub, (24, 104), 0.5, C_DIM)
        except Exception as exc:
            print(f"[rPPG] status panel: {exc}")

        try:
            face_msg = "FACE OK" if face_ok else (
                f"NO FACE ({face_age:.0f}s)" if face_age else "WAITING FOR FACE")
            face_col = C_ACCENT if face_ok else C_BAD
            sig_msg, sig_col = (
                ("SIGNAL OK", C_ACCENT) if hr_fresh
                else ("WARMING UP", C_WARN) if hr_age is None
                else (f"SIGNAL STALE ({hr_age:.0f}s)", C_WARN)
            )
            x0 = 320
            batch.add((x0, 10), (x0 + 280, 56))
            put_label(frame, face_msg, (x0 + 12, 32), 0.55, face_col)
            put_label(frame, sig_msg,  (x0 + 12, 50), 0.5,  sig_col)
        except Exception as exc:
            print(f"[rPPG] diagnostics: {exc}")

        try:
            elapsed = now - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            tag = f"{self.model_name}   {fps:4.1f} fps"
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            batch.add((w - tw - 28, 10), (w - 10, 44))
            put_label(frame, tag, (w - tw - 18, 32), 0.55, C_TEXT)
        except Exception as exc:
            print(f"[rPPG] fps tag: {exc}")

        try:
            if self.show_box:
                if box is not None:
                    (y1, y2), (x1, x2) = box
                    color = hr_to_color(snap.get("hr")) if hr_fresh else C_WARN
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                        cv2.circle(frame, (cx, cy), 3, color, -1)
                    label = (f"HR {snap['hr']:.1f}" if snap.get("hr") else "tracking…")
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 14, y1), color, -1)
                    put_label(frame, label, (x1 + 7, y1 - 7), 0.6, (0, 0, 0))
                else:
                    msg = "no face detected — center your face & hold still"
                    (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                    cx, cy = (w - mw) // 2, h // 2
                    batch.add((cx - 16, cy - mh - 10), (cx + mw + 16, cy + 14))
                    put_label(frame, msg, (cx, cy), 0.6, C_BAD)
        except Exception as exc:
            print(f"[rPPG] face box: {exc}")

        try:
            if self.show_bvp:
                draw_bvp(frame, snap.get("bvp"),
                         origin=(20, h - 120), size=(w // 2 - 40, 100),
                         batch=batch)
        except Exception as exc:
            print(f"[rPPG] bvp plot: {exc}")

        try:
            if self.show_hist and len(self.hr_history) >= 2:
                draw_hr_history(frame, list(self.hr_history),
                                origin=(w // 2 + 20, h - 120),
                                size=(w // 2 - 40, 100),
                                batch=batch)
        except Exception as exc:
            print(f"[rPPG] hr history: {exc}")

        batch.flush(frame)

        try:
            status = "PAUSED" if self.paused else "LIVE"
            line = (f"[{status}]  q quit · s save · p pause · r reset · "
                    f"g graph · h history · f face · m model")
            put_label(frame, line, (16, h - 14), 0.42, C_DIM)
            if snap.get("last_err"):
                put_label(frame, f"last error: {snap['last_err'][:90]}",
                          (16, h - 32), 0.42, C_BAD)
        except Exception as exc:
            print(f"[rPPG] footer: {exc}")

    def save_session(self, frame: np.ndarray):
        ts       = datetime.now().strftime("%Y%m%d-%H%M%S")
        png      = self.log_dir / f"snapshot-{ts}.png"
        csv_path = self.log_dir / f"session-{ts}.csv"
        cv2.imwrite(str(png), frame)
        try:
            bvp, t = self.model.bvp()
            with open(csv_path, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["t_seconds", "bvp"])
                for ti, bi in zip(t, bvp):
                    wr.writerow([f"{ti:.4f}", f"{bi:.6f}"])
            print(f"[rPPG] saved → {png}")
            print(f"[rPPG] saved → {csv_path}")
        except Exception as exc:
            print(f"[rPPG] BVP CSV failed ({exc}); image saved → {png}")

    def reset(self, quiet: bool = False):
        try:
            self.model.reset()
        except Exception:
            pass
        self._metrics.reset()
        self.hr_history.clear()
        self.frame_count = 0
        self.start_time  = time.time()
        if not quiet:
            print("[rPPG] buffer reset.")

    def run(self):
        _start_stream_server(self.port)

        win = "rPPG Ultimate"
        if not self.headless:
            try:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                print(f"[rPPG] cv2 window unavailable ({exc}); going headless.")
                self.headless = True

        last_good_frame = None
        exit_reason     = "user quit"

        try:
            with self.model.video_capture(self.camera):
                preview_iter = iter(self.model.preview)

                while True:
                    try:
                        frame, box = next(preview_iter)
                    except StopIteration:
                        exit_reason = ("preview stream ended "
                                       "(camera disconnected or face lost too long)")
                        break
                    except Exception as exc:
                        traceback.print_exc()
                        exit_reason = f"preview error: {exc!r}"
                        break

                    if frame is None:
                        if not self.headless:
                            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                                break
                        else:
                            time.sleep(0.01)
                        continue

                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except cv2.error as exc:
                        print(f"[rPPG] color-convert skipped: {exc}")
                        continue

                    self.frame_count += 1
                    last_good_frame = frame

                    if not self.paused:
                        self._snap = self._metrics.snapshot()
                        hr = self._snap.get("hr")
                        if hr and (not self.hr_history or self.hr_history[-1] != hr):
                            self.hr_history.append(hr)
                            snr_part = (f"   SNR {self._snap['snr']:5.1f} dB"
                                        if self._snap.get("snr") is not None else "")
                            print(f"[rPPG] HR {hr:6.1f} BPM{snr_part}")

                    try:
                        self.draw_hud(frame, box)
                    except Exception as exc:
                        print(f"[rPPG] HUD draw failed: {exc}")

                    _push_frame(frame)

                    if not self.headless:
                        cv2.imshow(win, frame)

                        try:
                            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                                exit_reason = "window closed by user"
                                break
                        except cv2.error:
                            exit_reason = "window was destroyed"
                            break

                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        elif key == ord("s") and last_good_frame is not None:
                            self.save_session(last_good_frame)
                        elif key == ord("r"):
                            self.reset()
                        elif key == ord("p"):
                            self.paused = not self.paused
                            print(f"[rPPG] {'paused' if self.paused else 'resumed'}.")
                        elif key == ord("g"):
                            self.show_bvp = not self.show_bvp
                        elif key == ord("h"):
                            self.show_hist = not self.show_hist
                        elif key == ord("f"):
                            self.show_box = not self.show_box
                        elif key == ord("m"):
                            try:
                                self.cycle_model()
                            except Exception as exc:
                                print(f"[rPPG] model switch failed: {exc}")

        except KeyboardInterrupt:
            exit_reason = "Ctrl+C"
        except Exception as exc:
            traceback.print_exc()
            exit_reason = f"unexpected error: {exc!r}"
        finally:
            if not self.headless:
                cv2.destroyAllWindows()
            print(f"[rPPG] exiting — {exit_reason}.")


# ─────────────────────────────── CLI ───────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="rPPG Ultimate — real-time heart-rate monitor (optimized).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model name (default {DEFAULT_MODEL}). "
                        f"Available: {', '.join(MODEL_ZOO)}.")
    p.add_argument("--camera", type=int, default=0,
                   help="camera index (default 0)")
    p.add_argument("--port", type=int, default=5050,
                   help="HTTP port for the dashboard (default 5050)")
    p.add_argument("--headless", action="store_true",
                   help="don't open the OpenCV preview window — useful when "
                        "running on a server or WSL without a display.")
    p.add_argument("--log-dir", default="rppg_logs",
                   help="directory for saved snapshots and CSV sessions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RPPGMonitor(model_name=args.model,
                camera=args.camera,
                log_dir=args.log_dir,
                port=args.port,
                headless=args.headless).run()