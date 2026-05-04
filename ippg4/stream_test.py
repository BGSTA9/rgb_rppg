"""
stream_test.py — minimal MJPEG test server.

Use this to confirm the basic plumbing (port, browser, HTML, MJPEG decoding)
works on your machine WITHOUT involving rppg, the camera, or any model.

Run:
    python stream_test.py

Then open:
    http://localhost:5050/         → animated test pattern
    http://localhost:5050/health   → JSON {"ok": true, ...}
    http://localhost:5050/video_feed → raw MJPEG stream

If you see the moving rectangle on /, your environment is fine and the
problem is inside run4.py. If you see "site can't be reached", the issue
is environmental (port already taken, firewall, wrong python).
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

PORT = 5050

_jpeg: bytes | None = None
_lock = threading.Lock()
_started_at = time.time()
_frames = 0


def _producer():
    """Generates a synthetic moving frame at ~30 fps."""
    global _jpeg, _frames
    t0 = time.time()
    while True:
        img = np.zeros((400, 720, 3), dtype=np.uint8)
        img[:] = (24, 26, 32)

        # moving green block (matches C_ACCENT in run4.py)
        x = int(((time.time() - t0) * 120) % 600) + 50
        cv2.rectangle(img, (x, 150), (x + 80, 250), (96, 232, 96), -1)

        cv2.putText(img, "stream test OK", (240, 330),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(img, time.strftime("%H:%M:%S"), (300, 365),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (140, 140, 150), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _lock:
                _jpeg = buf.tobytes()
                _frames += 1
        time.sleep(0.033)


CORS = (
    ("Access-Control-Allow-Origin",  "*"),
    ("Access-Control-Allow-Methods", "GET, OPTIONS"),
)


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        for k, v in CORS:
            self.send_header(k, v)

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/health":
            self._serve_health()
        elif self.path == "/video_feed":
            self._serve_mjpeg()
        else:
            self.send_response(404); self._cors(); self.end_headers()

    def _serve_html(self):
        body = b"""<!doctype html><meta charset=utf-8>
<title>stream test</title>
<style>
  body{margin:0;background:#0b0d10;color:#e8ecf2;font:14px/1.4 ui-monospace,monospace;
       display:grid;place-items:center;min-height:100vh;text-align:center}
  img{max-width:90vw;max-height:70vh;border:1px solid #2c4a2c;border-radius:8px}
  .ok{color:#60e860}
</style>
<h2 class="ok">stream test running</h2>
<p>If the rectangle below is moving, your MJPEG pipeline works.</p>
<img src="/video_feed?t=" alt="test stream"
     onload="document.querySelector('h2').textContent='\u2713 stream live'">
<p style=color:#8b94a3>also try
  <a style=color:#60e860 href=/health>/health</a> and
  <a style=color:#60e860 href=/video_feed>/video_feed</a></p>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_health(self):
        with _lock:
            payload = {
                "ok":          True,
                "uptime_s":    round(time.time() - _started_at, 1),
                "frames":      _frames,
                "has_jpeg":    _jpeg is not None,
            }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=rpframe")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self._cors()
        self.end_headers()
        try:
            while True:
                with _lock:
                    j = _jpeg
                if j:
                    try:
                        self.wfile.write(
                            b"--rpframe\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(j)).encode() + b"\r\n\r\n"
                        )
                        self.wfile.write(j)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        break
                time.sleep(0.033)
        except Exception:
            pass

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    threading.Thread(target=_producer, daemon=True).start()
    print(f"[test] http://localhost:{PORT}/")
    print(f"[test] http://localhost:{PORT}/health")
    print(f"[test] http://localhost:{PORT}/video_feed")
    try:
        ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    except OSError as e:
        print(f"\n[test] could not bind port {PORT}: {e}")
        print(f"[test] something is already using port {PORT} — close it or "
              f"change PORT at the top of this file.")