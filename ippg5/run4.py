from __future__ import annotations

import argparse
import json
import threading
import time
import io
from collections import deque

import numpy as np
import rppg
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image

# ───────────────────────── CONFIG ─────────────────────────

DEFAULT_MODEL = "ME-flow.rlap"
FRAME_BUFFER_SIZE = 150   # ~5 seconds @30 FPS

# ───────────────────────── SAFE MODEL LOAD ─────────────────────────

def load_model_safe(name):
    try:
        return rppg.Model(name)
    except Exception:
        print(f"[rPPG] Invalid model '{name}', falling back to {DEFAULT_MODEL}")
        return rppg.Model(DEFAULT_MODEL)

# ───────────────────────── MONITOR ─────────────────────────

class RPPGMonitor:
    def __init__(self, model_name):
        self.model = load_model_safe(model_name)

        self.frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
        self.hr = None

        self.lock = threading.Lock()

        # Background processing thread
        threading.Thread(target=self._process_loop, daemon=True).start()

    def add_frame(self, frame):
        with self.lock:
            self.frame_buffer.append(frame)

    def _process_loop(self):
        while True:
            time.sleep(1)

            with self.lock:
                if len(self.frame_buffer) < 30:
                    continue

                frames = np.array(self.frame_buffer, dtype=np.uint8)

            try:
                self.model.process_video_tensor(frames, fps=30)

                res = self.model.hr(start=-10)

                if res and "hr" in res:
                    self.hr = float(res["hr"])
                    print(f"[rPPG] HR: {self.hr:.1f} BPM")

            except Exception as e:
                print("[rPPG] processing error:", e)

# ───────────────────────── HTTP SERVER ─────────────────────────

class Handler(BaseHTTPRequestHandler):

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open("dash.html", "rb") as f:
                    html = f.read()

                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self._cors()
                self.end_headers()
                self.wfile.write(html)
            except:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/process_frame":
            self.handle_frame()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_frame(self):
        try:
            import cgi

            ctype, pdict = cgi.parse_header(self.headers.get('content-type'))

            if ctype != 'multipart/form-data':
                self.send_response(400)
                self.end_headers()
                return

            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            pdict['CONTENT-LENGTH'] = int(self.headers.get('content-length'))

            fields = cgi.parse_multipart(self.rfile, pdict)

            if 'frame' not in fields:
                self.send_response(400)
                self.end_headers()
                return

            file_data = fields['frame'][0]

            # ✅ Now decode correctly
            img = Image.open(io.BytesIO(file_data)).convert("RGB")
            frame = np.array(img)

            monitor.add_frame(frame)

            response = json.dumps({
                "hr": monitor.hr
            }).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            print("[rPPG] frame error:", e)
            self.send_response(500)
            self.end_headers()

# ───────────────────────── SERVER START ─────────────────────────

def start_server(port, monitor_instance):
    global monitor
    monitor = monitor_instance

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)

    print(f"[rPPG] Server running at http://localhost:{port}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

# ───────────────────────── MAIN ─────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--port", type=int, default=5050)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    monitor = RPPGMonitor(args.model)
    start_server(args.port, monitor)

    print("[rPPG] Waiting for browser connection...")

    while True:
        time.sleep(1)

# # ───────────────────────── CLI ─────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--model", default=DEFAULT_MODEL)
#     p.add_argument("--port", type=int, default=5050)
#     return p.parse_args()

# if __name__ == "__main__":
#     args = parse_args()

#     RPPGMonitor(
#         model_name=args.model,
#         port=args.port
#     ).run()