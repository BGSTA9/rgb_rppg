import rppg
import cv2
import time
import threading
import numpy as np
from collections import deque
from flask import Flask
from flask_socketio import SocketIO

# ─────────────────────────────────────────────
# Server setup
# ─────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)

model = rppg.Model()

state = {
    "running": False,
    "thread": None
}

lock = threading.Lock()


# ─────────────────────────────────────────────
# HR Stabilizer (unchanged, just cleaned)
# ─────────────────────────────────────────────
class HRStabilizer:
    def __init__(self):
        self.buf = deque(maxlen=8)
        self.last = None
        self.last_time = None

    def reset(self):
        self.buf.clear()
        self.last = None
        self.last_time = None

    def push(self, hr):
        if hr is None or not np.isfinite(hr):
            return self.last

        if not (30 <= hr <= 220):
            return self.last

        self.buf.append(hr)

        if len(self.buf) < 4:
            return None

        arr = np.array(self.buf)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        filtered = arr[np.abs(arr - med) < max(8, 3.5 * mad)]

        val = float(np.mean(filtered)) if len(filtered) else float(np.mean(arr))

        now = time.time()

        if self.last is not None:
            dt = max(0.05, now - self.last_time)
            max_delta = 25 * dt
            delta = val - self.last

            if abs(delta) > max_delta:
                val = self.last + (max_delta if delta > 0 else -max_delta)

        self.last = val
        self.last_time = now
        return val

    def quality(self):
        if len(self.buf) < 4:
            return 0.0
        std = np.std(self.buf)
        return float(np.clip(1.0 - (std - 2.0) / 10.0, 0, 1))


hr_stab = HRStabilizer()


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def vitals_loop():
    while state["running"]:
        time.sleep(1)

        try:
            result = model.hr(start=-15)
            hr_raw = float(result["hr"]) if result and result.get("hr") else None

            hr_smooth = hr_stab.push(hr_raw)

            bvp_vals = []
            timestamps = []

            try:
                bvp, ts = model.bvp(start=-10)
                if bvp is not None:
                    bvp_vals = bvp[-150:].tolist()
                    timestamps = ts[-150:].tolist()
            except:
                pass

            socketio.emit("vitals", {
                "hr": round(hr_smooth, 1) if hr_smooth else None,
                "hr_raw": round(hr_raw, 1) if hr_raw else None,
                "quality": round(hr_stab.quality(), 2),
                "bvp": bvp_vals,
                "timestamps": timestamps,
            })

        except Exception as e:
            print("Vitals error:", e)


# ─────────────────────────────────────────────
# SOCKET EVENTS
# ─────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print("Client connected")

    with lock:
        if not state["running"]:
            model.__enter__()
            hr_stab.reset()

            state["running"] = True
            t = threading.Thread(target=vitals_loop, daemon=True)
            t.start()
            state["thread"] = t


@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")

    with lock:
        state["running"] = False
        hr_stab.reset()

        try:
            model.__exit__(None, None, None)
        except Exception as e:
            print("Exit error:", e)


@socketio.on("frame")
def on_frame(data):
    try:
        import base64

        img_data = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_data)

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ts = float(data.get("ts", time.time()))

        model.update_face(img, ts=ts, hasface=True)

    except Exception as e:
        print("Frame error:", e)


@socketio.on("no_face")
def on_no_face(data):
    try:
        ts = float(data.get("ts", time.time()))
        model.update_face(None, ts=ts, hasface=False)
    except Exception as e:
        print("No face error:", e)


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Backend running → http://localhost:5050")
    socketio.run(app, host="0.0.0.0", port=5050)