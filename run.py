import rppg
import cv2
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model = rppg.Model()
state = {"running": False, "vitals_thread": None}
lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
#  HRStabilizer
#  Raw rPPG HR estimates are noisy. We apply, in order:
#    1. Plausibility gate     (30 ≤ HR ≤ 220 BPM)
#    2. Rolling buffer        (last N estimates)
#    3. Median + MAD outlier rejection
#    4. Slew-rate limit       (HR can't physically change >25 BPM/s)
#    5. Warm-up gate          (need min_fill samples before first publish)
#  Quality is reported as 1 − (buffer-std / 12), clipped to [0, 1].
# ─────────────────────────────────────────────────────────────
class HRStabilizer:
    def __init__(self, buffer_size=8, min_fill=4, max_slew_bpm_per_sec=25.0,
                 hr_min=30.0, hr_max=220.0):
        self.buf = deque(maxlen=buffer_size)
        self.min_fill = min_fill
        self.max_slew = max_slew_bpm_per_sec
        self.hr_min = hr_min
        self.hr_max = hr_max
        self._last_pub = None
        self._last_pub_time = None

    def reset(self):
        self.buf.clear()
        self._last_pub = None
        self._last_pub_time = None

    def push(self, hr_raw):
        """Feed a raw HR estimate. Returns smoothed value or None if not ready."""
        # 1) plausibility gate
        try:
            hr_raw = float(hr_raw) if hr_raw is not None else None
        except (TypeError, ValueError):
            hr_raw = None
        if hr_raw is None or not np.isfinite(hr_raw):
            return self._last_pub
        if hr_raw < self.hr_min or hr_raw > self.hr_max:
            return self._last_pub

        self.buf.append(hr_raw)

        # 2) warm-up
        if len(self.buf) < self.min_fill:
            return None

        # 3) median + MAD outlier rejection
        arr = np.array(self.buf, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        threshold = max(8.0, 3.5 * mad)  # always allow at least ±8 BPM
        keep = arr[np.abs(arr - med) <= threshold]
        if keep.size == 0:
            keep = arr
        smoothed = float(np.mean(keep))

        # 4) slew-rate limit
        now = time.time()
        if self._last_pub is not None and self._last_pub_time is not None:
            dt = max(0.05, now - self._last_pub_time)
            max_delta = self.max_slew * dt
            delta = smoothed - self._last_pub
            if abs(delta) > max_delta:
                smoothed = self._last_pub + (max_delta if delta > 0 else -max_delta)

        self._last_pub = smoothed
        self._last_pub_time = now
        return smoothed

    def quality(self):
        """0..1 — based on stability of recent estimates."""
        if len(self.buf) < self.min_fill:
            return 0.0
        std = float(np.std(self.buf))
        # std ≤ 2 BPM ⇒ ~1.0 quality, std ≥ 12 BPM ⇒ 0
        return float(np.clip(1.0 - (std - 2.0) / 10.0, 0.0, 1.0))


hr_stab = HRStabilizer()

# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")


def vitals_loop():
    while state["running"]:
        time.sleep(1.0)
        try:
            # Use a longer analysis window for stable peak detection
            # (10s gives ~10-17 cycles; 15s gives more dominant-freq stability)
            result = model.hr(start=-15)
            hr_raw = None
            if result and result.get("hr"):
                hr_raw = float(result["hr"])

            # Stabilize
            hr_smooth = hr_stab.push(hr_raw)
            hr_val = round(hr_smooth, 1) if hr_smooth is not None else None

            bvp_vals, timestamps = [], []
            try:
                bvp, ts = model.bvp(start=-10)
                if bvp is not None and len(bvp) >= 2:
                    bvp_vals = [round(float(v), 4) for v in bvp[-150:]]
                    timestamps = [round(float(t), 3) for t in ts[-150:]]
            except Exception:
                pass

            socketio.emit("vitals", {
                "hr": hr_val,                                      # smoothed (use this)
                "hr_raw": round(hr_raw, 1) if hr_raw else None,    # for debugging
                "quality": round(hr_stab.quality(), 2),            # 0..1 signal quality
                "bvp": bvp_vals,
                "timestamps": timestamps,
            })
        except Exception as e:
            print(f"vitals error: {e}")


# ─────────────────────────────────────────────────────────────
#  Socket handlers
# ─────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    with lock:
        if not state["running"]:
            model.__enter__()
            hr_stab.reset()
            state["running"] = True
            t = threading.Thread(target=vitals_loop, daemon=True)
            t.start()
            state["vitals_thread"] = t
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect():
    with lock:
        if state["running"]:
            state["running"] = False
            hr_stab.reset()
            try:
                model.__exit__(None, None, None)
            except Exception as e:
                print(f"model exit error: {e}")
    print("Client disconnected")


@socketio.on("face_frame")
def on_face_frame(data):
    try:
        img_bytes = data.get("img")
        ts = data.get("ts")
        if img_bytes is None:
            return
        if isinstance(img_bytes, str):
            import base64
            if "," in img_bytes:
                img_bytes = img_bytes.split(",", 1)[1]
            img_bytes = base64.b64decode(img_bytes)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if ts is not None:
            ts = float(ts)
        model.update_face(img, ts=ts, hasface=True)
    except Exception as e:
        print(f"face_frame error: {e}")


@socketio.on("no_face")
def on_no_face(data):
    try:
        ts = data.get("ts")
        if ts is not None:
            ts = float(ts)
        model.update_face(None, ts=ts, hasface=False)
    except Exception as e:
        print(f"no_face error: {e}")


if __name__ == "__main__":
    print("▶  rPPG Dashboard → http://localhost:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)