import rppg
import cv2
import time
import threading
import numpy as np
import base64
from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__, template_folder=".")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model = rppg.Model()
state = {"running": False, "capture_thread": None}
lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
#  HRStabilizer
#  Raw rPPG HR estimates are noisy. We apply, in order:
#    1. Plausibility gate     (30 ≤ HR ≤ 220 BPM)
#    2. Rolling buffer        (last N estimates)
#    3. Median + MAD outlier rejection
#    4. Slew-rate limit       (HR can't physically change >15 BPM/s)
#    5. Warm-up gate          (need min_fill samples before first publish)
#  Quality blends library SQI with buffer stability.
# ─────────────────────────────────────────────────────────────
class HRStabilizer:
    def __init__(self, buffer_size=12, min_fill=5, max_slew_bpm_per_sec=15.0,
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
        threshold = max(6.0, 3.0 * mad)  # always allow at least ±6 BPM
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


def capture_and_vitals():
    """
    Main pipeline thread: uses model.video_capture(0) to let open-rppg
    handle camera capture, face detection, and signal extraction internally.

    The model.preview generator yields (frame, box) pairs. We:
      1. Stream the camera frame to the browser for display
      2. Periodically query model.hr() and model.bvp() for vital signs
    """
    try:
        with model.video_capture(0):
            print("✓ Camera opened by open-rppg pipeline")
            last_hr_query = 0
            frame_count = 0

            for frame, box in model.preview:
                if not state["running"]:
                    break

                frame_count += 1
                now = time.time()

                # ── Stream video frame to browser (every 2nd frame ≈ 15fps) ──
                if frame_count % 2 == 0:
                    try:
                        # frame is RGB from open-rppg; convert to BGR for JPEG
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        _, jpeg = cv2.imencode('.jpg', bgr,
                                               [cv2.IMWRITE_JPEG_QUALITY, 70])
                        b64 = base64.b64encode(jpeg.tobytes()).decode('ascii')

                        # Send face bounding box if available
                        face_box = None
                        if box is not None:
                            # box format: [[row_min, row_max], [col_min, col_max]]
                            face_box = {
                                "y1": int(box[0][0]), "y2": int(box[0][1]),
                                "x1": int(box[1][0]), "x2": int(box[1][1]),
                            }

                        socketio.emit("camera_frame", {
                            "img": b64,
                            "box": face_box,
                            "w": frame.shape[1],
                            "h": frame.shape[0],
                        })
                    except Exception as e:
                        print(f"frame stream error: {e}")

                # ── Query vitals every ~1 second ──
                if now - last_hr_query >= 1.0:
                    last_hr_query = now
                    try:
                        result = model.hr(start=-15)
                        hr_raw = None
                        sqi = 0.0
                        if result:
                            if result.get("hr"):
                                hr_raw = float(result["hr"])
                            if result.get("SQI") is not None:
                                sqi = float(result["SQI"])

                        # Stabilize the HR
                        hr_smooth = hr_stab.push(hr_raw)
                        hr_val = round(hr_smooth, 1) if hr_smooth is not None else None

                        # Blend library SQI with our stabilizer quality
                        stab_q = hr_stab.quality()
                        combined_quality = (sqi * 0.6 + stab_q * 0.4) if sqi > 0 else stab_q

                        # Get BVP waveform for visualization
                        bvp_vals, timestamps = [], []
                        try:
                            bvp, ts = model.bvp(start=-10)
                            if bvp is not None and len(bvp) >= 2:
                                bvp_vals = [round(float(v), 4) for v in bvp[-150:]]
                                timestamps = [round(float(t), 3) for t in ts[-150:]]
                        except Exception:
                            pass

                        socketio.emit("vitals", {
                            "hr": hr_val,
                            "hr_raw": round(hr_raw, 1) if hr_raw else None,
                            "quality": round(combined_quality, 2),
                            "sqi": round(sqi, 2),
                            "bvp": bvp_vals,
                            "timestamps": timestamps,
                        })
                    except Exception as e:
                        print(f"vitals query error: {e}")

    except Exception as e:
        print(f"capture_and_vitals error: {e}")
    finally:
        print("capture_and_vitals thread exiting")
        with lock:
            state["running"] = False


# ─────────────────────────────────────────────────────────────
#  Socket handlers
# ─────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    with lock:
        if not state["running"]:
            hr_stab.reset()
            state["running"] = True
            t = threading.Thread(target=capture_and_vitals, daemon=True)
            t.start()
            state["capture_thread"] = t
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect():
    with lock:
        if state["running"]:
            state["running"] = False
            hr_stab.reset()
            try:
                model.stop()
            except Exception as e:
                print(f"model stop error: {e}")
    print("Client disconnected")


if __name__ == "__main__":
    print("▶  rPPG Dashboard → http://localhost:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)