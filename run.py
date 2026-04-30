import rppg
import cv2
import time
import base64
import threading
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model = rppg.Model()
running = False
lock = threading.Lock()


@app.route("/")
def index():
    return render_template("dashboard.html")


def rppg_loop():
    global running
    model.video_capture(0).__enter__()
    last_hr_time = 0

    for frame, box in model.preview:
        if not running:
            break

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw face bounding box
        box_data = None
        if box is not None:
            y1, y2 = int(box[0][0]), int(box[0][1])
            x1, x2 = int(box[1][0]), int(box[1][1])
            box_data = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        # Encode frame → base64 JPEG
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frame_b64 = base64.b64encode(buf).decode("utf-8")
        socketio.emit("frame", {"img": frame_b64, "box": box_data})

        now = time.time()
        if now - last_hr_time >= 1.0:
            # Heart rate (10-second window)
            result = model.hr(start=-10)
            hr_val = None
            if result and result.get("hr"):
                hr_val = round(float(result["hr"]), 1)

            # BVP signal
            bvp_vals, timestamps = [], []
            try:
                bvp, ts = model.bvp()
                if bvp is not None and len(bvp) >= 2:
                    # send last 150 samples
                    bvp_vals = [round(float(v), 4) for v in bvp[-150:]]
                    timestamps = [round(float(t), 3) for t in ts[-150:]]
            except Exception:
                pass

            # HR history (last 30 seconds)
            hr_history = []
            try:
                metrics = model.hr(start=-30)
                if metrics and metrics.get("hr"):
                    hr_history = [round(float(metrics["hr"]), 1)]
            except Exception:
                pass

            socketio.emit("vitals", {
                "hr": hr_val,
                "bvp": bvp_vals,
                "timestamps": timestamps,
                "hr_history": hr_history,
            })

            last_hr_time = now


@socketio.on("connect")
def on_connect():
    global running
    with lock:
        if not running:
            running = True
            t = threading.Thread(target=rppg_loop, daemon=True)
            t.start()
    print("Client connected")


@socketio.on("disconnect")
def on_disconnect():
    global running
    running = False
    print("Client disconnected")


if __name__ == "__main__":
    print("▶  rPPG Dashboard → http://localhost:5050")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False)