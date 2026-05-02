import rppg
import cv2
import time
import threading
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

model = rppg.Model()
state = {"running": False, "vitals_thread": None}
lock = threading.Lock()


@app.route("/")
def index():
    return render_template("dashboard.html")


def vitals_loop():
    while state["running"]:
        time.sleep(1.0)
        try:
            result = model.hr(start=-10)
            hr_val = None
            if result and result.get("hr"):
                hr_val = round(float(result["hr"]), 1)

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
                "bvp": bvp_vals,
                "timestamps": timestamps,
            })
        except Exception as e:
            print(f"vitals error: {e}")


@socketio.on("connect")
def on_connect():
    with lock:
        if not state["running"]:
            model.__enter__()
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
