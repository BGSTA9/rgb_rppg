"""
═══════════════════════════════════════════════════════════════════════
  rPPG Ultimate — Real-Time Heart Rate Monitor
  ───────────────────────────────────────────
  Model:  ME-flow  (state-space rPPG, low-latency flow inference,
                    arXiv 2025 — best choice for live streams)

  Features
    • Live camera feed with face tracking + animated HR overlay
    • Rolling BVP waveform (last 8 s)
    • HR-history sparkline with quality-based coloring
    • Per-second HR / HRV / SNR metrics via the advanced API
    • Pause, reset, snapshot + CSV session dump
    • Tensor-based offline modes available via class methods

  Controls
    q / ESC  quit              s  save snapshot + BVP CSV
    p        pause / resume    r  reset rolling buffer
    g        toggle BVP plot   h  toggle HR history
    f        toggle face box   m  cycle through models
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rppg


# ─────────────────────────────── Config ────────────────────────────────

# Ranked from best-for-live to oldest. ME-flow is the live-stream pick;
# ME-chunk is stronger for offline analysis.
MODEL_ZOO = [
    "ME-flow",        # arXiv 2025 — low-latency state-space  ★ default
    "ME-chunk",       # arXiv 2025 — chunked state-space (offline)
    "RhythmMamba",    # AAAI  2025 — frequency-constrained Mamba
    "PhysMamba",      # CCBR  2024 — dual-branch Mamba
    "FacePhys",       #             — optimized state-space
    "EfficientPhys",  # WACV  2023 — self-attention TSCAN variant
    "PhysFormer",     # CVPR  2022 — temporal-difference transformer
    "TSCAN",          # NeurIPS 2020
    "PhysNet",        # BMVC  2019
]
DEFAULT_MODEL    = "ME-flow"
HR_UPDATE_PERIOD = 1.0    # seconds between HR recomputations
HR_WINDOW        = 10     # seconds of signal used for HR
BVP_WINDOW       = 8      # seconds of BVP shown in the waveform
HR_HIST_LEN      = 120    # samples retained in HR-history sparkline

# Color palette (BGR)
C_PANEL  = (38, 38, 44)
C_ACCENT = (96, 232, 96)
C_TEXT   = (235, 235, 235)
C_DIM    = (140, 140, 150)
C_WARN   = (60, 200, 255)
C_BAD    = (60, 60, 240)
C_BVP    = (255, 180, 80)


# ─────────────────────────── Drawing helpers ───────────────────────────

def hr_to_color(hr):
    """Green when HR is reasonable, amber for borderline, red otherwise."""
    if hr is None:
        return C_DIM
    if 50 <= hr <= 110:
        return C_ACCENT
    if 40 <= hr < 50 or 110 < hr <= 140:
        return C_WARN
    return C_BAD


def put_label(img, text, org, scale=0.55, color=C_TEXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thick, cv2.LINE_AA)


def panel(img, p1, p2, color=C_PANEL, alpha=0.55):
    """Draw a translucent rectangular panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_bvp(img, signal, origin, size, color=C_BVP):
    x0, y0 = origin
    w, h   = size
    panel(img, (x0 - 6, y0 - 6), (x0 + w + 6, y0 + h + 6))
    put_label(img, "BVP (filtered)", (x0 + 8, y0 + 16), 0.5, C_DIM)
    if signal is None or len(signal) < 2:
        put_label(img, "waiting for signal…", (x0 + 8, y0 + h // 2),
                  0.5, C_DIM)
        return
    s = np.asarray(signal, dtype=np.float32)
    s = s - s.mean()
    rng = float(np.max(np.abs(s))) or 1.0
    s = s / rng
    n = len(s)
    xs = np.linspace(x0 + 4, x0 + w - 4, n).astype(np.int32)
    ys = (y0 + h / 2 - s * (h / 2 - 8)).astype(np.int32)
    cv2.polylines(img, [np.stack([xs, ys], axis=1)],
                  False, color, 2, cv2.LINE_AA)


def draw_hr_history(img, history, origin, size):
    x0, y0 = origin
    w, h   = size
    panel(img, (x0 - 6, y0 - 6), (x0 + w + 6, y0 + h + 6))
    put_label(img, "HR history (BPM)", (x0 + 8, y0 + 16), 0.5, C_DIM)
    if len(history) < 2:
        return
    hr = np.asarray(history, dtype=np.float32)
    lo = max(40.0,  float(hr.min()) - 5.0)
    hi = min(180.0, float(hr.max()) + 5.0)
    span = max(hi - lo, 1.0)
    n = len(hr)
    xs = np.linspace(x0 + 4, x0 + w - 4, n).astype(np.int32)
    ys = (y0 + h - (hr - lo) / span * (h - 24) - 4).astype(np.int32)
    cv2.polylines(img, [np.stack([xs, ys], axis=1)],
                  False, hr_to_color(hr[-1]), 2, cv2.LINE_AA)
    put_label(img, f"{hi:.0f}", (x0 + w - 30, y0 + 28), 0.4, C_DIM)
    put_label(img, f"{lo:.0f}", (x0 + w - 30, y0 + h - 6), 0.4, C_DIM)


# ─────────────────────────── Main monitor ──────────────────────────────

class RPPGMonitor:
    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 camera: int = 0,
                 log_dir: str = "rppg_logs"):
        self.model_name = model_name
        self.model      = self._load_model(model_name)
        self.camera     = camera
        self.log_dir    = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.hr_history       = deque(maxlen=HR_HIST_LEN)
        self.last_hr_time     = 0.0          # last time we *attempted* update
        self.last_hr_update   = None         # last time HR actually changed
        self.last_face_seen   = None         # last time box was not None
        self.last_error       = None         # last metric/draw error text
        self.current_hr       = None
        self.current_hrv      = None
        self.current_snr      = None
        self.frame_count      = 0
        self.start_time       = time.time()

        self.show_bvp  = True
        self.show_hist = True
        self.show_box  = True
        self.paused    = False

    # ---- Model -------------------------------------------------------------
    @staticmethod
    def _load_model(name: str) -> "rppg.Model":
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
        self.reset(quiet=True)
        print(f"[rPPG] switched to {self.model_name}")

    # ---- Metric queries (advanced API) ------------------------------------
    @staticmethod
    def _to_float(v, prefer_keys=("value", "rmssd", "sdnn", "mean", "median",
                                  "db", "snr", "hr")):
        """Coerce float / int / numeric-string / dict-with-numeric → float.

        rppg's hr() may return nested results (e.g. hrv as
        {'rmssd': ..., 'sdnn': ...}); this digs out the first usable number.
        """
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            for k in prefer_keys:
                if k in v:
                    out = RPPGMonitor._to_float(v[k])
                    if out is not None:
                        return out
            for sub in v.values():
                out = RPPGMonitor._to_float(sub)
                if out is not None:
                    return out
            return None
        if isinstance(v, (list, tuple)) and v:
            return RPPGMonitor._to_float(v[-1])
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def update_metrics(self):
        try:
            res = self.model.hr(start=-HR_WINDOW)
        except Exception as exc:
            self.last_error = f"hr(): {exc!r}"
            return
        if not res:
            return

        hr = self._to_float(res.get("hr"))
        if hr is not None and hr > 0:
            self.current_hr = hr
            self.hr_history.append(hr)
            self.last_hr_update = time.time()

        hrv = self._to_float(res.get("hrv") or res.get("rmssd"))
        if hrv is not None:
            self.current_hrv = hrv

        snr = self._to_float(res.get("snr") or res.get("snr_db"))
        if snr is not None:
            self.current_snr = snr

    def bvp_window(self, seconds: int = BVP_WINDOW):
        try:
            bvp, _ = self.model.bvp(start=-seconds)
            return bvp
        except Exception:
            return None

    # ---- Tensor-based modes (offline / batch) -----------------------------
    def process_video_tensor(self, tensor: np.ndarray, fps: float = 30.0):
        """Accepts (T,H,W,3) uint8 video frames; returns the model result."""
        return self.model.process_video_tensor(tensor, fps=fps)

    def process_faces_tensor(self, tensor: np.ndarray, fps: float = 30.0):
        """Accepts (T,128,128,3) uint8 face crops; returns the model result."""
        return self.model.process_faces_tensor(tensor, fps=fps)

    # ---- HUD --------------------------------------------------------------
    def _safe(self, label, fn):
        """Run a HUD section; remember the error instead of bubbling it."""
        try:
            fn()
        except Exception as exc:
            self.last_error = f"{label}: {exc!r}"
            print(f"[rPPG] {label} failed: {exc}")

    def draw_hud(self, frame, box):
        h, w = frame.shape[:2]
        now  = time.time()

        if box is not None:
            self.last_face_seen = now

        # Derived state
        face_age = (now - self.last_face_seen) if self.last_face_seen else None
        hr_age   = (now - self.last_hr_update) if self.last_hr_update else None
        face_ok  = face_age is not None and face_age < 1.5
        hr_fresh = hr_age   is not None and hr_age   < HR_UPDATE_PERIOD * 3

        # ===== Status panel (top-left) =====
        def _status_panel():
            panel(frame, (10, 10), (300, 122))
            color = hr_to_color(self.current_hr) if hr_fresh else C_DIM
            hr_text  = f"{self.current_hr:.1f}" if self.current_hr else "--"
            snr_text = f"{self.current_snr:.1f}" if self.current_snr is not None else "--"
            hrv_text = f"{self.current_hrv:.0f}" if self.current_hrv else "--"
            put_label(frame, "Heart Rate", (24, 36), 0.55, C_DIM)
            put_label(frame, hr_text, (24, 82), 1.5, color, 2)
            put_label(frame, "BPM", (160, 82), 0.7, C_DIM)
            sub = f"SNR {snr_text} dB    HRV {hrv_text} ms"
            if hr_age is not None and not hr_fresh:
                sub += f"   (stale {hr_age:.0f}s)"
            put_label(frame, sub, (24, 104), 0.5, C_DIM)
        self._safe("status panel", _status_panel)

        # ===== Diagnostics strip (top-center) =====
        def _diagnostics():
            face_msg = "FACE OK" if face_ok else (
                f"NO FACE ({face_age:.0f}s)" if face_age else "WAITING FOR FACE")
            face_col = C_ACCENT if face_ok else C_BAD
            sig_msg, sig_col = (
                ("SIGNAL OK", C_ACCENT) if hr_fresh
                else ("WARMING UP", C_WARN) if hr_age is None
                else (f"SIGNAL STALE ({hr_age:.0f}s)", C_WARN)
            )
            x0 = 320
            panel(frame, (x0, 10), (x0 + 280, 56))
            put_label(frame, face_msg, (x0 + 12, 32), 0.55, face_col, 1)
            put_label(frame, sig_msg,  (x0 + 12, 50), 0.5,  sig_col, 1)
        self._safe("diagnostics", _diagnostics)

        # ===== Model + FPS tag (top-right) =====
        def _fps_tag():
            elapsed = now - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            tag = f"{self.model_name}   {fps:4.1f} fps"
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            panel(frame, (w - tw - 28, 10), (w - 10, 44))
            put_label(frame, tag, (w - tw - 18, 32), 0.55, C_TEXT)
        self._safe("fps tag", _fps_tag)

        # ===== Face box =====
        def _face_box():
            if not self.show_box:
                return
            if box is not None:
                (y1, y2), (x1, x2) = box
                color = hr_to_color(self.current_hr) if hr_fresh else C_WARN
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    cv2.circle(frame, (cx, cy), 3, color, -1)
                label = (f"HR {self.current_hr:.1f}" if self.current_hr
                         else "tracking…")
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,
                                              0.6, 1)
                cv2.rectangle(frame, (x1, y1 - lh - 12),
                              (x1 + lw + 14, y1), color, -1)
                put_label(frame, label, (x1 + 7, y1 - 7), 0.6, (0, 0, 0), 1)
            else:
                msg = "no face detected — center your face & hold still"
                (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX,
                                              0.6, 1)
                cx, cy = (w - mw) // 2, h // 2
                panel(frame, (cx - 16, cy - mh - 10),
                      (cx + mw + 16, cy + 14))
                put_label(frame, msg, (cx, cy), 0.6, C_BAD, 1)
        self._safe("face box", _face_box)

        # ===== Live BVP waveform =====
        def _bvp():
            if self.show_bvp:
                draw_bvp(frame, self.bvp_window(),
                         origin=(20, h - 120),
                         size=(w // 2 - 40, 100))
        self._safe("bvp plot", _bvp)

        # ===== HR history sparkline =====
        def _history():
            if self.show_hist:
                draw_hr_history(frame, list(self.hr_history),
                                origin=(w // 2 + 20, h - 120),
                                size=(w // 2 - 40, 100))
        self._safe("hr history", _history)

        # ===== Footer (always last; reports last error if any) =====
        def _footer():
            status = "PAUSED" if self.paused else "LIVE"
            line = (f"[{status}]  q quit · s save · p pause · r reset · "
                    f"g graph · h history · f face · m model")
            put_label(frame, line, (16, h - 14), 0.42, C_DIM)
            if self.last_error:
                put_label(frame, f"last error: {self.last_error[:90]}",
                          (16, h - 32), 0.42, C_BAD)
        self._safe("footer", _footer)

    # ---- Persistence ------------------------------------------------------
    def save_session(self, frame):
        ts  = datetime.now().strftime("%Y%m%d-%H%M%S")
        png = self.log_dir / f"snapshot-{ts}.png"
        csv_path = self.log_dir / f"session-{ts}.csv"
        cv2.imwrite(str(png), frame)
        try:
            bvp, t = self.model.bvp()
            with open(csv_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_seconds", "bvp"])
                for ti, bi in zip(t, bvp):
                    w.writerow([f"{ti:.4f}", f"{bi:.6f}"])
            print(f"[rPPG] saved → {png}")
            print(f"[rPPG] saved → {csv_path}")
        except Exception as exc:
            print(f"[rPPG] BVP CSV failed ({exc}); image saved → {png}")

    def reset(self, quiet: bool = False):
        try:
            self.model.reset()
        except Exception:
            pass
        self.hr_history.clear()
        self.current_hr = self.current_hrv = self.current_snr = None
        self.frame_count = 0
        self.start_time  = time.time()
        if not quiet:
            print("[rPPG] buffer reset.")

    # ---- Main loop --------------------------------------------------------
    def run(self):
        win = "rPPG Ultimate"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_good_frame = None
        exit_reason = "user quit"

        try:
            with self.model.video_capture(self.camera):
                preview_iter = iter(self.model.preview)

                while True:
                    # ---- pull next frame from the rPPG generator ----------
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
                        # transient: skip the frame, keep the loop alive
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                        continue

                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except cv2.error as exc:
                        print(f"[rPPG] color-convert skipped: {exc}")
                        continue

                    self.frame_count += 1
                    last_good_frame = frame

                    # ---- metrics -----------------------------------------
                    if not self.paused:
                        now = time.time()
                        if now - self.last_hr_time > HR_UPDATE_PERIOD:
                            try:
                                self.update_metrics()
                            except Exception as exc:
                                print(f"[rPPG] metrics update failed: {exc}")
                            self.last_hr_time = now
                            if self.current_hr:
                                snr_part = (f"   SNR {self.current_snr:5.1f} dB"
                                            if self.current_snr is not None else "")
                                print(f"[rPPG] HR {self.current_hr:6.1f} BPM"
                                      + snr_part)

                    # ---- HUD (never let drawing kill the loop) -----------
                    try:
                        self.draw_hud(frame, box)
                    except Exception as exc:
                        print(f"[rPPG] HUD draw failed: {exc}")

                    cv2.imshow(win, frame)

                    # ---- detect window X-button --------------------------
                    try:
                        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                            exit_reason = "window closed by user"
                            break
                    except cv2.error:
                        exit_reason = "window was destroyed"
                        break

                    # ---- keys --------------------------------------------
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
            cv2.destroyAllWindows()
            print(f"[rPPG] exiting — {exit_reason}.")


# ─────────────────────────────── CLI ───────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="rPPG Ultimate — real-time heart-rate monitor.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"model name (default {DEFAULT_MODEL}). "
                        f"Append a checkpoint suffix like '.pure' or "
                        f"'.ubfc' to pick a trained variant. "
                        f"Available: {', '.join(MODEL_ZOO)}.")
    p.add_argument("--camera", type=int, default=0,
                   help="camera index (default 0)")
    p.add_argument("--log-dir", default="rppg_logs",
                   help="directory for saved snapshots and CSV sessions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RPPGMonitor(model_name=args.model,
                camera=args.camera,
                log_dir=args.log_dir).run()