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

        self.hr_history     = deque(maxlen=HR_HIST_LEN)
        self.last_hr_time   = 0.0
        self.current_hr     = None
        self.current_hrv    = None
        self.current_snr    = None
        self.frame_count    = 0
        self.start_time     = time.time()

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
    def update_metrics(self):
        try:
            res = self.model.hr(start=-HR_WINDOW)
        except Exception:
            res = None
        if not res:
            return
        if res.get("hr"):
            self.current_hr = float(res["hr"])
            self.hr_history.append(self.current_hr)
        # Tolerate keys that may or may not be present in this version.
        self.current_hrv = res.get("hrv")  or res.get("rmssd") or self.current_hrv
        self.current_snr = res.get("snr")  or res.get("snr_db") or self.current_snr

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
    def draw_hud(self, frame, box):
        h, w = frame.shape[:2]

        # Status panel (top-left)
        panel(frame, (10, 10), (300, 122))
        title_color = hr_to_color(self.current_hr)
        hr_text  = f"{self.current_hr:.1f}" if self.current_hr else "--"
        snr_text = f"{self.current_snr:.1f}" if self.current_snr else "--"
        hrv_text = f"{self.current_hrv:.0f}" if self.current_hrv else "--"
        elapsed  = time.time() - self.start_time
        fps      = self.frame_count / elapsed if elapsed > 0 else 0.0

        put_label(frame, "Heart Rate", (24, 36), 0.55, C_DIM)
        put_label(frame, hr_text, (24, 82), 1.5, title_color, 2)
        put_label(frame, "BPM",   (160, 82), 0.7, C_DIM)
        put_label(frame, f"SNR {snr_text} dB    HRV {hrv_text} ms",
                  (24, 104), 0.5, C_DIM)

        # Top-right tag with model + fps
        tag = f"{self.model_name}   {fps:4.1f} fps"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        panel(frame, (w - tw - 28, 10), (w - 10, 44))
        put_label(frame, tag, (w - tw - 18, 32), 0.55, C_TEXT)

        # Face box with corner markers
        if self.show_box and box is not None:
            (y1, y2), (x1, x2) = box
            color = hr_to_color(self.current_hr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.circle(frame, (cx, cy), 3, color, -1)
            label = f"HR {hr_text}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX,
                                          0.6, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 12),
                          (x1 + lw + 14, y1), color, -1)
            put_label(frame, label, (x1 + 7, y1 - 7), 0.6, (0, 0, 0), 1)

        # Live waveform + history
        if self.show_bvp:
            draw_bvp(frame, self.bvp_window(),
                     origin=(20, h - 120),
                     size=(w // 2 - 40, 100))
        if self.show_hist:
            draw_hr_history(frame, list(self.hr_history),
                            origin=(w // 2 + 20, h - 120),
                            size=(w // 2 - 40, 100))

        # Footer
        status = "PAUSED" if self.paused else "LIVE"
        put_label(frame,
                  f"[{status}]  q quit · s save · p pause · r reset · "
                  f"g graph · h history · f face · m model",
                  (16, h - 14), 0.42, C_DIM)

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

        with self.model.video_capture(self.camera):
            for frame, box in self.model.preview:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.frame_count += 1

                if not self.paused:
                    now = time.time()
                    if now - self.last_hr_time > HR_UPDATE_PERIOD:
                        self.update_metrics()
                        self.last_hr_time = now
                        if self.current_hr:
                            print(f"[rPPG] HR {self.current_hr:6.1f} BPM"
                                  + (f"   SNR {self.current_snr:5.1f} dB"
                                     if self.current_snr is not None else ""))

                self.draw_hud(frame, box)
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("s"):
                    self.save_session(frame)
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
                    self.cycle_model()

        cv2.destroyAllWindows()


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