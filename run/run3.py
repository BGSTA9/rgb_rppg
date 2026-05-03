"""
Medical-Grade rPPG Cardiac Monitor
====================================
Full signal processing pipeline — accuracy first, latency second.

  Stage 1  Multi-ROI Extraction
             Forehead (top 25%) + bilateral cheeks (mid 40%, outer 30% each)
             YCrCb skin-pixel mask applied per ROI before spatial averaging
             Pixel-count weighted fusion of ROI signals

  Stage 2  rPPG Signal Extraction  (both run, results fused)
             POS  — Plane-Orthogonal-to-Skin  (de Haan & Jeanne, TBME 2013)
             CHROM— Chrominance method         (de Haan & Jeanne, TBME 2013)

  Stage 3  Signal Preprocessing
             Linear detrend  (remove DC drift)
             Zero-phase 4th-order Butterworth bandpass  0.67–4.0 Hz (40–240 BPM)

  Stage 4  HR Estimation
             Welch PSD, 8× zero-padded for fine frequency resolution
             Harmonic-aware peak selection (rejects sub-harmonic artefacts)
             SNR = peak-window power / rest-of-band power
             POS and CHROM estimates fused by their individual SNR weights

  Stage 5  Adaptive Kalman Filter
             Measurement noise R ∝ 1/SNR  (bad signal → high noise → slow update)
             Physiological gate: rejects measurements >20 BPM from current state
             Reports uncertainty (±σ BPM) alongside point estimate

  Stage 6  HRV  (Heart Rate Variability)
             Zero-phase peak detection on bandpass signal
             IBI sequence, physiological plausibility gate (250–2000 ms)
             RMSSD, SDNN, pNN50

References
----------
  de Haan G., Jeanne V. (2013). Robust pulse rate from chrominance-based rPPG.
    IEEE Trans. Biomed. Eng. 60(10):2878–2886.
  Wang W. et al. (2017). Algorithmic principles of remote PPG.
    IEEE Trans. Biomed. Eng. 64(7):1479–1491.
"""

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, NamedTuple

import cv2
import numpy as np
from scipy import signal as sp_signal

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rppg_medical")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    # Camera
    camera_index: int = 0

    # Signal processing
    target_fps: float = 30.0           # Expected camera FPS (recalculated at runtime)
    window_seconds: float = 12.0       # Analysis window (longer → better freq resolution)
    min_window_seconds: float = 5.0    # Minimum before first report
    pipeline_interval: float = 0.5     # Background pipeline step interval (seconds)

    # Physiological band
    hr_freq_min: float = 0.67          # 40 BPM
    hr_freq_max: float = 4.0           # 240 BPM

    # Welch PSD
    welch_segment_seconds: float = 4.0 # nperseg length
    welch_nfft_mult: int = 8           # zero-padding factor → fine resolution

    # Signal quality gate
    min_snr_db: float = 1.5            # below this → Kalman ignores measurement

    # Kalman
    kalman_process_noise: float = 0.3  # BPM²/step process variance
    kalman_init_P: float = 100.0       # Initial uncertainty
    kalman_outlier_bpm: float = 20.0   # Gate width (reject if |z - x| > this)

    # HRV
    hrv_min_beats: int = 6

    # Face detector
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_min_size: tuple = (80, 80)
    box_ema_alpha: float = 0.25

    # HR zone thresholds (BPM)
    zone_resting_max: float = 60.0
    zone_normal_max: float = 100.0
    zone_elevated_max: float = 140.0

    # Display
    window_title: str = "rPPG Medical Monitor"
    quit_key: str = "q"
    font: int = cv2.FONT_HERSHEY_SIMPLEX


CFG = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────
class HRResult(NamedTuple):
    hr_bpm: float               # Kalman-filtered HR
    hr_raw_bpm: float           # Direct spectral HR (fused POS+CHROM)
    snr_db: float               # Spectral SNR
    uncertainty_bpm: float      # Kalman ±σ
    confidence: float           # 0–1 gate signal
    hrv_rmssd: Optional[float]  # ms
    hrv_sdnn: Optional[float]   # ms
    hrv_pnn50: Optional[float]  # %
    dominant_algo: str          # "POS" | "CHROM" | "FUSED"


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Face detector with EMA box smoothing
# ──────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    """
    OpenCV Haar cascade, EMA-smoothed bounding box.
    Runs every frame in the main loop — fast, non-blocking.
    """

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"Cascade not found: {path}")
        self._smooth: Optional[np.ndarray] = None   # [x, y, w, h] floats

    def detect(self, frame_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        """Returns (x1, y1, x2, y2) or None."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=CFG.face_scale_factor,
            minNeighbors=CFG.face_min_neighbors,
            minSize=CFG.face_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            self._smooth = None
            return None
        face = max(faces, key=lambda f: f[2] * f[3]).astype(float)
        if self._smooth is None:
            self._smooth = face.copy()
        else:
            a = CFG.box_ema_alpha
            self._smooth = a * face + (1 - a) * self._smooth
        x, y, w, h = self._smooth.astype(int)
        return x, y, x + w, y + h


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Multi-ROI skin-masked RGB extractor
# ──────────────────────────────────────────────────────────────────────────────
class ROIExtractor:
    """
    Splits the face bounding box into three physiologically meaningful ROIs,
    applies a YCrCb skin-pixel mask, and returns pixel-count-weighted mean RGB.

    ROI layout (relative to face box):
      Forehead  — top 25% of height, horizontal center 60%
      L. cheek  — rows 40–80% height, cols  5–35% width
      R. cheek  — rows 40–80% height, cols 65–95% width
    """

    # YCrCb skin-colour bounds (Peer et al. 2003)
    _SKIN_CR = (133, 173)
    _SKIN_CB = (77, 127)

    @classmethod
    def _skin_mask(cls, roi_bgr: np.ndarray) -> np.ndarray:
        if roi_bgr.size == 0:
            return np.zeros((0,), dtype=bool)
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        mask = (
            (cr >= cls._SKIN_CR[0]) & (cr <= cls._SKIN_CR[1]) &
            (cb >= cls._SKIN_CB[0]) & (cb <= cls._SKIN_CB[1])
        )
        return mask

    @classmethod
    def _roi_mean(cls, frame_bgr: np.ndarray, x1, y1, x2, y2) -> tuple[Optional[np.ndarray], int]:
        """Returns (rgb_mean [3,], pixel_count) after skin masking."""
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0
        mask = cls._skin_mask(roi)
        n = mask.sum()
        if n < 20:                               # too few skin pixels → unreliable
            return None, 0
        rgb = roi[:, :, ::-1].astype(float)     # BGR → RGB
        means = rgb[mask].mean(axis=0)           # shape (3,)
        return means, int(n)

    @classmethod
    def extract(
        cls,
        frame_bgr: np.ndarray,
        face_box: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Returns pixel-count-weighted mean RGB across all ROIs, shape (3,).
        Returns None if no ROI has enough skin pixels.
        """
        x1, y1, x2, y2 = face_box
        fx, fy = x2 - x1, y2 - y1          # face width, height

        rois = [
            # forehead
            (x1 + int(0.2 * fx), y1,
             x1 + int(0.8 * fx), y1 + int(0.25 * fy)),
            # left cheek
            (x1 + int(0.05 * fx), y1 + int(0.40 * fy),
             x1 + int(0.35 * fx), y1 + int(0.80 * fy)),
            # right cheek
            (x1 + int(0.65 * fx), y1 + int(0.40 * fy),
             x1 + int(0.95 * fx), y1 + int(0.80 * fy)),
        ]

        weighted_sum = np.zeros(3, dtype=float)
        total_weight = 0
        for rx1, ry1, rx2, ry2 in rois:
            rgb_mean, count = cls._roi_mean(frame_bgr, rx1, ry1, rx2, ry2)
            if rgb_mean is not None:
                weighted_sum += rgb_mean * count
                total_weight += count

        if total_weight == 0:
            return None
        return weighted_sum / total_weight


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — rPPG algorithms
# ──────────────────────────────────────────────────────────────────────────────
def pos_signal(rgb_buf: np.ndarray) -> np.ndarray:
    """
    Plane-Orthogonal-to-Skin (POS) — de Haan & Jeanne, TBME 2013.
    rgb_buf: (N, 3)  →  rPPG signal (N,)
    """
    C = rgb_buf.T                                       # (3, N)
    mu = C.mean(axis=1, keepdims=True) + 1e-8
    Cn = C / mu                                         # temporal normalisation

    # POS projection  H = [[0, 1, -1], [-2, 1, 1]]
    H1 = Cn[1] - Cn[2]                                 # G_n - B_n
    H2 = -2 * Cn[0] + Cn[1] + Cn[2]                   # -2R_n + G_n + B_n

    alpha = (H1.std() + 1e-8) / (H2.std() + 1e-8)
    return H1 + alpha * H2


def chrom_signal(rgb_buf: np.ndarray) -> np.ndarray:
    """
    Chrominance method (CHROM) — de Haan & Jeanne, TBME 2013.
    rgb_buf: (N, 3)  →  rPPG signal (N,)
    """
    C = rgb_buf.T
    mu = C.mean(axis=1, keepdims=True) + 1e-8
    Cn = C / mu
    R, G, B = Cn[0], Cn[1], Cn[2]

    Xc = 3 * R - 2 * G
    Yc = 1.5 * R + G - 1.5 * B

    alpha = (Xc.std() + 1e-8) / (Yc.std() + 1e-8)
    return Xc - alpha * Yc


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Zero-phase bandpass filter
# ──────────────────────────────────────────────────────────────────────────────
def bandpass(sig: np.ndarray, fs: float) -> np.ndarray:
    """4th-order zero-phase Butterworth bandpass 0.67–4.0 Hz."""
    nyq = fs / 2.0
    lo = CFG.hr_freq_min / nyq
    hi = CFG.hr_freq_max / nyq
    # Clamp to avoid numerical issues
    lo = np.clip(lo, 1e-4, 0.99)
    hi = np.clip(hi, lo + 1e-4, 0.999)
    b, a = sp_signal.butter(4, [lo, hi], btype="bandpass")
    # filtfilt requires at least padlen+1 samples
    if len(sig) < 15:
        return sig - sig.mean()
    return sp_signal.filtfilt(b, a, sig - sig.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Welch PSD HR estimator
# ──────────────────────────────────────────────────────────────────────────────
def welch_hr(sig: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Returns (hr_bpm, snr_db) using Welch PSD.
    snr_db: peak-window power vs. rest of cardiac band.
    Returns (0.0, -inf) on failure.
    """
    nperseg = min(len(sig), int(fs * CFG.welch_segment_seconds))
    nfft = max(nperseg * CFG.welch_nfft_mult, 1024)
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg, nfft=nfft)

    band = (freqs >= CFG.hr_freq_min) & (freqs <= CFG.hr_freq_max)
    bf, bp = freqs[band], psd[band]
    if len(bp) == 0:
        return 0.0, float("-inf")

    # --- harmonic-aware peak selection ---
    # Raw peak
    pk_idx = np.argmax(bp)
    hr_hz = bf[pk_idx]

    # Check if raw peak is actually a 2nd harmonic of a sub-harmonic candidate
    # (e.g. fundamental at 0.9 Hz, we mistakenly pick 1.8 Hz)
    half_hz = hr_hz / 2.0
    if half_hz >= CFG.hr_freq_min:
        half_candidates = np.abs(bf - half_hz) < 0.05
        if half_candidates.any():
            half_power = bp[half_candidates].max()
            if half_power > 0.6 * bp[pk_idx]:   # sub-harmonic is strong → prefer it
                hr_hz = bf[half_candidates][np.argmax(bp[half_candidates])]

    hr_bpm = hr_hz * 60.0

    # SNR: power within ±0.1 Hz of peak vs rest of band
    peak_win = np.abs(bf - hr_hz) <= 0.1
    sig_power = bp[peak_win].sum() + 1e-12
    noise_power = bp[~peak_win].sum() + 1e-12
    snr = sig_power / noise_power
    snr_db = 10 * np.log10(snr)

    return hr_bpm, snr_db


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — Adaptive Kalman filter (1D, HR tracking)
# ──────────────────────────────────────────────────────────────────────────────
class KalmanHR:
    """
    Constant-velocity 1D Kalman filter for HR.
    Measurement noise scales inversely with SNR → low-quality signals
    contribute weakly to the state estimate.
    """

    def __init__(self) -> None:
        self._x: Optional[float] = None    # state estimate (BPM)
        self._P: float = CFG.kalman_init_P # state uncertainty (BPM²)
        self._Q: float = CFG.kalman_process_noise

    @property
    def uncertainty_bpm(self) -> float:
        return float(np.sqrt(max(self._P, 0.0)))

    def update(self, z: float, snr_db: float) -> Optional[float]:
        """
        z       — spectral HR measurement (BPM)
        snr_db  — measurement quality
        Returns Kalman-filtered HR, or None if gate rejected.
        """
        if snr_db < CFG.min_snr_db:
            # Predict only (no measurement update)
            self._P += self._Q
            return self._x

        # Physiological outlier gate
        if self._x is not None and abs(z - self._x) > CFG.kalman_outlier_bpm:
            log.warning("Kalman gate: z=%.1f vs x=%.1f — rejected", z, self._x)
            self._P += self._Q
            return self._x

        # Adaptive R: SNR 1.5 dB → R=50, SNR 10 dB → R≈3
        R = 50.0 / max(10 ** (snr_db / 10.0), 0.1)

        if self._x is None:
            self._x = z
            self._P = R
            return self._x

        # Predict
        P_pred = self._P + self._Q

        # Update
        K = P_pred / (P_pred + R)
        self._x = self._x + K * (z - self._x)
        self._P = (1.0 - K) * P_pred

        return self._x


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — HRV analysis
# ──────────────────────────────────────────────────────────────────────────────
def compute_hrv(sig_filtered: np.ndarray, fs: float) -> dict:
    """
    Detects peaks in the filtered rPPG signal, extracts inter-beat intervals,
    and computes RMSSD, SDNN, pNN50.
    Returns empty dict if too few beats detected.
    """
    if len(sig_filtered) < int(fs * 3):
        return {}

    # Normalise
    s = sig_filtered.copy()
    s = (s - s.mean()) / (s.std() + 1e-8)

    min_dist = int(fs * 0.35)          # 0.35 s → max 171 BPM
    peaks, _ = sp_signal.find_peaks(
        s,
        distance=min_dist,
        prominence=0.4,
        width=max(2, int(fs * 0.04)),  # min width ≈ 40 ms
    )

    if len(peaks) < CFG.hrv_min_beats:
        return {}

    ibi_ms = np.diff(peaks) / fs * 1000.0       # inter-beat intervals in ms

    # Physiological plausibility gate
    ibi_ms = ibi_ms[(ibi_ms >= 250) & (ibi_ms <= 2000)]

    if len(ibi_ms) < 3:
        return {}

    diff_ibi = np.diff(ibi_ms)
    rmssd = float(np.sqrt(np.mean(diff_ibi ** 2)))
    sdnn = float(np.std(ibi_ms, ddof=1))
    pnn50 = float(100.0 * np.sum(np.abs(diff_ibi) > 50) / len(diff_ibi))

    return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50, "n_beats": len(peaks)}


# ──────────────────────────────────────────────────────────────────────────────
# Signal buffer — thread-safe, non-blocking push
# ──────────────────────────────────────────────────────────────────────────────
class SignalBuffer:
    """
    Circular buffer of (rgb_vec, timestamp) tuples.
    push() uses try-lock so the main loop NEVER blocks.
    snapshot() returns a consistent copy for the pipeline thread.
    """

    def __init__(self, maxlen: int) -> None:
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, rgb: np.ndarray, ts: float) -> None:
        if self._lock.acquire(blocking=False):
            try:
                self._buf.append((rgb.copy(), ts))
            finally:
                self._lock.release()
        # If lock is held by pipeline thread, silently skip (< 1 frame loss)

    def snapshot(self) -> list:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ──────────────────────────────────────────────────────────────────────────────
class rPPGPipeline:
    """
    Runs Stages 2-6 on a buffer snapshot.
    Called by the background worker thread.
    """

    def __init__(self) -> None:
        self._kalman = KalmanHR()

    def run(self, buf_snapshot: list) -> Optional[HRResult]:
        if len(buf_snapshot) < 2:
            return None

        rgb_arr = np.array([r for r, _ in buf_snapshot])   # (N, 3)
        ts_arr  = np.array([t for _, t in buf_snapshot])   # (N,)

        # Measured FPS from actual timestamps
        dt = np.diff(ts_arr)
        dt = dt[dt > 0]
        fs = float(1.0 / dt.mean()) if len(dt) > 0 else CFG.target_fps
        fs = np.clip(fs, 5.0, 120.0)                       # sanity clamp

        N = len(rgb_arr)
        min_n = int(fs * CFG.min_window_seconds)
        if N < min_n:
            return None                                     # not enough data yet

        # ── Stage 2: POS + CHROM signals ─────────────────────────────────────
        sig_pos   = pos_signal(rgb_arr)
        sig_chrom = chrom_signal(rgb_arr)

        # ── Stage 3: bandpass filter ──────────────────────────────────────────
        filt_pos   = bandpass(sig_pos, fs)
        filt_chrom = bandpass(sig_chrom, fs)

        # ── Stage 4: Welch HR + SNR for each algorithm ────────────────────────
        hr_pos,   snr_pos   = welch_hr(filt_pos,   fs)
        hr_chrom, snr_chrom = welch_hr(filt_chrom, fs)

        # SNR-weighted fusion of the two estimates
        w_pos   = max(0.0, snr_pos)
        w_chrom = max(0.0, snr_chrom)
        w_total = w_pos + w_chrom

        if w_total < 1e-6:
            hr_fused = 0.5 * (hr_pos + hr_chrom)
            snr_fused = max(snr_pos, snr_chrom)
            algo = "FUSED"
        else:
            hr_fused  = (w_pos * hr_pos + w_chrom * hr_chrom) / w_total
            snr_fused = (w_pos * snr_pos + w_chrom * snr_chrom) / w_total
            algo = "POS" if w_pos > w_chrom else "CHROM"

        # ── Stage 5: Kalman update ────────────────────────────────────────────
        hr_kalman = self._kalman.update(hr_fused, snr_fused)
        if hr_kalman is None:
            hr_kalman = hr_fused

        uncertainty = self._kalman.uncertainty_bpm
        confidence  = float(np.clip((snr_fused - CFG.min_snr_db) / 8.0, 0.0, 1.0))

        # ── Stage 6: HRV ─────────────────────────────────────────────────────
        # Use whichever filtered signal has higher SNR
        best_filt = filt_pos if snr_pos >= snr_chrom else filt_chrom
        hrv = compute_hrv(best_filt, fs)

        return HRResult(
            hr_bpm        = hr_kalman,
            hr_raw_bpm    = hr_fused,
            snr_db        = snr_fused,
            uncertainty_bpm = uncertainty,
            confidence    = confidence,
            hrv_rmssd     = hrv.get("rmssd"),
            hrv_sdnn      = hrv.get("sdnn"),
            hrv_pnn50     = hrv.get("pnn50"),
            dominant_algo = algo,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Background pipeline worker thread
# ──────────────────────────────────────────────────────────────────────────────
class PipelineWorker:
    """
    Runs rPPGPipeline.run() in a daemon thread every `pipeline_interval` seconds.
    Main loop reads results via a Lock — zero render-loop blocking.
    """

    def __init__(self, buf: SignalBuffer) -> None:
        self._buf      = buf
        self._pipeline = rPPGPipeline()
        self._lock     = threading.Lock()
        self._result: Optional[HRResult] = None
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, daemon=True, name="rppg-pipeline"
        )

    def start(self) -> None:
        self._thread.start()
        log.info("Pipeline worker started (step %.2fs)", CFG.pipeline_interval)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            snapshot = self._buf.snapshot()
            try:
                result = self._pipeline.run(snapshot)
                if result is not None:
                    with self._lock:
                        self._result = result
                    log.info(
                        "HR %.1f±%.1f BPM | raw %.1f | SNR %.1f dB | conf %.0f%% | %s%s",
                        result.hr_bpm, result.uncertainty_bpm,
                        result.hr_raw_bpm, result.snr_db,
                        result.confidence * 100,
                        result.dominant_algo,
                        (f" | RMSSD {result.hrv_rmssd:.1f}ms"
                         if result.hrv_rmssd is not None else ""),
                    )
            except Exception as exc:
                log.warning("Pipeline error: %s", exc, exc_info=True)

            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, CFG.pipeline_interval - elapsed))

    @property
    def result(self) -> Optional[HRResult]:
        with self._lock:
            return self._result


# ──────────────────────────────────────────────────────────────────────────────
# Minimal overlay (signal-focused, not cosmetic)
# ──────────────────────────────────────────────────────────────────────────────
def hr_zone(bpm: float) -> tuple[str, tuple[int, int, int]]:
    if bpm < CFG.zone_resting_max:  return "Resting",  (180, 180, 180)
    if bpm < CFG.zone_normal_max:   return "Normal",   (80, 200, 80)
    if bpm < CFG.zone_elevated_max: return "Elevated", (0, 165, 255)
    return                                 "High",     (60, 60, 220)


def draw_overlay(
    frame: np.ndarray,
    result: Optional[HRResult],
    face_box: Optional[tuple],
    fps: float,
    buf_len: int,
) -> None:
    h, w = frame.shape[:2]
    font = CFG.font

    # Face bounding box
    if face_box is not None:
        x1, y1, x2, y2 = face_box
        box_color = hr_zone(result.hr_bpm)[1] if result else (80, 200, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Semi-transparent panel
    slab = frame.copy()
    cv2.rectangle(slab, (6, 6), (260, 175), (15, 15, 15), -1)
    cv2.addWeighted(slab, 0.6, frame, 0.4, 0, frame)

    # Face / collecting status
    status = "Face detected" if face_box else "No face — searching…"
    dot_col = (80, 200, 80) if face_box else (60, 60, 220)
    cv2.circle(frame, (18, 22), 5, dot_col, -1)
    cv2.putText(frame, status, (28, 26), font, 0.42, (200, 200, 200), 1)

    if result is None or result.snr_db < CFG.min_snr_db:
        # No valid reading yet
        pct = int(100 * buf_len / max(
            int(CFG.target_fps * CFG.min_window_seconds), 1))
        cv2.putText(frame, "-- BPM", (10, 72), font, 1.4, (130, 130, 130), 2)
        cv2.putText(frame, f"Collecting signal {min(pct,100)}%",
                    (10, 95), font, 0.42, (130, 130, 130), 1)
    else:
        zone_label, zone_color = hr_zone(result.hr_bpm)

        # Large BPM
        cv2.putText(frame, f"{result.hr_bpm:.0f}", (10, 72), font, 1.9, zone_color, 3)
        cv2.putText(frame, "BPM", (105, 72), font, 0.55, (180, 180, 180), 1)

        # Zone + confidence bar
        conf_pct = int(result.confidence * 100)
        cv2.putText(frame, f"{zone_label}  conf {conf_pct}%",
                    (10, 92), font, 0.42, zone_color, 1)

        # SNR + uncertainty
        cv2.putText(frame,
                    f"SNR {result.snr_db:.1f}dB  ±{result.uncertainty_bpm:.1f}BPM  [{result.dominant_algo}]",
                    (10, 110), font, 0.38, (160, 160, 160), 1)

        # Raw spectral HR
        cv2.putText(frame, f"Spectral raw: {result.hr_raw_bpm:.1f} BPM",
                    (10, 126), font, 0.38, (120, 120, 120), 1)

        # HRV block
        if result.hrv_rmssd is not None:
            cv2.putText(frame,
                        f"RMSSD {result.hrv_rmssd:.0f}ms  SDNN {result.hrv_sdnn:.0f}ms  pNN50 {result.hrv_pnn50:.0f}%",
                        (10, 144), font, 0.36, (140, 200, 255), 1)

    # FPS + buffer depth
    info = f"FPS {fps:.0f}  buf {buf_len}fr"
    (iw, _), _ = cv2.getTextSize(info, font, 0.38, 1)
    cv2.putText(frame, info, (w - iw - 8, h - 8), font, 0.38, (100, 100, 100), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cap = cv2.VideoCapture(CFG.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CFG.camera_index}")
    detector    = FaceDetector()
    buf_maxlen  = int(CFG.target_fps * CFG.window_seconds)
    buf         = SignalBuffer(maxlen=buf_maxlen)
    worker      = PipelineWorker(buf)
    fps_times: deque = deque(maxlen=60)

    log.info("Medical-grade rPPG monitor — press '%s' to quit.", CFG.quit_key)
    worker.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame capture failed.")
                break

            # ── FPS ───────────────────────────────────────────────────────
            now = time.perf_counter()
            fps_times.append(now)
            fps = (len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-9)
                   if len(fps_times) > 1 else 0.0)

            # ── Face detection ────────────────────────────────────────────
            face_box = detector.detect(frame)

            # ── Signal extraction (main loop, ~µs) ───────────────────────
            if face_box is not None:
                rgb = ROIExtractor.extract(frame, face_box)
                if rgb is not None:
                    buf.push(rgb, now)

            # ── Read pipeline result (non-blocking) ───────────────────────
            result = worker.result

            # ── Draw ──────────────────────────────────────────────────────
            draw_overlay(frame, result, face_box, fps, len(buf))

            cv2.imshow(CFG.window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord(CFG.quit_key):
                log.info("Quit key pressed.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        log.info("Shut down cleanly.")


if __name__ == "__main__":
    main()