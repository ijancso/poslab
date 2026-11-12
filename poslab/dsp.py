"""Signal processing: spectrum (Welch PSD) and Butterworth low-pass filtering.

Typical use: find the engine vibration frequency on the accelerometer, then low-pass the
acceleration before it enters the fusion so the vibration does not leak into the estimate.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def psd(x: np.ndarray, fs: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of ``x`` sampled at ``fs`` [Hz].

    Welch averages the periodograms of overlapping (50 %), Hann-windowed segments: less
    variance than a single FFT, at the cost of frequency resolution fs / nperseg.
    Returns (frequencies [Hz], PSD [unit^2/Hz]). The mean is removed first.
    """
    x = np.asarray(x, float)
    return sps.welch(x, fs=fs, nperseg=min(nperseg, len(x)), detrend="constant")


def dominant_frequency(x: np.ndarray, fs: float, fmin: float = 0.5,
                       nperseg: int = 1024) -> float:
    """Frequency [Hz] of the highest Welch PSD peak at or above ``fmin``.

    ``fmin`` skips the near-DC bins, where vehicle motion (speed changes, turns) lives.
    The peak is refined by parabolic interpolation of the log-PSD over the three bins
    around the maximum, which gives sub-bin accuracy for a clean tone.
    """
    f, p = psd(x, fs, nperseg)
    band = np.flatnonzero(f >= fmin)
    if band.size == 0:
        raise ValueError(f"fmin={fmin} Hz is above the Nyquist frequency {fs / 2} Hz")
    k = band[np.argmax(p[band])]
    if 0 < k < len(p) - 1 and np.all(p[k - 1:k + 2] > 0):
        a, b, c = np.log(p[k - 1:k + 2])
        denom = a - 2 * b + c
        offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float(f[k] + offset * (f[1] - f[0]))
    return float(f[k])


def butter_lowpass(fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Butterworth low-pass design as second-order sections (numerically robust)."""
    if not 0 < cutoff < fs / 2:
        raise ValueError(f"cutoff {cutoff} Hz must be between 0 and fs/2 = {fs / 2} Hz")
    return sps.butter(order, cutoff, btype="low", fs=fs, output="sos")


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4,
            causal: bool = True) -> np.ndarray:
    """Butterworth low-pass filter.

    causal=True  -> ``sosfilt``: uses only past samples, so it can run in real time inside a
                    filter loop (this is what the fusion must use). Price: phase lag. For a
                    Butterworth filter the low-frequency group delay is roughly
                    order * 0.65 / (2 pi cutoff) s; e.g. 4th order at 5 Hz -> ~83 ms, i.e. the
                    filtered acceleration describes the vehicle ~0.08 s ago. Lower cutoff =
                    more vibration rejection but more delay; see ``group_delay``.
    causal=False -> ``sosfiltfilt``: forward + backward pass, zero phase, squared magnitude
                    response. Needs the whole signal (offline analysis/plots only).
    The causal filter starts in steady state at x[0] (``sosfilt_zi``) to avoid a start-up
    transient.
    """
    x = np.asarray(x, float)
    sos = butter_lowpass(fs, cutoff, order)
    if not causal:
        return sps.sosfiltfilt(sos, x)
    zi = sps.sosfilt_zi(sos) * (x[0] if len(x) else 0.0)
    y, _ = sps.sosfilt(sos, x, zi=zi)
    return y


def group_delay(fs: float, cutoff: float, order: int = 4, freq: float = 0.5) -> float:
    """Group delay [s] of the causal low-pass at ``freq`` [Hz] (default: vehicle-motion band)."""
    b, a = sps.sos2tf(butter_lowpass(fs, cutoff, order))
    _, gd = sps.group_delay((b, a), w=[freq], fs=fs)
    return float(gd[0] / fs)


def attenuation_db(fs: float, cutoff: float, freq: float, order: int = 4) -> float:
    """Magnitude response [dB] of the low-pass at ``freq`` (negative = attenuation)."""
    _, h = sps.sosfreqz(butter_lowpass(fs, cutoff, order), worN=[freq], fs=fs)
    return float(20 * np.log10(np.abs(h[0])))
