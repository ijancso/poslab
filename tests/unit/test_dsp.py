"""Spectrum and low-pass filter (poslab.dsp)."""
from __future__ import annotations

import numpy as np
import pytest

from poslab.dsp import attenuation_db, dominant_frequency, group_delay, lowpass, psd

FS = 100.0
T = np.arange(0, 60, 1 / FS)


def amplitude(x: np.ndarray, f: float, part: slice = slice(None)) -> float:
    """Amplitude of the ``f`` Hz component of ``x`` = signal over T[part]
    (least-squares fit of sin + cos)."""
    t = T[part]
    a = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    coef, *_ = np.linalg.lstsq(a, x, rcond=None)
    return float(np.hypot(*coef))


def test_dominant_frequency_finds_27hz_tone_in_noise_and_motion():
    rng = np.random.default_rng(0)
    motion = 0.3 * np.sign(np.sin(2 * np.pi * 0.02 * T))      # slow accel steps (vehicle)
    x = motion + 0.3 * np.sin(2 * np.pi * 27.0 * T + 1.0) + rng.normal(0, 0.1, T.size)
    assert dominant_frequency(x, FS) == pytest.approx(27.0, abs=0.5)


@pytest.mark.parametrize("f0", [12.3, 27.0, 31.7])
def test_dominant_frequency_sub_bin_accuracy(f0):
    assert dominant_frequency(np.sin(2 * np.pi * f0 * T), FS) == pytest.approx(f0, abs=0.05)


def test_dominant_frequency_rejects_fmin_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        dominant_frequency(np.sin(T), FS, fmin=60.0)


def test_psd_peak_power():
    f, p = psd(np.sin(2 * np.pi * 10.0 * T), FS)
    assert f[np.argmax(p)] == pytest.approx(10.0, abs=FS / 1024)


@pytest.mark.parametrize("causal", [True, False])
def test_lowpass_removes_vibration_and_keeps_motion(causal):
    """5 Hz 4th-order low-pass: 27 Hz vibration down > 20 dB, 1 Hz motion passes."""
    vib = np.sin(2 * np.pi * 27.0 * T)
    motion = np.sin(2 * np.pi * 1.0 * T)
    y_vib = lowpass(vib, FS, cutoff=5.0, causal=causal)
    y_mot = lowpass(motion, FS, cutoff=5.0, causal=causal)
    skip = slice(200, -200)                                    # ignore edge effects
    att_db = 20 * np.log10(amplitude(y_vib[skip], 27.0, skip) / 1.0)
    assert att_db < -20.0
    assert amplitude(y_mot[skip], 1.0, skip) > 0.95


def test_design_attenuation_matches_measured():
    assert attenuation_db(FS, 5.0, 27.0) < -50.0
    assert attenuation_db(FS, 5.0, 1.0) > -0.1


def test_offline_filter_has_zero_phase_causal_filter_lags_by_group_delay():
    x = np.sin(2 * np.pi * 0.5 * T)
    lags = np.arange(-50, 51)

    def best_lag(y: np.ndarray) -> int:
        core = slice(500, -500)
        return int(lags[np.argmax([np.dot(x[core], np.roll(y, -k)[core]) for k in lags])])

    assert best_lag(lowpass(x, FS, 5.0, causal=False)) == 0
    gd = group_delay(FS, 5.0, order=4, freq=0.5)
    assert 0.07 < gd < 0.10                                    # ~83 ms, see lowpass docstring
    assert best_lag(lowpass(x, FS, 5.0, causal=True)) == round(gd * FS)


def test_causal_filter_starts_without_transient():
    y = lowpass(np.full(100, 2.5), FS, 5.0, causal=True)
    np.testing.assert_allclose(y, 2.5)


def test_invalid_cutoff():
    with pytest.raises(ValueError, match="cutoff"):
        lowpass(np.zeros(10), FS, cutoff=50.0)
