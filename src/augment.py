"""Aumento de datos: duplica cada grabacion con variantes ligeras.

La idea es darle mas frames al LBG sin tener que regrabar. Las variantes
introducen variabilidad acustica razonable sin cambiar la identidad de
la palabra:

- copia original
- ruido blanco a SNR ~30 dB
- ruido blanco a SNR ~20 dB
- leve ganancia (+/- 3 dB)
- micro desplazamiento temporal (shift 40 muestras = 2.5 ms)
- leve resample (+/- 3% duracion) para imitar variacion de tempo
"""
from __future__ import annotations

import numpy as np

_RNG = np.random.default_rng(1234)


def _add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    power = float(np.mean(signal**2)) + 1e-12
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = _RNG.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise


def _gain(signal: np.ndarray, db: float) -> np.ndarray:
    return signal * (10 ** (db / 20.0))


def _shift(signal: np.ndarray, samples: int) -> np.ndarray:
    if samples == 0:
        return signal
    if samples > 0:
        pad = np.zeros(samples, dtype=signal.dtype)
        return np.concatenate([pad, signal[:-samples]])
    s = -samples
    pad = np.zeros(s, dtype=signal.dtype)
    return np.concatenate([signal[s:], pad])


def _resample_linear(signal: np.ndarray, factor: float) -> np.ndarray:
    """Cambia la duracion por un factor (0.97 = 3% mas rapido)."""
    n_new = max(1, int(round(signal.size * factor)))
    x_old = np.linspace(0.0, 1.0, signal.size)
    x_new = np.linspace(0.0, 1.0, n_new)
    return np.interp(x_new, x_old, signal).astype(signal.dtype)


def augment_signal(signal: np.ndarray) -> list[np.ndarray]:
    """Devuelve la lista de variantes (incluye la original)."""
    variants = [
        signal,
        _add_noise(signal, snr_db=30.0),
        _add_noise(signal, snr_db=20.0),
        _gain(signal, db=3.0),
        _gain(signal, db=-3.0),
        _shift(signal, samples=40),
        _shift(signal, samples=-40),
        _resample_linear(signal, factor=0.97),
        _resample_linear(signal, factor=1.03),
    ]
    return variants
