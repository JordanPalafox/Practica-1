"""Pre-enfasis y ventaneo Hamming."""
from __future__ import annotations

import numpy as np

from src.config import FRAME_SHIFT, FRAME_SIZE, PREEMPH_COEF


def preemphasis(signal: np.ndarray, coef: float = PREEMPH_COEF) -> np.ndarray:
    """Aplica Hp(z) = 1 - coef * z^-1."""
    if signal.size == 0:
        return signal
    out = np.empty_like(signal, dtype=np.float64)
    out[0] = signal[0]
    out[1:] = signal[1:] - coef * signal[:-1]
    return out


def frame_signal(
    signal: np.ndarray,
    frame_size: int = FRAME_SIZE,
    frame_shift: int = FRAME_SHIFT,
) -> np.ndarray:
    """Parte la senal en bloques solapados y aplica ventana Hamming.

    Devuelve matriz de (n_frames, frame_size).
    """
    if signal.size < frame_size:
        pad = np.zeros(frame_size - signal.size, dtype=signal.dtype)
        signal = np.concatenate([signal, pad])

    n_frames = 1 + (signal.size - frame_size) // frame_shift
    if n_frames <= 0:
        return np.zeros((0, frame_size))

    idx = (
        np.arange(frame_size)[None, :]
        + frame_shift * np.arange(n_frames)[:, None]
    )
    frames = signal[idx].astype(np.float64)
    window = np.hamming(frame_size)
    return frames * window[None, :]


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Energia log (dB aprox) por frame."""
    e = np.sum(frames**2, axis=1)
    return 10.0 * np.log10(e + 1e-12)
