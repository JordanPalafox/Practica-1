"""Deteccion de inicio/final de palabra por energia."""
from __future__ import annotations

import numpy as np

from src.config import FRAME_SHIFT, FRAME_SIZE, VAD_ENERGY_FACTOR, VAD_MIN_FRAMES
from src.preprocess import frame_energy, frame_signal


def detect_endpoints(
    signal: np.ndarray,
    noise_frames: int = 10,
    factor: float = VAD_ENERGY_FACTOR,
    min_frames: int = VAD_MIN_FRAMES,
) -> tuple[int, int]:
    """Devuelve (start_frame, end_frame) sobre la particion en frames.

    Usa un umbral adaptativo: media_ruido + factor * std_ruido (en dB)
    estimado de los primeros noise_frames bloques.
    """
    frames = frame_signal(signal)
    if frames.shape[0] == 0:
        return 0, 0

    energy = frame_energy(frames)
    noise_frames = max(1, min(noise_frames, energy.size))
    noise_mean = float(np.mean(energy[:noise_frames]))
    noise_std = float(np.std(energy[:noise_frames]) + 1e-6)
    threshold = noise_mean + factor * noise_std

    active = energy > threshold
    active = _smooth(active, min_frames)

    if not np.any(active):
        return 0, frames.shape[0]

    idx = np.where(active)[0]
    return int(idx[0]), int(idx[-1]) + 1


def trim_signal(signal: np.ndarray) -> np.ndarray:
    """Recorta la senal al segmento activo usando detect_endpoints."""
    start_f, end_f = detect_endpoints(signal)
    if end_f <= start_f:
        return signal
    start_s = start_f * FRAME_SHIFT
    end_s = min(signal.size, end_f * FRAME_SHIFT + FRAME_SIZE)
    return signal[start_s:end_s]


def _smooth(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Elimina activaciones cortas y rellena huecos pequenos."""
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if i > 0 and j < n and (j - i) < min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out
