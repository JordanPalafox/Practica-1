"""Pipeline completo de extraccion de features por archivo wav."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import LPC_ORDER
from src.endpoint import trim_signal
from src.lpc import autocorrelation, extract_lsf_matrix
from src.preprocess import frame_signal, preemphasis


def load_wav(path: Path) -> np.ndarray:
    signal, _ = sf.read(str(path))
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    return signal.astype(np.float64)


def extract_features(
    path: Path, order: int = LPC_ORDER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (lsf_frames, lpc_frames, autocorr_frames) para un wav."""
    signal = load_wav(path)
    pre = preemphasis(signal)
    pre = trim_signal(pre)
    frames = frame_signal(pre)
    lsf_mat, lpc_mat, _ = extract_lsf_matrix(frames, order)

    n = frames.shape[0]
    r_mat = np.zeros((n, order + 1))
    for i in range(n):
        r_mat[i] = autocorrelation(frames[i], order)

    return lsf_mat, lpc_mat, r_mat
