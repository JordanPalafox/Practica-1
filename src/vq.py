"""Cuantizador vectorial LBG sobre LSF.

Se agrupa en el espacio LSF (Euclidiano). La distancia Itakura-Saito se
usa en la fase de reconocimiento comparando los LPC reconstruidos contra
el frame de prueba.
"""
from __future__ import annotations

import numpy as np


def _nearest_neighbors(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    diff = data[:, None, :] - codebook[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    return np.argmin(dist, axis=1)


def _update_centroids(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    dim = data.shape[1]
    cb = np.zeros((k, dim))
    for i in range(k):
        mask = labels == i
        if np.any(mask):
            cb[i] = data[mask].mean(axis=0)
        else:
            cb[i] = data[np.random.randint(0, data.shape[0])]
    return cb


def kmeans(
    data: np.ndarray, k: int, max_iter: int = 50, tol: float = 1e-4
) -> np.ndarray:
    n = data.shape[0]
    if n <= k:
        cb = np.zeros((k, data.shape[1]))
        cb[:n] = data
        for i in range(n, k):
            cb[i] = data[i % n] + 1e-3 * np.random.randn(data.shape[1])
        return cb

    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=k, replace=False)
    codebook = data[idx].copy()

    prev_dist = np.inf
    for _ in range(max_iter):
        labels = _nearest_neighbors(data, codebook)
        codebook = _update_centroids(data, labels, k)
        diff = data - codebook[labels]
        d = float(np.mean(np.sum(diff * diff, axis=1)))
        if abs(prev_dist - d) / (d + 1e-12) < tol:
            break
        prev_dist = d
    return codebook


def lbg(data: np.ndarray, target_size: int, epsilon: float = 0.01) -> np.ndarray:
    """Algoritmo LBG: duplica centroides hasta alcanzar target_size (potencia de 2)."""
    centroid = data.mean(axis=0, keepdims=True)
    codebook = centroid.copy()

    while codebook.shape[0] < target_size:
        new_cb = np.vstack([codebook * (1 + epsilon), codebook * (1 - epsilon)])
        codebook = kmeans(data, new_cb.shape[0], max_iter=30)
    return codebook[:target_size]


def quantization_distortion(data: np.ndarray, codebook: np.ndarray) -> float:
    labels = _nearest_neighbors(data, codebook)
    diff = data - codebook[labels]
    return float(np.mean(np.sum(diff * diff, axis=1)))
