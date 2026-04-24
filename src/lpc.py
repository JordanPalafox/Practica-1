"""LPC de orden 12 + conversiones LPC <-> LSF y distancia Itakura-Saito."""
from __future__ import annotations

import numpy as np

from src.config import LPC_ORDER


def autocorrelation(frame: np.ndarray, order: int) -> np.ndarray:
    n = frame.size
    r = np.zeros(order + 1)
    for k in range(order + 1):
        r[k] = float(np.dot(frame[: n - k], frame[k:]))
    return r


def levinson_durbin(r: np.ndarray, order: int) -> tuple[np.ndarray, float]:
    """Resuelve Toeplitz para LPC. Devuelve (a, gain) con a[0] = 1."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = float(r[0]) + 1e-12
    a_prev = a.copy()

    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a_prev[j] * r[i - j]
        k = -(r[i] + acc) / e
        a = a_prev.copy()
        a[i] = k
        for j in range(1, i):
            a[j] = a_prev[j] + k * a_prev[i - j]
        e = (1.0 - k * k) * e
        a_prev = a.copy()

    gain = float(np.sqrt(max(e, 1e-12)))
    return a, gain


def lpc_from_frame(frame: np.ndarray, order: int = LPC_ORDER) -> tuple[np.ndarray, float]:
    r = autocorrelation(frame, order)
    return levinson_durbin(r, order)


def lpc_to_lsf(a: np.ndarray) -> np.ndarray:
    """LPC (a[0]=1) a Line Spectral Frequencies en rad.

    P(z) = A(z) + z^-(p+1) A(z^-1)
    Q(z) = A(z) - z^-(p+1) A(z^-1)
    Las raices de P y Q yacen en el circulo unitario; sus angulos (0..pi)
    son los LSF, entrelazados.
    """
    p = a.size - 1
    a_rev = a[::-1]
    P = np.concatenate([a, [0.0]]) + np.concatenate([[0.0], a_rev])
    Q = np.concatenate([a, [0.0]]) - np.concatenate([[0.0], a_rev])

    roots_p = np.roots(P)
    roots_q = np.roots(Q)

    angles_p = np.angle(roots_p)
    angles_q = np.angle(roots_q)
    angles_p = angles_p[(angles_p > 1e-6) & (angles_p < np.pi - 1e-6)]
    angles_q = angles_q[(angles_q > 1e-6) & (angles_q < np.pi - 1e-6)]

    lsf = np.sort(np.concatenate([angles_p, angles_q]))
    if lsf.size < p:
        lsf = np.pad(lsf, (0, p - lsf.size), constant_values=np.pi - 1e-3)
    return lsf[:p]


def lsf_to_lpc(lsf: np.ndarray) -> np.ndarray:
    """LSF a coeficientes LPC (a[0] = 1). Orden par.

    Descomposicion:
        P(z) = A(z) + z^-(p+1) A(z^-1)  -> raiz en z=-1, factor (1 + z^-1)
        Q(z) = A(z) - z^-(p+1) A(z^-1)  -> raiz en z=+1, factor (1 - z^-1)
    Los LSF ordenados se intercalan como P, Q, P, Q, ...
    A(z) = (P(z) + Q(z)) / 2; la ultima entrada es 0 por construccion.
    """
    p = lsf.size
    if p % 2 != 0:
        raise ValueError("Solo se soporta orden LPC par.")
    lsf_sorted = np.sort(lsf)

    p_angles = lsf_sorted[0::2]
    q_angles = lsf_sorted[1::2]

    P = np.array([1.0, 1.0])
    for theta in p_angles:
        P = np.convolve(P, [1.0, -2.0 * np.cos(theta), 1.0])

    Q = np.array([1.0, -1.0])
    for theta in q_angles:
        Q = np.convolve(Q, [1.0, -2.0 * np.cos(theta), 1.0])

    A = 0.5 * (P + Q)
    return A[: p + 1]


def extract_lsf_matrix(
    frames: np.ndarray, order: int = LPC_ORDER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (lsf_matrix, lpc_matrix, gains) para una coleccion de frames."""
    n = frames.shape[0]
    lsf_mat = np.zeros((n, order))
    lpc_mat = np.zeros((n, order + 1))
    gains = np.zeros(n)
    for i in range(n):
        a, g = lpc_from_frame(frames[i], order)
        lpc_mat[i] = a
        gains[i] = g
        lsf_mat[i] = lpc_to_lsf(a)
    return lsf_mat, lpc_mat, gains


def itakura_saito_distance(a_test: np.ndarray, a_ref: np.ndarray, r_test: np.ndarray) -> float:
    """Distancia Itakura-Saito entre dos modelos LPC sobre la autocorrelacion de test.

    d_IS(a_ref, a_test) = (a_ref^T R_test a_ref) / (a_test^T R_test a_test) - 1

    donde R_test es la matriz de autocorrelacion (Toeplitz) del frame de test.
    """
    p = a_test.size - 1
    R = _toeplitz_from_autocorr(r_test[: p + 1])
    num = float(a_ref @ R @ a_ref)
    den = float(a_test @ R @ a_test) + 1e-12
    return num / den - 1.0


def _toeplitz_from_autocorr(r: np.ndarray) -> np.ndarray:
    n = r.size
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = r[abs(i - j)]
    return M
