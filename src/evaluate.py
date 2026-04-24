"""Evalua el reconocimiento calculando la matriz de confusion.

Uso:
    python -m src.evaluate
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    CODEBOOK_SIZES,
    CODEBOOKS_DIR,
    RECORDINGS_DIR,
    RESULTS_DIR,
    TEST_PER_WORD,
    TRAIN_PER_WORD,
    WORDS,
)
from src.features import extract_features
from src.lpc import itakura_saito_distance


def load_codebooks(size: int) -> dict[str, np.ndarray]:
    cb: dict[str, np.ndarray] = {}
    for word in WORDS:
        path = CODEBOOKS_DIR / f"{word}_k{size}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Falta codebook entrenado: {path}")
        data = np.load(path)
        cb[word] = data["lpc"]
    return cb


def list_test_files(word: str) -> list[Path]:
    files = sorted((RECORDINGS_DIR / word).glob(f"{word}_*.wav"))
    return files[TRAIN_PER_WORD : TRAIN_PER_WORD + TEST_PER_WORD]


def score_file_against_word(
    lpc_frames: np.ndarray,
    r_frames: np.ndarray,
    codebook_lpc: np.ndarray,
) -> float:
    """Distancia acumulada: por frame de prueba, minimo IS sobre los codevectors."""
    total = 0.0
    count = 0
    for i in range(lpc_frames.shape[0]):
        a_test = lpc_frames[i]
        r_test = r_frames[i]
        best = np.inf
        for j in range(codebook_lpc.shape[0]):
            a_ref = codebook_lpc[j]
            d = itakura_saito_distance(a_test, a_ref, r_test)
            if d < best:
                best = d
        total += best
        count += 1
    return total / max(count, 1)


def classify_file(
    lpc_frames: np.ndarray,
    r_frames: np.ndarray,
    codebooks: dict[str, np.ndarray],
) -> tuple[str, dict[str, float]]:
    scores = {
        w: score_file_against_word(lpc_frames, r_frames, cb)
        for w, cb in codebooks.items()
    }
    pred = min(scores, key=lambda k: scores[k])
    return pred, scores


def evaluate_size(size: int) -> np.ndarray:
    codebooks = load_codebooks(size)
    n = len(WORDS)
    conf = np.zeros((n, n), dtype=int)
    idx = {w: i for i, w in enumerate(WORDS)}

    for true_word in WORDS:
        files = list_test_files(true_word)
        if not files:
            print(f"[WARN] sin archivos de prueba para '{true_word}'")
            continue
        for f in files:
            _, lpc_frames, r_frames = extract_features(f)
            if lpc_frames.shape[0] == 0:
                continue
            pred, _ = classify_file(lpc_frames, r_frames, codebooks)
            conf[idx[true_word], idx[pred]] += 1
    return conf


def plot_confusion(conf: np.ndarray, size: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(WORDS)))
    ax.set_yticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, rotation=45, ha="right")
    ax.set_yticklabels(WORDS)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (codebook k={size})")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            color = "white" if conf[i, j] > conf.max() / 2 else "black"
            ax.text(j, i, int(conf[i, j]), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_csv(conf: np.ndarray, size: int, out_path: Path) -> None:
    header = "," + ",".join(WORDS)
    lines = [header]
    for i, w in enumerate(WORDS):
        row = ",".join(str(int(x)) for x in conf[i])
        lines.append(f"{w},{row}")
    out_path.write_text("\n".join(lines))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[tuple[int, float]] = []

    for size in CODEBOOK_SIZES:
        print(f"\n=== Evaluando codebook k={size} ===")
        conf = evaluate_size(size)
        total = int(conf.sum())
        correct = int(np.trace(conf))
        acc = correct / total if total else 0.0
        summary.append((size, acc))
        print(conf)
        print(f"Accuracy (k={size}): {acc:.3f}  ({correct}/{total})")

        plot_confusion(conf, size, RESULTS_DIR / f"confusion_k{size}.png")
        save_csv(conf, size, RESULTS_DIR / f"confusion_k{size}.csv")

    print("\n=== Resumen ===")
    for size, acc in summary:
        print(f"  k={size:>3}  accuracy = {acc:.3f}")
    best = max(summary, key=lambda x: x[1])
    print(f"\nMejor tamano de codebook: k={best[0]} (accuracy={best[1]:.3f})")


if __name__ == "__main__":
    main()
