"""Entrena un codebook LBG por palabra para varios tamanos.

Uso:
    python -m src.train
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.augment import augment_signal
from src.config import (
    CODEBOOK_SIZES,
    CODEBOOKS_DIR,
    LPC_ORDER,
    RECORDINGS_DIR,
    TRAIN_PER_WORD,
    WORDS,
)
from src.features import load_wav, signal_to_features
from src.lpc import lsf_to_lpc
from src.vq import lbg, quantization_distortion


def list_training_files(word: str) -> list[Path]:
    files = sorted((RECORDINGS_DIR / word).glob(f"{word}_*.wav"))
    if len(files) < TRAIN_PER_WORD:
        raise FileNotFoundError(
            f"Se requieren al menos {TRAIN_PER_WORD} archivos para '{word}', hay {len(files)}."
        )
    return files[:TRAIN_PER_WORD]


def collect_lsf(files: list[Path]) -> np.ndarray:
    """Extrae LSF de cada archivo y tambien de variantes aumentadas."""
    chunks: list[np.ndarray] = []
    n_variants = 0
    for f in files:
        signal = load_wav(f)
        for variant in augment_signal(signal):
            lsf, _, _ = signal_to_features(variant)
            if lsf.shape[0]:
                chunks.append(lsf)
                n_variants += 1
    if not chunks:
        raise RuntimeError("No se extrajeron frames validos.")
    print(f"   variantes usadas: {n_variants} (de {len(files)} archivos originales)")
    return np.vstack(chunks)


def train_word(word: str) -> dict[int, dict[str, np.ndarray]]:
    files = list_training_files(word)
    lsf_all = collect_lsf(files)
    print(f"[{word}] frames de entrenamiento: {lsf_all.shape[0]}")

    result: dict[int, dict[str, np.ndarray]] = {}
    for size in CODEBOOK_SIZES:
        cb_lsf = lbg(lsf_all, size)
        cb_lpc = np.array([lsf_to_lpc(lsf) for lsf in cb_lsf])
        distortion = quantization_distortion(lsf_all, cb_lsf)
        print(f"   codebook k={size:>3}  distortion_LSF={distortion:.4f}")
        result[size] = {"lsf": cb_lsf, "lpc": cb_lpc, "distortion": np.array(distortion)}
    return result


def main() -> None:
    CODEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for word in WORDS:
        word_dir = RECORDINGS_DIR / word
        if not word_dir.exists() or len(list(word_dir.glob("*.wav"))) < TRAIN_PER_WORD:
            print(f"[SKIP] '{word}' no tiene suficientes grabaciones.")
            continue
        trained = train_word(word)
        for size, data in trained.items():
            out = CODEBOOKS_DIR / f"{word}_k{size}.npz"
            np.savez(out, lsf=data["lsf"], lpc=data["lpc"], distortion=data["distortion"])
        print(f"[{word}] guardado en {CODEBOOKS_DIR}")

    print(f"\nLPC order = {LPC_ORDER}, tamanos = {CODEBOOK_SIZES}")
    print("Listo.")


if __name__ == "__main__":
    main()
