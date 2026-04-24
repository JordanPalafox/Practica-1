"""Grabacion de palabras a 16 kHz.

Flujo por clip:
    1) ENTER -> empieza a grabar
    2) ENTER -> detiene y guarda

Uso:
    python -m src.record                 # graba todas las palabras, 15 reps
    python -m src.record --word start    # solo una palabra
    python -m src.record --word stop --reps 15 --start-index 0
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.config import (
    RECORDINGS_DIR,
    REPETITIONS,
    SAMPLE_RATE,
    WORDS,
)

MAX_SECONDS = 10.0


def record_until_enter(sr: int) -> np.ndarray:
    """Graba desde que se llama hasta que el usuario presiona ENTER."""
    q: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        if status:
            print(f"  (aviso: {status})", file=sys.stderr)
        q.put(indata.copy())

    chunks: list[np.ndarray] = []

    def wait_for_enter() -> None:
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    t = threading.Thread(target=wait_for_enter, daemon=True)
    t.start()

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback):
        print("  GRABANDO... presiona ENTER para detener.")
        total_samples = 0
        max_samples = int(MAX_SECONDS * sr)
        while not stop_event.is_set():
            try:
                data = q.get(timeout=0.1)
                chunks.append(data)
                total_samples += data.shape[0]
                if total_samples >= max_samples:
                    print(f"  (limite de {MAX_SECONDS:.0f}s alcanzado)")
                    break
            except queue.Empty:
                continue

    while not q.empty():
        chunks.append(q.get())

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).flatten()


def save_clip(audio: np.ndarray, word: str, index: int) -> Path:
    out_dir = RECORDINGS_DIR / word
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{word}_{index:02d}.wav"
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def record_word(word: str, reps: int, start_index: int) -> None:
    print(f"\n=== Palabra: '{word}' ({reps} repeticiones) ===")
    i = start_index
    end = start_index + reps
    while i < end:
        prompt = f"\n[{i + 1}/{end}] ENTER para empezar a grabar '{word}' (o 'q' + ENTER para saltar): "
        choice = input(prompt).strip().lower()
        if choice == "q":
            print("  Saltando palabra.")
            return
        audio = record_until_enter(SAMPLE_RATE)

        if audio.size == 0:
            print("  No se capturo audio. Reintenta.")
            continue

        peak = float(np.max(np.abs(audio)))
        dur = audio.size / SAMPLE_RATE
        if peak < 1e-3 or dur < 0.2:
            print(f"  AVISO: clip invalido (peak={peak:.3f}, dur={dur:.2f}s). Reintenta.")
            continue

        path = save_clip(audio, word, i)
        print(f"  Guardado {path.name}  (dur={dur:.2f}s, peak={peak:.3f})")
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Grabador de palabras para VQ.")
    parser.add_argument("--word", type=str, default=None, help="Palabra a grabar.")
    parser.add_argument("--reps", type=int, default=REPETITIONS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    print(f"Tasa de muestreo: {SAMPLE_RATE} Hz  |  Corte manual con ENTER (max {MAX_SECONDS:.0f}s)")
    print(f"Dispositivo de entrada: {sd.query_devices(kind='input')['name']}")

    words = [args.word] if args.word else WORDS
    for w in words:
        if w not in WORDS:
            print(f"Palabra '{w}' no esta en la lista. Palabras: {WORDS}")
            continue
        record_word(w, args.reps, args.start_index)

    print("\nListo.")


if __name__ == "__main__":
    main()
