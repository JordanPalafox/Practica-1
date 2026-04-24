"""Configuracion global de la practica."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"
CODEBOOKS_DIR = ROOT / "codebooks"
RESULTS_DIR = ROOT / "results"

SAMPLE_RATE = 16000
RECORD_SECONDS = 2.0
REPETITIONS = 15

PREEMPH_COEF = 0.95
FRAME_SIZE = 320
FRAME_SHIFT = 128
LPC_ORDER = 12

VAD_ENERGY_FACTOR = 3.0
VAD_MIN_FRAMES = 5

CODEBOOK_SIZES = [16, 32, 64]
TRAIN_PER_WORD = 10
TEST_PER_WORD = 5

WORDS = [
    "start",
    "stop",
    "lift",
    "drop",
    "forward",
    "backward",
    "left",
    "right",
    "pick",
    "place",
]

for d in (RECORDINGS_DIR, CODEBOOKS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
    for w in WORDS:
        (RECORDINGS_DIR / w).mkdir(parents=True, exist_ok=True)
