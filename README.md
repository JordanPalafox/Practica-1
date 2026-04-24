# Practica 1 - Reconocimiento de voz con Cuantizadores Vectoriales

Sistema de reconocimiento de palabras aisladas para comandos de un robot
de almacen, basado en LPC orden 12, LSF y cuantizacion vectorial LBG
con distancia Itakura-Saito.

## Palabras (10 comandos de almacen)

`start, stop, lift, drop, forward, backward, left, right, pick, place`

## Requisitos

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> En macOS, si `sounddevice` falla, instala PortAudio: `brew install portaudio`.

## Flujo

Todos los scripts se corren desde la raiz del proyecto con `-m` para que
los imports relativos funcionen.

### 1) Grabar (16 kHz, 15 repeticiones por palabra)

```bash
python -m src.record                   # graba todas las palabras
python -m src.record --word start      # solo una palabra
python -m src.record --list-devices    # ver microfonos
```

Los archivos se guardan en `recordings/<palabra>/<palabra>_NN.wav`
(indice 00..14). Los primeros 10 se usan para entrenar, los 5 restantes
para prueba.

### 2) Entrenar codebooks LBG con k = 16, 32, 64

```bash
python -m src.train
```

Cada codebook se guarda como `codebooks/<palabra>_k<K>.npz` con:

- `lsf`: K centroides en el dominio LSF
- `lpc`: los mismos centroides convertidos a coeficientes LPC
- `distortion`: distorsion promedio al final del LBG

### 3) Evaluar y generar matrices de confusion

```bash
python -m src.evaluate
```

Genera `results/confusion_k{16,32,64}.png` y `.csv`, y un resumen con
el accuracy por tamano de codebook.

## Pipeline implementado

1. **Pre-enfasis**: `Hp(z) = 1 - 0.95 z^-1` (`src/preprocess.py`).
2. **Ventaneo**: Hamming de 320 puntos (20 ms @ 16 kHz) cada 128
   muestras (8 ms de shift) (`src/preprocess.py`).
3. **Endpointing**: umbral adaptativo sobre la energia logaritmica
   por frame usando los primeros 10 bloques como estimacion de ruido
   (`src/endpoint.py`).
4. **LPC orden 12**: autocorrelacion + Levinson-Durbin
   (`src/lpc.py`).
5. **VQ**: clustering LBG en el espacio LSF (Euclidiano)
   (`src/vq.py`, `src/train.py`). El uso de LSF garantiza estabilidad
   al promediar centroides.
6. **Reconocimiento**: cada frame de prueba se compara contra cada
   codevector de cada codebook con distancia Itakura-Saito
   `d_IS(a_ref, a_test) = (a_ref^T R_test a_ref) / (a_test^T R_test a_test) - 1`;
   la palabra con menor distancia acumulada gana (`src/evaluate.py`).

## Pregunta del enunciado: tamano del codebook

El script `evaluate.py` imprime el accuracy para `k = 16, 32, 64` y
genera una matriz de confusion por cada uno. La respuesta se da en
funcion de tus propios datos:

- Si `k=16` ya alcanza accuracy alto (>90%), esta bien para este
  vocabulario corto.
- Si subir a 32 o 64 aumenta el accuracy de forma notable, el
  vocabulario tiene mas variabilidad y conviene un codebook mayor;
  el costo computacional crece linealmente con K.
- En general para 10 palabras y un solo hablante, `k=32` suele ser el
  mejor compromiso entre resolucion y sobreajuste.

## Estructura del repo

```
Practica 1/
├── requirements.txt
├── README.md
├── recordings/<palabra>/<palabra>_NN.wav
├── codebooks/<palabra>_k{16,32,64}.npz
├── results/confusion_k{16,32,64}.{png,csv}
└── src/
    ├── config.py
    ├── record.py
    ├── preprocess.py
    ├── endpoint.py
    ├── lpc.py
    ├── vq.py
    ├── features.py
    ├── train.py
    └── evaluate.py
```
