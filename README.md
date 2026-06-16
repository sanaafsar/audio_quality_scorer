# Audio Quality Scorer

An unsupervised audio-quality scoring system for speech. It learns what "good"
speech sounds like from a corpus of clean audio, then scores new audio by how far
it deviates from that learned normal. Higher scores mean lower quality / more
anomalous audio.

The system combines three complementary signals per audio chunk:

1. **Embedding anomaly** — Wav2Vec2 embeddings scored with a Mahalanobis distance
   model fit on clean speech (weight `0.6`).
2. **Signal features** — clipping ratio, RMS level, and silence ratio (weight `0.3`).
3. **Spectrogram reconstruction error** — a mel-spectrogram autoencoder trained on
   clean speech; high reconstruction error flags unusual audio (weight `0.1`).

Because the three raw signals live on very different scales (the embedding anomaly
is orders of magnitude larger than the others), each component is **standardized**
(z-scored against per-component mean/std measured on the training set) *before* the
weights are applied. This keeps the weights meaningful instead of letting the
largest-scale component dominate. The standardized, weighted score is then
normalized by the 95th-percentile (`p95`) of the training scores, so the output is
a relative anomaly score (~1.0 ≈ the boundary of normal training audio).

## Project layout

Training and inference are separate. The **inference** path is a single,
self-contained module (`model.py`) that subclasses `BaseQualityModel`, so it can
be dropped straight into the media-quality pipeline.

```
audio_quality_scorer/
├── base.py            # BaseQualityModel contract (shared with the ML pipeline)
├── model.py           # Inference: AudioQualityModel(BaseQualityModel) + shared building blocks
├── train.py           # Training entry point (writes models/ artifacts)
├── prepare_data.py    # Build the clean training set (download + VAD + filtering)
├── requirements.txt   # Python dependencies
├── data/
│   ├── train/         # Clean training chunks (clean_*.wav)
│   └── test/          # Audio to score
├── raw_audio/         # Downloaded source audio (LibriSpeech, YouTube)
├── models/            # Saved models + normalization stats
│   ├── normal_model.pkl   # Mahalanobis mean + inverse covariance
│   ├── autoencoder.pt     # Spectrogram autoencoder weights
│   └── norm_stats.npy     # p95 + per-component (anom/sig/ae) mean & std
├── result/            # Saved JSON scoring outputs
└── cache/             # Hugging Face model cache (created on first run)
```

`model.py` holds the shared building blocks (embedder, autoencoder, Mahalanobis
model, feature/chunking helpers) and the inference wrapper; `train.py` imports
those building blocks and adds the training/persistence logic. The two stay in
sync because the scoring core (`AudioQualityDetector`) lives in `model.py` and is
reused by both.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires `ffmpeg` for the data pipeline (provided via `imageio-ffmpeg`).

## Preparing training data

`prepare_data.py` builds a clean speech dataset by downloading LibriSpeech
(`dev-clean`) and an optional list of YouTube videos, then filtering audio into
3-second chunks that pass voice-activity detection (WebRTC VAD) and signal-quality
checks (RMS, silence ratio, clipping).

```bash
python prepare_data.py
```

Output clean chunks are written to `dataset_clean/`. Move/copy the chunks you want
to train on into `data/train/`.

> **Note:** YouTube downloads use `pytubefix` with OAuth. On first run it prints a
> Google device-auth URL and code; the token is cached for subsequent runs.

## Training

`train.py` trains the Mahalanobis embedding model and the spectrogram autoencoder
on the audio in `--train-dir`, then computes and saves the normalization
statistics (per-component mean/std + `p95`). All three artifacts are written to
`--models-dir` (default `models/`).

```bash
python train.py --train-dir data/train
```

## Scoring audio

### As a pipeline model (the real integration)

`model.py` exposes `AudioQualityModel`, a `BaseQualityModel` subclass. The
pipeline decodes a media file and hands `predict()` a waveform + sample rate:

```python
from model import AudioQualityModel

model = AudioQualityModel(models_dir="models")
model.load()                                  # load weights once

scores = model.predict(waveform, sample_rate)  # waveform: np.ndarray
# {
#   "overall": 0.29,            # quality in [0, 1], 1.0 = best
#   "anomaly_score": 2.46,      # mean per-chunk anomaly, higher = worse (~1.0 = edge of normal)
#   "max_chunk_anomaly": 5.39,  # worst single 3s chunk (less diluted than the mean)
#   "num_chunks": 42.0
# }
```

`predict` averages multi-channel input to mono and resamples to 16 kHz when
`sample_rate != 16000`. Audio is split into non-overlapping 3-second chunks;
chunks shorter than 3 seconds are dropped (so audio under 3s returns
`overall = 1.0`, `num_chunks = 0`). `overall` follows the `BaseQualityModel`
convention (`[0, 1]`, 1.0 = best); the raw anomaly fields keep the native
"higher = worse" semantics for debugging.

### Local test harness (CLI)

`model.py` has a `__main__` that decodes a file and runs the same `predict`,
useful for spot-checking before wiring into the pipeline:

```bash
python model.py --audio data/test/drunk.wav --output result/drunk.json
```

## CLI reference

```
# Training
python train.py [--train-dir DIR] [--models-dir DIR] [--device {cpu,cuda}]

# Local inference test harness
python model.py --audio PATH [--models-dir DIR] [--device {cpu,cuda}] [--output PATH]
```
