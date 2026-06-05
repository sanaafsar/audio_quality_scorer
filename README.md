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

Per-chunk scores are normalized by the 95th-percentile (`p95`) score of the
training data so the output is a relative anomaly score (~1.0 ≈ the boundary of
normal training audio).

## Project layout

```
audio_quality_scorer/
├── script.py          # Train models + score audio (main entry point)
├── prepare_data.py    # Build the clean training set (download + VAD + filtering)
├── requirements.txt   # Python dependencies
├── data/
│   ├── train/         # Clean training chunks (clean_*.wav)
│   └── test/          # Audio to score
├── raw_audio/         # Downloaded source audio (LibriSpeech, YouTube)
├── models/            # Saved models + normalization stats
│   ├── normal_model.pkl   # Mahalanobis mean + inverse covariance
│   ├── autoencoder.pt     # Spectrogram autoencoder weights
│   └── norm_stats.npy     # {p95, mean, std} of training scores
├── result/            # Saved JSON scoring outputs
└── cache/             # Hugging Face model cache (created on first run)
```

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

Trains the Mahalanobis embedding model and the spectrogram autoencoder on the
audio in `--train-dir`, then computes and saves normalization statistics
(`mean`, `std`, `p95`).

```bash
python script.py train --train-dir data/train
```

Artifacts are written to `models/`.

## Scoring audio

```bash
python script.py test --audio data/test/drunk.wav --output result/drunk.json
```

Output (also printed to console):

```json
{
  "audio_file": "data/test/drunk.wav",
  "duration_seconds": 3.0,
  "final_score": 0.646,
  "num_chunks": 1,
  "chunk_scores": [
    { "score": 0.646, "time": "0 sec" }
  ],
  "model_status": {
    "normal_model_loaded": true,
    "autoencoder_loaded": true
  }
}
```

`final_score` is the mean of the per-chunk normalized scores. Audio is split into
non-overlapping 3-second chunks; chunks shorter than 3 seconds are dropped.

## CLI reference

```
python script.py <mode> [options]

modes:
  train             Train models from a folder of clean .wav files
  test              Score a single audio file

options:
  --audio PATH      Audio file to score (required in test mode)
  --train-dir PATH  Training audio directory (default: data/train)
  --device {cpu,cuda}   Device (auto-detected if omitted)
  --output PATH     Write test results to a JSON file
```
