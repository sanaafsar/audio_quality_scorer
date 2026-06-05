# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

An unsupervised audio-quality scorer for speech. It learns a model of "good"
speech from clean training audio, then scores new audio by deviation from that
normal. Higher score = lower quality / more anomalous. There are no labels — it is
anomaly detection, not classification.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build clean training data (LibriSpeech + optional YouTube, VAD-filtered)
python prepare_data.py            # writes chunks to dataset_clean/

# Train (writes models/ artifacts)
python script.py train --train-dir data/train

# Score a file
python script.py test --audio data/test/drunk.wav --output result/out.json
```

There is no test suite, linter config, or build step in this repo.

## Architecture

Two scripts, no package structure.

### `script.py` — training + scoring

The scoring pipeline combines three per-chunk signals into a weighted final score
(`AudioQualityDetector.evaluate`):

```
final = 0.6 * mahalanobis_anomaly + 0.3 * signal_score + 0.1 * ae_recon_error
score = final / p95          # normalized by training p95
```

Key components:
- `Wav2Vec2Embedder` — `facebook/wav2vec2-base` → 768-dim mean-pooled embedding per
  chunk. Loads/caches the HF model under `cache/`.
- `NormalModel` — Mahalanobis distance (mean + inverse covariance) fit on clean
  embeddings. Persisted as `models/normal_model.pkl`.
- `SpectrogramAutoencoder` — small MLP autoencoder over 128-mel spectrograms;
  reconstruction MSE is the anomaly signal. Persisted as `models/autoencoder.pt`.
- `compute_signal_features` / `signal_score` — clipping, RMS, silence heuristics.
- `norm_stats.npy` — `{p95, mean, std}` of training scores, used to normalize at
  test time.

### `prepare_data.py` — dataset construction

Downloads LibriSpeech `dev-clean` and optional YouTube audio into `raw_audio/`,
chunks everything into 3-second segments, and keeps only chunks that pass WebRTC
VAD (`is_speech`) and signal-quality checks (`is_good_chunk`). Clean chunks land in
`dataset_clean/`.

## Conventions & invariants

- **Sample rate is 16 kHz everywhere** (`SR = 16000`). All audio is resampled on
  load; embeddings and spectrograms assume it.
- **Chunks are 3 seconds, non-overlapping.** Trailing chunks shorter than 3s are
  dropped (`chunk_audio`). This is intentional and matches between training and
  scoring.
- `chunk_audio` exists in *both* scripts with slightly different signatures — keep
  them behaviorally consistent if you touch one.
- Audio, models (`*.pt`, `*.pkl`), and JSON results are gitignored. Do not commit
  files from `data/`, `raw_audio/`, `models/`, or `result/`.
- Models must be trained before `test` mode works — it raises `FileNotFoundError`
  if `models/normal_model.pkl` or `norm_stats.npy` is missing.

## Known rough edges (verify before relying on them)

- `script.py` references `Path(...)` in `Wav2Vec2Embedder.__init__` but only
  imports `os`/`glob`, not `pathlib.Path`. If you trigger that path, add
  `from pathlib import Path`.
- `train_autoencoder` accumulates `total_loss` with `int(loss.item())` and prints
  inside the inner loop — loss reporting is coarse/noisy, not a correctness signal.

## Style

Match the existing code: type hints on signatures, Google-style docstrings, and
the `# ===` banner comments separating sections. Keep changes minimal and in the
idiom of the surrounding file.
