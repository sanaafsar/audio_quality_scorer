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
python train.py --train-dir data/train

# Score a file locally (inference test harness)
python model.py --audio data/test/drunk.wav --output result/out.json
```

There is no test suite, linter config, or build step in this repo.

## Architecture

Training and inference are **separate modules**; no package structure.

- `base.py` — the `BaseQualityModel` ABC, copied from the downstream media-quality
  pipeline (Chain-of-Responsibility). Defines the contract inference must satisfy:
  `load()`, `predict(*args, **kwargs) -> ScoreDict`, `model_name`, `is_loaded`.
  **Do not change it to fit this repo** — it mirrors the pipeline's interface.
- `model.py` — inference + shared building blocks. Self-contained drop-in for the
  pipeline (only needs `base.py` + the `models/` artifacts).
- `train.py` — training entry point. Imports the building blocks from `model.py`.
- `prepare_data.py` — dataset construction.

### `model.py` — inference + shared core

`AudioQualityModel(BaseQualityModel)` is the pipeline drop-in:
- `predict(self, audio: np.ndarray, sample_rate: int = 16000) -> ScoreDict` —
  narrows the base signature to a decoded waveform. Averages multi-channel to mono,
  resamples to 16 kHz, scores, and returns `{overall, anomaly_score,
  max_chunk_anomaly, num_chunks}` (all floats). `overall = 1/(1+max(anomaly,0))` in
  `[0,1]` (1.0 = best) per the base convention; the raw fields keep "higher = worse".
- `load()` reads `normal_model.pkl`, `autoencoder.pt` (optional), and
  `norm_stats.npy` from `models_dir`; raises `FileNotFoundError` if the normal model
  or stats are missing. `predict()` raises `RuntimeError` if called before `load()`.

The scoring core `AudioQualityDetector` lives here and is reused by `train.py`, so
training and inference cannot drift. It combines three per-chunk signals, each
**z-scored against per-component training mean/std before weighting** (the raw
embedding anomaly is ~1000× larger than the signal score and would otherwise
dominate):

```
# AudioQualityDetector.chunk_components -> raw (anom, sig, ae)
# AudioQualityDetector.combine          -> standardize + weight
z_x   = (x - stats["x_mean"]) / stats["x_std"]      for x in {anom, sig, ae}
final = 0.6 * z_anom + 0.3 * z_sig + 0.1 * z_ae     # WEIGHTS class constant
score = final / stats["p95"]                        # normalized by training p95
```

`evaluate(audio, stats)` takes the full `norm_stats` dict (per-component mean/std +
`p95`), not just `p95`.

Key components (all in `model.py`):
- `Wav2Vec2Embedder` — `facebook/wav2vec2-base` → 768-dim mean-pooled embedding per
  chunk. Loads/caches the HF model under `cache/`.
- `NormalModel` — Mahalanobis distance (mean + inverse covariance) fit on clean
  embeddings. Persisted as `models/normal_model.pkl`.
- `SpectrogramAutoencoder` — small MLP autoencoder over 128-mel spectrograms;
  reconstruction MSE is the anomaly signal. Persisted as `models/autoencoder.pt`.
- `compute_signal_features` / `signal_score` — clipping, RMS, silence heuristics.
- `norm_stats.npy` — `{p95, mean, std, anom_mean, anom_std, sig_mean, sig_std,
  ae_mean, ae_std}`. The per-component mean/std drive standardization; `p95`
  rescales the final score. Must be regenerated (retrain) whenever the scoring
  components or weights change.

### `train.py` — training

`train_embedding_model` + `train_autoencoder` fit the models; `compute_norm_stats`
runs the two-pass loop (pass 1 collects raw components, pass 2 standardizes to
derive `p95`). `save_*` persist the artifacts. CLI: `--train-dir`, `--models-dir`,
`--device`.

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
- `chunk_audio` exists in *both* `model.py` and `prepare_data.py` with slightly
  different signatures — keep them behaviorally consistent if you touch one.
- Audio, models (`*.pt`, `*.pkl`), and JSON results are gitignored. Do not commit
  files from `data/`, `raw_audio/`, `models/`, or `result/`.
- Models must be trained before inference works — `AudioQualityModel.load()` raises
  `FileNotFoundError` if `models/normal_model.pkl` or `norm_stats.npy` is missing,
  and `predict()` raises `RuntimeError` if called before `load()`.
- When dropping `model.py` into the pipeline, fix up the `from base import ...`
  line to the pipeline's import path for `BaseQualityModel`.

## Style

Match the existing code: type hints on signatures, Google-style docstrings, and
the `# ===` banner comments separating sections. Keep changes minimal and in the
idiom of the surrounding file.
