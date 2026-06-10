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

The scoring pipeline combines three per-chunk signals into a weighted final score.
Each component is **z-scored against per-component training mean/std before
weighting** so the weights are meaningful (the raw embedding anomaly is ~1000×
larger than the signal score and would otherwise dominate):

```
# AudioQualityDetector.chunk_components -> raw (anom, sig, ae)
# AudioQualityDetector.combine          -> standardize + weight
z_x   = (x - stats["x_mean"]) / stats["x_std"]      for x in {anom, sig, ae}
final = 0.6 * z_anom + 0.3 * z_sig + 0.1 * z_ae     # WEIGHTS class constant
score = final / stats["p95"]                        # normalized by training p95
```

`evaluate(audio, stats)` takes the full `norm_stats` dict (per-component mean/std +
`p95`), not just `p95`. Per-component stats and `p95` are computed at train time in
a two-pass loop in `__main__` (pass 1 collects raw components, pass 2 standardizes
to derive `p95`).

Key components:
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

## Style

Match the existing code: type hints on signatures, Google-style docstrings, and
the `# ===` banner comments separating sections. Keep changes minimal and in the
idiom of the surrounding file.
