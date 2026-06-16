"""Training entry point for the audio-quality scorer.

Trains the two models and computes the normalization statistics consumed at
inference time by ``model.AudioQualityModel``:

    python train.py --train-dir data/train

Writes three artifacts to ``--models-dir`` (default ``models/``):
    - normal_model.pkl   Mahalanobis mean + inverse covariance
    - autoencoder.pt     spectrogram autoencoder weights
    - norm_stats.npy     p95 + per-component (anom/sig/ae) mean & std

The shared model/feature code lives in ``model.py`` so training and inference
stay in sync.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray

from model import (
    MODELS_DIR,
    AudioQualityDetector,
    NormalModel,
    SpectrogramAutoencoder,
    Wav2Vec2Embedder,
    chunk_audio,
    get_spectrogram,
)


# =========================
# MODEL PERSISTENCE (training side)
# =========================
def ensure_models_dir(models_dir: str = MODELS_DIR) -> None:
    """Creates the models directory if it doesn't exist."""
    os.makedirs(models_dir, exist_ok=True)


def save_normal_model(model: NormalModel, models_dir: str = MODELS_DIR, filename: str = "normal_model.pkl") -> None:
    """Saves the normal model (mean and covariance inverse) to disk.

    Args:
        model (NormalModel): Trained normal model to save.
        models_dir (str): Output directory. Defaults to 'models'.
        filename (str): Output filename. Defaults to 'normal_model.pkl'.
    """
    ensure_models_dir(models_dir)
    filepath: str = os.path.join(models_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'mean': model.mean, 'cov_inv': model.cov_inv}, f)
    print(f"Normal model saved to {filepath}")


def save_autoencoder(model: SpectrogramAutoencoder, models_dir: str = MODELS_DIR, filename: str = "autoencoder.pt") -> None:
    """Saves the autoencoder model to disk.

    Args:
        model (SpectrogramAutoencoder): Trained autoencoder model to save.
        models_dir (str): Output directory. Defaults to 'models'.
        filename (str): Output filename. Defaults to 'autoencoder.pt'.
    """
    ensure_models_dir(models_dir)
    filepath: str = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Autoencoder model saved to {filepath}")


# =========================
# TRAIN FUNCTIONS
# =========================
def train_embedding_model(audio_folder: str) -> Tuple[Wav2Vec2Embedder, NormalModel]:
    """Trains the embedding-based anomaly detection model.

    Extracts embeddings from all audio files in a folder and fits a Mahalanobis
    distance model to detect anomalies.

    Args:
        audio_folder (str): Path to folder containing training audio files (.wav).

    Returns:
        Tuple[Wav2Vec2Embedder, NormalModel]: Trained embedder and anomaly detector.

    Raises:
        ValueError: If no usable training chunks are found.
    """
    embedder: Wav2Vec2Embedder = Wav2Vec2Embedder()
    embeddings: List[ndarray] = []

    for f in glob.glob(os.path.join(audio_folder, "*.wav")):
        audio: ndarray = embedder.load_audio(f)
        chunks: List[ndarray] = chunk_audio(audio)
        for c in chunks:
            emb: ndarray = embedder.get_embedding(c)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(
            f"No training embeddings produced from '{audio_folder}'. Ensure it "
            f"contains .wav files at least 3s long (shorter clips yield no chunks)."
        )

    model: NormalModel = NormalModel()
    model.fit(np.array(embeddings))

    return embedder, model


def train_autoencoder(audio_folder: str, device: str = "cpu", epochs: int = 5) -> SpectrogramAutoencoder:
    """Trains the spectrogram autoencoder.

    Learns to reconstruct mel-spectrograms from good audio. Reconstruction error
    serves as a quality indicator.

    Args:
        audio_folder (str): Path to folder containing training audio files (.wav).
        device (str): Device for training ('cpu' or 'cuda'). Defaults to 'cpu'.
        epochs (int): Number of training epochs. Defaults to 5.

    Returns:
        SpectrogramAutoencoder: Trained autoencoder model.
    """
    import librosa

    model: SpectrogramAutoencoder = SpectrogramAutoencoder().to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn: nn.MSELoss = nn.MSELoss()

    data: List[ndarray] = []
    for f in glob.glob(os.path.join(audio_folder, "*.wav")):
        audio, _ = librosa.load(f, sr=16000)
        chunks: List[ndarray] = chunk_audio(audio)
        for c in chunks:
            spec: ndarray = get_spectrogram(c)
            data.append(spec)

    for epoch in range(epochs):
        total_loss: float = 0.0
        for spec in data:
            spec_tensor: torch.Tensor = torch.tensor(spec, dtype=torch.float32).to(device)
            recon: torch.Tensor = model(spec_tensor)
            loss: torch.Tensor = loss_fn(recon, spec_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss: float = total_loss / len(data) if data else 0.0
        print(f"Epoch {epoch+1}, Loss: {mean_loss:.4f}")

    return model


def compute_norm_stats(detector: AudioQualityDetector, train_dir: str) -> Dict[str, Any]:
    """Computes per-component standardization stats and the combined p95.

    Two passes over the training chunks:
      1. Collect the raw (anom, sig, ae) components and derive per-component
         mean/std (used to z-score each component at scoring time).
      2. Standardize + weight each chunk and take the 95th percentile, which
         rescales test scores to the familiar ~1.0 range.

    Args:
        detector (AudioQualityDetector): Detector wrapping the trained models.
        train_dir (str): Directory with the clean training .wav files.

    Returns:
        Dict[str, Any]: norm_stats dict with 'p95', 'mean', 'std' and per-component
            '<name>_mean'/'<name>_std' for 'anom', 'sig', 'ae'.
    """
    # Pass 1: collect the RAW per-component values over every training chunk.
    anom_raw: List[float] = []
    sig_raw: List[float] = []
    ae_raw: List[float] = []
    for f in glob.glob(os.path.join(train_dir, "*.wav")):
        audio = detector.embedder.load_audio(f)
        for c in chunk_audio(audio):
            anom, sig, ae = detector.chunk_components(c)
            anom_raw.append(anom)
            sig_raw.append(sig)
            ae_raw.append(ae)

    if not anom_raw:
        raise ValueError(
            f"No training chunks found in '{train_dir}' for normalization stats."
        )

    def _mean_std(values: List[float]) -> Tuple[float, float]:
        """Returns (mean, sample-std) with std floored to 1.0 to avoid divide-by-zero.

        Uses ddof=1 (sample std) for consistency with ``np.cov`` in NormalModel.
        """
        arr = np.asarray(values, dtype=np.float64)
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return float(arr.mean()), std if std > 1e-12 else 1.0

    anom_mean, anom_std = _mean_std(anom_raw)
    sig_mean, sig_std = _mean_std(sig_raw)
    ae_mean, ae_std = _mean_std(ae_raw)

    component_stats: Dict[str, Any] = {
        "anom_mean": anom_mean, "anom_std": anom_std,
        "sig_mean": sig_mean, "sig_std": sig_std,
        "ae_mean": ae_mean, "ae_std": ae_std,
    }

    # Pass 2: standardize + weight each chunk, then derive the p95.
    finals: List[float] = [
        detector.combine(a, s, e, component_stats)
        for a, s, e in zip(anom_raw, sig_raw, ae_raw)
    ]
    p95 = float(np.percentile(finals, 95))
    p99 = float(np.percentile(finals, 99))

    norm_scores = [fv / p95 for fv in finals]
    mean_score = float(np.mean(norm_scores))
    std_score = float(np.std(norm_scores))

    print("Component means (anom, sig, ae):", anom_mean, sig_mean, ae_mean)
    print("Component stds  (anom, sig, ae):", anom_std, sig_std, ae_std)
    print("Mean (normalized):", mean_score)
    print("Std  (normalized):", std_score)
    print("P95 (combined):", p95)
    print("P99 (combined):", p99)

    return {"p95": p95, "mean": mean_score, "std": std_score, **component_stats}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the audio-quality models and normalization stats."
    )
    parser.add_argument(
        '--train-dir', type=str, default='data/train',
        help='Directory with training audio files (default: data/train)'
    )
    parser.add_argument(
        '--models-dir', type=str, default=MODELS_DIR,
        help='Output directory for trained artifacts (default: models)'
    )
    parser.add_argument(
        '--device', type=str, choices=['cpu', 'cuda'],
        help='Device to use (cpu or cuda). Auto-detected if not specified.'
    )
    args = parser.parse_args()

    device: str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 50)
    print("TRAINING MODE")
    print("=" * 50)

    print("Training embedding model...")
    embedder, normal_model = train_embedding_model(args.train_dir)
    save_normal_model(normal_model, args.models_dir)

    print("\nTraining autoencoder...")
    ae_model = train_autoencoder(args.train_dir, device=device)
    save_autoencoder(ae_model, args.models_dir)

    print("\n✓ Models trained and saved successfully!")
    print(f"  - Normal model: {os.path.join(args.models_dir, 'normal_model.pkl')}")
    print(f"  - Autoencoder:  {os.path.join(args.models_dir, 'autoencoder.pt')}")

    print("\nCalculating normalization statistics from training data...")
    detector = AudioQualityDetector(embedder, normal_model, ae_model, device=device)
    norm_stats = compute_norm_stats(detector, args.train_dir)

    ensure_models_dir(args.models_dir)
    norm_stats_path = os.path.join(args.models_dir, "norm_stats.npy")
    np.save(norm_stats_path, norm_stats)
    print(f"  - Norm stats:   {norm_stats_path}")
