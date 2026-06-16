"""Audio-quality inference model.

Self-contained inference module for the media-quality pipeline. It exposes
``AudioQualityModel``, a :class:`BaseQualityModel` subclass that loads the
trained artifacts from ``models/`` and scores decoded audio.

Drop-in usage (inside the Chain-of-Responsibility pipeline)::

    from model import AudioQualityModel

    model = AudioQualityModel(models_dir="models")
    model.load()                                # once
    scores = model.predict(waveform, sample_rate)   # per media file

Training lives in ``train.py`` and writes the artifacts this module consumes
(``normal_model.pkl``, ``autoencoder.pt``, ``norm_stats.npy``). The classes and
helpers below are the shared building blocks used by both modules.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from base import BaseQualityModel, ScoreDict

# Sample rate assumed everywhere downstream (embeddings + spectrograms).
SR: int = 16000
# Default location of trained artifacts, relative to the caller's CWD.
MODELS_DIR: str = "models"


# =========================
# WAV2VEC2 EMBEDDER
# =========================
class Wav2Vec2Embedder:
    """Extracts audio embeddings using pre-trained Wav2Vec2 model.

    This class loads audio files and generates embeddings using the Facebook
    Wav2Vec2 base model. Embeddings are normalized to 16kHz sampling rate.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Initializes the Wav2Vec2 embedder.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device: str = device

        # Cache directory: next to this module for portability.
        cache_dir = Path(__file__).parent / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base",
            cache_dir=str(cache_dir),
        )
        self.model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            cache_dir=str(cache_dir),
        ).to(device)
        self.model.eval()

    def load_audio(self, path: str) -> ndarray:
        """Loads audio file and resamples to 16kHz.

        Args:
            path (str): Path to the audio file.

        Returns:
            ndarray: Audio waveform as numpy array with shape (samples,).
        """
        waveform, sr = librosa.load(path, sr=SR)
        return waveform

    def get_embedding(self, audio: ndarray) -> ndarray:
        """Generates embedding from audio using Wav2Vec2 model.

        Args:
            audio (ndarray): Audio waveform array at 16kHz.

        Returns:
            ndarray: Embedding vector with shape (768,).
        """
        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device))
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy()[0]


# =========================
# AUTOENCODER (Spectrogram)
# =========================
class SpectrogramAutoencoder(nn.Module):
    """Autoencoder for spectrogram reconstruction and anomaly detection.

    This neural network encodes mel-spectrograms to a 32-dimensional latent space
    and reconstructs them. Reconstruction error indicates audio quality issues.
    """

    def __init__(self) -> None:
        """Initializes the autoencoder architecture."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input mel-spectrogram with shape (time, 128).

        Returns:
            torch.Tensor: Reconstructed spectrogram with shape (time, 128).
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def get_spectrogram(audio: ndarray) -> ndarray:
    """Converts audio to log mel-spectrogram.

    Args:
        audio (ndarray): Audio waveform array at 16kHz.

    Returns:
        ndarray: Log mel-spectrogram with shape (time, 128).
    """
    spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    log_spec = librosa.power_to_db(spec)
    return log_spec.T  # shape (time, 128)


# =========================
# NORMAL MODEL (Mahalanobis)
# =========================
class NormalModel:
    """Mahalanobis distance based anomaly detector for embeddings.

    Models the distribution of good audio embeddings and computes anomaly
    scores using Mahalanobis distance.
    """

    def __init__(self) -> None:
        """Initializes the normal distribution model."""
        self.mean: Optional[ndarray] = None
        self.cov_inv: Optional[ndarray] = None

    def fit(self, embeddings: ndarray) -> None:
        """Fits the model to embeddings from good audio.

        Uses trace-relative shrinkage so the inverse covariance stays
        well-conditioned even when the number of samples is not comfortably
        larger than the embedding dimension (otherwise the sample covariance is
        rank-deficient and ``cov_inv`` is numerically meaningless).

        Args:
            embeddings (ndarray): Array of embeddings with shape (n_samples, embedding_dim).
        """
        n_samples, dim = embeddings.shape
        if n_samples <= dim:
            print(
                f"WARNING: NormalModel fit on {n_samples} samples for {dim}-dim "
                f"embeddings (n <= dim); covariance is rank-deficient and the "
                f"Mahalanobis anomaly score may be unreliable. Use more training audio."
            )

        self.mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings, rowvar=False)
        # Shrink toward a scaled identity: lambda * (trace(cov) / dim) * I.
        # This is scale-aware (unlike a fixed 1e-6 ridge) and guarantees a
        # positive-definite, invertible matrix.
        shrinkage = 1e-3
        ridge = shrinkage * (np.trace(cov) / dim)
        cov += np.eye(dim) * ridge
        self.cov_inv = np.linalg.inv(cov)

    def score(self, x: ndarray) -> float:
        """Computes Mahalanobis distance anomaly score.

        Args:
            x (ndarray): Embedding vector.

        Returns:
            float: Anomaly score (higher = more anomalous).
        """
        diff = x - self.mean
        return float(diff.T @ self.cov_inv @ diff)


# =========================
# SIGNAL CHECKS
# =========================
def compute_signal_features(audio: ndarray) -> Dict[str, Any]:
    """Computes signal quality features from audio.

    Args:
        audio (ndarray): Audio waveform array.

    Returns:
        Dict[str, Any]: Dictionary with keys 'clipping_ratio', 'rms', 'silence_ratio'.
    """
    return {
        "clipping_ratio": np.mean(np.abs(audio) > 0.99),
        "rms": np.sqrt(np.mean(audio**2)),
        "silence_ratio": np.mean(np.abs(audio) < 1e-3)
    }


def signal_score(f: Dict[str, Any]) -> float:
    """Computes audio quality score from signal features.

    Args:
        f (Dict[str, Any]): Signal features dictionary with 'clipping_ratio', 'rms', 'silence_ratio'.

    Returns:
        float: Quality score (0.0 = good, up to 1.5 = poor).
    """
    score = 0.0
    if f["clipping_ratio"] > 0.01:
        score += 0.5
    if f["rms"] < 0.01:
        score += 0.5
    if f["silence_ratio"] > 0.5:
        score += 0.5
    return score


# =========================
# CHUNKING
# =========================
def chunk_audio(audio: ndarray, sr: int = SR, chunk_sec: int = 3) -> List[ndarray]:
    """Splits audio into fixed-length non-overlapping chunks.

    Args:
        audio (ndarray): Audio waveform array.
        sr (int): Sampling rate in Hz. Defaults to 16000.
        chunk_sec (int): Chunk duration in seconds. Defaults to 3.

    Returns:
        List[ndarray]: List of audio chunks with consistent length.
    """
    chunk_size: int = sr * chunk_sec
    chunks: List[ndarray] = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks


# =========================
# MODEL LOADING (inference side of persistence)
# =========================
def load_normal_model(models_dir: str = MODELS_DIR, filename: str = "normal_model.pkl") -> Optional[NormalModel]:
    """Loads a normal model from disk.

    Args:
        models_dir (str): Directory containing the artifact. Defaults to 'models'.
        filename (str): Input filename. Defaults to 'normal_model.pkl'.

    Returns:
        Optional[NormalModel]: Loaded model or None if file not found.
    """
    filepath: str = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    model: NormalModel = NormalModel()
    model.mean = data['mean']
    model.cov_inv = data['cov_inv']
    return model


def load_autoencoder(models_dir: str = MODELS_DIR, device: str = "cpu", filename: str = "autoencoder.pt") -> Optional[SpectrogramAutoencoder]:
    """Loads an autoencoder model from disk.

    Args:
        models_dir (str): Directory containing the artifact. Defaults to 'models'.
        device (str): Device to load model on. Defaults to 'cpu'.
        filename (str): Input filename. Defaults to 'autoencoder.pt'.

    Returns:
        Optional[SpectrogramAutoencoder]: Loaded model or None if file not found.
    """
    filepath: str = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        return None
    model: SpectrogramAutoencoder = SpectrogramAutoencoder().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model


# =========================
# DETECTOR (scoring core, shared by training + inference)
# =========================
class AudioQualityDetector:
    """Comprehensive audio quality detector using multiple metrics.

    Combines embedding-based anomaly detection, signal features, and spectrogram
    reconstruction error to assess audio quality.
    """

    # Weights for (anom, sig, ae) applied AFTER each component is standardized.
    WEIGHTS: Tuple[float, float, float] = (0.6, 0.3, 0.1)

    def __init__(self, embedder: Wav2Vec2Embedder, normal_model: NormalModel, ae_model: Optional[SpectrogramAutoencoder] = None, device: str = "cpu") -> None:
        """Initializes the audio quality detector.

        Args:
            embedder (Wav2Vec2Embedder): Embedder for extracting audio representations.
            normal_model (NormalModel): Anomaly detector for embeddings.
            ae_model (Optional[SpectrogramAutoencoder]): Optional autoencoder for reconstruction error.
            device (str): Device for model inference ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.embedder: Wav2Vec2Embedder = embedder
        self.normal_model: NormalModel = normal_model
        self.ae_model: Optional[SpectrogramAutoencoder] = ae_model
        self.device: str = device

    def ae_score(self, audio: ndarray) -> float:
        """Computes autoencoder reconstruction error.

        Args:
            audio (ndarray): Audio waveform array.

        Returns:
            float: Mean squared error between input and reconstructed spectrogram.
        """
        spec = get_spectrogram(audio)
        spec_tensor: torch.Tensor = torch.tensor(spec, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.ae_model(spec_tensor)
        loss: float = torch.mean((spec_tensor - recon) ** 2).item()
        return loss

    def chunk_components(self, c: ndarray) -> Tuple[float, float, float]:
        """Computes the three raw quality components for a single chunk.

        Args:
            c (ndarray): A single fixed-length audio chunk.

        Returns:
            Tuple[float, float, float]: Raw (anom, sig, ae) components, where
                anom is the Mahalanobis embedding anomaly, sig is the signal
                heuristic score, and ae is the autoencoder reconstruction error
                (0.0 when no autoencoder is loaded).
        """
        sig: float = signal_score(compute_signal_features(c))
        emb = self.embedder.get_embedding(c)
        anom: float = self.normal_model.score(emb)
        ae: float = self.ae_score(c) if self.ae_model else 0.0
        return anom, sig, ae

    def combine(self, anom: float, sig: float, ae: float, stats: Dict[str, Any]) -> float:
        """Combines raw components into a single standardized quality score.

        Each component is z-scored against its training-set mean/std so the
        WEIGHTS express intended importance rather than being dominated by the
        component with the largest raw scale (the embedding anomaly).

        Args:
            anom (float): Raw Mahalanobis embedding anomaly.
            sig (float): Raw signal heuristic score.
            ae (float): Raw autoencoder reconstruction error.
            stats (Dict[str, Any]): Normalization statistics containing per-component
                '<name>_mean' and '<name>_std' for 'anom', 'sig', 'ae'.

        Returns:
            float: Weighted, standardized score (NOT yet divided by p95).
        """
        z_anom = (anom - stats["anom_mean"]) / stats["anom_std"]
        z_sig = (sig - stats["sig_mean"]) / stats["sig_std"]
        z_ae = (ae - stats["ae_mean"]) / stats["ae_std"]
        return float(self.WEIGHTS[0] * z_anom + self.WEIGHTS[1] * z_sig + self.WEIGHTS[2] * z_ae)

    def evaluate(self, audio: ndarray, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates audio quality by analyzing chunks.

        Args:
            audio (ndarray): Audio waveform array at 16kHz.
            stats (Dict[str, Any]): Normalization statistics with per-component
                mean/std (see ``combine``) and 'p95', the 95th-percentile of the
                combined training score used to scale the output to ~1.0.

        Returns:
            Dict[str, Any]: Dictionary with 'final_score' (float) and 'chunk_scores' (List[Dict[str, Any]]).
        """
        chunks = chunk_audio(audio)

        if not chunks:
            # Audio is shorter than a single 3s chunk; nothing to score.
            return {
                "final_score": 0.0,
                "chunk_scores": [],
                "warning": "audio shorter than one 3s chunk; no chunks evaluated",
            }

        scores: List[Dict[str, Any]] = []

        for i, c in enumerate(chunks):
            anom, sig, ae = self.chunk_components(c)
            final: float = self.combine(anom, sig, ae, stats)
            # Normalize by p95 to get a relative score (~1.0 ≈ edge of normal).
            scores.append({"score": final / stats["p95"], "time": str(i * 3) + " sec"})

        return {
            "final_score": float(np.mean([s["score"] for s in scores])),
            "chunk_scores": scores
        }


# =========================
# PIPELINE MODEL (BaseQualityModel subclass)
# =========================
class AudioQualityModel(BaseQualityModel):
    """Unsupervised audio-quality model for the media-quality pipeline.

    Scores decoded audio by how far it deviates from a learned model of clean
    speech. The raw anomaly is *higher = worse*; ``predict`` also reports an
    ``"overall"`` value in ``[0, 1]`` (1.0 = best) per the BaseQualityModel
    convention.
    """

    MODEL_NAME: str = "audio-quality-scorer"
    MODEL_VERSION: str = "1.0"

    def __init__(self, models_dir: str = MODELS_DIR, device: Optional[str] = None) -> None:
        """Initializes the model wrapper (no weights loaded yet).

        Args:
            models_dir (str): Directory holding the trained artifacts
                (normal_model.pkl, autoencoder.pt, norm_stats.npy). Defaults to 'models'.
            device (Optional[str]): 'cpu' or 'cuda'. Auto-detected when None.
        """
        super().__init__()
        self._models_dir: str = str(models_dir)
        self._device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._detector: Optional[AudioQualityDetector] = None
        self._stats: Optional[Dict[str, Any]] = None

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (name + version)."""
        return f"{self.MODEL_NAME}@{self.MODEL_VERSION}"

    def load(self) -> None:
        """Loads the trained artifacts and initializes the embedder.

        Raises:
            FileNotFoundError: If the normal model or normalization stats are
                missing (i.e. the model has not been trained). The autoencoder
                is optional and silently skipped when absent.
        """
        normal_model = load_normal_model(self._models_dir)
        if normal_model is None:
            raise FileNotFoundError(
                f"normal_model.pkl not found in '{self._models_dir}'. "
                f"Train the model first (see train.py)."
            )

        stats_path = os.path.join(self._models_dir, "norm_stats.npy")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"norm_stats.npy not found in '{self._models_dir}'. "
                f"Train the model first (see train.py)."
            )
        self._stats = np.load(stats_path, allow_pickle=True).item()

        ae_model = load_autoencoder(self._models_dir, device=self._device)
        embedder = Wav2Vec2Embedder(device=self._device)
        self._detector = AudioQualityDetector(embedder, normal_model, ae_model, device=self._device)
        self._loaded = True

    def predict(self, audio: ndarray, sample_rate: int = 16000) -> ScoreDict:
        """Scores a decoded audio waveform.

        Args:
            audio (ndarray): Decoded waveform. Mono or multi-channel; multi-channel
                input is averaged to mono. Resampled to 16kHz when needed.
            sample_rate (int): Sample rate of ``audio`` in Hz. Defaults to 16000.

        Returns:
            ScoreDict: Metric dict with:
                - 'overall': quality in [0, 1] (1.0 = best). 1 / (1 + max(anomaly, 0)).
                - 'anomaly_score': mean per-chunk anomaly (higher = worse; ~1.0 = edge of normal).
                - 'max_chunk_anomaly': worst single 3s chunk (less diluted than the mean).
                - 'num_chunks': number of 3s chunks evaluated.

        Raises:
            RuntimeError: If called before ``load()``.
        """
        if not self._loaded or self._detector is None or self._stats is None:
            raise RuntimeError("AudioQualityModel.predict() called before load().")

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            # Average the channel axis (the shorter one) down to mono.
            channel_axis = int(np.argmin(audio.shape))
            audio = audio.mean(axis=channel_axis)
        if sample_rate != SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SR)

        result = self._detector.evaluate(audio, self._stats)
        chunk_scores = result["chunk_scores"]

        if not chunk_scores:
            # Too short to score; treat as no detected anomaly.
            return {
                "overall": 1.0,
                "anomaly_score": 0.0,
                "max_chunk_anomaly": 0.0,
                "num_chunks": 0.0,
            }

        anomaly = float(result["final_score"])
        max_chunk = float(max(s["score"] for s in chunk_scores))
        # Map unbounded "higher = worse" anomaly onto [0, 1] with 1.0 = best.
        overall = float(np.clip(1.0 / (1.0 + max(anomaly, 0.0)), 0.0, 1.0))

        return {
            "overall": overall,
            "anomaly_score": anomaly,
            "max_chunk_anomaly": max_chunk,
            "num_chunks": float(len(chunk_scores)),
        }


# =========================
# LOCAL TEST CLI
# =========================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Score an audio file with the trained AudioQualityModel (local test harness)."
    )
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file to score')
    parser.add_argument('--models-dir', type=str, default=MODELS_DIR, help='Directory with trained artifacts')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device (auto-detected if omitted)')
    parser.add_argument('--output', type=str, help='Save the score dict to a JSON file')
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        raise SystemExit(f"ERROR: audio file not found: {args.audio}")

    # Decode at native sample rate to exercise the predict() resampling path,
    # mirroring how the pipeline hands a decoded waveform to the model.
    waveform, native_sr = librosa.load(args.audio, sr=None)

    model = AudioQualityModel(models_dir=args.models_dir, device=args.device)
    model.load()
    scores = model.predict(waveform, sample_rate=native_sr)

    print("\n" + "-" * 50)
    print("AUDIO QUALITY SCORES")
    print("-" * 50)
    print(f"Model:        {model.model_name}")
    print(f"Audio File:   {args.audio}")
    print(f"Duration:     {len(waveform) / native_sr:.2f} s  (native sr={native_sr})")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({"audio_file": args.audio, "model_name": model.model_name, "scores": scores}, f, indent=2)
        print(f"\nResults saved to: {args.output}")
