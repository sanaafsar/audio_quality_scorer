
# audio_quality_system_v2.py

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
import glob
import librosa
from typing import Any, Dict, List, Tuple, Optional
from numpy import ndarray

# =========================
# WAV2VEC2 EMBEDDER
# =========================
class Wav2Vec2Embedder:
    """Extracts audio embeddings using pre-trained Wav2Vec2 model.
    
    This class loads audio files and generates embeddings using the Facebook
    Wav2Vec2 base model. Embeddings are normalized to 16kHz sampling rate.
    """
    
    def __init__(self, device="cpu") -> None:
        """Initializes the Wav2Vec2 embedder.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device: str = device
        self.processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        self.model.eval()

    def load_audio(self, path: str) -> ndarray:
        """Loads audio file and resamples to 16kHz.
        
        Args:
            path (str): Path to the audio file.
            
        Returns:
            ndarray: Audio waveform as numpy array with shape (samples,).
        """
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform: torch.Tensor = torchaudio.functional.resample(waveform, sr, 16000)
        return waveform.squeeze().numpy()

    def get_embedding(self, audio: ndarray) -> ndarray:
        """Generates embedding from audio using Wav2Vec2 model.
        
        Args:
            audio (ndarray): Audio waveform array at 16kHz.
            
        Returns:
            ndarray: Embedding vector with shape (768,).
        """
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
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
    spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
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
        
        Args:
            embeddings (ndarray): Array of embeddings with shape (n_samples, embedding_dim).
        """
        self.mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
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
def chunk_audio(audio: ndarray, sr: int = 16000, chunk_sec: int = 3) -> List[ndarray]:
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
# DETECTOR
# =========================
class AudioQualityDetector:
    """Comprehensive audio quality detector using multiple metrics.
    
    Combines embedding-based anomaly detection, signal features, and spectrogram
    reconstruction error to assess audio quality.
    """
    
    def __init__(self, embedder: Wav2Vec2Embedder, normal_model: 'NormalModel', ae_model: Optional['SpectrogramAutoencoder'] = None, device: str = "cpu") -> None:
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

    def evaluate(self, audio: ndarray) -> Dict[str, Any]:
        """Evaluates audio quality by analyzing chunks.
        
        Args:
            audio (ndarray): Audio waveform array.
            
        Returns:
            Dict[str, Any]: Dictionary with 'final_score' (float) and 'chunk_scores' (List[float]).
        """
        chunks = chunk_audio(audio)

        scores: List[float] = []

        for c in chunks:
            sig: float = signal_score(compute_signal_features(c))
            emb = self.embedder.get_embedding(c)
            anom = self.normal_model.score(emb)

            ae: float = 0.0
            if self.ae_model:
                ae = self.ae_score(c)

            final: float = 0.6 * anom + 0.3 * sig + 0.1 * ae
            scores.append(final)

        return {
            "final_score": float(np.mean(scores)),
            "chunk_scores": scores
        }


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
    """
    embedder: Wav2Vec2Embedder = Wav2Vec2Embedder()
    embeddings: List[ndarray] = []

    for f in glob.glob(os.path.join(audio_folder, "*.wav")):
        audio: ndarray = embedder.load_audio(f)
        chunks: List[ndarray] = chunk_audio(audio)
        for c in chunks:
            emb: ndarray = embedder.get_embedding(c)
            embeddings.append(emb)

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
        total_loss: int = 0
        for spec in data:
            spec_tensor: torch.Tensor = torch.tensor(spec, dtype=torch.float32).to(device)
            recon: torch.Tensor = model(spec_tensor)
            loss: torch.Tensor = loss_fn(recon, spec_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += int(loss.item())

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return model


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    DEVICE: str = "cpu"

    print("Training embedding model...")
    embedder: Wav2Vec2Embedder
    normal_model: NormalModel
    embedder, normal_model = train_embedding_model("data/train")

    print("Training autoencoder...")
    ae_model: SpectrogramAutoencoder = train_autoencoder("data/train", device=DEVICE)

    detector: AudioQualityDetector = AudioQualityDetector(embedder, normal_model, ae_model, device=DEVICE)

    audio: ndarray = embedder.load_audio("data/test/test.wav")
    result: Dict[str, Any] = detector.evaluate(audio)

    print("RESULT:", result)
