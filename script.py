
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
    def __init__(self, device="cpu") -> None:
        self.device: str = device
        self.processor: Wav2Vec2Processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model: Wav2Vec2Model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
        self.model.eval()

    def load_audio(self, path: str) -> ndarray:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform: torch.Tensor = torchaudio.functional.resample(waveform, sr, 16000)
        return waveform.squeeze().numpy()

    def get_embedding(self, audio: ndarray) -> ndarray:
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(inputs.input_values.to(self.device))
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.cpu().numpy()[0]


# =========================
# AUTOENCODER (Spectrogram)
# =========================
class SpectrogramAutoencoder(nn.Module):
    def __init__(self) -> None:
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
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def get_spectrogram(audio: ndarray) -> ndarray:
    spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
    log_spec = librosa.power_to_db(spec)
    return log_spec.T  # shape (time, 128)


# =========================
# NORMAL MODEL (Mahalanobis)
# =========================
class NormalModel:
    def __init__(self) -> None:
        self.mean: Optional[ndarray] = None
        self.cov_inv: Optional[ndarray] = None

    def fit(self, embeddings: ndarray) -> None:
        self.mean = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv = np.linalg.inv(cov)

    def score(self, x: ndarray) -> float:
        diff = x - self.mean
        return float(diff.T @ self.cov_inv @ diff)


# =========================
# SIGNAL CHECKS
# =========================
def compute_signal_features(audio: ndarray) -> Dict[str, Any]:
    return {
        "clipping_ratio": np.mean(np.abs(audio) > 0.99),
        "rms": np.sqrt(np.mean(audio**2)),
        "silence_ratio": np.mean(np.abs(audio) < 1e-3)
    }


def signal_score(f: Dict[str, Any]) -> float:
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
    def __init__(self, embedder: Wav2Vec2Embedder, normal_model: 'NormalModel', ae_model: Optional['SpectrogramAutoencoder'] = None, device: str = "cpu") -> None:
        self.embedder: Wav2Vec2Embedder = embedder
        self.normal_model: NormalModel = normal_model
        self.ae_model: Optional[SpectrogramAutoencoder] = ae_model
        self.device: str = device

    def ae_score(self, audio: ndarray) -> float:
        spec = get_spectrogram(audio)
        spec_tensor: torch.Tensor = torch.tensor(spec, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.ae_model(spec_tensor)
        loss: float = torch.mean((spec_tensor - recon) ** 2).item()
        return loss

    def evaluate(self, audio: ndarray) -> Dict[str, Any]:
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
    embedder, normal_model = train_embedding_model("good_audio")

    print("Training autoencoder...")
    ae_model: SpectrogramAutoencoder = train_autoencoder("good_audio", device=DEVICE)

    detector: AudioQualityDetector = AudioQualityDetector(embedder, normal_model, ae_model, device=DEVICE)

    audio: ndarray = embedder.load_audio("test.wav")
    result: Dict[str, Any] = detector.evaluate(audio)

    print("RESULT:", result)
