
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os
import glob
from pathlib import Path
import librosa
import pickle
import argparse
import json
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
               
        # Cache directory: inside the model folder for portability
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
        waveform, sr = librosa.load(path, sr=16000)
        return waveform

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
# MODEL PERSISTENCE
# =========================
MODELS_DIR: str = "models"

def ensure_models_dir() -> None:
    """Creates models directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)

def save_normal_model(model: 'NormalModel', filename: str = "normal_model.pkl") -> None:
    """Saves the normal model (mean and covariance inverse) to disk.
    
    Args:
        model (NormalModel): Trained normal model to save.
        filename (str): Output filename. Defaults to 'normal_model.pkl'.
    """
    ensure_models_dir()
    filepath: str = os.path.join(MODELS_DIR, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'mean': model.mean, 'cov_inv': model.cov_inv}, f)
    print(f"Normal model saved to {filepath}")

def load_normal_model(filename: str = "normal_model.pkl") -> Optional['NormalModel']:
    """Loads a normal model from disk.
    
    Args:
        filename (str): Input filename. Defaults to 'normal_model.pkl'.
        
    Returns:
        Optional[NormalModel]: Loaded model or None if file not found.
    """
    filepath: str = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    model: NormalModel = NormalModel()
    model.mean = data['mean']
    model.cov_inv = data['cov_inv']
    print(f"Normal model loaded from {filepath}")
    return model

def save_autoencoder(model: SpectrogramAutoencoder, filename: str = "autoencoder.pt") -> None:
    """Saves the autoencoder model to disk.
    
    Args:
        model (SpectrogramAutoencoder): Trained autoencoder model to save.
        filename (str): Output filename. Defaults to 'autoencoder.pt'.
    """
    ensure_models_dir()
    filepath: str = os.path.join(MODELS_DIR, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Autoencoder model saved to {filepath}")

def load_autoencoder(device: str = "cpu", filename: str = "autoencoder.pt") -> Optional[SpectrogramAutoencoder]:
    """Loads an autoencoder model from disk.
    
    Args:
        device (str): Device to load model on. Defaults to 'cpu'.
        filename (str): Input filename. Defaults to 'autoencoder.pt'.
        
    Returns:
        Optional[SpectrogramAutoencoder]: Loaded model or None if file not found.
    """
    filepath: str = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        return None
    model: SpectrogramAutoencoder = SpectrogramAutoencoder().to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    print(f"Autoencoder model loaded from {filepath}")
    return model

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

    # Weights for (anom, sig, ae) applied AFTER each component is standardized.
    WEIGHTS: Tuple[float, float, float] = (0.6, 0.3, 0.1)

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
            audio (ndarray): Audio waveform array.
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

    if not embeddings:
        raise ValueError(
            f"No training embeddings produced from '{audio_folder}'. Ensure it "
            f"contains .wav files at least {3}s long (shorter clips yield no chunks)."
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


# =========================
# TEST FUNCTION
# =========================
def test_audio(audio_path: str, device: str = "cpu") -> Dict[str, Any]:
    """Tests audio quality using saved models.
    
    Loads pre-trained models from disk and evaluates the audio quality of the
    provided audio file.
    
    Args:
        audio_path (str): Path to the audio file to test (.wav or .flac).
        device (str): Device for inference ('cpu' or 'cuda'). Defaults to 'cpu'.
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'final_score': Overall quality score
            - 'chunk_scores': Per-chunk quality scores
            - 'model_status': Status of model loading
            
    Raises:
        FileNotFoundError: If audio file or model files are not found.
    """
    print(f"Testing audio file: {audio_path}")
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load models
    print("Loading saved models...")
    normal_model: Optional[NormalModel] = load_normal_model()
    ae_model: Optional[SpectrogramAutoencoder] = load_autoencoder(device=device)
    
    if normal_model is None:
        raise FileNotFoundError("Normal model not found. Please train the model first using train mode.")
    
    # Initialize embedder
    embedder: Wav2Vec2Embedder = Wav2Vec2Embedder(device=device)
    
    # Create detector
    detector: AudioQualityDetector = AudioQualityDetector(embedder, normal_model, ae_model, device=device)

    norm_stats_path = os.path.join(MODELS_DIR, "norm_stats.npy")
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            "Normalization stats (norm_stats.npy) not found. Please train the model "
            "first using train mode."
        )
    stats = np.load(norm_stats_path, allow_pickle=True).item()

    # Load and evaluate audio
    print("Loading audio...")
    audio: ndarray = embedder.load_audio(audio_path)

    print("Evaluating audio quality...")
    result: Dict[str, Any] = detector.evaluate(audio, stats=stats)
    

    # Add metadata
    result['audio_file'] = audio_path
    result['audio_duration'] = len(audio) / 16000.0  # duration in seconds
    result['model_status'] = {
        'normal_model_loaded': True,
        'autoencoder_loaded': ae_model is not None
    }
    
    return result


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Audio Quality Detection System - Train embeddings or test audio files"
    )
    parser.add_argument(
        'mode',
        choices=['train', 'test'],
        help='Mode: train to train models, test to evaluate audio file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file for testing (required in test mode)'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/train',
        help='Directory with training audio files (default: data/train)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        help='Device to use (cpu or cuda). Auto-detected if not specified.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save test results to JSON file'
    )
    
    args: argparse.Namespace = parser.parse_args()
    
    # Auto-detect device if not specified
    device: str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print("\n" + "="*50)
        print("TRAINING MODE")
        print("="*50)
        
        print("Training embedding model...")
        embedder: Wav2Vec2Embedder
        normal_model: NormalModel
        embedder, normal_model = train_embedding_model(args.train_dir)
        save_normal_model(normal_model)

        print("\nTraining autoencoder...")
        ae_model: SpectrogramAutoencoder = train_autoencoder(args.train_dir, device=device)
        save_autoencoder(ae_model)
        
        print("\n✓ Models trained and saved successfully!")
        print(f"  - Normal model: models/normal_model.pkl")
        print(f"  - Autoencoder: models/autoencoder.pt")

        print("\nCalculating normalization statistics from training data...")
        # Create detector
        detector: AudioQualityDetector = AudioQualityDetector(embedder, normal_model, ae_model, device=device)

        # Pass 1: collect the RAW per-component values over every training chunk.
        anom_raw: List[float] = []
        sig_raw: List[float] = []
        ae_raw: List[float] = []
        for f in glob.glob(os.path.join(args.train_dir, "*.wav")):
            audio = embedder.load_audio(f)
            for c in chunk_audio(audio):
                anom, sig, ae = detector.chunk_components(c)
                anom_raw.append(anom)
                sig_raw.append(sig)
                ae_raw.append(ae)

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

        # Pass 2: standardize + weight each chunk, then derive the p95 used to
        # scale test-time scores to the familiar ~1.0 range.
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

        norm_stats_path = os.path.join(MODELS_DIR, "norm_stats.npy")
        np.save(norm_stats_path, {
            "p95": p95,
            "mean": mean_score,
            "std": std_score,
            **component_stats,
        })
        
    elif args.mode == 'test':
        print("\n" + "="*50)
        print("TEST MODE")
        print("="*50)
        
        if not args.audio:
            print("ERROR: --audio argument required in test mode")
            parser.print_help()
            exit(1)
        
        try:
            result: Dict[str, Any] = test_audio(args.audio, device=device)
            
            print("\n" + "-"*50)
            print("EVALUATION RESULTS")
            print("-"*50)
            print(f"Audio File: {result['audio_file']}")
            print(f"Duration: {result['audio_duration']:.2f} seconds")
            print(f"Final Quality Score: {result['final_score']:.4f}")
            print(f"Number of Chunks: {len(result['chunk_scores'])}")
            print(f"Models Loaded: {result['model_status']}")
            
            # Save to JSON if requested
            if args.output:
                output_data: Dict[str, Any] = {
                    'audio_file': result['audio_file'],
                    'duration_seconds': result['audio_duration'],
                    'final_score': result['final_score'],
                    'num_chunks': len(result['chunk_scores']),
                    'chunk_scores': result['chunk_scores'],
                    'model_status': result['model_status']
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            exit(1)
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            exit(1)
