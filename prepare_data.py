"""Audio dataset preparation pipeline with voice activity detection.

This module processes raw audio files from multiple sources (LibriSpeech, YouTube) and
filters them to create a clean dataset of good quality speech chunks. Uses WebRTC VAD
for voice activity detection and signal quality metrics for filtering.
"""

import os
import subprocess
import librosa
import numpy as np
import soundfile as sf
from glob import glob
import imageio_ffmpeg
from pytubefix import YouTube
from pytubefix.cli import on_progress
import webrtcvad

SR = 16000
CHUNK_SEC = 3
OUTPUT_DIR = "dataset_clean"
RAW_DIR = "raw_audio"

vad = webrtcvad.Vad(2)  # aggressiveness: 0-3

def download_librispeech():
    """Downloads and extracts LibriSpeech dev-clean dataset.

    Retrieves the LibriSpeech dev-clean data (approximately 5.4 hours of audio)
    from OpenSLR servers and extracts it to RAW_DIR. Skips the ~330 MB download
    and extraction if the dataset is already present.
    """
    extracted_dir = os.path.join(RAW_DIR, "LibriSpeech")
    if os.path.isdir(extracted_dir):
        print(f"LibriSpeech already present at {extracted_dir}, skipping download.")
        return

    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    subprocess.run(["wget", "-c", url, "-P", RAW_DIR])
    subprocess.run(["tar", "-xvf", f"{RAW_DIR}/dev-clean.tar.gz", "-C", RAW_DIR])

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def download_youtube(urls):
    """Downloads audio from YouTube URLs using pytubefix with OAuth.

    On the first run this will print a Google device-auth URL and a short code.
    Open that URL in your browser, enter the code, then press Enter here.
    The OAuth token is cached locally so subsequent runs are fully automatic.

    Args:
        urls (list): List of YouTube video URLs to download.

    Downloads audio-only streams and converts them to WAV files in RAW_DIR.
    Skips unavailable videos and reports per-URL errors.
    """
    for url in urls:
        print(f"Downloading: {url}")
        try:
            yt = YouTube(url, use_oauth=True, allow_oauth_cache=True,
                         on_progress_callback=on_progress)
            stream = yt.streams.get_audio_only()
            if stream is None:
                print(f"  No audio stream found for {url}, skipping.")
                continue

            safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in yt.title)
            tmp_path = os.path.join(RAW_DIR, f"{safe_title}.{stream.subtype}")
            wav_path = os.path.join(RAW_DIR, f"{safe_title}.wav")

            if os.path.exists(wav_path):
                print(f"  Already exists: {wav_path}, skipping.")
                continue

            stream.download(output_path=RAW_DIR, filename=f"{safe_title}.{stream.subtype}")
            subprocess.run(
                [FFMPEG, "-y", "-i", tmp_path, "-ar", str(SR), "-ac", "1", wav_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(tmp_path)
            print(f"  Saved: {wav_path}")
        except Exception as e:
            print(f"  Failed: {url} — {e}")

def chunk_audio(audio):
    """Splits audio into fixed-length non-overlapping chunks.
    
    Args:
        audio (ndarray): Audio waveform array at 16kHz sampling rate.
        
    Returns:
        list: List of audio chunks, each of length CHUNK_SEC seconds (samples ignored if shorter).
    """
    chunk_size = SR * CHUNK_SEC
    return [
        audio[i:i+chunk_size]
        for i in range(0, len(audio), chunk_size)
        if len(audio[i:i+chunk_size]) == chunk_size
    ]

def is_speech(chunk):
    """Detects presence of speech in audio chunk using WebRTC VAD.
    
    Args:
        chunk (ndarray): Audio chunk array.
        
    Returns:
        bool: True if detected speech ratio > 30%, False otherwise.
        
    Uses WebRTC Voice Activity Detection with 30ms frames for robust speech detection.
    """
    # VAD expects 16-bit PCM. Clip to [-1, 1) and scale by 32767 so a
    # full-scale +1.0 sample does not overflow int16 to -32768.
    pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    frame_duration = 30  # ms
    frame_size = int(SR * frame_duration / 1000) * 2

    speech_frames = 0
    total_frames = 0

    for i in range(0, len(pcm) - frame_size, frame_size):
        frame = pcm[i:i+frame_size]
        if vad.is_speech(frame, SR):
            speech_frames += 1
        total_frames += 1

    if total_frames == 0:
        return False

    return speech_frames / total_frames > 0.3

def is_good_chunk(chunk):
    """Validates audio chunk quality based on signal characteristics.
    
    Args:
        chunk (ndarray): Audio chunk array.
        
    Returns:
        bool: True if chunk meets all quality criteria, False otherwise.
        
    Quality criteria:
        - RMS > 0.01 (sufficient amplitude)
        - Silence ratio < 40% (mostly contains sound)
        - Clipping ratio < 1% (no distortion)
        - Contains speech (VAD detection)
    """
    rms = np.sqrt(np.mean(chunk**2))
    silence = np.mean(np.abs(chunk) < 1e-3)
    clipping = np.mean(np.abs(chunk) > 0.99)

    return (
        rms > 0.01 and
        silence < 0.4 and
        clipping < 0.01 and
        is_speech(chunk)
    )

def process_file(path, idx_start=0):
    """Processes a single audio file and extracts good quality chunks.
    
    Args:
        path (str): Path to audio file (.wav or .flac).
        idx_start (int): Starting index for output file naming. Defaults to 0.
        
    Returns:
        int: Next available index after processing (idx_start + number of good chunks).
        
    Loads audio, chunks it, filters by quality criteria, and saves valid chunks
    as individual WAV files in OUTPUT_DIR.
    """
    try:
        audio, _ = librosa.load(path, sr=SR)
        chunks = chunk_audio(audio)
        count = idx_start

        for c in chunks:
            if is_good_chunk(c):
                out_path = os.path.join(OUTPUT_DIR, f"clean_{count}.wav")
                sf.write(out_path, c, SR)
                count += 1

        return count
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return idx_start

def run_pipeline(youtube_urls=[]):
    """Main pipeline: downloads audio and processes into clean dataset.
    
    Args:
        youtube_urls (list): List of YouTube URLs to download. Defaults to empty list.
        
    Executes the complete pipeline:
        1. Optionally download YouTube audio
        2. Discover all .flac and .wav files in RAW_DIR
        3. Process each file and filter for quality
        4. Output clean chunks to OUTPUT_DIR
        
    Prints total number of clean chunks created upon completion.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    download_librispeech()

    if youtube_urls:
        download_youtube(youtube_urls)

    files = glob(f"{RAW_DIR}/**/*.flac", recursive=True) + \
            glob(f"{RAW_DIR}/**/*.wav", recursive=True)

    idx = 0
    for f in files:
        idx = process_file(f, idx)

    print(f"Done. Total clean chunks: {idx}")

if __name__ == "__main__":
    # Main entry point for the dataset preparation pipeline.
    # Runs the pipeline with a predefined list of YouTube video URLs containing
    # high-quality speech content for dataset creation.
    youtube_urls = [
    "https://www.youtube.com/watch?v=DxREm3s1scA",
    "https://www.youtube.com/watch?v=8ZcmTl_1ER8",
    "https://www.youtube.com/watch?v=U6oJxS0Q0lU",
    "https://www.youtube.com/watch?v=Y7G5CKX0je0",
    "https://www.youtube.com/watch?v=5qap5aO4i9A",
    "https://www.youtube.com/watch?v=2OEL4P1Rz04",

    "https://www.youtube.com/watch?v=Jtq1p7ZCkbs",
    "https://www.youtube.com/watch?v=8lZzJ2g5y9M",
    "https://www.youtube.com/watch?v=6ZfuNTqbHE8",
    "https://www.youtube.com/watch?v=9bZkp7q19f0",
    "https://www.youtube.com/watch?v=3GwjfUFyY6M",

    "https://www.youtube.com/watch?v=9Auq9mYxFEE",
    "https://www.youtube.com/watch?v=frW4G6u7k6E",
    "https://www.youtube.com/watch?v=Z1BCujX3pw8",
    "https://www.youtube.com/watch?v=VYOjWnS4cMY",

    "https://www.youtube.com/watch?v=0JdJe8g5k6A",
    "https://www.youtube.com/watch?v=7Xf-Lesrkuc",
    "https://www.youtube.com/watch?v=JGwWNGJdvx8",
    
    "https://www.youtube.com/watch?v=HhjHYkPQ8F0",
    "https://www.youtube.com/watch?v=Kx7B-XvmFtE"
    ]

    run_pipeline(youtube_urls)
