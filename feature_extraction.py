"""Shared audio feature extraction used by both training and inference."""
import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 8.0
N_MFCC = 20

def extract_features(audio_path: str) -> np.ndarray:
    """Return a fixed-length feature vector for an audio file."""
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    if y.size == 0:
        raise ValueError("The audio file is empty or could not be decoded.")

    # Pad very short clips so spectral features remain well-defined.
    min_samples = int(0.25 * sr)
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)))

    # Trim leading/trailing silence, but retain a valid signal if it is all silence.
    trimmed, _ = librosa.effects.trim(y, top_db=35)
    if trimmed.size:
        y = trimmed

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    def stats(matrix):
        return np.concatenate([
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
        ])

    features = np.concatenate([
        stats(mfcc),
        stats(delta),
        stats(delta2),
        stats(spectral_centroid),
        stats(spectral_bandwidth),
        stats(spectral_rolloff),
        stats(zero_crossing),
        stats(rms),
        stats(chroma),
        stats(mel_db),
    ]).astype(np.float32)

    if not np.all(np.isfinite(features)):
        raise ValueError("Audio produced invalid feature values.")
    return features
