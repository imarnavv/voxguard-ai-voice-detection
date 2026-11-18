"""
Train an audio classifier.

Expected dataset layout:
dataset/
  human/       # genuine human speech
  ai_generated/ # synthetic / deepfake speech

Supported formats: .wav, .mp3, .flac, .ogg, .m4a
Example:
  python train.py --data_dir dataset
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import extract_features

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
LABELS = {"human": 0, "ai_generated": 1}

def load_dataset(data_dir: Path):
    X, y, skipped = [], [], []
    for folder, label in LABELS.items():
        class_dir = data_dir / folder
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing folder: {class_dir}. Create human/ and ai_generated/."
            )
        paths = sorted(p for p in class_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        if not paths:
            raise ValueError(f"No supported audio files found in {class_dir}")
        print(f"[INFO] {folder}: {len(paths)} files")
        for i, path in enumerate(paths, 1):
            try:
                X.append(extract_features(str(path)))
                y.append(label)
            except Exception as exc:
                skipped.append((str(path), str(exc)))
            if i % 25 == 0:
                print(f"  processed {i}/{len(paths)}")
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} unreadable file(s).")
        for path, error in skipped[:10]:
            print(f"  - {path}: {error}")
    if len(set(y)) != 2:
        raise ValueError("Both classes need at least one successfully decoded file.")
    return np.vstack(X), np.asarray(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="dataset")
    parser.add_argument("--model_out", default="model.joblib")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X, y = load_dataset(data_dir)
    counts = np.bincount(y)
    if counts.min() < 2:
        raise ValueError("At least two valid samples per class are required for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Two complementary models in a soft-voting ensemble.
    extra = ExtraTreesClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced", random_state=args.random_state, n_jobs=-1
    )
    svm = make_pipeline(
        StandardScaler(),
        SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced",
            random_state=args.random_state)
    )
    model = VotingClassifier(
        estimators=[("extra_trees", extra), ("svm", svm)],
        voting="soft", weights=[2, 1], n_jobs=-1
    )
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nHold-out accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test, predictions, target_names=["Human", "AI-Generated"], zero_division=0
    ))
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_test, predictions))

    artifact = {
        "model": model,
        "labels": {0: "Human", 1: "AI-Generated"},
        "feature_version": 1,
        "sample_rate": 22050,
        "duration_seconds": 8.0,
        "validation_accuracy": float(accuracy),
        "training_samples": int(len(y)),
    }
    joblib.dump(artifact, args.model_out)
    print(f"\n[SUCCESS] Saved model to {args.model_out}")
    print("[NOTE] This score is only meaningful if the test set is independent.")
    print("Avoid placing clips from the same speaker, source recording, or generator")
    print("in both train and test sets; use a speaker/source-disjoint split for evaluation.")

if __name__ == "__main__":
    main()
