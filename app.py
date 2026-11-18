import os
import uuid
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from feature_extraction import extract_features

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model.joblib"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_MB = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_DIR.mkdir(exist_ok=True)

artifact = None
if MODEL_PATH.exists():
    artifact = joblib.load(MODEL_PATH)

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

@app.get("/")
def home():
    return render_template("index.html", model_ready=artifact is not None)

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": artifact is not None})

@app.post("/api/predict")
def predict():
    global artifact
    if artifact is None:
        return jsonify({
            "error": "Model not found. Train it first with: python train.py --data_dir dataset"
        }), 503
    if "audio" not in request.files:
        return jsonify({"error": "Choose an audio file first."}), 400

    file = request.files["audio"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported format. Upload MP3, WAV, FLAC, OGG or M4A."}), 400

    safe_name = secure_filename(file.filename)
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    try:
        file.save(temp_path)
        features = extract_features(str(temp_path)).reshape(1, -1)
        model = artifact["model"]
        probabilities = model.predict_proba(features)[0]
        class_index = int(np.argmax(probabilities))
        prediction = artifact["labels"][class_index]
        human_prob = float(probabilities[0]) * 100
        ai_prob = float(probabilities[1]) * 100

        return jsonify({
            "prediction": prediction,
            "class": "ai_generated" if class_index == 1 else "human",
            "confidence": round(float(probabilities[class_index]) * 100, 2),
            "probabilities": {
                "human": round(human_prob, 2),
                "ai_generated": round(ai_prob, 2)
            },
            "filename": safe_name,
            "disclaimer": "Model estimate only; not forensic proof of authenticity."
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": f"Could not analyse this audio: {str(exc)}"}), 422
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."}), 413

if __name__ == "__main__":
    # Use debug=False outside local development.
    app.run(host="127.0.0.1", port=5000, debug=True)
