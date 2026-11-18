# VoxGuard — AI Voice Detection

A Flask + librosa + scikit-learn college-project prototype that estimates whether an audio clip belongs to the **Human** or **AI-Generated** class.

## 1. Project structure

```text
voiceguard_ai_detector/
├── app.py
├── train.py
├── feature_extraction.py
├── requirements.txt
├── model.joblib             # created after training
├── dataset/
│   ├── human/
│   └── ai_generated/
├── templates/
│   └── index.html
└── uploads/                 # temporary uploads; files are deleted after inference
```

## 2. Dataset preparation

Create these folders:

```bash
mkdir -p dataset/human dataset/ai_generated
```

Put genuine human speech recordings in `dataset/human/` and AI-generated / synthetic speech in `dataset/ai_generated/`. WAV and MP3 are supported, as are FLAC, OGG and M4A.

**Dataset quality matters more than the algorithm.** Aim for hundreds or thousands of varied clips per class if possible. Include multiple speakers, languages, recording devices, codecs, clip lengths and TTS/deepfake generators. Keep classes reasonably balanced.

For a credible evaluation, split by **speaker, original recording and generator**, not just random clips. Otherwise near-duplicate clips can leak into both train and test sets and make accuracy look much higher than real-world performance. The included script uses a stratified random hold-out split for a simple baseline; it does not automatically perform speaker-disjoint splitting.

Do not use private recordings without consent. Only use datasets whose licences permit your intended academic use.

## 3. Environment setup

Python 3.10 or 3.11 is recommended.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If MP3 decoding fails on your machine, install FFmpeg and ensure it is on PATH. `soundfile` is used for common audio formats; librosa relies on an audio decoder backend for MP3.

## 4. Train the model

After adding files to both dataset folders:

```bash
python train.py --data_dir dataset
```

Optional arguments:

```bash
python train.py --data_dir dataset --model_out model.joblib --test_size 0.2 --random_state 42
```

The script:
1. Loads audio and resamples it to 22,050 Hz mono.
2. Extracts MFCCs, MFCC deltas, spectral centroid/bandwidth/rolloff, zero-crossing rate, RMS energy, chroma and mel-spectrogram statistics.
3. Trains a soft-voting ensemble (Extra Trees + scaled RBF SVM).
4. Prints hold-out accuracy, precision, recall, F1 and a confusion matrix.
5. Saves the trained model and metadata to `model.joblib`.

Training can take time depending on dataset size and CPU. A `model.joblib` file is required before the web app can classify audio.

## 5. Run the web application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Upload an MP3 or WAV file and click **Analyse voice**.

Useful endpoints:
- `GET /` — web interface
- `GET /api/health` — checks server/model readiness
- `POST /api/predict` — multipart upload with field name `audio`; returns JSON

Example response:

```json
{
  "prediction": "AI-Generated",
  "class": "ai_generated",
  "confidence": 91.25,
  "probabilities": {
    "human": 8.75,
    "ai_generated": 91.25
  },
  "filename": "sample.mp3",
  "disclaimer": "Model estimate only; not forensic proof of authenticity."
}
```

## 6. Important limitations (include in your viva)

- There is no universally reliable audio-only detector. Generators evolve, and compression, noise, language, microphones and post-processing can change the signal.
- The confidence score is the classifier's estimated class probability, **not a calibrated guarantee** that the answer is correct.
- MFCC/spectral features are a useful educational baseline, but modern spoofing detection often uses larger datasets and learned representations (for example, wav2vec 2.0 or RawNet-style models).
- A high random-split test score may be misleading if the same speakers, source recordings or generators appear in both partitions.
- Treat the output as a research estimate, never as proof in legal, employment or other high-stakes decisions.

## 7. Troubleshooting

**`Model not found`** — run `python train.py --data_dir dataset` from the project directory first.

**`No module named ...`** — activate `.venv` and run `pip install -r requirements.txt`.

**MP3 decode error** — install FFmpeg and restart the terminal so PATH refreshes.

**Very low accuracy** — check labels, remove duplicates, balance classes, add more diverse samples, and evaluate on speakers/generators not seen during training. Do not tune only against the test set.

**Upload rejected** — only supported audio extensions are accepted; maximum upload size is 20 MB.

## 8. Suggested college demonstration

1. Show the dataset class folders and explain the labels.
2. Run `train.py` and show the printed metrics/confusion matrix.
3. Start Flask and open the dashboard.
4. Test one genuine recording and one generated recording.
5. Explain the feature pipeline and why the result is an estimate rather than definitive proof.
