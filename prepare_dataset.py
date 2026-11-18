
from datasets import load_dataset, Audio
from pathlib import Path
import shutil

print("Loading In-the-Wild dataset...")

dataset = load_dataset(
    "SpeechAntiSpoofingBenchmarks/InTheWild",
    split="test"
)

# Disable automatic audio decoding
dataset = dataset.cast_column("audio", Audio(decode=False))

human_dir = Path("dataset/human")
ai_dir = Path("dataset/ai_generated")

human_dir.mkdir(parents=True, exist_ok=True)
ai_dir.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 1000

human_count = 0
ai_count = 0
skipped = 0

# Read the actual class names from the dataset metadata
label_names = dataset.features["label"].names

print("Available labels:", label_names)
print("Starting extraction...\n")

for index, sample in enumerate(dataset):

    label = label_names[sample["label"]]
    audio = sample["audio"]

    if label == "bonafide":
        if human_count >= MAX_PER_CLASS:
            continue
        destination = human_dir
        human_count += 1

    elif label == "spoof":
        if ai_count >= MAX_PER_CLASS:
            continue
        destination = ai_dir
        ai_count += 1

    else:
        continue

    try:
        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")

        extension = (
            Path(audio_path).suffix
            if audio_path else ".flac"
        )

        if not extension:
            extension = ".flac"

        filename = destination / f"sample_{index}{extension}"

        if audio_bytes is not None:
            filename.write_bytes(audio_bytes)

        elif audio_path and Path(audio_path).is_file():
            shutil.copy2(audio_path, filename)

        else:
            raise ValueError("Audio bytes and local file path unavailable")

    except Exception as error:
        skipped += 1

        if label == "bonafide":
            human_count -= 1
        else:
            ai_count -= 1

        print(f"Skipped sample {index}: {error}")
        continue

    if (human_count + ai_count) % 100 == 0:
        print(
            f"Human: {human_count} | "
            f"AI: {ai_count}"
        )

    if human_count >= MAX_PER_CLASS and ai_count >= MAX_PER_CLASS:
        break

print("\n========== DATASET SUMMARY ==========")
print("Human recordings:", human_count)
print("AI recordings:", ai_count)
print("Skipped:", skipped)
print("=====================================")

if human_count == 0 or ai_count == 0:
    print("WARNING: One or both classes are empty!")
else:
    print("Dataset preparation successful!")
