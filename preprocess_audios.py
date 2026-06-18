import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

# -----------------------------------------------
# Paths — relative to this script's location
# -----------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
input_root  = os.path.join(BASE_DIR, "VoiceGuard_Dataset")
output_root = os.path.join(BASE_DIR, "Preprocessed_Audios")
os.makedirs(output_root, exist_ok=True)

# -----------------------------------------------
# Parameters
# -----------------------------------------------
TARGET_SR    = 16000   # YAMNet requires 16 kHz
MAX_DURATION = 10      # seconds — skip anything longer
MIN_DURATION = 0.3     # seconds — skip anything shorter
AUGMENT      = True    # set False if dataset is already large enough

# -----------------------------------------------
# Stats tracker
# -----------------------------------------------
stats = {"processed": 0, "skipped": 0, "errors": 0}

# -----------------------------------------------
# Bandpass filter — keeps human speech (300–3400 Hz)
# removes AC hum and very high-freq noise
# -----------------------------------------------
def bandpass_filter(audio, lowcut=300, highcut=3400, fs=16000):
    nyq = fs / 2
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, audio)

# -----------------------------------------------
# Data augmentation — creates extra training samples
# -----------------------------------------------
def augment_audio(y, sr):
    augmented = []
    try:
        augmented.append(("_aug_pitchup",   librosa.effects.pitch_shift(y, sr=sr, n_steps=2)))
        augmented.append(("_aug_pitchdown", librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)))
        augmented.append(("_aug_slower",    librosa.effects.time_stretch(y, rate=0.9)))
        augmented.append(("_aug_faster",    librosa.effects.time_stretch(y, rate=1.1)))
        noise = np.random.normal(0, 0.005, len(y))
        augmented.append(("_aug_noise", y + noise))
    except Exception as e:
        print(f"  ⚠️ Augmentation warning: {e}")
    return augmented

# -----------------------------------------------
# Core preprocessing function
# -----------------------------------------------
def preprocess_audio(file_path, output_path):
    # Skip if already processed
    if os.path.exists(output_path):
        return "skip"

    try:
        # Load with librosa (handles mono/stereo, resamples automatically)
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True, duration=MAX_DURATION)

        # Check for silent / corrupt file
        max_val = np.max(np.abs(y))
        if max_val < 1e-6:
            print(f"  ⏩ Silent file skipped: {os.path.basename(file_path)}")
            return "skip"

        # Normalize to [-1, 1]
        y = y / max_val

        # Bandpass filter — remove noise outside speech range
        y = bandpass_filter(y, fs=sr)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)

        # Skip if too short after trimming
        duration = len(y_trimmed) / sr
        if duration < MIN_DURATION:
            print(f"  ⏩ Too short ({duration:.2f}s), skipped: {os.path.basename(file_path)}")
            return "skip"

        # Save processed original
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_trimmed, sr)

        # Save augmented versions (only for training data)
        if AUGMENT:
            for suffix, aug_audio in augment_audio(y_trimmed, sr):
                aug_path = output_path.replace(".wav", f"{suffix}.wav")
                if not os.path.exists(aug_path):
                    sf.write(aug_path, aug_audio, sr)

        return "ok"

    except Exception as e:
        print(f"  ❌ Error: {os.path.basename(file_path)} → {e}")
        return "error"

# -----------------------------------------------
# Walk through dataset and process
# -----------------------------------------------
SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')

print("🚀 Starting audio preprocessing...\n")

for root, _, files in os.walk(input_root):
    for file in files:
        if not file.lower().endswith(SUPPORTED_FORMATS):
            continue

        input_file  = os.path.join(root, file)
        rel_path    = os.path.relpath(input_file, input_root)
        # Always save as .wav
        output_file = os.path.join(output_root, rel_path)
        output_file = os.path.splitext(output_file)[0] + ".wav"

        result = preprocess_audio(input_file, output_file)

        if result == "ok":
            stats["processed"] += 1
        elif result == "skip":
            stats["skipped"] += 1
        elif result == "error":
            stats["errors"] += 1

# -----------------------------------------------
# Summary
# -----------------------------------------------
print(f"\n{'='*45}")
print(f"✅ Preprocessing complete!")
print(f"   Processed : {stats['processed']}")
print(f"   Skipped   : {stats['skipped']}")
print(f"   Errors    : {stats['errors']}")
print(f"{'='*45}")
