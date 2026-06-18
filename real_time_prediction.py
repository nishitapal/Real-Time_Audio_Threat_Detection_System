import os
import json
import time
import logging
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import speech_recognition as sr
import warnings
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
from joblib import load as joblib_load
from intent_detector import detect_intent
warnings.filterwarnings("ignore")

# -----------------------------------------------
# Paths
# -----------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "DNN_ThreatDetector.h5")
SCALER_PATH  = os.path.join(BASE_DIR, "scaler.save")
CONFIG_PATH  = os.path.join(BASE_DIR, "model_config.json")
YAMNET_LOCAL = os.path.join(BASE_DIR, "yamnet_local")
LOG_PATH     = os.path.join(BASE_DIR, "voiceguard.log")

# -----------------------------------------------
# Logging
# -----------------------------------------------
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------------------------
# Recording config
# -----------------------------------------------
SAMPLE_RATE      = 16000
CHUNK_DURATION   = 4
OVERLAP_DURATION = 0
COOLDOWN_SECONDS = 4
SILENCE_RMS      = 0.001

# -----------------------------------------------
# Load threshold
# -----------------------------------------------
THRESHOLD = 0.90
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    saved_thresh = config.get("best_threshold", 0.55)
    THRESHOLD = min(saved_thresh, 0.90)
    print(f"✅ Tone threshold: {THRESHOLD:.2f}")
else:
    print(f"⚠️  Config not found, using default: {THRESHOLD}")

# -----------------------------------------------
# Load models
# -----------------------------------------------
print("🔹 Loading models...")

if os.path.exists(YAMNET_LOCAL):
    yamnet_model = tf.saved_model.load(YAMNET_LOCAL)
else:
    print("⚠️ Downloading YAMNet...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    tf.saved_model.save(yamnet_model, YAMNET_LOCAL)

classifier = load_model(MODEL_PATH)
scaler     = joblib_load(SCALER_PATH)
recognizer = sr.Recognizer()
print("✅ All models loaded.\n")

# -----------------------------------------------
# Total alerts counter
# -----------------------------------------------
total_alerts = 0

# -----------------------------------------------
# Noise reduction
# -----------------------------------------------
def reduce_noise(audio):
    rms = np.sqrt(np.mean(audio**2))
    gate_threshold = rms * 0.5
    audio[np.abs(audio) < gate_threshold] = 0
    return audio

# -----------------------------------------------
# Bandpass filter
# -----------------------------------------------
def bandpass_filter(audio, lowcut=300, highcut=3400, fs=16000):
    nyq = fs / 2
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, audio)

# -----------------------------------------------
# Extract YAMNet features
# -----------------------------------------------
def extract_features(audio):
    waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    return tf.reduce_mean(embeddings, axis=0).numpy()

# -----------------------------------------------
# Layer 1 — Tone Detection
# -----------------------------------------------
def detect_tone(audio):
    audio = bandpass_filter(audio)
    audio = reduce_noise(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    features    = extract_features(audio).reshape(1, -1)
    features    = scaler.transform(features)
    probability = classifier.predict(features, verbose=0)[0][0]
    return probability >= THRESHOLD, float(probability)

# -----------------------------------------------
# Layer 2 — Speech to Text
# -----------------------------------------------
def get_transcript(audio):
    try:
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        audio_data  = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        text        = recognizer.recognize_google(audio_data).lower()
        return text
    except:
        return ""

# -----------------------------------------------
# Alert
# -----------------------------------------------
def send_alert(probability, reason="", transcript=""):
    global total_alerts
    total_alerts += 1
    message = (
        f"\n{'='*55}\n"
        f"🚨 THREAT DETECTED!\n"
        f"   Heard       : \"{transcript}\"\n"
        f"   Confidence  : {probability:.0%}\n"
        f"   Reason      : {reason}\n"
        f"   Time        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"   Total Alerts: {total_alerts}\n"
        f"{'='*55}"
    )
    print(message)
    logging.warning(message)

# -----------------------------------------------
# Combined prediction — Tone + Intent
# -----------------------------------------------
def predict_chunk(audio):
    # Silence check
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < SILENCE_RMS:
        return None, None, None, ""

    # Layer 1 — Tone
    tone_threat, tone_prob = detect_tone(audio)

    # Layer 2 — Intent (via STT + ML classifier)
    transcript     = get_transcript(audio)
    intent_threat  = False
    intent_prob    = 0.0

    if transcript:
        print(f"\n🗣️  Heard: \"{transcript}\"")
        intent_threat, intent_prob = detect_intent(transcript)

    # Final decision
    if intent_threat or (tone_threat and tone_prob >= 0.90):
        reason = []
        if tone_threat:
            reason.append(f"aggressive tone ({tone_prob:.0%})")
        if intent_threat:
            reason.append(f"threatening intent ({intent_prob:.0%})")
        final_prob = max(tone_prob, intent_prob)
        return "THREAT", final_prob, " + ".join(reason), transcript

    return "SAFE", tone_prob, "clean", transcript

# -----------------------------------------------
# Main monitoring loop
# -----------------------------------------------
def run_voiceguard():
    print("🎙️  VoiceGuard is ACTIVE — listening continuously...")
    print(f"    Tone Threshold : {THRESHOLD:.2f}")
    print(f"    Chunk size     : {CHUNK_DURATION}s")
    print(f"    Logs saved     : {LOG_PATH}")
    print("    Press Ctrl+C to stop.\n")

    last_alert_time = 0
    chunk_samples   = int(SAMPLE_RATE * CHUNK_DURATION)

    while True:
        try:
            audio = sd.rec(chunk_samples, samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()

            label, prob, reason, transcript = predict_chunk(audio)

            if label is None:
                print("🔇 Silence...", end="\r")
                continue

            now = time.time()

            if label == "THREAT":
                if now - last_alert_time > COOLDOWN_SECONDS:
                    send_alert(prob, reason=reason, transcript=transcript)
                    last_alert_time = now
                    logging.warning(f"THREAT | prob={prob:.2f} | reason={reason}")
                else:
                    remaining = int(COOLDOWN_SECONDS - (now - last_alert_time))
                    print(f"⚠️  Threat ({prob:.0%}) — cooldown {remaining}s left", end="\r")
            else:
                print(f"🟢 SAFE ({prob:.0%} threat prob)   ", end="\r")
                logging.info(f"SAFE | prob={prob:.2f} | heard={transcript}")

            time.sleep(0.5)

        except sd.PortAudioError as e:
            print(f"\n🎤 Mic error: {e}")
            logging.error(f"Mic error: {e}")
            time.sleep(2)

        except KeyboardInterrupt:
            print("\n\n👋 VoiceGuard stopped.")
            logging.info("VoiceGuard stopped by user.")
            break

        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            logging.error(f"Error: {e}")
            time.sleep(1)

# -----------------------------------------------
# Entry point
# -----------------------------------------------
if __name__ == "__main__":
    run_voiceguard()



