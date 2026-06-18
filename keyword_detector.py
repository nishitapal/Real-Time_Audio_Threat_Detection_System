import speech_recognition as sr
import numpy as np

# -----------------------------------------------
# Threat Keywords — category based
# covers most banking threat scenarios
# -----------------------------------------------
THREAT_KEYWORDS = [
    "give me", "hand over", "empty the",
    "gun", "shoot", "knife", "bomb", "weapon", "fire", "blast",
    "freeze", "hands up",
    "kill", "hurt", "dead", "die", "hostage", "rob", "robbery",
    "murder", "attack",
    "vault", "valuables",
    "help me", "let me go", "please don't"
]
# -----------------------------------------------
# Setup recognizer
# -----------------------------------------------
recognizer = sr.Recognizer()
SAMPLE_RATE = 16000

# -----------------------------------------------
# Detect keywords in audio chunk
# Returns: (list of found keywords, transcript text)
# -----------------------------------------------
def detect_keywords(audio):
    try:
        audio_bytes = (audio * 32767).astype(np.int16).tobytes()
        audio_data  = sr.AudioData(audio_bytes, SAMPLE_RATE, 2)
        text        = recognizer.recognize_google(audio_data).lower()
        found       = [w for w in THREAT_KEYWORDS if w in text]
        return found, text
    except sr.UnknownValueError:
        return [], ""   # could not understand
    except sr.RequestError:
        return [], ""   # no internet
    except Exception:
        return [], ""

