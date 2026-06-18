import os
from joblib import load

# -----------------------------------------------
# Load saved model and vectorizer
# -----------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "intent_model.save")
TFIDF_PATH = os.path.join(BASE_DIR, "intent_tfidf.save")

intent_model = None
tfidf        = None

if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
    intent_model = load(MODEL_PATH)
    tfidf        = load(TFIDF_PATH)
    print("✅ Intent model loaded.")
else:
    print("⚠️  Intent model not found — run train_intent_model.py first!")

# -----------------------------------------------
# Predict intent from text
# Returns: (is_threat, confidence)
# -----------------------------------------------
def detect_intent(text):
    if not text or intent_model is None:
        return False, 0.0
    try:
        vec   = tfidf.transform([text.lower()])
        proba = intent_model.predict_proba(vec)[0]
        classes = intent_model.classes_.tolist()
        threat_prob = proba[classes.index('threat')]
        is_threat   = threat_prob >= 0.6
        return is_threat, float(threat_prob)
    except Exception:
        return False, 0.0
