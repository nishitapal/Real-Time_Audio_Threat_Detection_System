import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

# -----------------------------------------------
# Paths
# -----------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "intent_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "intent_model.save")
TFIDF_PATH  = os.path.join(BASE_DIR, "intent_tfidf.save")

# -----------------------------------------------
# Load data
# -----------------------------------------------
print("🔹 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna()
df = df[df['label'].isin(['safe', 'threat'])]
print(f"✅ Loaded {len(df)} sentences")
print(df['label'].value_counts())

X = df['sentence'].values
y = df['label'].values

# -----------------------------------------------
# Train/Test split
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------------------------
# TF-IDF Vectorizer
# -----------------------------------------------
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)

# -----------------------------------------------
# Train Logistic Regression
# -----------------------------------------------
print("\n🚀 Training intent classifier...")
model = LogisticRegression(max_iter=1000, class_weight='balanced',C=5)
model.fit(X_train_vec, y_train)

# -----------------------------------------------
# Evaluate
# -----------------------------------------------
y_pred = model.predict(X_test_vec)
print("\n📈 Results:")
print(classification_report(y_test, y_pred, target_names=['safe', 'threat']))

# -----------------------------------------------
# Save model and vectorizer
# -----------------------------------------------
dump(model, MODEL_PATH)
dump(tfidf,  TFIDF_PATH)
print(f"✅ Intent model saved!")
print(f"✅ TF-IDF vectorizer saved!")
