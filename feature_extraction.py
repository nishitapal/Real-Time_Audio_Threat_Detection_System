import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------------------------
# Paths
# -----------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED    = os.path.join(BASE_DIR, "Preprocessed_Audios")
YAMNET_LOCAL    = os.path.join(BASE_DIR, "yamnet_local")
OUTPUT_EMB      = os.path.join(PREPROCESSED, "embeddings.npy")
OUTPUT_LABELS   = os.path.join(PREPROCESSED, "labels.npy")
OUTPUT_PATHS    = os.path.join(PREPROCESSED, "file_paths.npy")

# Explicit folder → label mapping (no fragile string matching)
CATEGORY_MAP = {
    "all_safe":   "safe",
    "all_threat": "threat"
}

# -----------------------------------------------
# Load YAMNet — locally if available, else download
# -----------------------------------------------
print("🔹 Loading YAMNet model...")
if os.path.exists(YAMNET_LOCAL):
    yamnet_model = tf.saved_model.load(YAMNET_LOCAL)
    print("✅ Loaded YAMNet from local cache.")
else:
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    tf.saved_model.save(yamnet_model, YAMNET_LOCAL)
    print("✅ YAMNet downloaded and saved locally for future use.")

# -----------------------------------------------
# Extract YAMNet embedding from a file
# Uses librosa — same as preprocessing and inference
# -----------------------------------------------
def extract_embedding(file_path):
    try:
        # Load exactly the same way as inference script
        y, _ = librosa.load(file_path, sr=16000, mono=True)

        # Skip silent files
        if np.max(np.abs(y)) < 1e-6:
            return None

        # Normalize
        y = y / np.max(np.abs(y))

        waveform = tf.convert_to_tensor(y, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform)
        mean_emb = tf.reduce_mean(embeddings, axis=0).numpy()

        # Validate embedding
        if mean_emb.shape[0] != 1024:
            return None
        if np.any(np.isnan(mean_emb)) or np.any(np.isinf(mean_emb)):
            return None
        if np.all(mean_emb == 0):
            return None

        return mean_emb

    except Exception as e:
        print(f"  ⚠️ Error: {os.path.basename(file_path)} → {e}")
        return None

# -----------------------------------------------
# Load existing progress if resuming
# -----------------------------------------------
processed_paths = set()
all_data = []

if os.path.exists(OUTPUT_PATHS):
    existing_paths  = np.load(OUTPUT_PATHS,  allow_pickle=True)
    existing_labels = np.load(OUTPUT_LABELS, allow_pickle=True)
    existing_embs   = np.load(OUTPUT_EMB,    allow_pickle=True)
    processed_paths = set(existing_paths.tolist())
    for p, l, e in zip(existing_paths, existing_labels, existing_embs):
        all_data.append({"file_path": p, "label": l, "embedding": e})
    print(f"🔄 Resuming — {len(processed_paths)} files already processed.")

# -----------------------------------------------
# Extract features
# -----------------------------------------------
for category, label in CATEGORY_MAP.items():
    folder = os.path.join(PREPROCESSED, category)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    all_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                all_files.append(os.path.join(root, f))

    print(f"\n📂 {category} ({label}): {len(all_files)} files found")

    for path in tqdm(all_files, desc=f"Extracting {label}"):
        if path in processed_paths:
            continue

        emb = extract_embedding(path)
        if emb is not None:
            all_data.append({"file_path": path, "label": label, "embedding": emb})
            processed_paths.add(path)

# -----------------------------------------------
# Save as numpy binary (much faster than CSV)
# -----------------------------------------------
embeddings  = np.vstack([d["embedding"] for d in all_data])
labels      = np.array([d["label"] for d in all_data])
file_paths  = np.array([d["file_path"] for d in all_data])

np.save(OUTPUT_EMB,    embeddings)
np.save(OUTPUT_LABELS, labels)
np.save(OUTPUT_PATHS,  file_paths)

print(f"\n✅ Saved {len(labels)} embeddings to .npy files")

# -----------------------------------------------
# Class distribution report
# -----------------------------------------------
unique, counts = np.unique(labels, return_counts=True)
print("\n📊 Dataset Distribution:")
for cls, cnt in zip(unique, counts):
    pct = cnt / len(labels) * 100
    print(f"   {cls}: {cnt} samples ({pct:.1f}%)")

# -----------------------------------------------
# PCA Visualization — tells you if features are separable
# -----------------------------------------------
print("\n🔍 Generating PCA visualization...")
pca     = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

plt.figure(figsize=(8, 6))
colors = {"safe": "green", "threat": "red"}
for lbl, color in colors.items():
    mask = labels == lbl
    plt.scatter(reduced[mask, 0], reduced[mask, 1],
                c=color, label=lbl, alpha=0.5, s=15)
plt.legend()
plt.title("YAMNet Embedding Space — PCA (check if clusters are separate)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "embedding_pca.png"))
print("✅ PCA plot saved as embedding_pca.png — open it to check separability")
plt.show()
