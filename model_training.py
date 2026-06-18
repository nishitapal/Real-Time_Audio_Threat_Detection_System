import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Recall, Precision
from joblib import dump

# -----------------------------------------------
# Paths
# -----------------------------------------------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED  = os.path.join(BASE_DIR, "Preprocessed_Audios")
MODEL_PATH    = os.path.join(BASE_DIR, "DNN_ThreatDetector.h5")
SCALER_PATH   = os.path.join(BASE_DIR, "scaler.save")
CONFIG_PATH   = os.path.join(BASE_DIR, "model_config.json")

# -----------------------------------------------
# Load embeddings
# -----------------------------------------------
print("🔹 Loading embeddings...")
embeddings = np.load(os.path.join(PREPROCESSED, "embeddings.npy"), allow_pickle=True)
labels_raw = np.load(os.path.join(PREPROCESSED, "labels.npy"),    allow_pickle=True)

# Validate labels
print("Unique labels:", np.unique(labels_raw))
assert set(np.unique(labels_raw)).issubset({"safe", "threat"}), \
    "❌ Unexpected label values — check your dataset folders!"

y = (labels_raw == "threat").astype(int)
X = embeddings

print(f"✅ Loaded {len(y)} samples | Shape: {X.shape}")
print(f"\n📊 Class Distribution (before balancing):")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    name = "threat" if cls == 1 else "safe"
    print(f"   {name}: {cnt} ({cnt/len(y)*100:.1f}%)")

# -----------------------------------------------
# Balance dataset — undersample safe to 10000
# so model focuses more on threat patterns
# -----------------------------------------------
from sklearn.utils import resample

safe_idx   = np.where(y == 0)[0]
threat_idx = np.where(y == 1)[0]

safe_downsampled = resample(safe_idx, n_samples=10000, random_state=42)
balanced_idx     = np.concatenate([safe_downsampled, threat_idx])

X = X[balanced_idx]
y = y[balanced_idx]

print(f"\n📊 Class Distribution (after balancing):")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    name = "threat" if cls == 1 else "safe"
    print(f"   {name}: {cnt} ({cnt/len(y)*100:.1f}%)")
print(f"✅ Balanced dataset: {len(y)} total samples")

# -----------------------------------------------
# Proper 3-way split: train / val / test
# val  → used during training (EarlyStopping)
# test → final evaluation only (never seen during training)
# -----------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"\nSplit → Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")

# -----------------------------------------------
# Scale features
# -----------------------------------------------
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
dump(scaler, SCALER_PATH)
print("✅ Scaler saved.")

# -----------------------------------------------
# Class weights (handles imbalanced dataset)
# -----------------------------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print(f"\n⚖️  Class Weights: {class_weights}")

# -----------------------------------------------
# Model
# -----------------------------------------------
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],),
          kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        AUC(name='auc'),
        Recall(name='recall'),
        Precision(name='precision')
    ]
)

model.summary()

# -----------------------------------------------
# Callbacks
# -----------------------------------------------
callbacks = [
    # Stop if val_auc stops improving (AUC is better than loss for imbalanced data)
    EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6)
]

# -----------------------------------------------
# Train
# -----------------------------------------------
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------------------------
# Find best threshold using F1 score
# -----------------------------------------------
print("\n🔍 Finding best classification threshold...")
probs = model.predict(X_val, verbose=0).flatten()
thresholds = np.arange(0.2, 0.9, 0.05)
best_thresh = max(thresholds, key=lambda t: f1_score(y_val, (probs >= t).astype(int)))
print(f"✅ Best threshold: {best_thresh:.2f}")

# -----------------------------------------------
# Final evaluation on truly unseen test set
# -----------------------------------------------
print("\n📈 Final Evaluation on Test Set:")
test_probs = model.predict(X_test, verbose=0).flatten()
y_pred     = (test_probs >= best_thresh).astype(int)

print(classification_report(y_test, y_pred, target_names=['safe', 'threat']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"AUC Score: {roc_auc_score(y_test, test_probs):.4f}")

# -----------------------------------------------
# Save model and config
# -----------------------------------------------
model.save(MODEL_PATH)
print(f"\n✅ Model saved: {MODEL_PATH}")

config = {
    "best_threshold": float(best_thresh),
    "input_shape": int(X_train.shape[1]),
    "trained_on": str(np.datetime64('today'))
}
with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)
print(f"✅ Config saved: {CONFIG_PATH}")

# -----------------------------------------------
# Training curves
# -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history.history['loss'],     label='Train')
axes[0].plot(history.history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].legend()

axes[1].plot(history.history['auc'],     label='Train')
axes[1].plot(history.history['val_auc'], label='Val')
axes[1].set_title('AUC')
axes[1].legend()

axes[2].plot(history.history['recall'],     label='Train')
axes[2].plot(history.history['val_recall'], label='Val')
axes[2].set_title('Recall (Threat Detection Rate)')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "training_curves.png"))
print("✅ Training curves saved as training_curves.png")
plt.show()
