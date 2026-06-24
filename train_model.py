"""
=======================================================
  LSTM Training — Thai Sign Language
  เทรน model จาก dataset ที่เก็บด้วย collect_data.py
=======================================================
วิธีใช้:  python train_model.py
Output:   model/lstm_model.h5
          model/label_map.json
          model/confusion_matrix.png
=======================================================
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

GESTURES        = ["sawatdi", "khob_khun", "kho_thot"]
DATA_DIR        = "data"
MODEL_DIR       = "model"
SEQUENCE_LENGTH = 30
FEATURES        = 126   # 2 hands × 21 landmarks × 3 (x,y,z)
EPOCHS          = 200
BATCH_SIZE      = 32
VAL_SPLIT       = 0.20
WARN_THRESHOLD  = 0.70


def load_dataset():
    X, y = [], []
    for label, gesture in enumerate(GESTURES):
        folder = os.path.join(DATA_DIR, gesture)
        if not os.path.exists(folder):
            print(f"  [WARN] folder not found: {folder}")
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        skipped = 0
        for fname in files:
            seq = np.load(os.path.join(folder, fname))
            if seq.shape == (SEQUENCE_LENGTH, FEATURES):
                X.append(seq)
                y.append(label)
            else:
                skipped += 1
        print(f"  {gesture}: {len(files) - skipped} loaded"
              + (f", {skipped} skipped (wrong shape)" if skipped else ""))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, FEATURES)),
        LSTM(32),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(len(GESTURES), activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(model, X_val, y_val, best_val_acc):
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm     = confusion_matrix(y_val, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GESTURES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (validation) — best val_acc = {best_val_acc:.1%}")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved: {path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 56)
    print("  LSTM Training — Thai Sign Language")
    print("=" * 56)

    print("\nLoading dataset...")
    X, y = load_dataset()
    if len(X) == 0:
        print("\n[ERROR] No data found. Run collect_data.py first.")
        return
    print(f"  Total: {len(X)} sequences  shape={X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}   Val: {len(X_val)}")

    model = build_model()
    model.summary()

    model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
    callbacks = [
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc  = max(history.history["val_accuracy"])
    final_acc     = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    print("\n" + "=" * 56)
    print(f"  Final train accuracy : {final_acc:.4f}  ({final_acc:.1%})")
    print(f"  Final val accuracy   : {final_val_acc:.4f}  ({final_val_acc:.1%})")
    print(f"  Best  val accuracy   : {best_val_acc:.4f}  ({best_val_acc:.1%})")
    print("=" * 56)

    label_map = {str(i): g for i, g in enumerate(GESTURES)}
    label_map_path = os.path.join(MODEL_DIR, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {label_map_path}")

    best_model = tf.keras.models.load_model(model_path)
    save_confusion_matrix(best_model, X_val, y_val, best_val_acc)

    print(f"\n  >>> Change a applied — val_accuracy: {best_val_acc:.1%}")

    if best_val_acc < WARN_THRESHOLD:
        print("\n" + "=" * 56)
        print(f"  [WARNING] val_accuracy = {best_val_acc:.1%} — ต่ำกว่า 70%")
        print("  แนะนำ:")
        print("    1. เก็บ data เพิ่ม (ควร >= 90 sequences/gesture)")
        print("    2. บันทึกจากหลายคน / หลายมุม / หลายแสง")
        print("    3. ปรึกษา Claude Code ก่อนแก้ architecture")
        print("=" * 56)
    else:
        print(f"\n  [OK] val_accuracy = {best_val_acc:.1%} — พร้อมใช้งาน")
        print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
