import json
import os
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TRAIN_DIR = Path(os.environ["TRAIN_DIR"])
VAL_DIR = Path(os.environ["VAL_DIR"])
MODEL_OUTPUT = Path(os.environ["MODEL_OUTPUT"])
CLASS_INDICES_OUTPUT = Path(os.environ.get("CLASS_INDICES_OUTPUT", MODEL_OUTPUT.parent / "class_indices.json"))
TRAINING_HISTORY_OUTPUT = Path(os.environ.get("TRAINING_HISTORY_OUTPUT", MODEL_OUTPUT.parent / "training_history.pkl"))

IMG_SIZE = tuple(int(part.strip()) for part in os.environ.get("RETRAIN_IMG_SIZE", "256,256").split(","))
BATCH_SIZE = int(os.environ.get("RETRAIN_BATCH_SIZE", "16"))
EPOCHS = int(os.environ.get("RETRAIN_EPOCHS", "10"))

MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
CLASS_INDICES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
TRAINING_HISTORY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

if train_gen.samples == 0 or val_gen.samples == 0:
    raise RuntimeError("Training and validation folders must contain images.")

with CLASS_INDICES_OUTPUT.open("w", encoding="utf-8") as class_file:
    json.dump(train_gen.class_indices, class_file, indent=4)

num_classes = len(train_gen.class_indices)
print(f"Class indices saved to {CLASS_INDICES_OUTPUT}")
print(f"Number of classes: {num_classes}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print(f"Starting training for {EPOCHS} epoch(s).")
print(f"Best model output: {MODEL_OUTPUT}")

callbacks = [
    ModelCheckpoint(str(MODEL_OUTPUT), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
]

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
)

model.save(str(MODEL_OUTPUT))
with TRAINING_HISTORY_OUTPUT.open("wb") as history_file:
    pickle.dump(history.history, history_file)

print(f"Model saved to: {MODEL_OUTPUT}")
print(f"Training history saved to: {TRAINING_HISTORY_OUTPUT}")
