# AER850 – Project 2 (Steps 1–4)
# Hajaanan Maheskumar 501099977

import os, io, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

#Config
SEED        = 42
IMG_SIZE    = (500, 500)
BATCH_SIZE  = 32
EPOCHS      = 20

MODEL_PATH      = "p2_best_model.keras"
CLASS_MAP_JSON  = "class_indices.json"
HPARAMS_JSON    = "p2_hparams.json"
TRAIN_LOG_CSV   = "training_log.csv"
MODEL_SUMMARY   = "model_summary.txt"

tf.keras.utils.set_random_seed(SEED)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
for d in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(d, True)
    except: pass

# Data roots
def _root():
    for p in [Path("/content/Data"), Path("./Data"), Path("./data")]:
        if p.exists(): return p.resolve()
    return Path("./Data").resolve()

DATA      = _root()
TRAIN_DIR = (DATA / "Train") if (DATA/"Train").exists() else (DATA / "train")
VAL_DIR   = (DATA / "Validation") if (DATA/"Validation").exists() else (DATA / "valid")

# Step 1: Datasets
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_ds.class_names)
    with open(CLASS_MAP_JSON, "w") as f:
        json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.05, 0.05),
    ])
    rescale = layers.Rescaling(1/255.)

    def map_train(x, y): return rescale(aug(x, training=True)), y
    def map_eval(x, y):  return rescale(x), y

    train_ds = train_ds.map(map_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(map_eval,  num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    counts = [sum(1 for p in (TRAIN_DIR/c).iterdir() if p.is_file()) for c in class_names]
    total  = float(sum(counts))
    raw_w  = [total / (len(counts) * max(1, c)) for c in counts]
    scale  = len(raw_w) / sum(raw_w)
    class_weights = {i: float(w * scale) for i, w in enumerate(raw_w)}

    return train_ds, val_ds, class_names, class_weights

# Step 2: Model
def build_model(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inp, out, name="cnn_baseline")

# Step 3: Compile
def compile_model(model: tf.keras.Model):
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    with open(HPARAMS_JSON, "w") as f:
        json.dump({
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "optimizer": "Adam(1e-4)",
            "loss": "categorical_crossentropy",
            "augment": {"flip":"H","rot":0.05,"zoom":0.08,"translate":0.05}
        }, f, indent=2)

    return model

# Step 4: Train & Plot
def train_and_plot(model, train_ds, val_ds, class_weights):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(TRAIN_LOG_CSV, append=False),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    s = io.StringIO()
    model.summary(print_fn=lambda t: s.write(t + "\n"))
    with open(MODEL_SUMMARY, "w") as f:
        f.write(s.getvalue())

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history["accuracy"], label="Train")
    plt.plot(hist.history["val_accuracy"], label="Validation")
    plt.title("Model Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history["loss"], label="Train")
    plt.plot(hist.history["val_loss"], label="Validation")
    plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout(); plt.show()

# Main 
if __name__ == "__main__":
    print("Train dir:", TRAIN_DIR)
    print("Valid dir:", VAL_DIR)

    train_ds, val_ds, class_names, class_weights = load_datasets()
    model = build_model(num_classes=len(class_names))
    model = compile_model(model)
    train_and_plot(model, train_ds, val_ds, class_weights)
