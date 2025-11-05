# AER850 - Project 2 (Step 5)
# Hajaanan Maheskumar 501099977
# Step 5: Model Testing

import json, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image

IMG_SIZE = (500, 500)
MODEL_NAME = "p2_best_model.keras"
CLASS_NAME = "class_indices.json"

def find_file(name: str):
    candidates = [
        Path.cwd() / name,
        Path("/content") / name,
        Path("/content/Data") / name,
        Path("/content/Data/Data") / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Could not find {name}. Looked in: " +
        ", ".join(str(c) for c in candidates)
    )

def find_test_dir():
    for p in [
        Path("/content/Data/test"),
        Path("/content/Data/Test"),
        Path.cwd() / "Data" / "test",
        Path.cwd() / "Data" / "Test",
        Path("/content/Data/Data/test"),
        Path("/content/Data/Data/Test"),
    ]:
        if p.is_dir():
            return p
        
    listing = [str(x) for x in Path("/content/Data").glob("**/*")] if Path("/content/Data").exists() else []
    raise FileNotFoundError("Could not find Data/test. Checked common locations. "
                            f"Current /content/Data contents sample: {listing[:12]} ...")

MODEL_PATH = find_file(MODEL_NAME)
CLASS_JSON = find_file(CLASS_NAME)
TEST_DIR   = find_test_dir()

REQ_FILES = {
    "crack":        "test_crack.jpg",
    "missing-head": "test_missinghead.jpg",
    "paint-off":    "test_paintoff.jpg",
}

def load_x(path: Path):
    img = image.load_img(path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0), img 

# main
if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_JSON) as f:
        name_to_idx = json.load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    for true_name, fname in REQ_FILES.items():
        path = TEST_DIR / true_name / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        x, pil_img = load_x(path)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = idx_to_name[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        plt.figure(figsize=(4,4))
        plt.imshow(pil_img); plt.axis("off")
        plt.title(f"Actual: {true_name}\nPredicted: {pred_name} ({conf:.1f}%)")
        plt.tight_layout(); plt.show()

        print(f"{path.name:18s} | actual: {true_name:12s} | "
              f"predicted: {pred_name:12s} | confidence: {conf:5.1f}%")
