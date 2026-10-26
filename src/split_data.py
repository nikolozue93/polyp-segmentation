
from pathlib import Path
import random
import json

ROOT = Path("data/Kvasir-SEG")
IMAGE_DIR = ROOT / "images"
MASK_DIR = ROOT / "masks"

image_names = {p.name for p in IMAGE_DIR.glob("*.jpg")}
mask_names = {p.name for p in MASK_DIR.glob("*.jpg")}

assert image_names == mask_names, "Image/mask mismatch"

filenames = sorted(image_names)

# use a fixed seed so the same split can be reproduced
rng = random.Random(42)
rng.shuffle(filenames)

n = len(filenames)
n_train = int(n * 0.8) # 80% for training
n_val = int(n * 0.1) # 10% for validation

splits = {
    "train": filenames[:n_train], # first 800
    "val": filenames[n_train:n_train + n_val], # 800-900
    "test": filenames[n_train + n_val:], # last 10%
}

assert not (set(splits["train"]) & set(splits["val"]))
assert not (set(splits["train"]) & set(splits["test"]))
assert not (set(splits["val"]) & set(splits["test"]))

Path("configs").mkdir(exist_ok=True)

with open("configs/splits.json", "w") as f:
    json.dump(splits, f, indent=2)

for name, files in splits.items():
    print(f"{name}: {len(files)} images")
