from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path("data/Kvasir-SEG")

image_dir = ROOT / "images"
mask_dir = ROOT / "masks"

image_paths = sorted(image_dir.glob("*.jpg"))
mask_paths = sorted(mask_dir.glob("*.jpg"))

print("Images:", len(image_paths))
print("Masks:", len(mask_paths))

assert image_paths, "No images found. Check ROOT."

image_path = image_paths[0]
mask_path = mask_dir / image_path.name

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# tuple of width/height dimensions, (622, 529)
print("Image dimensions:", image.size)
print("Mask dimensions:", mask.size)

# transforming to pixel array
mask_array = np.asarray(mask)

print("Mask minimum:", mask_array.min())
print("Mask maximum:", mask_array.max())
print("Unique mask values:", len(np.unique(mask_array)))

# checking that every image has its own corresponding mask
#for img, msk in zip(image_paths, mask_paths):
#    if img.name != msk.name:
#        raise ValueError("Not matching image and mask pair")
assert {p.name for p in image_paths} == {p.name for p in mask_paths}

# converting the mask to a binary array
# pixels > 127 should become 1, other pixels 0
binary_arr = (mask_array > 127).astype(int)

# displaying original image and binary mask side by side

# figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# plot original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")  # hide grid ticks

# plot mask
axes[1].imshow(binary_arr, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()
