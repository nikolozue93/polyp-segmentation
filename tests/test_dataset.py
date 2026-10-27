from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.dataset import PolypDataset

image_paths = sorted(Path("data/Kvasir-SEG/images").glob("*.jpg"))

# work with 16 samples for a fast initial test
dataset = PolypDataset(image_paths[:16])

image, mask = dataset[0]

print("Image:", image.shape, image.dtype)
print("Mask:", mask.shape, mask.dtype)

assert image.shape == (3, 256, 256)
assert mask.shape == (1, 256, 256)

assert image.dtype == torch.float32
assert mask.dtype == torch.float32

assert 0 <= image.min() and image.max() <= 1
assert torch.all((mask == 0) | (mask == 1))

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

images, masks = next(iter(loader))

print("Batch images:", images.shape)
print("Batch masks:", masks.shape)
