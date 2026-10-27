from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"


class PolypDataset(Dataset):
    def __init__(self, image_paths, image_size=256):
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_dir = DATA_ROOT / "masks"
        self.image_size = image_size

        assert self.image_paths, "Dataset contains no images"

        assert {p.name for p in self.image_paths} <= {p.name for p in self.mask_dir.glob("*.jpg") 
                }, "Some images have no corresponding mask"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        returning one (image_tensor, mask_tensor) pair
        """
        # get image path at index idx
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name


        # RGB image and grayscale mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # resizing both to 256x256
        image_res = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask_res = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        # converting image to float32 and normalizing on [0,1]
        image_normalized = (np.asarray(image_res).astype('float32') / 255.0)

        # converting mask to binary float32
        binary_array = (np.asarray(mask_res) > 127).astype('float32')

        # converting both arrays to PyTorch tensors
        # rearraning image dims to channel height width for CNN/PyTorch
        # add adding a new dimension of size one for the mask's single channel
        t_image = torch.from_numpy(image_normalized).permute(2, 0, 1)
        t_mask = torch.from_numpy(binary_array).unsqueeze(0)

        return (t_image, t_mask)
