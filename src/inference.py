from pathlib import Path
import numpy as np, torch
import segmentation_models_pytorch as smp
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "weights" / "model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 256

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    weights = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(weights)
    model = model.to(DEVICE)
    model.eval()

    return model


def predict(model, image: Image.Image):
    original = image.convert("RGB")

    # matching the resolution and normalization used in training
    resized = original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR,)

    pixels = np.asarray(resized, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    tensor = (tensor - MEAN) / STD
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)

    # Logit > 0 is equivalent to sigmoid(logit) > 0.5.
    mask_array = ((logits[0, 0] > 0).to(torch.uint8).cpu().numpy() * 255)

    mask = Image.fromarray(mask_array)

    # Restore the output to the input image's original dimensions.
    mask = mask.resize(original.size, Image.Resampling.NEAREST,)

    # Highlight predicted polyp pixels in red.
    overlay_array = np.asarray(original).copy()
    foreground = np.asarray(mask) > 0

    red = np.array([255, 0, 0])
    overlay_array[foreground] = (0.6 * overlay_array[foreground] + 0.4 * red).astype(np.uint8)

    overlay = Image.fromarray(overlay_array)

    return mask, overlay
