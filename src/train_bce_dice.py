import csv, json, random
from pathlib import Path
import numpy as np, torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from src.dataset import PolypDataset
from src.losses import BCEDiceLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"
IMAGE_DIR = DATA_ROOT / "images"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)


# load reproducible splits
with open(PROJECT_ROOT / "configs" / "splits.json") as f:
    splits = json.load(f)

train_paths = [IMAGE_DIR / name for name in splits["train"]]
val_paths = [IMAGE_DIR / name for name in splits["val"]]

train_loader = DataLoader(
    PolypDataset(train_paths),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    PolypDataset(val_paths),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# model, loss, optimizer
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# standard ImageNet normalization stats for RGB images 
# and then viewing each pixel 
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def run_epoch(loader, training=False):
    model.train(training)

    total_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0
    num_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # normalizing RGB channels for pretrained encoder
        images = (images - mean) / std

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                loss.backward()
                optimizer.step()

        # measuring segmentation quality without gradients
        with torch.no_grad():
            predicted = logits > 0
            target = masks > 0.5

            # for each image in the batch, count how many pixels were predicted as foreground 
            # and are also foreground in the ground truth mask
            intersection = (predicted & target).sum(dim=(1, 2, 3))

            # count the predicted and actual foreground pixels separately for each image
            # important cause because we're calculating dice and iou per image, 
            # then averaging those scores over the dataset
            # if we summed across all four dimensions, 
            # we'd merge the foreground pixels of all eight images together 
            # and compute a batch level aggregate rather than individual image scores
            predicted_pixels = predicted.sum(dim=(1, 2, 3))
            target_pixels = target.sum(dim=(1, 2, 3))

            eps = 1e-7

            # just dice and iou funcs
            dice = (2 * intersection + eps) / (
                predicted_pixels + target_pixels + eps
            )

            iou = (intersection + eps) / (
                predicted_pixels + target_pixels - intersection + eps
            )

            n = images.size(0)
            total_loss += loss.item() * n
            dice_total += dice.sum().item()
            iou_total += iou.sum().item()
            num_samples += n

    return {
        "loss": total_loss / num_samples,
        "dice": dice_total / num_samples,
        "iou": iou_total / num_samples,
    }


# main training loop
best_dice = -1.0
history = []

for epoch in range(EPOCHS):
    train_metrics = run_epoch(train_loader, training=True)
    val_metrics = run_epoch(val_loader, training=False)

    record = {
        "epoch": epoch + 1,
        "train_loss": train_metrics["loss"],
        "val_loss": val_metrics["loss"],
        "val_dice": val_metrics["dice"],
        "val_iou": val_metrics["iou"],
    }
    history.append(record)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_metrics['loss']:.4f} | "
        f"Val loss: {val_metrics['loss']:.4f} | "
        f"Dice: {val_metrics['dice']:.4f} | "
        f"IoU: {val_metrics['iou']:.4f}",
        flush=True,
    )

    # save the best validation checkpoint
    if val_metrics["dice"] > best_dice:
        best_dice = val_metrics["dice"]

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_dice,
                "model_name": "unet_resnet18",
                "image_size": 256,
            },
            CHECKPOINT_DIR / "bce_dice_best.pt",
        )
        print("  Best checkpoint saved.", flush=True)

    # persist metrics after each epoch
    with open(RESULTS_DIR / "bce_dice_metrics.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

print("Training finished. Best validation Dice:", best_dice)
