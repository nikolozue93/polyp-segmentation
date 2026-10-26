import json, csv, torch
from pathlib import Path
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from src.dataset import PolypDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

drive_root = Path("/content/drive/MyDrive/polyp-segmentation")
checkpoint_path = drive_root / "checkpoints/bce_dice_best.pt"
results_dir = drive_root / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# loading the untouched test split
with open("configs/splits.json") as f:
    splits = json.load(f)

test_paths = [Path("data/Kvasir-SEG/images") / name for name in splits["test"]]

test_loader = DataLoader(
    PolypDataset(test_paths),
    batch_size=8,
    shuffle=False
)

# reconstructing the same architecture
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
).to(device)

# restoring learned weights
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])

# identical normalization to training
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)

std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


# evaluation
records = []
offset = 0

with torch.inference_mode():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        images = (images - mean) / std

        logits = model(images)

        predicted = logits > 0
        target = masks > 0.5

        intersection = (predicted & target).sum(dim=(1, 2, 3))
        p_count = predicted.sum(dim=(1, 2, 3))
        t_count = target.sum(dim=(1, 2, 3))

        eps = 1e-7

        dice = (2 * intersection + eps) / (p_count + t_count + eps)

        iou = (intersection + eps) / (p_count + t_count - intersection + eps)

        for j in range(images.size(0)):
            records.append({
                "image": test_paths[offset + j].name,
                "dice": dice[j].item(),
                "iou": iou[j].item(),
            })

        offset += images.size(0)

assert len(records) == len(test_paths)

# summary
mean_dice = sum(r["dice"] for r in records) / len(records)
mean_iou = sum(r["iou"] for r in records) / len(records)

print(f"Test Dice: {mean_dice:.4f}")
print(f"Test IoU:  {mean_iou:.4f}")

# saving per-image metrics
with open(results_dir / "test_metrics.csv", "w") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "dice", "iou"]
    )
    writer.writeheader()
    writer.writerows(records)

# identifying the worst predictions for later inspection
worst = sorted(records, key=lambda r: r["dice"])[:5]

print("\nFive lowest-Dice images:")
for record in worst:
    print(record)
