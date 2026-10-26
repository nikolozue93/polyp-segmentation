# initial benchmark implementation generated with ChatGPT assistance
# see AI_USAGE.md for development and validation details

import statistics, torch

def benchmark(model, loader, mean, std, use_amp=False):
    model.eval()

    times = []
    dice_scores = []
    iou_scores = []

    with torch.inference_mode():
        for images, masks in loader:
            images = (images.cuda() - mean) / std
            masks = masks.cuda()

            # warm up the GPU before measuring
            if not times:
                for _ in range(10):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp,):
                        _ = model(images)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            with torch.autocast("cuda", dtype=torch.float16,enabled=use_amp,):
                logits = model(images)

            end.record()
            torch.cuda.synchronize()

            times.append(start.elapsed_time(end))

            predicted = logits > 0
            target = masks > 0.5

            intersection = (predicted & target).sum().item()
            p = predicted.sum().item()
            t = target.sum().item()

            dice_scores.append(
                (2 * intersection + 1e-7) / (p + t + 1e-7)
            )
            iou_scores.append(
                (intersection + 1e-7) /
                (p + t - intersection + 1e-7)
            )

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "dice": statistics.mean(dice_scores),
        "iou": statistics.mean(iou_scores),
    }



if __name__ == "__main__":
    import csv
    import json
    from pathlib import Path

    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader
    from src.dataset import PolypDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this benchmark.")

    device = torch.device("cuda")

    DRIVE_ROOT = Path("/content/drive/MyDrive/polyp-segmentation")
    CHECKPOINT = DRIVE_ROOT / "checkpoints/bce_dice_best.pt"
    RESULTS = DRIVE_ROOT / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Load validation images, not test images.
    with open("configs/splits.json") as f:
        splits = json.load(f)

    val_paths = [
        Path("data/Kvasir-SEG/images") / name
        for name in splits["val"]
    ]

    val_loader = DataLoader(
        PolypDataset(val_paths),
        batch_size=1,
        shuffle=False,
    )

    # 2. Reconstruct and load the trained model.
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Same normalization as training.
    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device,
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device,
    ).view(1, 3, 1, 1)

    # 4. Benchmark both configurations.
    print("Running FP32 benchmark...", flush=True)

    fp32_results = benchmark(
        model, val_loader, mean, std, use_amp=False
    )

    print("Running mixed-precision benchmark...", flush=True)

    amp_results = benchmark(
        model, val_loader, mean, std, use_amp=True
    )

    # 5. Compare results.
    print("\nFP32 results:", fp32_results)
    print("Mixed precision results:", amp_results)

    speedup = (
        fp32_results["median_ms"] /
        amp_results["median_ms"]
    )

    print(f"\nMedian latency speedup: {speedup:.2f}x")

    # 6. Save measurements.
    records = [
        {"mode": "FP32", **fp32_results},
        {"mode": "AMP_FP16", **amp_results},
    ]

    output_path = RESULTS / "optimization.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    print("Results saved to:", output_path)
