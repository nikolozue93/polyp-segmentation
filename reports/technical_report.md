# Technical Report — Polyp Segmentation with U-Net

**Project:** Binary segmentation of gastrointestinal polyps in endoscopic images  
**Framework:** PyTorch and `segmentation_models_pytorch`  
**Status:** Research/educational prototype; not a medical device

## 1. Problem and objective

The goal was to build an end-to-end computer-vision pipeline that segments polyp regions in colonoscopy images and exposes a usable inference interface. In addition to segmentation quality, the project assesses repeatability, failure cases, and inference efficiency. The output is a binary foreground mask and a visual overlay on an uploaded image. This system is **not clinically validated** and must not be used for diagnosis.

![Training and inference workflow](../assets/polyp_pipeline.png)

## 2. Data and preprocessing

**Dataset.** Kvasir-SEG contains 1,000 endoscopic images paired with segmentation masks. The dataset was divided deterministically with Python's `random.Random(42)` into 800 training, 100 validation, and 100 held-out test images. Original filenames are recorded in `configs/splits.json`; matching image and mask filenames are used to prevent pairing errors. The held-out test split was not used for model selection.

**Preprocessing.** Images were converted to RGB, resized to **256 × 256** using bilinear interpolation, and scaled to `[0, 1]`. Inputs were subsequently normalized using ImageNet channel statistics (mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]`) to match the pretrained encoder. Ground-truth masks were read as grayscale, resized using nearest-neighbor interpolation, and binarized by thresholding values above 127. Image/mask tensors had shapes `(3, 256, 256)` and `(1, 256, 256)` respectively. The same input preprocessing was applied for validation, test evaluation, and interactive inference.

**Reproducibility.** The source code, committed split manifest, metrics CSV files, and an accompanying Colab notebook document the workflow. Fixed seeds were used for Python, NumPy, and PyTorch; GPU-level bitwise determinism was not independently established. The dataset must be downloaded separately under its original terms of use.

## 3. Model and training design

A **U-Net** segmentation model with an **ImageNet-pretrained ResNet-18 encoder** was implemented using `segmentation_models_pytorch`. A pretrained, lightweight encoder was selected as a practical compromise between available GPU resources and image representation quality. The decoder outputs one logit per pixel, representing a binary foreground/background prediction. At inference, `logit > 0` corresponds to a sigmoid probability greater than 0.5.

Both experiments used the same model architecture, dataset split, 256 × 256 input resolution, Adam optimizer, learning rate **0.0001**, batch size **8**, seed **42**, and **5 training epochs**. Model weights were updated using backpropagation during training only. At each epoch, validation was run without gradient tracking and the best checkpoint was selected by **mean per-image validation Dice**.

### Experiment A — BCE baseline

The baseline objective was `torch.nn.BCEWithLogitsLoss()`, combining a sigmoid activation and binary cross-entropy in a numerically stable implementation. This evaluates foreground classification at the pixel level.

### Experiment B — BCE + soft Dice loss

The second objective was the sum of BCE and soft Dice loss:

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{BCE}} + (1 - \mathrm{Dice}_{\mathrm{soft}}).
$$

The soft Dice term uses sigmoid probabilities rather than thresholded binary predictions so gradients can propagate. The implementation computes per-image soft overlap, averages over the batch, and uses a smoothing constant of **1.0**. BCE and Dice were combined with equal coefficients. The motivation was to complement pixel-wise classification with a region-overlap objective while changing only one component of the baseline experiment.

## 4. Training results and model selection

| Epoch | BCE validation Dice | BCE + Dice validation Dice |
|---:|---:|---:|
| 1 | 0.7077 | 0.7308 |
| 2 | 0.7985 | 0.8158 |
| 3 | 0.8248 | 0.8424 |
| 4 | 0.8495 | 0.8503 |
| 5 | 0.8495 | **0.8521** |

| Metric at selected checkpoint | BCE baseline | BCE + Dice |
|---|---:|---:|
| Validation Dice | 0.8495 | **0.8521** |
| Validation IoU | 0.7637 | **0.7675** |
| Training epochs | 5 | 5 |

Both models improved across the five epochs. The combined-loss model achieved approximately **0.0026 absolute Dice** (0.26 percentage points) higher validation Dice than BCE alone and was selected for test evaluation. The difference is small; a single seeded run per configuration does **not** establish statistical significance or general superiority. The absolute values of BCE loss and BCE + Dice loss cannot be compared directly because the optimization objectives differ.

Training histories are saved in `results/baseline_metrics.csv` and `results/bce_dice_metrics.csv`.

## 5. Held-out evaluation

The selected model was loaded from the **epoch-5 checkpoint** and evaluated on the 100 previously held-out test images. The predicted binary masks were derived by thresholding logits at zero. Dice and Intersection over Union (IoU) were calculated independently for each image and then averaged.

| Metric | Held-out test result |
|---|---:|
| Mean Dice | **0.8508** |
| Mean IoU | **0.7642** |

Test and validation metrics were close in this particular split (test Dice 0.8508 vs. validation Dice 0.8521), which is encouraging but does not demonstrate cross-institution or cross-device generalization. Per-image scores are in `results/test_metrics.csv`.

### Error analysis

The five lowest-scoring test images had Dice scores of approximately **0.231**, **0.373**, **0.377**, **0.481**, and **0.481**. Visual inspection indicated that the model generally identified at least part of the polyp but often under-segmented the true region, sometimes incorrectly labeling surrounding tissue as foreground. Difficult viewing angles and weak separation between polyp and adjacent mucosa appeared in some failures. These are observational hypotheses, not verified causal explanations.

Where redistribution is permitted under the dataset terms, `results/worst_predictions.png` shows the original image, annotated mask, predicted mask, and overlay for each case. A high mean score therefore coexists with poor performance on some individual images — an important limitation for sensitive applications.

## 6. Measured inference optimization

To assess computational efficiency, the same selected checkpoint was benchmarked with **FP32** inference and **automatic mixed precision (CUDA FP16 autocast)**. Both modes used the 100 validation images, batch size **1**, and the same input preprocessing. GPU forward passes were timed using `torch.cuda.Event`, with warm-up iterations and synchronization before reading elapsed times. This measures **model forward-pass latency with tensors already on the GPU**, excluding file decoding, preprocessing, host-to-device transfer, and Gradio/network latency.

| Measurement | FP32 | Mixed precision (FP16 autocast) |
|---|---:|---:|
| Mean latency per image | 7.919 ms | 5.469 ms |
| Median latency per image | 7.692 ms | 5.334 ms |
| Validation Dice | 0.85214 | 0.85216 |
| Validation IoU | 0.76751 | 0.76753 |

Median latency improved by **1.44×**, equivalent to an approximately **30.6% decrease** under the measured conditions, with no meaningful change in validation quality. The minor Dice/IoU variation does not demonstrate an accuracy improvement. GPU architecture, software versions, load, and benchmarking repetitions influence timing, so the measurement should not be interpreted as a universal speedup.

**Benchmark environment:** Google Colab GPU name **Tesla T4, 15360 MiB**, Python/PyTorch versions **3.13.16/2.11.0+cu130**. The raw comparison is in `results/optimization.csv`.

## 7. Practical application and artifacts

A **Gradio** application (`app.py`) accepts an endoscopic image and displays a predicted binary segmentation mask and a red-overlay visualization. The inference module (`src/inference.py`) loads the selected model parameters, reproduces the training-time input preprocessing, runs a forward pass, and resizes the predicted mask back to the original image dimensions using nearest-neighbor interpolation. A local smoke test confirmed that the app produces visible outputs.

The selected model is distributed as an inference-only `model.pt` file, separate from the repository source code. Users obtain it from the repository's **GitHub Release `v1.0.0`**, place it under `weights/model.pt`, and run `python app.py`. Reproducing training requires obtaining Kvasir-SEG separately. The evaluation and benchmark scripts currently expect complete training checkpoints; they do not automatically load the inference-only file.

## 8. Engineering challenges, limitations, and improvements

**Challenges addressed.** During development, attention was required to correctly match masks to images by filename, preserve binary mask labels when resizing, handle PyTorch package/module imports, use paths that work outside Colab, maintain exact preprocessing parity at inference, and persist model artifacts reliably. The updated repository was verified by running the dataset and model smoke tests after reorganizing the project structure and correcting import paths. The trained model weights were also downloaded from the GitHub Release and successfully tested with the Gradio inference application.

**Limitations.** Only one dataset and one fixed split were evaluated. There was no external validation, calibration study, repeated-seed analysis, or assessment of clinical risk. The 256 × 256 resolution may obscure fine boundaries. The benchmark measures isolated GPU model latency rather than complete service latency and may not predict CPU or deployment performance. The model can fail on difficult or visually ambiguous images; its average test score does not justify medical use.

**Future work.** Evaluate synchronized image/mask augmentations, longer training with early stopping, repeated seeds and confidence intervals, higher input resolution, boundary-focused objectives, calibration, and cross-dataset testing. Improve the inference benchmark with repeated interleaved runs, explicit device specifications, and end-to-end application timings.

## 9. AI assistance and provenance

OpenAI ChatGPT was used to explain PyTorch concepts, guide debugging, draft portions of the training/evaluation code, implement the combined loss and the benchmarking scaffold, and assist with documentation. Some source files are predominantly AI-generated. The code was executed during the project; real metrics, measured latencies, and qualitative observations were obtained from those runs and reviewed by the project author. See `AI-USAGE.md` for a precise account of assistance and verification. This work does not claim a novel neural-network architecture or loss function.

## References

Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). **Kvasir-SEG: A Segmented Polyp Dataset**. *International Conference on Multimedia Modeling*, 451–462. Springer. https://doi.org/10.1007/978-3-030-37734-2_37

Dataset and terms: https://datasets.simula.no/kvasir-seg/
