# Satellite Image Super-Resolution (4×) on EuroSAT

A PyTorch project that upscales low-resolution Sentinel-2 satellite patches by 4× (16×16 → 64×64) and measures how much a small residual CNN improves on plain bicubic interpolation.

On the held-out test set (2,700 images) the model reaches **33.94 dB PSNR / 0.831 SSIM**, against **32.69 dB / 0.783** for bicubic.

> **Scope note.** The low-resolution inputs are made by downsampling the original patches, so this is a *synthetic* super-resolution setup, not real sensor blur. See [Limitations](#limitations).

---

## Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Method](#method)
- [Results](#results)
- [Figures](#figures)
- [How to run](#how-to-run)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Overview

Free satellite imagery such as Sentinel-2 has a ground resolution of about 10 m per pixel, so thin roads, field edges and small buildings blur together. Super-resolution tries to recover some of that lost structure with a learned model.

This project builds the full pipeline end to end:

1. Load EuroSAT patches and build (low-res, high-res) training pairs.
2. Measure a bicubic baseline.
3. Train a residual CNN with an L1 loss.
4. Evaluate on a held-out test set with PSNR and SSIM.
5. Analyse where the model helps most, per land-cover class.

## Dataset

**EuroSAT (RGB)**: 27,000 Sentinel-2 patches, 64×64 pixels, 10 land-use / land-cover classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).

The dataset's own split files are used:

| Split | File | Images |
|---|---|---|
| Train | `train.csv` | 18,900 |
| Validation | `validation.csv` | 5,400 |
| Test | `test.csv` | 2,700 |

The dataset is **not included** in this repository. Download EuroSAT separately (see [Acknowledgements](#acknowledgements)) and follow its licence.

**Training pairs.** The 64×64 original is the target. The input is the same patch downsampled to 16×16 with bicubic interpolation (`PIL.Image.BICUBIC`). Pixel values are scaled to [0, 1].

## Method

### Model: `SRNet2` (324,912 parameters)

```
input 16×16×3
  ├─ bicubic ×4 upsample ───────────────────────────────┐ (global skip)
  └─ Conv 3→64
       └─ 4 × ResBlock (Conv → ReLU → Conv, + skip)
            └─ (+ skip around the blocks)
                 └─ Conv 64→48 → PixelShuffle(4)  ───────┤
                                                          ▼
                                                       sum → output 64×64×3
```

- The network predicts only the **residual** on top of the bicubic upsampling, so it starts from the baseline and learns the missing detail.
- `PixelShuffle(4)` rearranges 48 channels into 3 channels at 4× the height and width.

An earlier, smaller model (3 conv layers, 103,344 parameters, no residual blocks) reached 33.64 dB validation PSNR and then stopped improving. It is kept in the notebook as `SRNet`.

### Training

| Setting | Value |
|---|---|
| Loss | L1 |
| Optimizer | Adam, lr = 1e-3 |
| Schedule | Cosine annealing, `T_max = 10` |
| Epochs | 10 |
| Batch size | 16 |
| Hardware | CPU (PyTorch 2.11.0, CPU build) |
| Checkpoint | Best validation PSNR (`srnet2_best.pth`) |

### Evaluation

- **PSNR** is computed per image on [0, 1] images (peak value 1.0) and averaged.
- **SSIM** uses `skimage.metrics.structural_similarity` (`channel_axis=2`, `data_range=1.0`).
- **Baseline** is `torch.nn.functional.interpolate(..., mode="bicubic")` on the same 16×16 input, clamped to [0, 1].
- Validation was used to choose the checkpoint. The test set was evaluated once, at the end.

## Results

### Test set (2,700 images)

| Metric | Bicubic | `SRNet2` | Gain |
|---|---|---|---|
| PSNR (dB) | 32.69 | **33.94** | +1.25 |
| SSIM | 0.783 | **0.831** | +0.048 |

### Validation PSNR across model versions

| Model | Parameters | Val PSNR (dB) |
|---|---|---|
| Bicubic | – | 32.64 |
| `SRNet` (3 conv layers) | 103,344 | 33.64 |
| `SRNet2` (residual blocks + bicubic skip) | 324,912 | **33.91** |

### Gain per land-cover class (validation, PSNR in dB)

| Class | Model | Bicubic | Gain |
|---|---|---|---|
| AnnualCrop | 36.24 | 32.93 | **+3.32** |
| River | 33.35 | 31.45 | +1.90 |
| Highway | 30.68 | 28.88 | +1.80 |
| Industrial | 23.41 | 22.07 | +1.34 |
| Pasture | 36.83 | 35.57 | +1.26 |
| PermanentCrop | 29.81 | 28.65 | +1.16 |
| HerbaceousVegetation | 32.83 | 32.11 | +0.72 |
| Forest | 41.07 | 40.57 | +0.50 |
| Residential | 25.62 | 25.12 | +0.50 |
| SeaLake | 47.15 | 46.73 | +0.43 |

The model helps most on classes with regular linear structure (crop fields, rivers, roads) and least on smooth classes (sea, forest), where bicubic is already close to the original. Residential gains little despite being hard: its fine, irregular detail is largely absent from a 16×16 input.

## Figures

- `gain_per_class.png` – PSNR gain over bicubic for each class.
- `class_comparison.png` – one test image per class: input, bicubic, model output, original.

*(Add the images here once they are in the repository, for example `![Gain per class](gain_per_class.png)`.)*

## How to run

```bash
pip install torch torchvision numpy pandas pillow matplotlib scikit-image jupyter
jupyter notebook satellite-super-resolution.ipynb
```

1. Download and unzip EuroSAT (RGB version with `train.csv`, `validation.csv`, `test.csv`).
2. In the notebook, set the dataset path (the notebook currently uses a local Windows path, `E:\archive\EuroSAT`). Change it to your own.
3. Run the cells from top to bottom: data → baseline → model → training → evaluation.

Training for 10 epochs runs on a CPU; a GPU will be faster.

Saved weights: `srnet2_best.pth` (best validation epoch) and `srnet2_final.pth`.

### Upscaling your own images

The notebook includes a tiled inference function (`upscale_tensor`) that upscales every image in a folder by 4× and writes PNGs to a new folder. It works on images of any size. Results are best on Sentinel-2-like imagery at about 10 m per pixel.

## Limitations

- **Synthetic degradation.** Inputs are produced by downsampling the targets. A model trained this way learns to undo that specific operation, and it is not guaranteed to transfer to real low-resolution sensor imagery.
- **Information limit.** Detail that is absent from the 16×16 input (small buildings, thin tracks, boats) cannot be recovered. The model sharpens large structures and edges.
- **L1-only training.** The output tends to be smooth. Perceptual or adversarial losses would give sharper texture but usually lower PSNR.
- **Single run.** No random seed is fixed and results come from one training run, so there is no variance estimate.
- **Baselines.** The only comparison is bicubic interpolation. Published models (SRCNN, EDSR, SwinIR) are not evaluated.
- **Small images.** EuroSAT patches are 64×64. Behaviour on large scenes, or on imagery at a different ground resolution, is untested.

## Future work

- Data augmentation (random flips and 90° rotations on the training pairs).
- Multiple seeds with mean and standard deviation, and an ablation (bicubic skip, residual blocks, scheduler).
- Comparison with published super-resolution baselines under the same training budget.
- Perceptual and adversarial losses (ESRGAN-style) with LPIPS reported alongside PSNR.
- A downstream test: classification accuracy on bicubic-upscaled versus model-upscaled images.
- Real Sentinel-2 to high-resolution pairs (the WorldStrat dataset) in place of synthetic degradation.

## Acknowledgements

- **EuroSAT:** P. Helber, B. Bischke, A. Dengel, D. Borth, *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification*, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019. Contains modified Copernicus Sentinel-2 data.
- Built with PyTorch, NumPy, pandas, Pillow, Matplotlib and scikit-image.

*Author: add your name here.*
