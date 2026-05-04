# Diffusion Model for Face Generation

## EEEM068: Applied Machine Learning – Part B

This project implements a Denoising Diffusion Probabilistic Model (DDPM) using a UNet architecture for unconditional face image generation.

---

## Overview

The model learns to generate realistic human faces by reversing a gradual noise corruption process. The pipeline includes training, sampling, and evaluation using standard generative metrics.

---

## Project Structure

```
partB/
├── src/
│   ├── model.py              # UNet architecture
│   ├── scheduler.py          # DDPM scheduler (cosine schedule)
│   ├── train.py              # Training script
│   └── generate.py           # Sampling (DDIM)
│
├── evaluation/
│   ├── compute_fid.py        # FID computation
│   └── compute_kid.py        # KID computation
│
├── visualization/
│   ├── diffusion_visualize.py   # Forward & reverse diffusion
│   └── plot_loss.py             # Training loss curve
│
├── outputs/
│   ├── images/               # Generated samples
│   └── metrics/              # FID & KID scores
│
├── checkpoints/              # Model weights (not included)
├── data/                     # Dataset (not included)
│   └── celeba_hq/
│
├── requirements.txt
└── README.md
```

---

## Objective

* Train a diffusion model on face images
* Generate realistic samples from noise
* Evaluate generation quality using FID and KID

---

## Methodology

### Model

* UNet-based encoder-decoder architecture
* Residual blocks with skip connections
* Sinusoidal time embeddings

### Diffusion Process

* Forward diffusion: gradual addition of Gaussian noise
* Reverse diffusion: learned denoising process

### Noise Schedule

* Cosine schedule for improved stability

### Sampling

* DDIM sampling for faster and stable generation

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python src/train.py \
  --data_dir data \
  --output_dir checkpoints \
  --epochs 120 \
  --batch_size 4
```

---

## Generation

```bash
python src/generate.py \
  --checkpoint checkpoints/unet_epoch_60.pt \
  --output_dir outputs/images \
  --num_images 100 \
  --batch_size 4 \
  --sample_steps 100
```

---

## Evaluation

### FID

```bash
python evaluation/compute_fid.py
```

### KID

```bash
python evaluation/compute_kid.py
```

---

## Results

| Metric | Value           |
| ------ | --------------- |
| FID    | ~75             |
| KID    | 0.0369 ± 0.0027 |

Generated samples:

```
outputs/images/generated_epoch60.png
```

---

## Notes

* Dataset and checkpoints are excluded due to size constraints
* Place trained model in `checkpoints/` before generation
* Dataset should be located in `data/celeba_hq/`

---

## Conclusion

This implementation demonstrates diffusion-based image generation using DDPM, including forward and reverse processes, stable sampling, and quantitative evaluation using FID and KID.
