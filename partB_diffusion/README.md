# Diffusion Model for Face Generation  
## EEEM068: Applied Machine Learning – Part B

This project implements a Denoising Diffusion Probabilistic Model (DDPM) using a UNet architecture for unconditional face image generation.

---

## Project Structure

```
partB_diffusion/
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
│   ├── diffusion_visualize.py   # Forward and reverse diffusion
│   └── plot_loss.py             # Training loss curve
│
├── outputs/
│   ├── images/               # Generated samples
│   └── metrics/              # FID and KID scores
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

- Train a diffusion model on face images  
- Generate realistic samples from noise  
- Evaluate generation quality using FID and KID  

---

## Methodology

### Model
- UNet-based architecture  
- Residual blocks with skip connections  
- Sinusoidal time embeddings  

### Diffusion Process
- Forward diffusion: gradual addition of Gaussian noise  
- Reverse diffusion: learned denoising process  

### Noise Schedule
- Cosine schedule for improved stability  

### Sampling
- DDIM sampling for faster and more stable generation  

---

## Training

```bash
python train.py \
  --data_dir ./data/celeba_hq \
  --output_dir ./checkpoints \
  --epochs 60 \
  --batch_size 4
  --learning_rate 1e-4 \
  --num_workers 4
```
Generation
python generate.py \
  --checkpoint checkpoints/unet_epoch_60.pt \
  --num_images 300

Outputs:

outputs/images/generated/ (individual images)
outputs/images/generated_epoch60.png (grid)
Diffusion Visualization
python diffusion_visualize_final.py

Outputs:

forward_diffusion.png – image to noise
reverse_diffusion.png – noise to image
Evaluation
FID (Fréchet Inception Distance)
python fid_final.py

Saved to:

outputs/metrics/fid.json
KID (Kernel Inception Distance)
python kid.py

Saved to:

outputs/metrics/kid.json
Results
FID: ~75.6
KID: ~0.045

The model produces recognizable face-like structures. Reverse diffusion demonstrates gradual denoising from Gaussian noise to structured images.

Key Observations
Cosine noise schedule improves stability
DDIM sampling provides clearer intermediate denoising
EMA improves generation quality
Limited training results in slight blur and artifacts
Limitations
Moderate FID due to limited training duration
Some color bias in outputs
No conditional generation (unconditional model only)
Requirements
torch >= 2.0
torchvision >= 0.15
numpy
Pillow
tqdm
scipy
scikit-learn

Conclusion

This implementation demonstrates diffusion-based image generation using DDPM, including forward and reverse processes, stable sampling, and evaluation using FID and KID.
