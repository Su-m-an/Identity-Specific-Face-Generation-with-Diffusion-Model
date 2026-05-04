"""
Fréchet Inception Distance (FID) Evaluation Script

Computes FID between real and generated image distributions using
InceptionV3 features.

Steps:
1. Load images from real and generated directories
2. Extract deep features using pretrained InceptionV3
3. Compute mean and covariance of features
4. Calculate FID score

FID measures similarity between two distributions. Lower is better.
"""

import os
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from scipy.linalg import sqrtm


# -----------------------------------------------------
# Device setup
# -----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------
# InceptionV3 feature extractor
# -----------------------------------------------------
weights = Inception_V3_Weights.DEFAULT
model = inception_v3(weights=weights, transform_input=False).to(device)

# Replace classification head with identity to extract features
model.fc = torch.nn.Identity()
model.eval()


# -----------------------------------------------------
# Image preprocessing
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])


# -----------------------------------------------------
# Load images from directory
# -----------------------------------------------------
def load_images(path, max_images=1000):
    """
    Load images from a directory.

    Args:
        path (str): Directory containing images
        max_images (int): Maximum number of images to load

    Returns:
        torch.Tensor: Image batch (N, 3, 299, 299)
    """
    files = []

    for root, _, fs in os.walk(path):
        for f in fs:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                files.append(os.path.join(root, f))

    files = files[:max_images]

    if len(files) == 0:
        raise ValueError(f"No images found in: {path}")

    imgs = []
    for p in files:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img))

    return torch.stack(imgs)


# -----------------------------------------------------
# Feature extraction
# -----------------------------------------------------
@torch.no_grad()
def get_features(images, batch_size=32):
    """
    Extract Inception features.

    Args:
        images (Tensor): Input images
        batch_size (int): Batch size for processing

    Returns:
        np.ndarray: Feature matrix
    """
    feats = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        f = model(batch)
        feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0).astype(np.float64)


# -----------------------------------------------------
# Compute statistics
# -----------------------------------------------------
def compute_stats(feats):
    """
    Compute mean and covariance of features.

    Args:
        feats (np.ndarray)

    Returns:
        tuple: (mean, covariance)
    """
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# -----------------------------------------------------
# Compute FID
# -----------------------------------------------------
def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Fréchet Inception Distance.

    Args:
        mu1, sigma1: Statistics of real images
        mu2, sigma2: Statistics of generated images
        eps (float): Small value for numerical stability

    Returns:
        float: FID score
    """
    # Add epsilon to diagonals for numerical stability
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2

    # Matrix square root
    covmean = sqrtm(sigma1 @ sigma2)

    # Handle numerical issues
    if not np.isfinite(covmean).all():
        covmean = sqrtm(
            (sigma1 + np.eye(len(sigma1)) * eps) @
            (sigma2 + np.eye(len(sigma2)) * eps)
        )

    # Remove small imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


# -----------------------------------------------------
# Main execution
# -----------------------------------------------------
if __name__ == "__main__":
    print("Loading images...")

    real_path = "data/celeba_hq"
    fake_path = "outputs/images/generated"

    real = load_images(real_path, max_images=500)
    fake = load_images(fake_path, max_images=500)

    print(f"Real: {len(real)} | Fake: {len(fake)}")

    print("Extracting features...")
    f1 = get_features(real)
    f2 = get_features(fake)

    m1, s1 = compute_stats(f1)
    m2, s2 = compute_stats(f2)

    print("Computing FID...")
    fid = compute_fid(m1, s1, m2, s2)

    print(f"\nFinal FID: {fid:.4f}")

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    save_path = "outputs/metrics/fid.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {
        "FID": float(fid),
        "real_images": int(len(real)),
        "generated_images": int(len(fake)),
    }

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved FID to {save_path}")