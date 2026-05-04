"""
Kernel Inception Distance (KID) Evaluation Script

Computes KID between real and generated image distributions using
InceptionV3 features and a polynomial kernel.

KID is an unbiased alternative to FID and is more stable for small sample sizes.

Steps:
1. Load images from real and generated directories
2. Extract deep features using pretrained InceptionV3
3. Compute kernel-based distance between feature distributions
"""

import os
import json
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from PIL import Image
from sklearn.metrics.pairwise import polynomial_kernel


# -----------------------------------------------------
# Device setup
# -----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------
# InceptionV3 feature extractor
# -----------------------------------------------------
weights = Inception_V3_Weights.DEFAULT
model = inception_v3(weights=weights, transform_input=False).to(device)

# Replace classification layer with identity
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
# Load images
# -----------------------------------------------------
def load_images(path, max_images=500):
    """
    Load images from directory.

    Args:
        path (str): Directory path
        max_images (int): Maximum number of images

    Returns:
        torch.Tensor: Image batch
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
def get_features(images):
    """
    Extract Inception features.

    Args:
        images (Tensor)

    Returns:
        np.ndarray: Feature vectors
    """
    feats = []

    for i in range(0, len(images), 32):
        batch = images[i:i + 32].to(device)
        f = model(batch)
        feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)


# -----------------------------------------------------
# Compute KID
# -----------------------------------------------------
def compute_kid(real_feats, fake_feats, subsets=10, subset_size=100):
    """
    Compute Kernel Inception Distance (KID).

    Uses polynomial kernel and subset sampling for unbiased estimation.

    Args:
        real_feats (np.ndarray): Features from real images
        fake_feats (np.ndarray): Features from generated images
        subsets (int): Number of subsets
        subset_size (int): Size of each subset

    Returns:
        tuple: (mean KID, std deviation)
    """
    n = min(len(real_feats), len(fake_feats))
    scores = []

    for _ in range(subsets):
        idx_r = np.random.choice(n, subset_size, replace=False)
        idx_f = np.random.choice(n, subset_size, replace=False)

        r = real_feats[idx_r]
        f = fake_feats[idx_f]

        # Kernel matrices
        k_rr = polynomial_kernel(r, r)
        k_ff = polynomial_kernel(f, f)
        k_rf = polynomial_kernel(r, f)

        m = subset_size

        score = (
            (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
            + (k_ff.sum() - np.trace(k_ff)) / (m * (m - 1))
            - 2 * k_rf.mean()
        )

        scores.append(score)

    return float(np.mean(scores)), float(np.std(scores))


# -----------------------------------------------------
# Main execution
# -----------------------------------------------------
if __name__ == "__main__":
    print("Loading images...")

    real = load_images("data/celeba_hq")
    fake = load_images("outputs/images/generated")

    print("Extracting features...")
    f1 = get_features(real)
    f2 = get_features(fake)

    print("Computing KID...")
    kid_mean, kid_std = compute_kid(f1, f2)

    print(f"\nFinal KID: {kid_mean:.4f} ± {kid_std:.4f}")

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    save_path = "outputs/metrics/kid.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(
            {
                "KID_mean": kid_mean,
                "KID_std": kid_std,
            },
            f,
            indent=4,
        )

    print(f"Saved KID to {save_path}")