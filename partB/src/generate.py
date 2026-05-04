"""
Image Generation Script for Diffusion Model

Generates images using a trained UNet diffusion model with DDIM sampling.
Supports batch generation, reproducibility via seeds, and grid visualization.

Usage:
    python generate.py --checkpoint checkpoints/unet_epoch_60.pt
"""

import argparse
import random
from pathlib import Path
import math

import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

from model import UNet
from scheduler import DDPMScheduler


# -----------------------------------------------------
# Argument parsing
# -----------------------------------------------------
def parse_args():
    """
    Parse command-line arguments for generation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs/images")

    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--sample_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


# -----------------------------------------------------
# Load trained model
# -----------------------------------------------------
def load_model(path, device, image_size):
    """
    Load trained UNet model from checkpoint.

    Args:
        path (str): Path to checkpoint
        device (str): Device for inference
        image_size (int): Target resolution

    Returns:
        tuple: (model, scheduler_config)
    """
    ckpt = torch.load(path, map_location=device)

    config = ckpt.get("model_config", {})
    config.setdefault("image_size", image_size)

    model = UNet(**config).to(device)

    # Prefer EMA weights for better quality
    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict)

    return model.eval(), ckpt.get("scheduler_config", {})


# -----------------------------------------------------
# DDIM sampling
# -----------------------------------------------------
@torch.no_grad()
def sample(model, scheduler, batch, size, device, steps):
    """
    Generate images using DDIM sampling.

    Args:
        model (nn.Module): Trained diffusion model
        scheduler (DDPMScheduler): Noise scheduler
        batch (int): Batch size
        size (int): Image resolution
        device (str): Device
        steps (int): Number of sampling steps

    Returns:
        torch.Tensor: Generated images in range [0, 1]
    """
    # Initialize with Gaussian noise (x_T)
    x = torch.randn(batch, 3, size, size, device=device)

    # Create timestep schedule
    times = torch.linspace(
        scheduler.timesteps - 1, 0, steps, device=device
    ).long()

    # Iterative denoising
    for i in range(len(times)):
        t = int(times[i])
        t_prev = int(times[i + 1]) if i + 1 < len(times) else -1

        x = scheduler.ddim_step(model, x, t, t_prev, eta=0.0)

    # Convert from [-1, 1] to [0, 1]
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2

    # Remove global color bias
    mean = x.mean(dim=[2, 3], keepdim=True)
    x = x - (mean - 0.5)

    x = torch.clamp(x, 0, 1)

    return x.cpu()


# -----------------------------------------------------
# Save grid visualization
# -----------------------------------------------------
def save_clean_grid(images, save_path):
    import math
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    n = len(images)  # use ALL images

    # Automatically choose grid size
    nrow = int(math.sqrt(n))

    grid = make_grid(images, nrow=nrow, padding=2)

    plt.figure(figsize=(16, 16))  # bigger = clearer grid
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")

    plt.title(
        f"Generated Images ({n} samples)",
        fontsize=16,
        weight="bold",
        pad=12,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -----------------------------------------------------
# Main execution
# -----------------------------------------------------
def main():
    """
    Generate images using a trained diffusion model.
    """
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(random.randint(0, 1_000_000))

    # Prepare output directories
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen_dir = out / "generated"
    gen_dir.mkdir(exist_ok=True)

    # Load model and scheduler
    model, sched_cfg = load_model(args.checkpoint, device, args.image_size)

    scheduler = DDPMScheduler(
        timesteps=sched_cfg.get("timesteps", 1000),
        device=device,
        schedule=sched_cfg.get("schedule", "cosine"),
    )

    # Generate images in batches
    all_imgs = []
    total = 0

    while total < args.num_images:
        batch = min(args.batch_size, args.num_images - total)

        imgs = sample(
            model,
            scheduler,
            batch,
            args.image_size,
            device,
            args.sample_steps,
        )

        all_imgs.append(imgs)
        total += batch

        print(f"Generated {total}/{args.num_images}")

    all_imgs = torch.cat(all_imgs)

    # Save individual images
    for i, img in enumerate(all_imgs):
        save_image(img, gen_dir / f"{i:04d}.png")

    # Save grid visualization
    save_clean_grid(all_imgs, out / "generated_epoch60.png")

    print("Generation complete")


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
if __name__ == "__main__":
    main()