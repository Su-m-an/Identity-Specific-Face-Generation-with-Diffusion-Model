"""
Training Script for Diffusion Model (Part B)

This script trains a UNet-based denoising diffusion model (DDPM).
It includes exponential moving average (EMA) stabilization,
cosine learning rate scheduling, checkpointing, and sample generation.

Usage:
    python train.py --data_dir ./data --output_dir ./checkpoints --epochs 120
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import UNet
from scheduler import DDPMScheduler


class EMA:
    """
    Exponential Moving Average (EMA) of model parameters.

    Maintains a smoothed version of model weights during training.
    EMA weights generally produce higher quality samples than raw weights.

    Args:
        model (nn.Module): Model to track
        beta (float): Decay rate for EMA updates
    """

    def __init__(self, model, beta=0.9999):
        self.beta = beta
        self.model = UNet(**model.config).to(next(model.parameters()).device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()

    @torch.no_grad()
    def update(self, model):
        """
        Update EMA weights using current model parameters.
        """
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(self.beta).add_(p, alpha=1.0 - self.beta)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train diffusion model")

    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")

    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--ema_decay", type=float, default=0.9999)

    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--sample_steps", type=int, default=100)

    parser.add_argument("--resume", default=None)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


def make_dataloader(args):
    """
    Create dataset and dataloader.

    Applies:
    - Resize and crop to fixed resolution
    - Horizontal flip augmentation
    - Normalization to [-1, 1] range
    """
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    return dataset, loader


def save_checkpoint(path, model, ema, optimizer, lr_scheduler, epoch, global_step, args, scheduler):
    """
    Save full training checkpoint.
    """
    payload = {
        "model": model.state_dict(),
        "ema": ema.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "model_config": model.config,
        "scheduler_config": {
            "timesteps": scheduler.timesteps,
            "schedule": scheduler.schedule,
        },
        "args": vars(args),
    }

    torch.save(payload, path)


@torch.no_grad()
def save_sample_grid(path, model, scheduler, args, device):
    """
    Generate sample images using DDIM sampling and save as a grid.
    """
    was_training = model.training
    model.eval()

    sample_count = max(1, args.sample_count)

    # Start from Gaussian noise
    x = torch.randn(sample_count, 3, args.image_size, args.image_size, device=device)

    steps = min(args.sample_steps, scheduler.timesteps)
    times = torch.linspace(scheduler.timesteps - 1, 0, steps, dtype=torch.long).tolist()
    times = sorted(set(int(t) for t in times), reverse=True)

    # DDIM sampling loop
    for idx, t in enumerate(times):
        t_prev = times[idx + 1] if idx + 1 < len(times) else -1

        x = scheduler.ddim_step(
            model,
            x,
            t,
            t_prev,
            eta=0.1  # small stochasticity improves visual stability
        )

    # Convert from [-1, 1] to [0, 1]
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2

    # Mild color normalization (prevents bias)
    mean = x.mean(dim=[2, 3], keepdim=True)
    x = x - (mean - 0.5)
    x = torch.clamp(x, 0, 1)

    save_image(x.cpu(), path, nrow=int(sample_count**0.5), padding=2)

    if was_training:
        model.train()


def load_checkpoint(path, model, ema, optimizer, lr_scheduler, device):
    """
    Load checkpoint for resuming training.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    ema.model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    optimizer.load_state_dict(checkpoint["optimizer"])

    if "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    start_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", 0))

    return start_epoch, global_step


def main():
    """
    Main training loop.
    """
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    dataset, loader = make_dataloader(args)

    print(f"Dataset size: {len(dataset)}")
    print(f"Device: {device}")

    model = UNet(
        base=args.base_channels,
        time_dim=args.time_dim,
        image_size=args.image_size,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    ).to(device)

    ema = EMA(model, beta=args.ema_decay)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs * len(loader), 1),
        eta_min=args.lr * 0.05,
    )

    diffusion = DDPMScheduler(
        timesteps=args.timesteps,
        device=device,
        schedule=args.schedule,
    )

    global_step = 0
    start_epoch = 0

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, ema, optimizer, lr_scheduler, device
        )
        print(f"Resumed from {args.resume}")

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        running_loss = 0.0

        for imgs, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)

            # Sample random timestep
            t = torch.randint(0, args.timesteps, (imgs.size(0),), device=device)

            # Forward diffusion
            x_t, noise = diffusion.add_noise(imgs, t)

            # Predict noise
            pred = model(x_t, t)

            # MSE loss
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()

            ema.update(model)

            global_step += 1
            running_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}: avg loss {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                output_dir / f"unet_epoch_{epoch + 1}.pt",
                model, ema, optimizer, lr_scheduler,
                epoch + 1, global_step, args, diffusion
            )

        # Generate samples
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            save_sample_grid(
                samples_dir / f"epoch_{epoch + 1:04d}.png",
                ema.model,
                diffusion,
                args,
                device,
            )

    # Final save
    save_checkpoint(
        output_dir / "unet_best.pt",
        model, ema, optimizer, lr_scheduler,
        args.epochs, global_step, args, diffusion
    )

    torch.save(ema.model.state_dict(), output_dir / "unet_ema_state_dict.pth")

    print("Training complete.")


if __name__ == "__main__":
    main()