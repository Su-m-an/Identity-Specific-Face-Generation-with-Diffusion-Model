#!/usr/bin/env python3
"""
Evaluate LoRA checkpoints and generate analysis plots for Part A.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import torch
from diffusers import StableDiffusionPipeline
import textwrap
import time

# Configuration
OUTPUT_DIR = "./checkpoints"
RESULTS_DIR = "./outputs"
HF_CACHE = "./hf_cache"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def plot_loss_curve():
    """Plot training loss curve from TensorBoard logs."""
    print("Generating loss curve...")

    log_dir = os.path.join(OUTPUT_DIR, "logs")
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)

    if not event_files:
        print("No TensorBoard logs found.")
        return

    latest = sorted(event_files)[-1]
    print(f"Reading: {latest}")

    ea = event_accumulator.EventAccumulator(latest)
    ea.Reload()

    tags = ea.Tags()["scalars"]
    loss_tag = [t for t in tags if "loss" in t][0] if tags else None

    if loss_tag:
        events = ea.Scalars(loss_tag)
        steps = np.array([e.step for e in events])
        losses = np.array([e.value for e in events])

        print(f"Steps logged: {len(steps)}")
        print(f"Loss start: {losses[:10].mean():.4f}, end: {losses[-10:].mean():.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, alpha=0.7, label='Training Loss', color='#4ecdc4', linewidth=2)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('LoRA Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{RESULTS_DIR}/loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved loss_curve.png")
    else:
        print("No loss data found.")

def load_lora_pipeline(base_model, lora_path=None):
    """Load Stable Diffusion pipeline with optional LoRA weights."""
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")

    if lora_path:
        pipe.load_lora_weights(lora_path)

    pipe.set_progress_bar_config(disable=True)
    return pipe

def generate_images(pipe, prompts, steps=30, cfg=7.5, seed=42):
    """Generate images for given prompts."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    images = []
    for prompt in prompts:
        img = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]
        images.append(img)
    return images

def cfg_ablation():
    """Run CFG scale ablation study."""
    print("Running CFG ablation...")

    pipe = load_lora_pipeline("CompVis/stable-diffusion-v1-4", OUTPUT_DIR)

    CFG_SCALES = [1.0, 3.0, 5.0, 7.5, 10.0, 12.0, 15.0]
    test_prompt = "Naruto Uzumaki portrait, detailed anime illustration, sharp lines, vibrant colors"
    cfg_images = []

    for cfg in CFG_SCALES:
        img = pipe(
            test_prompt,
            num_inference_steps=30,
            guidance_scale=cfg,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
        cfg_images.append(img)

    fig, axes = plt.subplots(1, len(CFG_SCALES), figsize=(28, 4))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, img, cfg in zip(axes, cfg_images, CFG_SCALES):
        ax.imshow(img)
        ax.set_title(f"CFG = {cfg}", fontsize=11, color='white', fontweight='bold')
        ax.axis("off")

    plt.suptitle(
        f'CFG Scale Ablation\nPrompt: "{test_prompt}"',
        fontsize=12, fontweight='bold', color='white'
    )
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cfg_ablation.png",
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved cfg_ablation.png")

def steps_ablation():
    """Run inference steps ablation study."""
    print("Running steps ablation...")

    pipe = load_lora_pipeline("CompVis/stable-diffusion-v1-4", OUTPUT_DIR)

    STEP_COUNTS = [10, 20, 30, 50, 75, 100]
    test_prompt = "red-haired kunoichi with fierce green eyes, detailed anime illustration"
    step_images = []
    step_times = []

    for n_steps in STEP_COUNTS:
        t0 = time.time()
        img = pipe(
            test_prompt,
            num_inference_steps=n_steps,
            guidance_scale=8.5,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).images[0]
        step_images.append(img)
        step_times.append(time.time() - t0)

    fig, axes = plt.subplots(1, len(STEP_COUNTS), figsize=(28, 4))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, img, n, t in zip(axes, step_images, STEP_COUNTS, step_times):
        ax.imshow(img)
        ax.set_title(f"{n} steps\n({t:.1f}s)", fontsize=10,
                     color='white', fontweight='bold')
        ax.axis("off")

    plt.suptitle(
        f'Inference Steps Ablation\nPrompt: "{test_prompt}"',
        fontsize=12, fontweight='bold', color='white'
    )
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/steps_ablation.png",
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved steps_ablation.png")

def final_outputs():
    """Generate final model outputs."""
    print("Generating final outputs...")

    pipe = load_lora_pipeline("CompVis/stable-diffusion-v1-4", OUTPUT_DIR)

    EVAL_PROMPTS = [
        "Naruto Uzumaki, anime style, sharp lines, vibrant colors, high quality",
        "anime ninja warrior, orange outfit, dynamic pose, cinematic lighting, detailed",
        "young ninja with spiky blonde hair, blue eyes, anime style, ultra detailed, 4k",
        "red-haired kunoichi with fierce green eyes, detailed anime illustration",
        "mysterious ninja wearing dark mask and cloak, anime style",
        "peaceful village hidden in misty mountains, anime landscape",
        "battle scene with powerful chakra energy bursts, anime style",
        "old sage with white hair and wooden staff, wise expression, anime"
    ]

    generator = torch.Generator(device="cuda").manual_seed(2024)
    generated = []
    for prompt in EVAL_PROMPTS:
        img = pipe(
            prompt,
            num_inference_steps=50,
            guidance_scale=8.5,
            generator=generator,
        ).images[0]
        generated.append((prompt, img))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#0f0f23')

    for ax, (prompt, img) in zip(axes.flat, generated):
        ax.imshow(img)
        ax.set_title(textwrap.fill(prompt, 28), fontsize=8, color='white', pad=4)
        ax.axis("off")

    plt.suptitle(
        "Part A — Final LoRA Model Outputs\n"
        "50 inference steps · CFG=8.5 · seed=2024",
        fontsize=14, fontweight='bold', color='white'
    )
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/final_outputs.png",
                dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print("Saved final_outputs.png")

if __name__ == "__main__":
    print("Starting Part A evaluation...")

    # Generate loss curve
    plot_loss_curve()

    # Generate final outputs
    final_outputs()

    # Run ablations
    cfg_ablation()
    steps_ablation()

    print("Evaluation complete!")