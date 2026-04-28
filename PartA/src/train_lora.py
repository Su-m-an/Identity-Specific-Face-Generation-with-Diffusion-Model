# Part A - LoRA Fine-tuning for Stable Diffusion on Naruto Dataset
# Clean source code for reproducibility

import os
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np

def setup_environment():
    """Set up paths and environment variables."""
    HF_CACHE = "./cache"
    OUTPUT_DIR = "./checkpoints"
    RESULTS_DIR = "./outputs"

    os.environ["HF_HOME"] = HF_CACHE
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for path in [HF_CACHE, OUTPUT_DIR, RESULTS_DIR]:
        os.makedirs(path, exist_ok=True)

    return HF_CACHE, OUTPUT_DIR, RESULTS_DIR

def load_lora_pipeline(base_model="CompVis/stable-diffusion-v1-4", lora_path=None):
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

def generate_image(pipe, prompt, steps=30, cfg=7.5, seed=42):
    """Generate a single image with fixed seed."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
    ).images[0]

def plot_grid(images, titles, filename, cols=4):
    """Plot a grid of images."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.patch.set_facecolor('#0f0f23')

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = axes.flat[i] if rows > 1 else axes[i]
        ax.imshow(img)
        ax.set_title(title, fontsize=10, color='white')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# Example usage
if __name__ == "__main__":
    hf_cache, output_dir, results_dir = setup_environment()

    # Load model
    pipe = load_lora_pipeline(lora_path=output_dir)

    # Generate sample
    prompt = "Naruto Uzumaki, anime style"
    img = generate_image(pipe, prompt)

    # Save
    img.save(f"{results_dir}/sample.png")
    print("Sample generated and saved.")