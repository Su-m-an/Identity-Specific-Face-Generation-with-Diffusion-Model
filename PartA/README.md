# LoRA Fine-tuning for Naruto Character Generation


**Author:** Suman R  
---

This project demonstrates **efficient fine-tuning** of Stable Diffusion v1.4 using **Low-Rank Adaptation (LoRA)** on the Naruto anime dataset.

### Objective
Adapt Stable Diffusion to generate high-quality Naruto-style anime characters while keeping training fast and resource-efficient.

### Key Highlights

- Base Model: **Stable Diffusion v1.4**
- Fine-tuning Method: **LoRA** (rank-16)
- Dataset: `lambdalabs/naruto-blip-captions` (1,221 images)
- Training time: ~2 hours on RTX A4000
- Target modules: Attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`)
- Strong prompt adherence with distinct anime style

### What’s Included

- Full training and evaluation notebook: `run_partA.ipynb`
- Comprehensive hyperparameter experiments:
  - CFG scale ablation
  - Inference steps analysis (10–100 steps)
  - Training loss visualization
  - Progressive sample generation
  - Before vs After style transfer comparison

### Visualizations
- Dataset samples and statistics
- Training loss curve
- Sample grids across training checkpoints
- Final generated Naruto characters
- CFG and steps ablation results

---

## Quick Inference

You can load the fine-tuned model directly from Hugging Face (if uploaded) or run the notebook to train from scratch.

---

**Current Status:** Part A completed.  
Working towards full identity-specific face generation pipeline.

---

Made by **Suman R**  
