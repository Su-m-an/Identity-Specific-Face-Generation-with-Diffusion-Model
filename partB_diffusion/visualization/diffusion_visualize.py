# diffusion_visualize_final.py

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as transforms
import os

from model import UNet
from scheduler import DDPMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs/images", exist_ok=True)

# ---- Load model ----
ckpt = torch.load("checkpoints/unet_epoch_60.pt", map_location=device)
config = ckpt.get("model_config", {})
model = UNet(**config).to(device)
model.load_state_dict(ckpt.get("ema", ckpt.get("model", ckpt)))
model.eval()

# ---- Scheduler ----
scheduler = DDPMScheduler(timesteps=1000, device=device)

# ---- Load one image for forward ----
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img_path = None
for root, _, files in os.walk("data"):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(root, f)
            break
    if img_path:
        break

x0 = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

# ---- FORWARD ----
steps = [0, 200, 400, 600, 800, 999]
forward_imgs = []
for t in steps:
    t_tensor = torch.tensor([t], device=device)
    xt, _ = scheduler.add_noise(x0, t_tensor)
    xt = (xt.clamp(-1, 1) + 1) / 2
    forward_imgs.append(xt.cpu())

grid = make_grid(torch.cat(forward_imgs), nrow=len(steps))
plt.imshow(grid.permute(1, 2, 0))
plt.title("Forward Diffusion (Image → Noise)")
plt.axis("off")
plt.savefig("outputs/images/forward_diffusion.png", dpi=300)
plt.close()

# ---- REVERSE (DDIM — THIS FIXES YOUR ISSUE) ----
print("Generating reverse diffusion (DDIM)...")

x = torch.randn_like(x0)

# use many steps for smooth denoising
num_steps = 100
times = torch.linspace(999, 0, num_steps, device=device).long()

# pick evenly spaced frames to save
save_ids = torch.linspace(0, num_steps - 1, 6).long().tolist()

reverse_imgs = []

for i in range(len(times)):
    t = int(times[i])
    t_prev = int(times[i + 1]) if i + 1 < len(times) else -1

    x = scheduler.ddim_step(model, x, t, t_prev, eta=0.0)

    if i in save_ids:
        x_vis = (x.clamp(-1, 1) + 1) / 2

        # remove residual color shift (important)
        mean = x_vis.mean(dim=[2, 3], keepdim=True)
        x_vis = x_vis - (mean - 0.5)
        x_vis = torch.clamp(x_vis, 0, 1)

        reverse_imgs.append(x_vis.cpu())

grid = make_grid(torch.cat(reverse_imgs), nrow=len(reverse_imgs))
plt.imshow(grid.permute(1, 2, 0))
plt.title("Reverse Diffusion (DDIM)")
plt.axis("off")
plt.savefig("outputs/images/reverse_diffusion.png", dpi=300)
plt.close()

print("✅ Done")