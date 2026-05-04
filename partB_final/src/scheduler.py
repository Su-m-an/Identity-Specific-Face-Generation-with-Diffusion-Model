"""
DDPM Scheduler for Diffusion Models

Implements forward diffusion and reverse denoising processes.

Key Components:
- Beta schedule (linear or cosine)
- Forward noise addition (q(x_t | x_0))
- Reverse denoising (p(x_{t-1} | x_t))
- DDIM sampling for efficient inference
"""

import torch
import math


class DDPMScheduler:
    """
    Scheduler for diffusion processes (DDPM / DDIM).

    Handles:
    - Noise schedule (betas, alphas)
    - Forward diffusion (adding noise)
    - Reverse denoising (sampling)

    Args:
        timesteps (int): Total diffusion steps
        device (str): Device for tensor storage
        beta_start (float): Initial beta value (linear schedule)
        beta_end (float): Final beta value (linear schedule)
        schedule (str): "linear" or "cosine"
    """

    def __init__(
        self,
        timesteps=1000,
        device="cpu",
        beta_start=1e-4,
        beta_end=0.02,
        schedule="cosine",
    ):
        self.timesteps = timesteps
        self.device = device
        self.schedule = schedule

        # -----------------------------
        # Beta schedule
        # -----------------------------
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif schedule == "cosine":
            betas = self._cosine_betas(timesteps, device)
        else:
            raise ValueError("Unknown schedule")

        self.betas = betas
        self.alphas = 1.0 - betas

        # -----------------------------
        # Cumulative product of alphas
        # α̂_t = ∏_{i=1}^{t} α_i
        # -----------------------------
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

        self.alpha_hat_prev = torch.cat(
            [torch.ones(1, device=device), self.alpha_hats[:-1]]
        )

        # -----------------------------
        # Posterior variance
        # Used in reverse diffusion
        # -----------------------------
        self.posterior_variance = (
            betas * (1 - self.alpha_hat_prev) / (1 - self.alpha_hats)
        ).clamp(min=1e-20)

    # -----------------------------------------------------
    # Cosine noise schedule (Nichol & Dhariwal, 2021)
    # -----------------------------------------------------
    def _cosine_betas(self, timesteps, device, s=0.008):
        """
        Compute cosine-based beta schedule.

        Produces smoother noise transitions compared to linear schedule.

        Args:
            timesteps (int)
            device (str)
            s (float): Small offset for stability

        Returns:
            torch.Tensor: Beta values
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)

        alpha_hat = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_hat = alpha_hat / alpha_hat[0]

        betas = 1 - (alpha_hat[1:] / alpha_hat[:-1])
        return betas.clamp(1e-4, 0.999)

    # -----------------------------------------------------
    # Forward diffusion: q(x_t | x_0)
    # -----------------------------------------------------
    def add_noise(self, x0, t, noise=None):
        """
        Add Gaussian noise to clean images.

        x_t = sqrt(α̂_t) * x_0 + sqrt(1 - α̂_t) * ε

        Args:
            x0 (Tensor): Clean images (B, C, H, W)
            t (Tensor): Timesteps (B,)
            noise (Tensor, optional): Pre-sampled noise

        Returns:
            Tuple[Tensor, Tensor]: Noisy image and noise used
        """
        if noise is None:
            noise = torch.randn_like(x0)

        t = t.to(self.device)

        alpha_hat = self.alpha_hats.gather(0, t).view(-1, 1, 1, 1)

        x_t = (
            torch.sqrt(alpha_hat) * x0 +
            torch.sqrt(1 - alpha_hat) * noise
        )

        return x_t, noise

    # -----------------------------------------------------
    # Reverse diffusion (DDPM sampling)
    # -----------------------------------------------------
    @torch.no_grad()
    def reverse(self, model, x, t):
        """
        Perform one DDPM reverse step.

        x_{t-1} ~ N(mean, variance)

        Args:
            model (nn.Module): Noise predictor
            x (Tensor): Current noisy image (B, C, H, W)
            t (Tensor): Current timestep (B,)

        Returns:
            Tensor: Denoised sample at t-1
        """
        t = t.to(self.device)

        # Predict noise ε_θ(x_t, t)
        eps = model(x, t)

        # Gather timestep-dependent parameters
        alpha = self.alphas.gather(0, t).view(-1, 1, 1, 1)
        alpha_hat = self.alpha_hats.gather(0, t).view(-1, 1, 1, 1)
        beta = self.betas.gather(0, t).view(-1, 1, 1, 1)

        # Compute mean (DDPM equation)
        mean = (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * eps) / torch.sqrt(alpha)

        # At t = 0, return deterministic output
        if t[0].item() == 0:
            return mean

        # Add stochastic noise
        var = self.posterior_variance.gather(0, t).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)

        return mean + torch.sqrt(var) * noise

    # -----------------------------------------------------
    # DDIM sampling (deterministic / fast)
    # -----------------------------------------------------
    @torch.no_grad()
    def ddim_step(self, model, x, t, t_prev, eta=0.0):
        """
        Perform one DDIM sampling step.

        Allows faster sampling with fewer steps than DDPM.

        Args:
            model (nn.Module): Noise predictor
            x (Tensor): Current image (B, C, H, W)
            t (int): Current timestep
            t_prev (int): Previous timestep
            eta (float): Stochasticity (0 = deterministic)

        Returns:
            Tensor: Updated image at timestep t_prev
        """
        batch = x.shape[0]
        t_batch = torch.full((batch,), t, device=self.device, dtype=torch.long)

        eps = model(x, t_batch)

        alpha_hat_t = self.alpha_hats[t]
        alpha_hat_prev = (
            self.alpha_hats[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=self.device)
        )

        alpha_hat_t = alpha_hat_t.view(1, 1, 1, 1)
        alpha_hat_prev = alpha_hat_prev.view(1, 1, 1, 1)

        # Estimate clean image x_0
        x0 = (x - torch.sqrt(1 - alpha_hat_t) * eps) / torch.sqrt(alpha_hat_t)
        x0 = x0.clamp(-1, 1)

        # Compute variance term
        sigma = eta * torch.sqrt(
            (1 - alpha_hat_prev) / (1 - alpha_hat_t)
        ) * torch.sqrt(1 - alpha_hat_t / alpha_hat_prev)

        noise = torch.randn_like(x) if eta > 0 else 0

        # Direction pointing to x_t
        direction = torch.sqrt(1 - alpha_hat_prev - sigma**2) * eps

        x_prev = torch.sqrt(alpha_hat_prev) * x0 + direction + sigma * noise

        return x_prev