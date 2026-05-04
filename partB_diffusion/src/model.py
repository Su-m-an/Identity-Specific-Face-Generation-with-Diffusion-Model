"""
UNet Model for Diffusion-based Image Generation

This module implements a UNet architecture for denoising diffusion models (DDPM).
The network predicts noise given a noisy image and timestep.

Key Features:
- Sinusoidal time embeddings for timestep conditioning
- Residual blocks with Group Normalization
- Multi-head self-attention for global context
- Encoder-decoder structure with skip connections
- Configurable channel scaling across resolutions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels):
    """
    Select an appropriate number of groups for GroupNorm.

    Prefers higher group counts (32 → 1) that evenly divide channels.

    Args:
        channels (int): Number of feature channels

    Returns:
        int: Number of groups for normalization
    """
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding module.

    Converts discrete timesteps into continuous embeddings using sinusoidal
    functions (similar to Transformer positional encodings), followed by an MLP.

    Args:
        dim (int): Embedding dimension
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        """
        Generate timestep embeddings.

        Args:
            t (torch.Tensor): Shape (B,)

        Returns:
            torch.Tensor: Shape (B, dim)
        """
        device = t.device
        half = self.dim // 2

        # Frequency scaling (log space)
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000) / max(half - 1, 1))
        )

        # Apply sinusoidal encoding
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # Pad if needed
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))

        return self.mlp(emb)


class ResBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    Applies normalization → activation → convolution twice,
    with time embedding injected between layers.

    Args:
        in_ch (int): Input channels
        out_ch (int): Output channels
        time_dim (int): Time embedding dimension
        dropout (float): Dropout rate
    """
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0):
        super().__init__()

        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch)
        )

        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t):
        """
        Forward pass.

        Args:
            x (Tensor): (B, in_ch, H, W)
            t (Tensor): (B, time_dim)

        Returns:
            Tensor: (B, out_ch, H, W)
        """
        h = self.conv1(F.silu(self.norm1(x)))

        # Inject time embedding
        h = h + self.time(t)[:, :, None, None]

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        # Residual connection (scaled for stability)
        return (h + self.skip(x)) / math.sqrt(2.0)


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention over spatial features.

    Enables global context modeling across the image.

    Args:
        channels (int): Feature channels
        num_heads (int): Number of attention heads
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()

        if channels % num_heads != 0:
            num_heads = 1

        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_groups(channels), channels)

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, H, W)

        Returns:
            Tensor: (B, C, H, W)
        """
        b, c, h, w = x.shape

        qkv = self.qkv(self.norm(x)).reshape(
            b, 3, self.num_heads, c // self.num_heads, h * w
        )

        q, k, v = qkv.unbind(dim=1)

        scale = (c // self.num_heads) ** -0.5

        attn = torch.softmax(
            torch.einsum("bhcn,bhcm->bhnm", q * scale, k),
            dim=-1
        )

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)

        return x + self.proj(out)


class Downsample(nn.Module):
    """Spatial downsampling via strided convolution."""
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Upsampling using nearest-neighbor interpolation + convolution."""
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    """
    UNet backbone for diffusion models.

    Consists of:
    - Encoder (downsampling path)
    - Bottleneck with attention
    - Decoder (upsampling path)

    Skip connections preserve spatial information across scales.

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        base (int): Base channel width
        time_dim (int): Time embedding dimension
        image_size (int): Input resolution
        channel_mults (tuple): Channel scaling per level
        num_res_blocks (int): Residual blocks per level
        attention_resolutions (tuple): Resolutions for attention
        dropout (float): Dropout rate
        attention_heads (int): Number of attention heads
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base=64,
        base_channels=None,
        time_dim=256,
        image_size=256,
        channel_mults=(1, 2, 4, 8, 8),
        channel_multipliers=None,
        num_res_blocks=2,
        attention_resolutions=(32, 16),
        dropout=0.1,
        attention_heads=4,
    ):
        super().__init__()

        if base_channels is not None:
            base = base_channels
        if channel_multipliers is not None:
            channel_mults = channel_multipliers

        self.config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base": base,
            "time_dim": time_dim,
            "image_size": image_size,
            "channel_mults": tuple(channel_mults),
            "num_res_blocks": num_res_blocks,
            "attention_resolutions": tuple(attention_resolutions),
            "dropout": dropout,
            "attention_heads": attention_heads,
        }

        self.time = TimeEmbedding(time_dim)
        self.conv_in = nn.Conv2d(in_channels, base, 3, padding=1)

        channels = [base]
        in_ch = base
        resolution = image_size

        self.downs = nn.ModuleList()

        # Encoder
        for level, mult in enumerate(channel_mults):
            out_ch = base * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(in_ch, out_ch, time_dim, dropout)]

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, attention_heads))

                self.downs.append(nn.ModuleList(layers))
                in_ch = out_ch
                channels.append(in_ch)

            if level != len(channel_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(in_ch)]))
                resolution //= 2
                channels.append(in_ch)

        # Bottleneck
        self.mid1 = ResBlock(in_ch, in_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(in_ch, attention_heads)
        self.mid2 = ResBlock(in_ch, in_ch, time_dim, dropout)

        # Decoder
        self.ups = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base * mult

            for _ in range(num_res_blocks + 1):
                skip_ch = channels.pop()

                layers = [ResBlock(in_ch + skip_ch, out_ch, time_dim, dropout)]

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, attention_heads))

                self.ups.append(nn.ModuleList(layers))
                in_ch = out_ch

            if level != 0:
                self.ups.append(nn.ModuleList([Upsample(in_ch)]))
                resolution *= 2

        self.out = nn.Sequential(
            nn.GroupNorm(_groups(in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Forward pass.

        Args:
            x (Tensor): Noisy image (B, C, H, W)
            t (Tensor): Timesteps (B,)

        Returns:
            Tensor: Predicted noise
        """
        t = self.time(t.to(x.device))

        h = self.conv_in(x)
        skips = [h]

        # Encoder
        for layers in self.downs:
            if isinstance(layers[0], Downsample):
                h = layers[0](h)
            else:
                h = layers[0](h, t)
                if len(layers) > 1:
                    h = layers[1](h)
            skips.append(h)

        # Bottleneck
        h = self.mid1(h, t)
        h = self.mid_attn(h)
        h = self.mid2(h, t)

        # Decoder
        for layers in self.ups:
            if isinstance(layers[0], Upsample):
                h = layers[0](h)
            else:
                skip = skips.pop()

                # Ensure spatial alignment
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")

                h = layers[0](torch.cat([h, skip], dim=1), t)

                if len(layers) > 1:
                    h = layers[1](h)

        return self.out(h)


def count_parameters(model):
    """
    Count trainable parameters.

    Args:
        model (nn.Module)

    Returns:
        int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)