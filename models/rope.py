import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from core.registry import MODULES


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of the input tensor:
    (-x2, x1) for each adjacent pair (x1, x2).
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


@MODULES.register()
class FrequencyAdaptive2DRoPE(nn.Module):
    """
    Frequency-Adaptive 2D Rotary Position Embedding (RoPE) for Vision Transformers.

    Adapts coordinates frequency-wise:
        p_k = alpha_k * p_raw + (1 - alpha_k) * p_norm
    where alpha_k is a function of the frequency omega_k:
        - Low frequencies (global structure) -> alpha -> 0 (normalized coordinates)
        - High frequencies (local structure) -> alpha -> 1 (raw coordinates)

    Supports modes:
        - 'adaptive_2d': Frequency-dependent linear ramp for alpha_k
        - 'standard_2d': alpha_k = 1.0 (pure raw coordinates)
        - 'normalized_2d': alpha_k = 0.0 (pure normalized coordinates)
        - 'learnable_2d': alpha_k is an nn.Parameter initialized with adaptive schedule
    """

    def __init__(
        self,
        dim: int,
        ref_img_size: int = 224,
        patch_size: int = 16,
        base: float = 10000.0,
        mode: str = "adaptive_2d",
    ):
        super().__init__()
        assert dim % 4 == 0, f"RoPE head dim must be divisible by 4, got {dim}"
        self.dim = dim
        self.ref_img_size = ref_img_size
        self.patch_size = patch_size
        self.base = base
        self.mode = mode

        # Half of head_dim allocated to x (width), half to y (height)
        self.dim_x = dim // 2
        self.dim_y = dim // 2
        self.num_bands = self.dim_x // 2  # Frequency bands per axis

        # Base frequencies: omega_m = base^(-2m / dim_x) for m in [0, num_bands - 1]
        freq_indices = torch.arange(self.num_bands, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (2.0 * freq_indices / self.dim_x))
        self.register_buffer("inv_freq", inv_freq)

        # Alpha initialization: m=0 is highest freq (raw, alpha=1), m=M-1 is lowest freq (norm, alpha=0)
        if self.num_bands > 1:
            alpha_init = 1.0 - (freq_indices / (self.num_bands - 1))
        else:
            alpha_init = torch.tensor([0.5], dtype=torch.float32)

        if mode == "learnable_2d":
            # Reparameterize with logit so sigmoid(param) stays in (0, 1)
            clamped_init = torch.clamp(alpha_init, 1e-4, 1.0 - 1e-4)
            logit_init = torch.log(clamped_init / (1.0 - clamped_init))
            self.alpha_param = nn.Parameter(logit_init)
        else:
            self.register_buffer("alpha_fixed", alpha_init)

    def get_alpha(self) -> torch.Tensor:
        """Returns alpha values of shape (num_bands,)."""
        if self.mode == "learnable_2d":
            return torch.sigmoid(self.alpha_param)
        elif self.mode == "standard_2d":
            return torch.ones_like(self.alpha_fixed)
        elif self.mode == "normalized_2d":
            return torch.zeros_like(self.alpha_fixed)
        else:  # 'adaptive_2d'
            return self.alpha_fixed

    def compute_freqs_cis(
        self,
        grid_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        has_cls_token: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes (cos, sin) tensors for the given 2D patch grid.
        Returns:
            cos: (N, dim)
            sin: (N, dim)
        where N = (1 + H*W) if has_cls_token else (H*W).
        """
        H_g, W_g = grid_size
        ref_grid_h = self.ref_img_size // self.patch_size
        ref_grid_w = self.ref_img_size // self.patch_size

        # Raw coordinate meshgrid: shape (H_g, W_g)
        y_raw = torch.arange(H_g, device=device, dtype=dtype)
        x_raw = torch.arange(W_g, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_raw, x_raw, indexing="ij")
        grid_y = grid_y.flatten()  # (H_g * W_g,)
        grid_x = grid_x.flatten()  # (H_g * W_g,)

        # Normalized coordinates relative to reference resolution
        scale_x = float(ref_grid_w) / float(W_g) if W_g > 0 else 1.0
        scale_y = float(ref_grid_h) / float(H_g) if H_g > 0 else 1.0
        grid_x_norm = grid_x * scale_x
        grid_y_norm = grid_y * scale_y

        alpha = self.get_alpha().to(device=device, dtype=dtype)  # (num_bands,)
        inv_freq = self.inv_freq.to(device=device, dtype=dtype)  # (num_bands,)

        # Compute adaptive coordinates per frequency band: (H_g * W_g, num_bands)
        # x_k = alpha_k * x_raw + (1 - alpha_k) * x_norm
        x_adaptive = grid_x.unsqueeze(-1) * alpha + grid_x_norm.unsqueeze(-1) * (1.0 - alpha)
        y_adaptive = grid_y.unsqueeze(-1) * alpha + grid_y_norm.unsqueeze(-1) * (1.0 - alpha)

        # Compute rotation angles theta = coordinate * inv_freq
        theta_x = x_adaptive * inv_freq.unsqueeze(0)  # (H_g * W_g, num_bands)
        theta_y = y_adaptive * inv_freq.unsqueeze(0)  # (H_g * W_g, num_bands)

        # Duplicate each frequency angle for the 2D rotation matrix: [cos, sin] pairs
        theta_x = theta_x.repeat_interleave(2, dim=-1)  # (H_g * W_g, dim_x)
        theta_y = theta_y.repeat_interleave(2, dim=-1)  # (H_g * W_g, dim_y)

        # Concatenate x and y axes along the head dimension
        theta_spatial = torch.cat([theta_x, theta_y], dim=-1)  # (H_g * W_g, dim)

        if has_cls_token:
            # CLS token at index 0 has angle 0 -> cos=1, sin=0 (unrotated)
            theta_cls = torch.zeros(1, self.dim, device=device, dtype=dtype)
            theta = torch.cat([theta_cls, theta_spatial], dim=0)  # (1 + H_g * W_g, dim)
        else:
            theta = theta_spatial

        cos = torch.cos(theta)  # (N, dim)
        sin = torch.sin(theta)  # (N, dim)
        return cos, sin

    def apply_rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies rotary position embedding to input tensor x.
        x: (B, num_heads, N, dim)
        cos: (N, dim) or broadcastable
        sin: (N, dim) or broadcastable
        """
        # Reshape cos and sin for broadcasting: (1, 1, N, dim)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        return (x * cos) + (rotate_half(x) * sin)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_size: Tuple[int, int],
        has_cls_token: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates Query and Key tensors using Frequency-Adaptive 2D RoPE.

        Args:
            q: (B, num_heads, N, head_dim)
            k: (B, num_heads, N, head_dim)
            grid_size: (H_patches, W_patches)
            has_cls_token: Whether index 0 is the CLS token
        Returns:
            q_rot, k_rot: rotated tensors of identical shapes
        """
        cos, sin = self.compute_freqs_cis(
            grid_size=grid_size,
            device=q.device,
            dtype=q.dtype,
            has_cls_token=has_cls_token,
        )
        q_rot = self.apply_rope(q, cos, sin)
        k_rot = self.apply_rope(k, cos, sin)
        return q_rot, k_rot
