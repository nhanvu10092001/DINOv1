"""Unsupervised Saliency and Pseudo-Mask Extraction Utilities.

Combines spatial Sobel gradient magnitude, color contrast against border background tokens,
and spectral TokenCut affinity to isolate salient foreground objects without human labels.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def derive_salience_pseudo_mask(
    images: torch.Tensor,
    patch_features: Optional[torch.Tensor] = None,
    grid_size: Tuple[int, int] = (14, 14),
    target_sparsity: float = 0.35,
) -> torch.Tensor:
    """Computes an unsupervised TokenCut spectral + contour salience mask on the patch grid.

    Combines:
    1. Spatial gradient magnitude (object contours via Sobel-like filter).
    2. Color contrast against border background tokens.
    3. Spectral TokenCut degree centrality from normalized patch token affinities (when provided).

    Args:
        images: Normalized image tensor of shape (B, 3, H, W).
        patch_features: Optional patch embeddings of shape (B, N, D).
        grid_size: Patch grid size (H_g, W_g).
        target_sparsity: Fraction of foreground patches.

    Returns:
        Binary foreground target mask of shape (B, H_g * W_g).
    """
    B, _, _, _ = images.shape
    H_g, W_g = grid_size
    N = H_g * W_g

    # 1. Spatial gradient magnitude (object contours)
    dx = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    dy = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    grad_x = F.adaptive_avg_pool2d(dx, output_size=(H_g, W_g)).squeeze(1)
    grad_y = F.adaptive_avg_pool2d(dy, output_size=(H_g, W_g)).squeeze(1)
    edge_energy = (grad_x.pow(2) + grad_y.pow(2)).sqrt()
    edge_energy = edge_energy / (edge_energy.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # 2. Color contrast against border background tokens
    patches = F.adaptive_avg_pool2d(images, output_size=(H_g, W_g))
    border_tokens = torch.cat([
        patches[:, :, 0, :],
        patches[:, :, -1, :],
        patches[:, :, :, 0],
        patches[:, :, :, -1],
    ], dim=-1)
    border_mean = border_tokens.mean(dim=-1, keepdim=True).unsqueeze(-1)
    border_std = border_tokens.std(dim=-1, keepdim=True).unsqueeze(-1) + 0.1
    color_contrast = ((patches - border_mean) / border_std).pow(2).sum(dim=1).sqrt()
    color_contrast = color_contrast / (color_contrast.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    yy = torch.linspace(-1.0, 1.0, H_g, device=images.device)
    xx = torch.linspace(-1.0, 1.0, W_g, device=images.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    vignette = torch.clamp(1.15 - 0.35 * (gx.pow(2) + gy.pow(2)), min=0.4, max=1.0).unsqueeze(0)

    img_salience = ((0.55 * edge_energy + 0.45 * color_contrast) * vignette).view(B, N)

    if patch_features is not None:
        # Spectral TokenCut centrality: patches with strong affinity to foreground cluster
        # and low affinity to image border tokens
        f_norm = F.normalize(patch_features.detach().float(), dim=-1, p=2)  # (B, N, D)
        f_2d = f_norm.view(B, H_g, W_g, -1)
        bg_proto = F.normalize(
            torch.cat([
                f_2d[:, 0, :, :],
                f_2d[:, -1, :, :],
                f_2d[:, :, 0, :],
                f_2d[:, :, -1, :],
            ], dim=1).mean(dim=1, keepdim=True),
            dim=-1,
            p=2,
        )  # (B, 1, D)
        bg_sim = (f_norm * bg_proto).sum(dim=-1)  # (B, N)
        fg_spectral = 1.0 - torch.clamp(bg_sim, min=0.0, max=1.0)
        fg_spectral = (fg_spectral - fg_spectral.amin(dim=-1, keepdim=True)) / (
            fg_spectral.amax(dim=-1, keepdim=True) - fg_spectral.amin(dim=-1, keepdim=True) + 1e-6
        )
        salience = 0.55 * fg_spectral + 0.45 * img_salience
    else:
        salience = img_salience

    k_fg = max(1, int(round(N * target_sparsity)))
    topk_idx = torch.topk(salience, k=k_fg, dim=-1).indices
    mask = torch.zeros((B, N), device=images.device, dtype=torch.float32)
    mask.scatter_(-1, topk_idx, 1.0)
    return mask

