"""DINO Projection Head implementation with weight normalization."""

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from core.registry import HEADS, MODULES


@HEADS.register()
@MODULES.register()
class DINOHead(nn.Module):
    """Projection head for DINO self-supervised learning.

    Projects token representations from the Vision Transformer backbone into a
    high-dimensional latent prototype space (e.g., out_dim=16384 or 65536).
    Consists of an MLP with GELU activations, L2 normalization, and a Weight-Normalized
    linear output layer with frozen magnitude (norm_last_layer=True).

    Args:
        in_dim: Input dimension from the backbone (e.g., 384 for ViT-Small).
        out_dim: Output dimension of projection space (e.g., 16384).
        use_bn: Whether to use Batch Normalization between MLP layers (default: False).
        norm_last_layer: If True, freezes the weight norm magnitude of the last layer to 1.
        nlayers: Total number of MLP layers (default: 3).
        hidden_dim: Hidden feature dimension of intermediate MLP layers (default: 2048).
        bottleneck_dim: Dimension of penultimate bottleneck layer (default: 256).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())

            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input features into DINO latent prototype logits.

        Args:
            x: Input feature tensor of shape (B, in_dim).

        Returns:
            Output logits of shape (B, out_dim).
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
