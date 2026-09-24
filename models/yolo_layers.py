import math
import torch
import torch.nn as nn
from core.registry import MODULES


@MODULES.register()
class PosEmbedAndClassToken(nn.Module):
    """
    Appends learnable CLS token to sequence and adds positional embeddings.
    Supports dynamic 2D bicubic interpolation if input crop spatial size differs from base num_patches.
    """

    def __init__(self, embed_dim: int, num_patches: int = 196):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, N_patches, embed_dim)
        B, N, C = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, embed_dim)

        if x.shape[1] == self.pos_embed.shape[1]:
            x = x + self.pos_embed
        else:
            # Interpolate positional embeddings for different crop sizes (e.g. 96x96 local crops)
            cls_pos = self.pos_embed[:, 0:1]
            patch_pos = self.pos_embed[:, 1:]
            dim = patch_pos.shape[-1]

            orig_size = int(math.sqrt(self.pos_embed.shape[1] - 1))
            new_size = int(math.sqrt(x.shape[1] - 1))

            patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
            patch_pos = nn.functional.interpolate(
                patch_pos,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
            pos_embed = torch.cat((cls_pos, patch_pos), dim=1)
            x = x + pos_embed

        return x


@MODULES.register()
class ExtractToken(nn.Module):
    """
    Extracts a specific token along sequence dimension (dim=1).
    Default index=0 extracts the [CLS] token representation.
    """

    def __init__(self, index: int = 0):
        super().__init__()
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.index]


@MODULES.register()
class LayerNorm1d(nn.Module):
    """
    Standard LayerNorm wrapper for 1D feature tokens.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


@MODULES.register()
class Concat(nn.Module):
    """
    Concatenate a list of tensors along dimension dim.
    """

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list) -> torch.Tensor:
        return torch.cat(x, self.d)
