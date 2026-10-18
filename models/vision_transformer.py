import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import MODULES, BACKBONES
from models.rope import FrequencyAdaptive2DRoPE
from models.sparse_attention import SpatialSemanticAttention, MultiScale2DWindowAttention

@MODULES.register()
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

@MODULES.register()
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn

@MODULES.register()
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

@MODULES.register()
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        rope: Optional[nn.Module] = None,
        sparse_attn_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if sparse_attn_cfg is not None:
            cfg_copy = dict(sparse_attn_cfg)
            attn_type = cfg_copy.pop("type", None)
            if attn_type == "MultiScale2DWindowAttention" or cfg_copy.get("sparse_mode") == "multiscale_2d":
                cfg_copy.pop("sparse_mode", None)
                self.attn = MultiScale2DWindowAttention(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    rope=rope,
                    **cfg_copy,
                )
            else:
                self.attn = SpatialSemanticAttention(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    rope=rope,
                    **cfg_copy,
                )
        elif rope is not None:
            self.attn = SpatialSemanticAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                sparse_mode="dense",
                rope=rope,
            )
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        return_attention: bool = False,
    ):
        if isinstance(self.attn, MultiScale2DWindowAttention):
            y, attn = self.attn(self.norm1(x), grid_size=grid_size, return_attention=return_attention)
        elif isinstance(self.attn, SpatialSemanticAttention):
            y, attn = self.attn(self.norm1(x), grid_size=grid_size, return_attention=True)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        if return_attention:
            return x, attn
        return x

@BACKBONES.register()
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        pos_embed_type: str = "learned_abs",
        rope_cfg: Optional[dict] = None,
        sparse_attn_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.pos_embed_type = pos_embed_type

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        head_dim = embed_dim // num_heads
        if rope_cfg is not None or pos_embed_type in ("rope", "both"):
            rope_kwargs = dict(ref_img_size=img_size, patch_size=patch_size)
            if rope_cfg:
                rope_kwargs.update(rope_cfg)
            self.rope = FrequencyAdaptive2DRoPE(dim=head_dim, **rope_kwargs)
        else:
            self.rope = None

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                rope=self.rope,
                sparse_attn_cfg=sparse_attn_cfg,
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # DINO head replaces standard classifier, so we use num_classes=0 and output features directly
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Init pos_embed and cls_token
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        # Determine the grid size from the expected number of patches
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size

        # Using a slight epsilon for math sqrt safely
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        grid_size = (H // self.patch_embed.patch_size, W // self.patch_embed.patch_size)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed_type == "learned_abs":
            x = x + self.interpolate_pos_encoding(x, W, H)
        elif self.pos_embed_type == "both":
            x = x + self.interpolate_pos_encoding(x, W, H)
        elif self.pos_embed_type == "rope":
            # Preserve learned CLS positional embedding for CLS token representation stability
            x[:, :1] = x[:, :1] + self.pos_embed[:, :1]

        return x, grid_size

    def forward_features(self, x):
        x, grid_size = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x, grid_size=grid_size)
        x = self.norm(x)
        return x[:, 0]  # Return CLS token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_last_selfattention(self, x):
        x, grid_size = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, grid_size=grid_size)
            else:
                _, attn = blk(x, grid_size=grid_size, return_attention=True)
                return attn

@BACKBONES.register("vit_tiny")
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

@BACKBONES.register("vit_small")
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

@BACKBONES.register("vit_tiny_sparse_rope")
def vit_tiny_sparse_rope(patch_size=16, pos_embed_type="rope", **kwargs):
    rope_cfg = kwargs.pop("rope_cfg", {"mode": "adaptive_2d", "ref_img_size": 224, "patch_size": patch_size})
    sparse_attn_cfg = kwargs.pop("sparse_attn_cfg", {"sparse_mode": "sparse", "k_local_window": 3, "k_semantic": 8})
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        pos_embed_type=pos_embed_type,
        rope_cfg=rope_cfg,
        sparse_attn_cfg=sparse_attn_cfg,
        **kwargs,
    )
    return model

@BACKBONES.register("vit_small_sparse_rope")
def vit_small_sparse_rope(patch_size=16, pos_embed_type="rope", **kwargs):
    rope_cfg = kwargs.pop("rope_cfg", {"mode": "adaptive_2d", "ref_img_size": 224, "patch_size": patch_size})
    sparse_attn_cfg = kwargs.pop("sparse_attn_cfg", {"sparse_mode": "sparse", "k_local_window": 3, "k_semantic": 16})
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        pos_embed_type=pos_embed_type,
        rope_cfg=rope_cfg,
        sparse_attn_cfg=sparse_attn_cfg,
        **kwargs,
    )
    return model

@BACKBONES.register("vit_tiny_multiscale")
def vit_tiny_multiscale(patch_size=16, pos_embed_type="learned_abs", **kwargs):
    head_window_sizes = kwargs.pop("head_window_sizes", [3, 5, -1])
    cls_bidirectional = kwargs.pop("cls_bidirectional", True)
    sparse_attn_cfg = kwargs.pop(
        "sparse_attn_cfg",
        {
            "type": "MultiScale2DWindowAttention",
            "head_window_sizes": head_window_sizes,
            "cls_bidirectional": cls_bidirectional,
        },
    )
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        pos_embed_type=pos_embed_type,
        rope_cfg=None,
        sparse_attn_cfg=sparse_attn_cfg,
        **kwargs,
    )
    return model

@BACKBONES.register("vit_small_multiscale")
def vit_small_multiscale(patch_size=16, pos_embed_type="learned_abs", **kwargs):
    head_window_sizes = kwargs.pop("head_window_sizes", [3, 5, 7, -1, -1, -1])
    cls_bidirectional = kwargs.pop("cls_bidirectional", True)
    sparse_attn_cfg = kwargs.pop(
        "sparse_attn_cfg",
        {
            "type": "MultiScale2DWindowAttention",
            "head_window_sizes": head_window_sizes,
            "cls_bidirectional": cls_bidirectional,
        },
    )
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        pos_embed_type=pos_embed_type,
        rope_cfg=None,
        sparse_attn_cfg=sparse_attn_cfg,
        **kwargs,
    )
    return model

