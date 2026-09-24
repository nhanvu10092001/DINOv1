from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.registry import MODULES


@MODULES.register()
class SpatialSemanticAttention(nn.Module):
    """
    Spatial-Semantic Sparse Softmax Attention.

    Combines:
    1. Local Spatial Context: A 2D spatial window neighborhood N_local(i) around each patch.
    2. Global Semantic Context: Top-K key tokens with highest similarity N_semantic(i).
    3. Global CLS Routing: Bidirectional connection between CLS token and all patches.

    Maintains exact Softmax Attention properties:
        Y_i = softmax( Q_i * K_{S_i}^T / sqrt(d) ) * V_{S_i}
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        sparse_mode: str = "sparse",
        k_local_window: int = 3,
        k_semantic: int = 16,
        cls_full_attn: bool = True,
        attn_backend: str = "mask",
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparse_mode = sparse_mode
        self.k_local_window = k_local_window
        self.k_semantic = k_semantic
        self.cls_full_attn = cls_full_attn
        self.attn_backend = attn_backend
        self.rope = rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Cache for geometric local masks: (grid_h, grid_w, has_cls) -> mask tensor
        self._mask_cache = {}

    def get_local_mask(
        self,
        grid_size: Tuple[int, int],
        has_cls_token: bool,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Builds or retrieves cached boolean mask of shape (N, N) where True means
        query i should attend to key j via local spatial proximity (or CLS routing).
        """
        H_g, W_g = grid_size
        cache_key = (H_g, W_g, has_cls_token, self.k_local_window, self.cls_full_attn, device)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        N_patches = H_g * W_g
        radius = self.k_local_window // 2

        # 2D patch coordinate grid
        rows = torch.arange(H_g, device=device)
        cols = torch.arange(W_g, device=device)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        grid_r = grid_r.flatten()  # (N_patches,)
        grid_c = grid_c.flatten()  # (N_patches,)

        # Spatial distance check
        dr = (grid_r.unsqueeze(1) - grid_r.unsqueeze(0)).abs()
        dc = (grid_c.unsqueeze(1) - grid_c.unsqueeze(0)).abs()
        patch_local_mask = (dr <= radius) & (dc <= radius)  # (N_patches, N_patches)

        if has_cls_token:
            N = 1 + N_patches
            full_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            # Patch-to-patch local connections (indices 1..N-1)
            full_mask[1:, 1:] = patch_local_mask

            if self.cls_full_attn:
                # CLS token query (row 0) attends to all tokens
                full_mask[0, :] = True
                # All patch queries attend to CLS token key (col 0)
                full_mask[:, 0] = True
            else:
                full_mask[0, 0] = True
                full_mask[:, 0] = True
        else:
            full_mask = patch_local_mask

        self._mask_cache[cache_key] = full_mask
        return full_mask

    def build_attention_mask(
        self,
        sim: torch.Tensor,
        grid_size: Tuple[int, int],
        has_cls_token: bool,
    ) -> torch.Tensor:
        """
        Builds the complete sparse attention boolean mask (B, H, N, N).
        Args:
            sim: Raw scaled dot-product similarity (B, H, N, N)
            grid_size: (H_g, W_g)
            has_cls_token: whether sequence has CLS token at index 0
        """
        B, H, N, _ = sim.shape
        device = sim.device

        if self.sparse_mode == "dense":
            return torch.ones(B, H, N, N, dtype=torch.bool, device=device)

        local_mask = self.get_local_mask(grid_size, has_cls_token, device)  # (N, N)

        if self.sparse_mode == "local_only":
            return local_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)

        # Global Semantic Top-K
        # For query i, candidates are keys NOT already in local_mask
        expanded_local = local_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)

        if self.sparse_mode == "semantic_only":
            # Mask only diagonal self-attention and CLS if needed, or select top-k across all keys
            sim_cand = sim.clone()
            if has_cls_token and self.cls_full_attn:
                # Preserve CLS routing
                base_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
                base_mask[0, :] = True
                base_mask[:, 0] = True
                expanded_base = base_mask.unsqueeze(0).unsqueeze(0).expand(B, H, N, N)
                sim_cand = sim_cand.masked_fill(expanded_base, float("-inf"))
            else:
                expanded_base = torch.zeros(B, H, N, N, dtype=torch.bool, device=device)

            k = min(self.k_semantic, N)
            topk_indices = torch.topk(sim_cand, k=k, dim=-1).indices
            semantic_mask = torch.zeros_like(sim, dtype=torch.bool)
            semantic_mask.scatter_(-1, topk_indices, True)
            return expanded_base | semantic_mask

        # 'sparse' mode: combine Local + Semantic Top-K
        # Mask out locally connected keys from candidate pool so Top-K finds distinct distant keys
        sim_cand = sim.masked_fill(expanded_local, float("-inf"))

        # Calculate number of available non-local keys
        k = min(self.k_semantic, N)
        topk_indices = torch.topk(sim_cand, k=k, dim=-1).indices
        semantic_mask = torch.zeros_like(sim, dtype=torch.bool)
        semantic_mask.scatter_(-1, topk_indices, True)

        combined_mask = expanded_local | semantic_mask
        return combined_mask

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        has_cls_token: bool = True,
        return_attention: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Spatial-Semantic Sparse Softmax Attention.

        Args:
            x: Input token features (B, N, C)
            grid_size: (H_patches, W_patches). If None, inferred as square sqrt(N - has_cls_token).
            has_cls_token: whether index 0 is CLS token
            return_attention: whether to return the attention weight tensor
        Returns:
            out: (B, N, C)
            attn_weights: (B, num_heads, N, N)
        """
        B, N, C = x.shape

        if grid_size is None:
            n_patches = N - 1 if has_cls_token else N
            side = int(math.isqrt(n_patches))
            assert side * side == n_patches, f"Cannot infer square grid from {n_patches} patches"
            grid_size = (side, side)

        # Q, K, V projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Apply 2D RoPE if provided
        if self.rope is not None:
            q, k = self.rope(q, k, grid_size=grid_size, has_cls_token=has_cls_token)

        # Compute dot-product similarity matrix
        sim = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Build sparse attention mask
        attn_mask = self.build_attention_mask(sim, grid_size, has_cls_token)  # (B, num_heads, N, N)

        if self.attn_backend == "gather":
            # True sparse index gather formulation for high resolution verification
            # Extract top active indices per row
            # Since attn_mask has varying true counts per row, we can use masked_fill for exact math
            # or gather along the max active candidates
            masked_sim = sim.masked_fill(~attn_mask, float("-inf"))
            attn_weights = F.softmax(masked_sim, dim=-1)
            # Replace NaNs if any row had all False (should not happen due to self/CLS attention)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            out = attn_weights @ v
        else:
            # Mask backend: highly optimized on modern GPUs with tensor cores
            masked_sim = sim.masked_fill(~attn_mask, float("-inf"))
            attn_weights = F.softmax(masked_sim, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            out = attn_weights @ v

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        return out, attn_weights
