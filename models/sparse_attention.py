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


@MODULES.register()
class MultiScale2DWindowAttention(nn.Module):
    """
    Multi-Scale 2D Sliding-Window Sparse Attention with Bidirectional [CLS] Global Bridge.

    - Divides H attention heads into groups with distinct 2D sliding-window sizes
      extracted directly via F.unfold (e.g., for ViT-Small H=6:
      Group 1 (1 head): 3x3 window,
      Group 2 (1 head): 5x5 window,
      Group 3 (1 head): 7x7 window,
      Group 4 (3 heads): Full Dense Attention).
    - Bidirectional [CLS] Global Bridge:
      1. Global Aggregation: q_CLS attends to all N keys K_all.
      2. Global Broadcast: Every spatial query q_i in window group k_g x k_g attends to
         [k_CLS || {k_j}_{j in N_{k_g}(i)}] of size (1 + k_g^2).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        head_window_sizes: Optional[Tuple[int, ...]] = None,
        cls_bidirectional: bool = True,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cls_bidirectional = cls_bidirectional
        self.rope = rope

        if head_window_sizes is None:
            if num_heads == 6:
                # Default Hybrid 4-group config for ViT-Small (H=6):
                # 1 head 3x3, 1 head 5x5, 1 head 7x7, 3 heads Full Dense (-1)
                head_window_sizes = (3, 5, 7, -1, -1, -1)
            elif num_heads == 3:
                # Default Hybrid config for ViT-Tiny (H=3):
                head_window_sizes = (3, 5, -1)
            else:
                base_pattern = (3, 5, 7, -1, -1, -1)
                head_window_sizes = tuple(base_pattern[i % len(base_pattern)] for i in range(num_heads))
        else:
            head_window_sizes = tuple(head_window_sizes)

        assert len(head_window_sizes) == num_heads, (
            f"Length of head_window_sizes {len(head_window_sizes)} must match num_heads {num_heads}"
        )
        self.head_window_sizes = head_window_sizes

        # Group head indices by window size while preserving group order
        groups = {}
        for h_idx, w_size in enumerate(self.head_window_sizes):
            groups.setdefault(w_size, []).append(h_idx)
        self.head_groups = []
        for w_size, indices in groups.items():
            is_contig = indices == list(range(indices[0], indices[-1] + 1))
            h_sel = slice(indices[0], indices[-1] + 1) if is_contig else indices
            self.head_groups.append((w_size, h_sel, len(indices)))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Cache for border validity masks of F.unfold: (H_g, W_g, k_g, device) -> valid_mask & index_map
        self._unfold_cache = {}

    def _get_unfold_meta(
        self,
        grid_size: Tuple[int, int],
        k_g: int,
        has_cls_token: bool,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cached border validity mask and token index map for F.unfold of kernel k_g x k_g.
        - valid_mask: (1, 1, N_p, 1 + k_g^2) bool (True for valid in-bounds neighbors and CLS)
        - unfold_idx: (N_p, k_g^2) long (1-based token index in [1..N_p], or 0 for out-of-bounds)
        """
        H_g, W_g = grid_size
        cache_key = (H_g, W_g, k_g, has_cls_token, self.cls_bidirectional, device)
        if cache_key in self._unfold_cache:
            return self._unfold_cache[cache_key]

        N_p = H_g * W_g
        pad = k_g // 2

        # Create 1-based spatial index grid (1 .. N_p), 0 indicates padded out-of-bounds border
        idx_grid = torch.arange(1, N_p + 1, device=device, dtype=torch.float32).view(1, 1, H_g, W_g)
        idx_unf = F.unfold(idx_grid, kernel_size=k_g, padding=pad)  # (1, k_g^2, N_p)
        idx_unf = idx_unf.squeeze(0).transpose(0, 1).long()  # (N_p, k_g^2)

        valid_spatial = idx_unf > 0  # (N_p, k_g^2)
        if has_cls_token and self.cls_bidirectional:
            cls_valid = torch.ones((N_p, 1), dtype=torch.bool, device=device)
            valid_mask = torch.cat([cls_valid, valid_spatial], dim=-1).unsqueeze(0).unsqueeze(0)
        else:
            valid_mask = valid_spatial.unsqueeze(0).unsqueeze(0)

        self._unfold_cache[cache_key] = (valid_mask, idx_unf)
        return valid_mask, idx_unf

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
        has_cls_token: bool = True,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, C = x.shape
        device = x.device

        if grid_size is None:
            n_patches = N - 1 if has_cls_token else N
            side = int(math.isqrt(n_patches))
            assert side * side == n_patches, f"Cannot infer square grid from {n_patches} patches"
            grid_size = (side, side)

        H_g, W_g = grid_size
        N_p = H_g * W_g

        # 1. Project Q, K, V: (B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q, k = self.rope(q, k, grid_size=grid_size, has_cls_token=has_cls_token)

        # Allocate output tensor for all heads: (B, num_heads, N, head_dim)
        out = torch.empty_like(q)
        attn_full = (
            torch.zeros((B, self.num_heads, N, N), dtype=q.dtype, device=device)
            if return_attention
            else None
        )

        if has_cls_token:
            # --- Chiều 1 (Global Aggregation): q_CLS attends to all N keys across all heads ---
            q_cls = q[:, :, :1, :]  # (B, H, 1, d_k)
            sim_cls = (q_cls @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
            attn_cls = F.softmax(sim_cls, dim=-1).to(q.dtype)
            out[:, :, :1, :] = (attn_cls @ v).to(out.dtype)  # (B, H, 1, d_k)
            if return_attention:
                attn_full[:, :, :1, :] = attn_cls

            q_patch = q[:, :, 1:, :]  # (B, H, N_p, d_k)
            k_patch = k[:, :, 1:, :]  # (B, H, N_p, d_k)
            v_patch = v[:, :, 1:, :]  # (B, H, N_p, d_k)
            k_cls = k[:, :, :1, :]    # (B, H, 1, d_k)
            v_cls = v[:, :, :1, :]    # (B, H, 1, d_k)
        else:
            q_patch = q
            k_patch = k
            v_patch = v
            k_cls = None
            v_cls = None

        # --- Chiều 2: Process each head group (2D Sliding Windows via F.unfold or Full Dense) ---
        for w_size, h_sel, H_sub in self.head_groups:
            q_g = q_patch[:, h_sel, :, :]  # (B, H_sub, N_p, d_k)

            # Full Dense head group (w_size <= 0 or window covers entire grid)
            if w_size <= 0 or (w_size >= H_g and w_size >= W_g):
                k_all_g = k[:, h_sel, :, :]  # (B, H_sub, N, d_k)
                v_all_g = v[:, h_sel, :, :]  # (B, H_sub, N, d_k)
                sim_g = (q_g @ k_all_g.transpose(-2, -1)) * self.scale  # (B, H_sub, N_p, N)
                attn_g = F.softmax(sim_g, dim=-1).to(q.dtype)
                y_g = (attn_g @ v_all_g).to(out.dtype)  # (B, H_sub, N_p, d_k)

                if has_cls_token:
                    out[:, h_sel, 1:, :] = y_g
                    if return_attention:
                        attn_full[:, h_sel, 1:, :] = attn_g
                else:
                    out[:, h_sel, :, :] = y_g
                    if return_attention:
                        attn_full[:, h_sel, :, :] = attn_g
            else:
                # True 2D Sliding-Window extraction via fused F.unfold on [K, V]
                pad = w_size // 2
                W_sq = w_size * w_size

                k_p_g = k_patch[:, h_sel, :, :]  # (B, H_sub, N_p, d_k)
                v_p_g = v_patch[:, h_sel, :, :]  # (B, H_sub, N_p, d_k)

                # Fuse K and V along channel dim to run a single F.unfold kernel: (B * H_sub, 2 * d_k, H_g, W_g)
                kv_2d = torch.cat([k_p_g, v_p_g], dim=-1).transpose(-2, -1).reshape(
                    B * H_sub, 2 * self.head_dim, H_g, W_g
                )
                kv_unf = F.unfold(kv_2d, kernel_size=w_size, padding=pad)  # (B * H_sub, 2 * d_k * W_sq, N_p)
                kv_win = kv_unf.view(B, H_sub, 2, self.head_dim, W_sq, N_p).permute(2, 0, 1, 5, 4, 3)
                k_win, v_win = kv_win[0], kv_win[1]  # each (B, H_sub, N_p, W_sq, d_k)

                if has_cls_token and self.cls_bidirectional:
                    # Prepend k_CLS and v_CLS to every local window -> (B, H_sub, N_p, 1 + W_sq, d_k)
                    k_cls_g = k_cls[:, h_sel, :, :].unsqueeze(-2).expand(B, H_sub, N_p, 1, self.head_dim)
                    v_cls_g = v_cls[:, h_sel, :, :].unsqueeze(-2).expand(B, H_sub, N_p, 1, self.head_dim)
                    k_tilde = torch.cat([k_cls_g, k_win], dim=-2)
                    v_tilde = torch.cat([v_cls_g, v_win], dim=-2)
                else:
                    k_tilde = k_win
                    v_tilde = v_win

                # Batched GEMM within the (1 + k_g^2) window only: (B, H_sub, N_p, 1, d_k) @ (B, H_sub, N_p, d_k, 1 + W_sq)
                sim_g = torch.matmul(q_g.unsqueeze(-2), k_tilde.transpose(-2, -1)).squeeze(-2) * self.scale

                # Mask out-of-bounds image border padding positions in the local window
                valid_mask, idx_unf = self._get_unfold_meta(grid_size, w_size, has_cls_token, device)
                sim_g = sim_g.masked_fill(~valid_mask, float("-inf"))

                attn_g = F.softmax(sim_g, dim=-1).to(q.dtype)  # (B, H_sub, N_p, 1 + W_sq)
                y_g = torch.matmul(attn_g.unsqueeze(-2), v_tilde).squeeze(-2).to(out.dtype)  # (B, H_sub, N_p, d_k)

                if has_cls_token:
                    out[:, h_sel, 1:, :] = y_g
                    if return_attention:
                        sub_attn = torch.zeros((B, H_sub, N_p, N), dtype=q.dtype, device=device)
                        if self.cls_bidirectional:
                            sub_attn[:, :, :, 0] = attn_g[:, :, :, 0]
                            win_probs = attn_g[:, :, :, 1:]
                        else:
                            win_probs = attn_g
                        # Scatter valid window probabilities into full N x N matrix for visualization
                        safe_idx = idx_unf.unsqueeze(0).unsqueeze(0).expand(B, H_sub, N_p, W_sq)
                        valid_win_probs = (win_probs * (idx_unf > 0).unsqueeze(0).unsqueeze(0)).to(q.dtype)
                        sub_attn.scatter_add_(-1, safe_idx, valid_win_probs)
                        attn_full[:, h_sel, 1:, :] = sub_attn
                else:
                    out[:, h_sel, :, :] = y_g

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, attn_full
