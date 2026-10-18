import pytest
import torch
import torch.nn as nn
from models.sparse_attention import SpatialSemanticAttention


def test_sparse_attention_output_shape():
    B, N, C = 2, 197, 384  # 1 CLS + 14x14 patches, embed_dim=384
    dim = C
    attn = SpatialSemanticAttention(
        dim=dim,
        num_heads=6,
        sparse_mode="sparse",
        k_local_window=3,
        k_semantic=16,
    )
    x = torch.randn(B, N, C)
    out, weights = attn(x, grid_size=(14, 14), has_cls_token=True)
    assert out.shape == (B, N, C)
    assert weights.shape == (B, 6, N, N)


def test_sparse_attention_softmax_sum():
    B, N, C = 2, 37, 192  # 1 CLS + 6x6 patches
    attn = SpatialSemanticAttention(
        dim=C,
        num_heads=3,
        sparse_mode="sparse",
        k_local_window=3,
        k_semantic=8,
    )
    x = torch.randn(B, N, C)
    out, weights = attn(x, grid_size=(6, 6), has_cls_token=True)

    # Every row of attention weights must sum to 1.0 (exact softmax)
    row_sums = weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)


def test_sparse_attention_cls_bidirectional():
    B, N, C = 2, 37, 192
    attn = SpatialSemanticAttention(
        dim=C,
        num_heads=3,
        sparse_mode="sparse",
        k_local_window=3,
        k_semantic=8,
        cls_full_attn=True,
    )
    x = torch.randn(B, N, C)
    _, weights = attn(x, grid_size=(6, 6), has_cls_token=True)

    # 1. CLS token query (row 0) attends to all tokens (> 0 weight)
    assert (weights[:, :, 0, :] > 0).all()

    # 2. All patch queries (rows 1..N-1) attend to the CLS token (column 0 > 0 weight)
    assert (weights[:, :, 1:, 0] > 0).all()


def test_sparse_attention_gradient_flow():
    B, N, C = 2, 37, 192
    attn = SpatialSemanticAttention(
        dim=C,
        num_heads=3,
        sparse_mode="sparse",
        k_local_window=3,
        k_semantic=8,
    )
    x = torch.randn(B, N, C, requires_grad=True)
    out, _ = attn(x, grid_size=(6, 6), has_cls_token=True)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert attn.qkv.weight.grad is not None and not torch.isnan(attn.qkv.weight.grad).any()
    assert attn.proj.weight.grad is not None and not torch.isnan(attn.proj.weight.grad).any()


def test_sparse_attention_modes():
    B, N, C = 1, 37, 192
    x = torch.randn(B, N, C)

    modes = ["dense", "local_only", "semantic_only", "sparse"]
    for mode in modes:
        attn = SpatialSemanticAttention(
            dim=C,
            num_heads=3,
            sparse_mode=mode,
            k_local_window=3,
            k_semantic=8,
        )
        out, weights = attn(x, grid_size=(6, 6), has_cls_token=True)
        assert out.shape == (B, N, C)
        assert not torch.isnan(out).any()
        row_sums = weights.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)


def test_sparse_attention_backends():
    B, N, C = 2, 37, 192
    x = torch.randn(B, N, C)

    attn_mask = SpatialSemanticAttention(
        dim=C,
        num_heads=3,
        sparse_mode="sparse",
        k_local_window=3,
        k_semantic=8,
        attn_backend="mask",
    )
    attn_gather = SpatialSemanticAttention(
        dim=C,
        num_heads=3,
        sparse_mode="sparse",
        k_local_window=3,
        k_semantic=8,
        attn_backend="gather",
    )
    # Share weights
    attn_gather.load_state_dict(attn_mask.state_dict())

    out_mask, _ = attn_mask(x, grid_size=(6, 6), has_cls_token=True)
    out_gather, _ = attn_gather(x, grid_size=(6, 6), has_cls_token=True)

    assert out_mask.shape == (B, N, C)
    assert out_gather.shape == (B, N, C)
    torch.testing.assert_close(out_mask, out_gather, rtol=1e-4, atol=1e-4)


def test_multiscale_2d_window_attention_exact_math():
    """
    Verifies that MultiScale2DWindowAttention (using F.unfold without N x N matrix for local heads)
    matches an explicit reference per-head mask implementation to 1e-5 precision.
    """
    from models.sparse_attention import MultiScale2DWindowAttention

    torch.manual_seed(42)
    B, H_g, W_g, C = 2, 14, 14, 384
    N = 1 + H_g * W_g
    num_heads = 6
    head_windows = (3, 5, 7, -1, -1, -1)

    attn = MultiScale2DWindowAttention(
        dim=C,
        num_heads=num_heads,
        qkv_bias=True,
        head_window_sizes=head_windows,
        cls_bidirectional=True,
        rope=None,
    )
    x = torch.randn(B, N, C)
    out_unfold, weights_unfold = attn(x, grid_size=(H_g, W_g), has_cls_token=True, return_attention=True)

    # Reference computation per head
    qkv = attn.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    sim_ref = (q @ k.transpose(-2, -1)) * attn.scale

    rows = torch.arange(H_g)
    cols = torch.arange(W_g)
    gr, gc = torch.meshgrid(rows, cols, indexing="ij")
    gr, gc = gr.flatten(), gc.flatten()
    dr = (gr.unsqueeze(1) - gr.unsqueeze(0)).abs()
    dc = (gc.unsqueeze(1) - gc.unsqueeze(0)).abs()

    mask_ref = torch.zeros(1, num_heads, N, N, dtype=torch.bool)
    for h_idx, w_size in enumerate(head_windows):
        if w_size <= 0:
            mask_ref[0, h_idx, :, :] = True
        else:
            rad = w_size // 2
            mask_ref[0, h_idx, 1:, 1:] = (dr <= rad) & (dc <= rad)
            mask_ref[0, h_idx, 0, :] = True
            mask_ref[0, h_idx, :, 0] = True

    weights_ref = torch.softmax(sim_ref.masked_fill(~mask_ref, float("-inf")), dim=-1)
    out_ref = (weights_ref @ v).transpose(1, 2).reshape(B, N, C)
    out_ref = attn.proj(out_ref)

    torch.testing.assert_close(out_unfold, out_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weights_unfold, weights_ref, rtol=1e-5, atol=1e-5)


def test_vit_small_multiscale_no_rope():
    from models.vision_transformer import vit_small_multiscale

    model = vit_small_multiscale(patch_size=16)
    assert model.rope is None
    assert model.pos_embed_type == "learned_abs"

    # Global crop (224x224) and Local crop (96x96)
    x_global = torch.randn(2, 3, 224, 224, requires_grad=True)
    x_local = torch.randn(2, 3, 96, 96)

    out_g = model(x_global)
    out_l = model(x_local)
    assert out_g.shape == (2, 384)
    assert out_l.shape == (2, 384)

    attn_map = model.get_last_selfattention(x_global)
    assert attn_map.shape == (2, 6, 197, 197)

    loss = out_g.sum() + out_l.sum()
    loss.backward()
    assert x_global.grad is not None and not torch.isnan(x_global.grad).any()

