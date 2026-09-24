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
