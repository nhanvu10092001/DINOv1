import pytest
import torch
import torch.nn as nn
from models.rope import FrequencyAdaptive2DRoPE


def test_rope_output_shape():
    rope = FrequencyAdaptive2DRoPE(dim=64, ref_img_size=224, patch_size=16)
    B, num_heads, N, head_dim = 2, 6, 197, 64  # 1 CLS + 196 patches
    q = torch.randn(B, num_heads, N, head_dim)
    k = torch.randn(B, num_heads, N, head_dim)

    q_rot, k_rot = rope(q, k, grid_size=(14, 14), has_cls_token=True)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_cls_token_unrotated():
    rope = FrequencyAdaptive2DRoPE(dim=64, ref_img_size=224, patch_size=16)
    B, num_heads, N, head_dim = 2, 6, 197, 64
    q = torch.randn(B, num_heads, N, head_dim)
    k = torch.randn(B, num_heads, N, head_dim)

    q_rot, k_rot = rope(q, k, grid_size=(14, 14), has_cls_token=True)
    # CLS token at index 0 must be unchanged
    torch.testing.assert_close(q_rot[:, :, 0], q[:, :, 0])
    torch.testing.assert_close(k_rot[:, :, 0], k[:, :, 0])

    # Spatial patch tokens (index > 0) should be modified by rotation
    assert not torch.allclose(q_rot[:, :, 1:], q[:, :, 1:])
    assert not torch.allclose(k_rot[:, :, 1:], k[:, :, 1:])


def test_rope_without_cls_token():
    rope = FrequencyAdaptive2DRoPE(dim=64, ref_img_size=224, patch_size=16)
    B, num_heads, N, head_dim = 2, 6, 196, 64
    q = torch.randn(B, num_heads, N, head_dim)
    k = torch.randn(B, num_heads, N, head_dim)

    q_rot, k_rot = rope(q, k, grid_size=(14, 14), has_cls_token=False)
    assert q_rot.shape == (B, num_heads, 196, head_dim)
    assert not torch.allclose(q_rot, q)


def test_rope_modes_distinction():
    dim = 64
    grid_size = (6, 6)  # Local crop resolution (96x96 -> 6x6 patches != 14x14 ref)
    N = 37  # 1 CLS + 36 patches
    q = torch.randn(1, 1, N, dim)
    k = torch.randn(1, 1, N, dim)

    rope_adapt = FrequencyAdaptive2DRoPE(dim=dim, mode="adaptive_2d", ref_img_size=224, patch_size=16)
    rope_std = FrequencyAdaptive2DRoPE(dim=dim, mode="standard_2d", ref_img_size=224, patch_size=16)
    rope_norm = FrequencyAdaptive2DRoPE(dim=dim, mode="normalized_2d", ref_img_size=224, patch_size=16)

    q_adapt, _ = rope_adapt(q, k, grid_size=grid_size, has_cls_token=True)
    q_std, _ = rope_std(q, k, grid_size=grid_size, has_cls_token=True)
    q_norm, _ = rope_norm(q, k, grid_size=grid_size, has_cls_token=True)

    # When grid_size != ref_grid_size, adaptive, standard, and normalized should yield distinct rotations
    assert not torch.allclose(q_adapt[:, :, 1:], q_std[:, :, 1:])
    assert not torch.allclose(q_adapt[:, :, 1:], q_norm[:, :, 1:])
    assert not torch.allclose(q_std[:, :, 1:], q_norm[:, :, 1:])


def test_rope_learnable_mode_gradients():
    dim = 64
    rope = FrequencyAdaptive2DRoPE(dim=dim, mode="learnable_2d", ref_img_size=224, patch_size=16)
    assert hasattr(rope, "alpha_param")

    q = torch.randn(2, 4, 37, dim, requires_grad=True)
    k = torch.randn(2, 4, 37, dim, requires_grad=True)

    q_rot, k_rot = rope(q, k, grid_size=(6, 6), has_cls_token=True)
    loss = (q_rot * k_rot).sum()
    loss.backward()

    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert rope.alpha_param.grad is not None and not torch.isnan(rope.alpha_param.grad).any()


def test_rope_multiple_resolutions():
    rope = FrequencyAdaptive2DRoPE(dim=64, ref_img_size=224, patch_size=16)
    B, H, dim = 2, 4, 64

    # Test varying resolutions: 96x96 (6x6), 224x224 (14x14), 448x448 (28x28)
    for grid in [(6, 6), (14, 14), (28, 28)]:
        N = 1 + grid[0] * grid[1]
        q = torch.randn(B, H, N, dim)
        k = torch.randn(B, H, N, dim)
        q_rot, k_rot = rope(q, k, grid_size=grid, has_cls_token=True)
        assert q_rot.shape == (B, H, N, dim)
        assert not torch.isnan(q_rot).any()
