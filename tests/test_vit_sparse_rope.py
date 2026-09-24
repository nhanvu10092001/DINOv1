import pytest
import torch
from core.registry import BACKBONES
import models
from models.vision_transformer import VisionTransformer, vit_tiny_sparse_rope, vit_small_sparse_rope


def test_vit_sparse_rope_creation():
    model_tiny = BACKBONES.build(dict(type="vit_tiny_sparse_rope", patch_size=16))
    assert isinstance(model_tiny, VisionTransformer)
    assert model_tiny.embed_dim == 192

    model_small = BACKBONES.build(dict(type="vit_small_sparse_rope", patch_size=16))
    assert isinstance(model_small, VisionTransformer)
    assert model_small.embed_dim == 384


def test_vit_sparse_rope_forward():
    model = vit_tiny_sparse_rope(patch_size=16)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 192)


def test_vit_sparse_rope_multiple_resolutions():
    model = vit_small_sparse_rope(patch_size=16)
    resolutions = [96, 224, 336, 448]

    for res in resolutions:
        x = torch.randn(1, 3, res, res)
        out = model(x)
        assert out.shape == (1, 384)
        assert not torch.isnan(out).any()


def test_vit_get_last_selfattention():
    model = vit_tiny_sparse_rope(patch_size=16)
    x = torch.randn(1, 3, 96, 96)  # 6x6 patches -> 37 tokens
    attn = model.get_last_selfattention(x)
    assert attn.shape == (1, 3, 37, 37)
    row_sums = attn.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-5, atol=1e-5)


def test_vit_backward_compatibility():
    # Verify standard vit_tiny and vit_small without sparse/rope work unchanged
    base_tiny = BACKBONES.build(dict(type="vit_tiny", patch_size=16))
    base_small = BACKBONES.build(dict(type="vit_small", patch_size=16))

    x = torch.randn(2, 3, 224, 224)
    assert base_tiny(x).shape == (2, 192)
    assert base_small(x).shape == (2, 384)


def test_vit_sparse_rope_gradient_flow():
    model = vit_tiny_sparse_rope(patch_size=16)
    x = torch.randn(2, 3, 96, 96)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Verify gradients exist for blocks
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert not torch.isnan(model.blocks[0].attn.qkv.weight.grad).any()
