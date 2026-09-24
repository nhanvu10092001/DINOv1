import pytest
import torch
from core.registry import MODULES, BACKBONES, HEADS, LOSSES, TRANSFORMS
import models
import losses
import dino_datasets


def test_modules_and_backbones_registered():
    assert "PatchEmbed" in MODULES
    assert "Attention" in MODULES
    assert "MLP" in MODULES
    assert "Block" in MODULES
    assert "VisionTransformer" in BACKBONES
    assert "vit_tiny" in BACKBONES
    assert "vit_small" in BACKBONES


def test_heads_registered():
    assert "DINOHead" in HEADS
    assert "DINOHead" in MODULES
    head = HEADS.build({"type": "DINOHead", "in_dim": 192, "out_dim": 1024, "nlayers": 2})
    x = torch.randn(2, 192)
    assert head(x).shape == (2, 1024)


def test_losses_registered():
    assert "DINOLoss" in LOSSES
    loss_fn = LOSSES.build({
        "type": "DINOLoss",
        "out_dim": 1024,
        "ncrops": 8,
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 1,
        "nepochs": 10,
    })
    s_out = torch.randn(8 * 2, 1024)
    t_out = torch.randn(2 * 2, 1024)
    l = loss_fn(s_out, t_out, epoch=0)
    assert isinstance(l, torch.Tensor)


def test_transforms_registered():
    assert "DataAugmentationDINO" in TRANSFORMS
    transform = TRANSFORMS.build({
        "type": "DataAugmentationDINO",
        "global_crops_scale": [0.4, 1.0],
        "local_crops_scale": [0.05, 0.4],
        "local_crops_number": 4,
    })
    assert transform.local_crops_number == 4
