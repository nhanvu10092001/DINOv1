import os
import yaml
import pytest
import torch
import models
import losses
from core.builder import build_loss, build_model, build_optimizer


def test_build_sparse_rope_config():
    config_path = "/home/si/study/xla/DINOv1/configs/vit_small_sparse_rope.yaml"
    assert os.path.exists(config_path), f"Config file missing: {config_path}"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    student, teacher = build_model(cfg, device="cpu")
    loss_fn = build_loss(cfg, device="cpu")
    optimizer = build_optimizer(student, cfg)

    assert student is not None
    assert teacher is not None
    assert loss_fn is not None
    assert optimizer is not None


def test_sparse_rope_multicrop_training_step():
    config_path = "/home/si/study/xla/DINOv1/configs/vit_small_sparse_rope.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    student, teacher = build_model(cfg, device="cpu")
    loss_fn = build_loss(cfg, device="cpu")
    optimizer = build_optimizer(student, cfg)

    # 2 global crops (224x224) and 6 local crops (96x96)
    crops = [
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
    ]

    # Teacher forwards 2 global crops
    with torch.no_grad():
        teacher_output = teacher(crops[:2])
    assert teacher_output.shape == (4, cfg["model"]["head"]["out_dim"])

    # Student forwards all 8 crops
    student_output = student(crops)
    total_samples = 2 * 2 + 6 * 2  # 16
    assert student_output.shape == (total_samples, cfg["model"]["head"]["out_dim"])

    # Loss computation & backward pass
    loss = loss_fn(student_output, teacher_output, epoch=0)
    assert not torch.isnan(loss)
    assert loss.item() > 0

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients flow into student parameters
    has_grad = any(p.grad is not None and (p.grad != 0).any() for p in student.parameters())
    assert has_grad
