import pytest
import torch
import models  # Ensure all models are registered
import losses  # Ensure losses are registered
from core.builder import (
    build_loss,
    build_model,
    build_optimizer,
    build_schedulers,
)


def test_build_model_standard_config():
    cfg = {
        "model": {
            "backbone": {
                "type": "vit_tiny",
                "patch_size": 16,
            },
            "head": {
                "type": "DINOHead",
                "in_dim": 192,
                "out_dim": 1024,
                "use_bn": False,
            },
        }
    }
    student, teacher = build_model(cfg, device="cpu")
    assert student is not None and teacher is not None

    # Check teacher frozen
    for p in teacher.parameters():
        assert not p.requires_grad

    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    out = student(x)
    assert out.shape == (2, 1024)


def test_build_model_yolo_config():
    cfg = {
        "model": {
            "backbone": {
                "type": "YOLOModel",
                "layers": [
                    [-1, 1, "PatchEmbed", [224, 16, 3, 64]],
                    [-1, 1, "PosEmbedAndClassToken", [64, 196]],
                    [-1, 2, "Block", [64, 2, 2.0, True]],
                    [-1, 1, "LayerNorm1d", [64]],
                    [-1, 1, "ExtractToken", [0]],
                ],
            },
            "head": {
                "type": "DINOHead",
                "out_dim": 512,
                "use_bn": False,
            },
        }
    }
    student, teacher = build_model(cfg, device="cpu")
    x = torch.randn(2, 3, 224, 224)
    out = student(x)
    assert out.shape == (2, 512)


def test_build_loss_optimizer_and_schedulers():
    cfg = {
        "model": {
            "out_dim": 512,
        },
        "dataset": {
            "local_crops_number": 6,
        },
        "loss": {
            "type": "DINOLoss",
            "out_dim": 512,
            "ncrops": 8,
            "warmup_teacher_temp": 0.04,
            "teacher_temp": 0.04,
            "warmup_teacher_temp_epochs": 1,
            "nepochs": 5,
        },
        "train": {
            "epochs": 5,
            "batch_size_per_gpu": 32,
            "gradient_accumulation_steps": 4,
            "lr": 0.0005,
            "min_lr": 0.00001,
            "warmup_epochs": 1,
            "weight_decay": 0.04,
            "weight_decay_end": 0.4,
            "momentum_teacher": 0.996,
            "optimizer": {
                "type": "AdamW",
                "lr": 0.0005,
                "weight_decay": 0.04,
            },
        },
    }

    dummy_model = torch.nn.Linear(64, 512)
    loss_fn = build_loss(cfg, device="cpu")
    assert loss_fn is not None

    optimizer = build_optimizer(dummy_model, cfg)
    assert isinstance(optimizer, torch.optim.AdamW)

    lr_scheduler, momentum_schedule = build_schedulers(optimizer, cfg, niter_per_ep=10)
    assert len(momentum_schedule) == 5 * 10
