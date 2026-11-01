import os
import yaml
import pytest
import torch
import models
import losses
from core.builder import build_loss, build_model, build_optimizer


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/vit_small_modular.yaml",
        "configs/vit_small_200k.yaml",
        "configs/vit_small_sparse_rope.yaml",
    ],
)
def test_load_and_build_configs(config_path):
    assert os.path.exists(config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    student, teacher = build_model(cfg, device="cpu")
    loss_fn = build_loss(cfg, device="cpu")
    opt = build_optimizer(student, cfg)

    assert student is not None
    assert teacher is not None
    assert loss_fn is not None
    assert opt is not None

    # Test forward pass with single dummy image
    x = torch.randn(2, 3, 224, 224)
    out = student(x)
    assert out.shape[0] == 2
