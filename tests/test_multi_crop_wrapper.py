import torch
import torch.nn as nn
from models.multi_crop_wrapper import MultiCropWrapper
from core.registry import MODELS


def test_multi_crop_wrapper_registered():
    assert "MultiCropWrapper" in MODELS


def test_multi_crop_wrapper_single_tensor():
    backbone = nn.Linear(10, 20)
    head = nn.Linear(20, 5)
    model = MultiCropWrapper(backbone, head)

    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 5)


def test_multi_crop_wrapper_multicrop_list():
    class DummyBackbone(nn.Module):
        def forward(self, x):
            # x is (B, C, H, W), returns (B, 16)
            return torch.zeros(x.shape[0], 16)

    head = nn.Linear(16, 8)
    model = MultiCropWrapper(DummyBackbone(), head)

    # 2 global crops (224x224) and 4 local crops (96x96)
    crops = [
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 224, 224),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
        torch.randn(2, 3, 96, 96),
    ]

    out = model(crops)
    # Total batch elements = 2 * 6 = 12
    assert out.shape == (12, 8)
