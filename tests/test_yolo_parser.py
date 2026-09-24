import pytest
import torch
from models.yolo_parser import parse_model_layers, YOLOModel
from core.registry import BACKBONES


def test_yolo_parser_linear_chain():
    layer_configs = [
        [-1, 1, "PatchEmbed", [224, 16, 3, 64]],
        [-1, 1, "PosEmbedAndClassToken", [64, 196]],
        [-1, 2, "Block", [64, 2, 2.0, True]],
        [-1, 1, "LayerNorm1d", [64]],
        [-1, 1, "ExtractToken", [0]],
    ]
    model = YOLOModel(layer_configs)
    assert model.embed_dim == 64

    # 1. Test 224x224 input
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 64)

    # 2. Test 96x96 local crop input (tests interpolation in PosEmbedAndClassToken)
    x_local = torch.randn(2, 3, 96, 96)
    out_local = model(x_local)
    assert out_local.shape == (2, 64)


def test_yolo_model_registered_in_backbones():
    assert "YOLOModel" in BACKBONES
    cfg = {
        "type": "YOLOModel",
        "layers": [
            [-1, 1, "PatchEmbed", [224, 16, 3, 32]],
            [-1, 1, "PosEmbedAndClassToken", [32, 196]],
            [-1, 1, "Block", [32, 1, 2.0, True]],
            [-1, 1, "LayerNorm1d", [32]],
            [-1, 1, "ExtractToken", [0]],
        ],
    }
    model = BACKBONES.build(cfg)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 32)
