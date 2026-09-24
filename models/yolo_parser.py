from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn as nn
from core.registry import BACKBONES, MODULES


def parse_model_layers(layers_cfg: List[list]) -> Tuple[nn.ModuleList, list]:
    """
    Parses a YOLO-style layer definition list:
    [[from, repeats, module_name, args], ...]

    from: index of layer output to pass as input (-1 = preceding layer)
    repeats: number of times to repeat block sequentially
    module_name: registered name in MODULES
    args: arguments list or dictionary
    """
    layers = []
    save = []

    for i, (f, n, m, args) in enumerate(layers_cfg):
        # Resolve class from registry if passed as string
        if isinstance(m, str):
            module_cls = MODULES.get(m)
        else:
            module_cls = m

        # Instantiate arguments
        if isinstance(args, dict):
            instantiate = lambda: module_cls(**args)
        elif isinstance(args, (list, tuple)):
            instantiate = lambda: module_cls(*args)
        else:
            instantiate = lambda: module_cls(args)

        if n == 1:
            module_instance = instantiate()
        else:
            module_instance = nn.Sequential(*[instantiate() for _ in range(n)])

        module_instance.i = i
        module_instance.f = f
        layers.append(module_instance)

        # Track layer outputs needed for non-sequential / branching layers
        if isinstance(f, int):
            if f != -1:
                save.append(i + 1 + f if f < 0 else f)
        elif isinstance(f, list):
            for x in f:
                if x != -1:
                    save.append(i + 1 + x if x < 0 else x)

    return nn.ModuleList(layers), sorted(list(set(save)))


@BACKBONES.register()
class YOLOModel(nn.Module):
    """
    Executes a DAG / sequence of layers defined in YOLO format.
    """

    def __init__(self, layers: List[list], embed_dim: int = None):
        super().__init__()
        self.layers_cfg = layers
        self.model_layers, self.save = parse_model_layers(layers)

        # Infer embed_dim if not explicitly provided
        if embed_dim is not None:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = self._infer_embed_dim()

    def _infer_embed_dim(self) -> int:
        for item in reversed(self.layers_cfg):
            if len(item) >= 4:
                m_name = item[2]
                if m_name in ("ExtractToken", "Concat"):
                    continue
                args = item[3]
                if isinstance(args, (list, tuple)) and len(args) > 0:
                    if m_name == "PatchEmbed" and len(args) >= 4 and isinstance(args[3], int):
                        return args[3]
                    if isinstance(args[0], int) and args[0] > 0:
                        return args[0]
                elif isinstance(args, dict):
                    for k in ("dim", "embed_dim", "in_features", "out_features"):
                        if k in args:
                            return args[k]
        return 384  # default fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = []
        for m in self.model_layers:
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                elif isinstance(m.f, list):
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x
