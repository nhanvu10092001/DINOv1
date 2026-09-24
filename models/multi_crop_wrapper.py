from typing import List, Optional, Union
import torch
import torch.nn as nn
from core.registry import MODELS


@MODELS.register()
class MultiCropWrapper(nn.Module):
    """
    Convenience wrapper to forward multi-crop input images through the backbone and head.
    Groups crops by spatial resolution to batch process them efficiently.
    Also supports forwarding a single tensor for evaluation/inference.
    """

    def __init__(self, backbone: nn.Module, head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # Case 1: Single tensor input (evaluation or inference)
        if isinstance(x, torch.Tensor):
            out = self.backbone(x)
            if self.head is not None:
                out = self.head(out)
            return out

        # Case 2: List of crop tensors (multi-crop training)
        resolutions = torch.tensor([inp.shape[-1] for inp in x])
        counts = torch.unique_consecutive(resolutions, return_counts=True)[1]
        idx_crops = torch.cumsum(counts, dim=0)

        start_idx = 0
        output = None
        for end_idx in idx_crops:
            chunk = torch.cat(x[start_idx:end_idx], dim=0)
            _out = self.backbone(chunk)
            if output is None:
                output = _out
            else:
                output = torch.cat((output, _out), dim=0)
            start_idx = end_idx

        if self.head is not None:
            output = self.head(output)
        return output
