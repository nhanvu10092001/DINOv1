import copy
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from core.registry import BACKBONES, HEADS, LOSSES, MODELS, OPTIMIZERS, SCHEDULERS
from utils.schedulers import cosine_scheduler, get_params_groups, LRScheduler

# Register standard PyTorch optimizers
OPTIMIZERS.register("AdamW", force=True)(torch.optim.AdamW)
OPTIMIZERS.register("SGD", force=True)(torch.optim.SGD)
OPTIMIZERS.register("Adam", force=True)(torch.optim.Adam)


def build_backbone(backbone_cfg: Dict[str, Any]) -> nn.Module:
    """Builds the backbone network from Registry."""
    return BACKBONES.build(backbone_cfg)


def build_head(head_cfg: Dict[str, Any], in_dim: Optional[int] = None) -> nn.Module:
    """Builds the projection head, injecting in_dim if required."""
    cfg = copy.deepcopy(head_cfg)
    if in_dim is not None and "in_dim" not in cfg:
        cfg["in_dim"] = in_dim
    return HEADS.build(cfg)


def build_model(
    cfg: Dict[str, Any],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, nn.Module]:
    """
    Constructs Student and Teacher MultiCropWrapper models from configuration.
    Initializes Teacher with identical weights to Student and freezes parameters.
    """
    model_cfg = cfg.get("model", cfg)

    # 1. Build Student & Teacher Backbones
    if "backbone" in model_cfg:
        student_backbone = build_backbone(model_cfg["backbone"])
        teacher_backbone = build_backbone(model_cfg["backbone"])
    elif "arch" in model_cfg:
        arch = model_cfg["arch"]
        patch_size = model_cfg.get("patch_size", 16)
        student_backbone = BACKBONES.get(arch)(patch_size=patch_size)
        teacher_backbone = BACKBONES.get(arch)(patch_size=patch_size)
    else:
        raise ValueError("Config model section must contain 'backbone' or 'arch'")

    # Determine embed dimension
    embed_dim = getattr(student_backbone, "embed_dim", None)
    if embed_dim is None and hasattr(student_backbone, "_infer_embed_dim"):
        embed_dim = student_backbone._infer_embed_dim()

    # 2. Build Student & Teacher Heads
    if "head" in model_cfg:
        student_head = build_head(model_cfg["head"], in_dim=embed_dim)
        teacher_head = build_head(model_cfg["head"], in_dim=embed_dim)
    else:
        out_dim = model_cfg.get("out_dim", 16384)
        use_bn = model_cfg.get("use_bn_in_head", False)
        norm_last = model_cfg.get("norm_last_layer", True)
        student_head = HEADS.build({
            "type": "DINOHead",
            "in_dim": embed_dim,
            "out_dim": out_dim,
            "use_bn": use_bn,
            "norm_last_layer": norm_last,
        })
        teacher_head = HEADS.build({
            "type": "DINOHead",
            "in_dim": embed_dim,
            "out_dim": out_dim,
            "use_bn": use_bn,
            "norm_last_layer": True,
        })

    # 3. Assemble MultiCropWrapper
    MultiCropWrapper = MODELS.get("MultiCropWrapper")
    student = MultiCropWrapper(student_backbone, student_head).to(device)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head).to(device)

    # Initialize Teacher weights equal to Student and freeze
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher


def build_loss(
    cfg: Dict[str, Any],
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Builds DINOLoss from config."""
    if "loss" in cfg:
        loss_fn = LOSSES.build(cfg["loss"])
    else:
        loss_fn = LOSSES.build({
            "type": "DINOLoss",
            "out_dim": cfg["model"]["out_dim"],
            "ncrops": cfg["dataset"]["local_crops_number"] + 2,
            "warmup_teacher_temp": cfg["train"]["warmup_teacher_temp"],
            "teacher_temp": cfg["train"]["teacher_temp"],
            "warmup_teacher_temp_epochs": cfg["train"]["warmup_teacher_temp_epochs"],
            "nepochs": cfg["train"]["epochs"],
        })
    return loss_fn.to(device)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """Builds parameter groups and optimizer dynamically."""
    train_cfg = cfg["train"]
    opt_cfg = train_cfg.get("optimizer", {"type": "AdamW"})

    params_groups = get_params_groups(model)
    opt_type = opt_cfg.get("type", "AdamW")
    opt_cls = OPTIMIZERS.get(opt_type)

    kwargs = {k: v for k, v in opt_cfg.items() if k != "type"}
    if "lr" not in kwargs and "lr" in train_cfg:
        kwargs["lr"] = train_cfg["lr"]
    if "weight_decay" not in kwargs and "weight_decay" in train_cfg:
        kwargs["weight_decay"] = train_cfg["weight_decay"]

    return opt_cls(params_groups, **kwargs)


def build_schedulers(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    niter_per_ep: int,
):
    """Builds LR, Weight Decay, and Momentum cosine schedules."""
    train_cfg = cfg["train"]
    batch_size = train_cfg["batch_size_per_gpu"]
    grad_acc = train_cfg["gradient_accumulation_steps"]
    epochs = train_cfg["epochs"]

    effective_lr = train_cfg["lr"] * (batch_size * grad_acc) / 256.0
    lr_schedule = cosine_scheduler(
        effective_lr,
        train_cfg["min_lr"],
        epochs,
        niter_per_ep,
        warmup_epochs=train_cfg.get("warmup_epochs", 0),
    )
    wd_schedule = cosine_scheduler(
        train_cfg["weight_decay"],
        train_cfg["weight_decay_end"],
        epochs,
        niter_per_ep,
    )
    momentum_schedule = cosine_scheduler(
        train_cfg["momentum_teacher"],
        1.0,
        epochs,
        niter_per_ep,
    )
    lr_scheduler = LRScheduler(optimizer, lr_schedule, wd_schedule)
    return lr_scheduler, momentum_schedule
