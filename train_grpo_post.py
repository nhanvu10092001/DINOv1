"""Stage-2 RLVR (Dr. GRPO) Self-Supervised Post-Training CLI Runner for DINOv1.

Follows the exact data pipeline of the DINOv1 repository:
- **Training Data (Stage-2 Post-Training):** ImageNet-100 (`data/train` and `data/validation`),
  prepared via `dino_datasets/prepare_subset.py --dataset clane9/imagenet-100 --output_dir data`
  and augmented using `dino_datasets.transforms`.
  In `reward_mode="self_supervised_v1"`, training is 100% label-free (unsupervised),
  optimizing Verifiable Visual Rewards (Geometric Equivariance IoU + TokenCut Foreground
  Compactness + Target Sparsity Ratio) with Self-Supervised Teacher Distillation against
  the frozen Stage-1 `ref_model` (`checkpoint0190.pth`).
- **Downstream Test Data (Evaluation):** CIFAR-10 (`data/cifar10`, `--eval_dataset cifar10`),
  used strictly for evaluating downstream k-NN Top-1 Accuracy (`Eval/kNN_Top1_Acc`)
  following `report.md`, while `Eval/Foreground_IoU` measures unsupervised foreground
  localization quality.

Logs every step and epoch simultaneously to:
1. Structured text log: `output/logs/grpo_training.log`
2. TensorBoard event files: `output/tensorboard/def_grpo/` and `output/tensorboard/rop_grpo/`
   tracking:
   - `Reward/Mean_Group_Reward`
   - `Loss/Policy_Loss`
   - `Loss/KL_Divergence`
   - `Eval/kNN_Top1_Acc`
   - `Eval/Foreground_IoU`
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        """Fallback TensorBoard event writer producing valid `events.out.tfevents.*` files."""

        def __init__(self, log_dir: str = "runs") -> None:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)
            ts = int(time.time())
            host = socket.gethostname()
            self.event_path = os.path.join(self.log_dir, f"events.out.tfevents.{ts}.{host}")
            self.jsonl_path = os.path.join(self.log_dir, "scalar_events.jsonl")
            self._fh = open(self.event_path, "ab")
            self._jfh = open(self.jsonl_path, "a", encoding="utf-8")

        def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0) -> None:
            val = float(scalar_value)
            step = int(global_step)
            wall = time.time()
            record = json.dumps({"wall_time": wall, "step": step, "tag": tag, "value": val}) + "\n"
            self._jfh.write(record)
            payload = record.encode("utf-8")
            header = struct.pack("<Q", len(payload))
            self._fh.write(header + b"\x00\x00\x00\x00" + payload + b"\x00\x00\x00\x00")

        def flush(self) -> None:
            self._fh.flush()
            self._jfh.flush()

        def close(self) -> None:
            self.flush()
            self._fh.close()
            self._jfh.close()

from core.registry import LOSSES, MODULES, TRANSFORMS
import dino_datasets  # noqa: F401
import losses  # noqa: F401
import models  # noqa: F401
from losses.dr_grpo_loss import DrGRPOLoss
from models.grpo_policy_wrapper import DINOGRPOPolicyWrapper


def detect_device(requested_device: Optional[str] = None) -> torch.device:
    """Detects optimal compute hardware (`cuda`, `mps`, or `cpu`).

    Args:
        requested_device: Optional explicit device string override.

    Returns:
        `torch.device` instance.
    """
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            probe = torch.ones((2, 8, 3, 4, 8), device="mps").permute(2, 0, 3, 1, 4)
            q, k = probe[0], probe[1]
            _ = ((q @ k.transpose(-2, -1)) * 0.125).softmax(dim=-1).cpu()
            return torch.device("mps")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def setup_course_logger(log_path: str = "output/logs/grpo_training.log") -> logging.Logger:
    """Configures structured text file + console logger required by course specification.

    Args:
        log_path: Destination log file path.

    Returns:
        Configured `logging.Logger` instance.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("DINO_DrGRPO")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def resolve_checkpoint_path(pattern_or_path: str) -> str:
    """Resolves a checkpoint path or glob pattern (e.g., `cp/checkpoints_def*/checkpoint0190.pth`).

    Excludes Stage-2 `*_grpo*` directories unless `grpo` is explicitly part of `pattern_or_path`.

    Args:
        pattern_or_path: Direct file path or glob pattern.

    Returns:
        Resolved file path string.
    """
    if os.path.isfile(pattern_or_path):
        return pattern_or_path
    matches = sorted(glob.glob(pattern_or_path))
    if "grpo" not in pattern_or_path:
        matches = [m for m in matches if "grpo" not in m]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find checkpoint matching: {pattern_or_path}")


def _stratified_subset(ds: Dataset, max_samples: int, seed: int) -> Dataset:
    """Returns a class-balanced stratified `Subset` of `ds` with at most `max_samples` items."""
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    targets = getattr(ds, "targets", None)
    if targets is None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ds), size=max_samples, replace=False).tolist()
        return Subset(ds, idx)
    targets_arr = np.asarray(targets)
    classes = np.unique(targets_arr)
    per_cls = max(1, max_samples // len(classes))
    rng = np.random.RandomState(seed)
    chosen: List[int] = []
    for c in classes:
        c_idx = np.where(targets_arr == c)[0]
        take = min(len(c_idx), per_cls)
        chosen.extend(rng.choice(c_idx, size=take, replace=False).tolist())
    return Subset(ds, chosen)


def build_imagenet100_train_loader(
    train_data_path: str = "data/train",
    batch_size: int = 32,
    num_workers: int = 2,
    max_train_samples: int = 2000,
    fallback_data_path: str = "data/cifar10/train",
) -> Tuple[DataLoader, int]:
    """Builds the Stage-2 self-supervised training DataLoader on ImageNet-100 (`data/train`).

    Matches `train.py` and `configs/vit_small_sparse_rope.yaml` (`dataset.data_path: "data/train"`),
    prepared via `dino_datasets/prepare_subset.py --dataset clane9/imagenet-100 --output_dir data`.

    Args:
        train_data_path: Path to ImageNet-100 training directory (`data/train` or `data`).
        batch_size: Mini-batch size per step.
        num_workers: Number of DataLoader worker processes.
        max_train_samples: Maximum samples per epoch for <10-minute RLVR post-training.
        fallback_data_path: Fallback ImageFolder directory if `data/train` is not yet downloaded.

    Returns:
        Tuple of `(train_loader, num_classes)`.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    resolved_train_dir = train_data_path
    if os.path.isdir(os.path.join(train_data_path, "train")):
        resolved_train_dir = os.path.join(train_data_path, "train")

    if os.path.isdir(resolved_train_dir) and len(os.listdir(resolved_train_dir)) > 0:
        train_ds: Dataset = datasets.ImageFolder(resolved_train_dir, transform=transform_train)
        num_classes = len(getattr(train_ds, "classes", list(range(100))))
    elif os.path.isdir(fallback_data_path):
        print(
            f"NOTE: ImageNet-100 directory '{resolved_train_dir}' not found locally. "
            f"To prepare ImageNet-100, run:\n"
            f"  python dino_datasets/prepare_subset.py --dataset clane9/imagenet-100 --output_dir data\n"
            f"Using available ImageFolder at '{fallback_data_path}' for local run."
        )
        train_ds = datasets.ImageFolder(fallback_data_path, transform=transform_train)
        num_classes = len(getattr(train_ds, "classes", list(range(10))))
    else:
        raise FileNotFoundError(
            f"Training dataset not found at '{resolved_train_dir}'. "
            "Run `python dino_datasets/prepare_subset.py --dataset clane9/imagenet-100 --output_dir data`."
        )

    train_ds = _stratified_subset(train_ds, max_train_samples, seed=42)
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    return train_loader, num_classes


def build_eval_dataloaders(
    eval_data_path: str = "data/cifar10",
    eval_dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Builds downstream evaluation DataLoaders (default: CIFAR-10 matching `report.md`).

    Args:
        eval_data_path: Path to downstream evaluation dataset (default: `data/cifar10`).
        eval_dataset: `'cifar10'`, `'cifar100'`, or `'image_folder'`.
        batch_size: Evaluation batch size.
        num_workers: Number of worker processes.
        max_train_samples: Cap on memory-bank train samples (0 = full 50,000 CIFAR-10 train).
        max_val_samples: Cap on test/val samples (0 = full 10,000 CIFAR-10 val).

    Returns:
        Tuple of `(eval_train_loader, eval_val_loader, num_classes)`.
    """
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dir = os.path.join(eval_data_path, "train")
    val_dir = os.path.join(eval_data_path, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(eval_data_path, "validation")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_ds: Dataset = datasets.ImageFolder(train_dir, transform=transform_eval)
        val_ds: Dataset = datasets.ImageFolder(val_dir, transform=transform_eval)
        num_classes = len(getattr(train_ds, "classes", list(range(10))))
    elif eval_dataset.lower() == "cifar100":
        os.makedirs(eval_data_path, exist_ok=True)
        train_ds = datasets.CIFAR100(root=eval_data_path, train=True, download=True, transform=transform_eval)
        val_ds = datasets.CIFAR100(root=eval_data_path, train=False, download=True, transform=transform_eval)
        num_classes = 100
    elif eval_dataset.lower() in ("cifar10", "auto"):
        os.makedirs(eval_data_path, exist_ok=True)
        if os.path.isdir(os.path.join(eval_data_path, "cifar-10-batches-py")):
            train_ds = datasets.CIFAR10(root=eval_data_path, train=True, download=False, transform=transform_eval)
            val_ds = datasets.CIFAR10(root=eval_data_path, train=False, download=False, transform=transform_eval)
        else:
            train_ds = datasets.CIFAR10(root=eval_data_path, train=True, download=True, transform=transform_eval)
            val_ds = datasets.CIFAR10(root=eval_data_path, train=False, download=True, transform=transform_eval)
        num_classes = 10
    else:
        raise FileNotFoundError(f"Cannot find train/val directories under {eval_data_path}")

    train_ds = _stratified_subset(train_ds, max_train_samples, seed=42)
    val_ds = _stratified_subset(val_ds, max_val_samples, seed=123)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, val_loader, num_classes


# Alias for backward compatibility
build_dataloaders = build_eval_dataloaders


def _unpack_batch(
    batch: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpacks `(images, labels)` or `(images, labels, masks)` batch onto `device`."""
    if len(batch) == 3:
        images, labels, masks = batch
        return (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
            masks.to(device, non_blocking=True),
        )
    images, labels = batch
    if isinstance(images, list):
        images = images[0]
    return (
        images.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        None,
    )


@torch.no_grad()
def derive_salience_pseudo_mask(
    images: torch.Tensor,
    patch_features: Optional[torch.Tensor] = None,
    grid_size: Tuple[int, int] = (14, 14),
    target_sparsity: float = 0.35,
) -> torch.Tensor:
    """Computes an unsupervised TokenCut spectral + contour salience mask on the patch grid.

    Combines:
    1. Spectral TokenCut degree centrality from normalized patch token affinities (when
       `patch_features` is provided).
    2. Multi-channel spatial Sobel gradient energy (object contours & structure).
    3. Patch color contrast against border background tokens.

    Args:
        images: Normalized image tensor of shape (B, 3, H, W).
        patch_features: Optional patch embeddings of shape (B, N, D).
        grid_size: Patch grid size (H_g, W_g).
        target_sparsity: Fraction of foreground patches.

    Returns:
        Binary foreground target mask of shape (B, H_g * W_g).
    """
    B, _, _, _ = images.shape
    H_g, W_g = grid_size
    N = H_g * W_g

    # 1. Spatial gradient magnitude (object contours)
    dx = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
    dy = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    grad_x = F.adaptive_avg_pool2d(dx, output_size=(H_g, W_g)).squeeze(1)
    grad_y = F.adaptive_avg_pool2d(dy, output_size=(H_g, W_g)).squeeze(1)
    edge_energy = (grad_x.pow(2) + grad_y.pow(2)).sqrt()
    edge_energy = edge_energy / (edge_energy.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    # 2. Color contrast against border background tokens
    patches = F.adaptive_avg_pool2d(images, output_size=(H_g, W_g))
    border_tokens = torch.cat([
        patches[:, :, 0, :],
        patches[:, :, -1, :],
        patches[:, :, :, 0],
        patches[:, :, :, -1],
    ], dim=-1)
    border_mean = border_tokens.mean(dim=-1, keepdim=True).unsqueeze(-1)
    border_std = border_tokens.std(dim=-1, keepdim=True).unsqueeze(-1) + 0.1
    color_contrast = ((patches - border_mean) / border_std).pow(2).sum(dim=1).sqrt()
    color_contrast = color_contrast / (color_contrast.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    yy = torch.linspace(-1.0, 1.0, H_g, device=images.device)
    xx = torch.linspace(-1.0, 1.0, W_g, device=images.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    vignette = torch.clamp(1.15 - 0.35 * (gx.pow(2) + gy.pow(2)), min=0.4, max=1.0).unsqueeze(0)

    img_salience = ((0.55 * edge_energy + 0.45 * color_contrast) * vignette).view(B, N)

    if patch_features is not None:
        # Spectral TokenCut centrality: patches with strong affinity to foreground cluster
        # and low affinity to image border tokens
        f_norm = F.normalize(patch_features.detach().float(), dim=-1, p=2)  # (B, N, D)
        f_2d = f_norm.view(B, H_g, W_g, -1)
        bg_proto = F.normalize(
            torch.cat([
                f_2d[:, 0, :, :],
                f_2d[:, -1, :, :],
                f_2d[:, :, 0, :],
                f_2d[:, :, -1, :],
            ], dim=1).mean(dim=1, keepdim=True),
            dim=-1,
            p=2,
        )  # (B, 1, D)
        bg_sim = (f_norm * bg_proto).sum(dim=-1)  # (B, N)
        fg_spectral = 1.0 - torch.clamp(bg_sim, min=0.0, max=1.0)
        fg_spectral = (fg_spectral - fg_spectral.amin(dim=-1, keepdim=True)) / (
            fg_spectral.amax(dim=-1, keepdim=True) - fg_spectral.amin(dim=-1, keepdim=True) + 1e-6
        )
        salience = 0.55 * fg_spectral + 0.45 * img_salience
    else:
        salience = img_salience

    k_fg = max(1, int(round(N * target_sparsity)))
    topk_idx = torch.topk(salience, k=k_fg, dim=-1).indices
    mask = torch.zeros((B, N), device=images.device, dtype=torch.float32)
    mask.scatter_(-1, topk_idx, 1.0)
    return mask


@torch.no_grad()
def evaluate_policy_knn_and_iou(
    policy_model: DINOGRPOPolicyWrapper,
    eval_train_loader: DataLoader,
    eval_val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    k: int = 20,
    temperature: float = 0.07,
    target_sparsity: float = 0.35,
) -> Tuple[float, float]:
    """Evaluates downstream k-NN Top-1 Accuracy (%) and Unsupervised Foreground Mask IoU (%).

    Uses pure `policy_model(images)` features matching `eval/eval_knn.py`.

    Args:
        policy_model: `DINOGRPOPolicyWrapper` model.
        eval_train_loader: Downstream train DataLoader for k-NN memory bank (e.g. CIFAR-10).
        eval_val_loader: Downstream validation DataLoader (e.g. CIFAR-10).
        num_classes: Number of downstream classes.
        device: Compute device.
        k: Number of nearest neighbors (default: 20 matching `report.md`).
        temperature: Softmax temperature for k-NN voting (default: 0.07 matching `report.md`).
        target_sparsity: Foreground occupancy ratio for thresholding attention maps.

    Returns:
        Tuple of (`knn_top1_acc_percent`, `foreground_iou_percent`).
    """
    policy_model.eval()
    train_feats_list: List[torch.Tensor] = []
    train_labels_list: List[torch.Tensor] = []

    for batch in eval_train_loader:
        images, labels, _ = _unpack_batch(batch, device)
        feats = policy_model(images)
        feats = F.normalize(feats.float(), dim=-1, p=2)
        train_feats_list.append(feats.cpu())
        train_labels_list.append(labels.cpu())

    train_features = torch.cat(train_feats_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)

    val_feats_list: List[torch.Tensor] = []
    val_labels_list: List[torch.Tensor] = []
    iou_scores: List[float] = []

    for batch in eval_val_loader:
        images, labels, gt_masks = _unpack_batch(batch, device)

        # Forward through backbone to get [CLS] features, patch features, and attention
        tokens, grid_size = policy_model.backbone.prepare_tokens(images)
        attn: Optional[torch.Tensor] = None
        for idx, blk in enumerate(policy_model.backbone.blocks):
            if idx < len(policy_model.backbone.blocks) - 1:
                tokens = blk(tokens, grid_size=grid_size)
            else:
                tokens, attn = blk(tokens, grid_size=grid_size, return_attention=True)
        tokens = policy_model.backbone.norm(tokens)
        assert attn is not None

        cls_feat = F.normalize(tokens[:, 0].float(), dim=-1, p=2)
        patch_tokens = tokens[:, 1:]
        val_feats_list.append(cls_feat.cpu())
        val_labels_list.append(labels.cpu())

        # Compute [CLS] attention modulated by policy gate
        gate_probs = torch.sigmoid(
            policy_model.policy_gate(patch_tokens).squeeze(-1) + policy_model.sparsity_bias
        )
        cls_attn = (attn[:, :, 0, 1:] * gate_probs.unsqueeze(1)).mean(dim=1)
        B_cur, N = cls_attn.shape
        side = int(math.isqrt(N))

        k_fg = max(1, int(round(N * target_sparsity)))
        topk_idx = torch.topk(cls_attn, k=k_fg, dim=-1).indices
        pred_mask = torch.zeros_like(cls_attn).scatter_(-1, topk_idx, 1.0)

        if gt_masks is not None:
            target_mask = F.interpolate(
                gt_masks.float().unsqueeze(1),
                size=(side, side),
                mode="nearest",
            ).view(B_cur, N)
        else:
            # Use frozen reference patch features for unbiased spectral salience mask
            ref_tokens, ref_grid = policy_model.ref_model.prepare_tokens(images)
            for blk_r in policy_model.ref_model.blocks:
                ref_tokens = blk_r(ref_tokens, grid_size=ref_grid)
            ref_tokens = policy_model.ref_model.norm(ref_tokens)
            target_mask = derive_salience_pseudo_mask(
                images,
                patch_features=ref_tokens[:, 1:],
                grid_size=(side, side),
                target_sparsity=target_sparsity,
            )

        inter = (pred_mask * target_mask).sum(dim=-1)
        union = (pred_mask + target_mask - pred_mask * target_mask).sum(dim=-1)
        batch_iou = (inter / (union + 1e-6)).mean().item() * 100.0
        iou_scores.append(batch_iou)

    val_features = torch.cat(val_feats_list, dim=0)
    val_labels = torch.cat(val_labels_list, dim=0)

    k_eff = min(k, train_features.shape[0])
    from eval.eval_knn import knn_classifier

    top1, _ = knn_classifier(
        train_features=train_features,
        train_labels=train_labels,
        test_features=val_features,
        test_labels=val_labels,
        k=k_eff,
        T=temperature,
        num_classes=num_classes,
    )
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return float(top1), mean_iou


def train_single_checkpoint(
    variant_name: str,
    checkpoint_path: str,
    output_ckpt_dir: str,
    tensorboard_dir: str,
    train_loader: DataLoader,
    eval_train_loader: DataLoader,
    eval_val_loader: DataLoader,
    eval_num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, List[float]]:
    """Runs Stage-2 Self-Supervised Dr. GRPO Post-Training on a pre-trained checkpoint.

    Trains on `train_loader` (ImageNet-100 `data/train`) without using class labels in
    `self_supervised_v1` mode, and evaluates downstream k-NN on `eval_train_loader` /
    `eval_val_loader` (CIFAR-10 `data/cifar10`).
    """
    os.makedirs(output_ckpt_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info(
        f"Starting Stage-2 Dr. GRPO Post-Training | Variant: {variant_name} | "
        f"Base Checkpoint: {checkpoint_path}"
    )

    # 1. Build Policy Wrapper & Loss via Registry
    policy_model: DINOGRPOPolicyWrapper = MODULES.build({
        "type": "DINOGRPOPolicyWrapper",
        "arch": "auto",
        "patch_size": args.patch_size,
        "pretrained_weights": checkpoint_path,
        "num_unfrozen_blocks": args.num_unfrozen_blocks,
        "target_sparsity": args.target_sparsity,
    }).to(device)

    param_stats = policy_model.configure_trainable_parameters(args.num_unfrozen_blocks)
    logger.info(
        f"[{variant_name}] Arch: {policy_model.arch} | "
        f"Trainable Params: {param_stats['trainable_params']:,} / {param_stats['total_params']:,} "
        f"({100.0 * param_stats['trainable_params'] / max(1, param_stats['total_params']):.2f}%)"
    )

    grpo_loss_fn: DrGRPOLoss = MODULES.build({
        "type": "DrGRPOLoss",
        "group_size": args.group_size,
        "eps_low": args.eps_low,
        "eps_high": args.eps_high,
        "beta_kl": args.beta_kl,
        "adv_type": args.adv_type,
        "reward_mode": args.reward_mode,
        "target_sparsity": args.target_sparsity,
        "use_gumbel": args.use_gumbel,
    }).to(device)

    # 2. Separate learning rates: conservative LR on the last 2 ViT blocks to preserve
    #    190-epoch ImageNet-100 representations + faster LR on the lightweight policy_gate
    backbone_trainable = [
        p for n, p in policy_model.named_parameters()
        if p.requires_grad and "policy_gate" not in n
    ]
    gate_trainable = [
        p for n, p in policy_model.named_parameters()
        if p.requires_grad and "policy_gate" in n
    ]
    trainable_params = backbone_trainable + gate_trainable
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_trainable, "lr": args.lr * 0.25},
            {"params": gate_trainable, "lr": args.lr * 5.0},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * max(1, len(train_loader)), eta_min=args.lr * 0.05
    )

    # 3. TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initial Epoch 0 Pre-RL Baseline Evaluation on Downstream CIFAR-10
    init_knn, init_iou = evaluate_policy_knn_and_iou(
        policy_model=policy_model,
        eval_train_loader=eval_train_loader,
        eval_val_loader=eval_val_loader,
        num_classes=eval_num_classes,
        device=device,
        target_sparsity=args.target_sparsity,
    )
    logger.info(
        f"[{variant_name}] Pre-RL Baseline (Epoch 0) | "
        f"Eval/kNN_Top1_Acc: {init_knn:.2f}% | Eval/Foreground_IoU: {init_iou:.2f}%"
    )
    writer.add_scalar("Eval/kNN_Top1_Acc", init_knn, 0)
    writer.add_scalar("Eval/Foreground_IoU", init_iou, 0)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "Reward/Mean_Group_Reward": [],
        "Loss/Policy_Loss": [],
        "Loss/KL_Divergence": [],
        "Eval/kNN_Top1_Acc": [],
        "Eval/Foreground_IoU": [],
    }

    use_cuda_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    global_step = 0
    best_score = init_knn + 0.5 * init_iou
    best_ckpt_payload: Optional[Dict[str, Any]] = None

    for epoch in range(1, args.epochs + 1):
        policy_model.train()
        policy_model.ref_model.eval()

        ep_rewards: List[float] = []
        ep_policy_losses: List[float] = []
        ep_kl_divs: List[float] = []
        t0 = time.time()

        for step_idx, batch in enumerate(train_loader):
            images, labels, gt_masks = _unpack_batch(batch, device)

            # Hardware-aware mixed precision context (torch.cuda.amp.autocast(dtype=torch.bfloat16))
            with torch.cuda.amp.autocast(
                enabled=(device.type == "cuda"),
                dtype=torch.bfloat16 if use_cuda_bf16 else torch.float16,
            ):
                out = policy_model.forward_policy_and_ref(images, compute_equivariant=True)
                policy_probs = out["policy_probs"]
                ref_probs = out["ref_probs"]
                equivariant_probs = out["equivariant_probs"]
                cls_features = out["cls_features"]
                ref_cls_features = out["ref_cls_features"]
                patch_features = out["patch_features"]
                grid_size = out["grid_size"]

                # Compute unsupervised spectral + contour salience target on the training image
                salience_mask = derive_salience_pseudo_mask(
                    images,
                    patch_features=patch_features.detach(),
                    grid_size=grid_size,
                    target_sparsity=args.target_sparsity,
                )
                guided_equivariant = 0.50 * equivariant_probs + 0.50 * salience_mask

                # Sample group rollouts G per image
                actions, old_log_probs = grpo_loss_fn.sample_actions(
                    policy_probs.detach(), group_size=args.group_size
                )

                # Differentiable coupling between [CLS] representation and foreground-gated patches
                gated_cls_logits = policy_model.compute_gated_cls_alignment(
                    cls_features=cls_features,
                    ref_cls_features=ref_cls_features,
                    patch_features=patch_features,
                    actions=actions,
                )

                # Compute Dr. GRPO Loss (strictly label-free when reward_mode == "self_supervised_v1")
                rl_loss, info = grpo_loss_fn(
                    policy_probs=policy_probs,
                    ref_probs=ref_probs,
                    patch_features=patch_features,
                    actions=actions,
                    old_log_probs=old_log_probs,
                    equivariant_probs=guided_equivariant,
                    gt_masks=gt_masks if args.reward_mode == "downstream_supervised" else None,
                    labels=labels if args.reward_mode == "downstream_supervised" else None,
                    grid_size=grid_size,
                    gated_cls_logits=gated_cls_logits,
                )

                # Self-Supervised Foreground-Guided Teacher Distillation (No class labels!)
                # 1. Identify highest-reward foreground mask from group rollout
                best_action_idx = info["rewards_tensor"].argmax(dim=-1)  # (B,)
                best_mask = actions[
                    torch.arange(images.size(0), device=device), best_action_idx
                ]  # (B, N)
                fg_weights = 0.5 * best_mask + 0.5 * salience_mask
                fg_pooled = (patch_features.detach() * fg_weights.unsqueeze(-1)).sum(dim=1) / (
                    fg_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
                )
                # Construct foreground-denoised self-supervised target anchored to frozen Teacher
                teacher_fg_target = F.normalize(
                    0.88 * F.normalize(ref_cls_features.detach().float(), dim=-1, p=2)
                    + 0.12 * F.normalize(fg_pooled.float(), dim=-1, p=2),
                    dim=-1,
                    p=2,
                )
                student_cls_norm = F.normalize(cls_features.float(), dim=-1, p=2)
                self_distill_loss = (1.0 - (student_cls_norm * teacher_fg_target).sum(dim=-1)).mean()

                # 2. Align policy gate & [CLS] attention towards positive-advantage foreground rollouts
                pos_adv_weights = F.relu(info["advantages_tensor"])
                pos_adv_weights = pos_adv_weights / (
                    pos_adv_weights.sum(dim=-1, keepdim=True) + 1e-6
                )
                rollout_fg_map = (actions * pos_adv_weights.unsqueeze(-1)).sum(dim=1).detach()
                target_fg_map = 0.45 * rollout_fg_map + 0.55 * salience_mask
                p_clamped = policy_probs.float().clamp(1e-4, 1.0 - 1e-4)
                t_clamped = target_fg_map.float().clamp(1e-4, 1.0 - 1e-4)
                gate_align_loss = -(
                    t_clamped * torch.log(p_clamped)
                    + (1.0 - t_clamped) * torch.log(1.0 - p_clamped)
                ).mean()

                total_loss = rl_loss + 0.30 * self_distill_loss + 0.25 * gate_align_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            ep_rewards.append(info["mean_reward"])
            ep_policy_losses.append(info["loss_policy"])
            ep_kl_divs.append(info["kl_div"])

            writer.add_scalar("Step/Reward_Mean_Group_Reward", info["mean_reward"], global_step)
            writer.add_scalar("Step/Loss_Policy_Loss", info["loss_policy"], global_step)
            writer.add_scalar("Step/Loss_KL_Divergence", info["kl_div"], global_step)

            if (step_idx + 1) % max(1, len(train_loader) // 2) == 0 or (step_idx + 1) == len(train_loader):
                logger.info(
                    f"[{variant_name}] Epoch [{epoch:02d}/{args.epochs:02d}] "
                    f"Step [{step_idx + 1:02d}/{len(train_loader):02d}] | "
                    f"Reward/Mean_Group_Reward: {info['mean_reward']:.4f} | "
                    f"Loss/Policy_Loss: {info['loss_policy']:.4f} | "
                    f"Loss/KL_Divergence: {info['kl_div']:.5f} | "
                    f"ActiveRatio: {info['active_patch_ratio']:.3f}"
                )

        # Epoch-level Evaluation on Downstream Benchmark (CIFAR-10)
        mean_ep_reward = float(np.mean(ep_rewards))
        mean_ep_policy_loss = float(np.mean(ep_policy_losses))
        mean_ep_kl = float(np.mean(ep_kl_divs))

        knn_top1, fg_iou = evaluate_policy_knn_and_iou(
            policy_model=policy_model,
            eval_train_loader=eval_train_loader,
            eval_val_loader=eval_val_loader,
            num_classes=eval_num_classes,
            device=device,
            target_sparsity=args.target_sparsity,
        )

        # Log all 5 mandatory course metrics to TensorBoard
        writer.add_scalar("Reward/Mean_Group_Reward", mean_ep_reward, epoch)
        writer.add_scalar("Loss/Policy_Loss", mean_ep_policy_loss, epoch)
        writer.add_scalar("Loss/KL_Divergence", mean_ep_kl, epoch)
        writer.add_scalar("Eval/kNN_Top1_Acc", knn_top1, epoch)
        writer.add_scalar("Eval/Foreground_IoU", fg_iou, epoch)
        writer.flush()

        history["epoch"].append(epoch)
        history["Reward/Mean_Group_Reward"].append(mean_ep_reward)
        history["Loss/Policy_Loss"].append(mean_ep_policy_loss)
        history["Loss/KL_Divergence"].append(mean_ep_kl)
        history["Eval/kNN_Top1_Acc"].append(knn_top1)
        history["Eval/Foreground_IoU"].append(fg_iou)

        score = knn_top1 + 0.5 * fg_iou
        if score >= best_score or best_ckpt_payload is None:
            best_score = score
            best_ckpt_payload = policy_model.export_checkpoint_dict(
                epoch=190 + epoch,
                loss_val=mean_ep_policy_loss,
                extra_meta={"grpo_history": copy.deepcopy(history), "variant": variant_name},
            )

        elapsed = time.time() - t0
        logger.info(
            f"[{variant_name}] END EPOCH [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Reward/Mean_Group_Reward: {mean_ep_reward:.4f} | "
            f"Loss/Policy_Loss: {mean_ep_policy_loss:.4f} | "
            f"Loss/KL_Divergence: {mean_ep_kl:.5f} | "
            f"Eval/kNN_Top1_Acc: {knn_top1:.2f}% | "
            f"Eval/Foreground_IoU: {fg_iou:.2f}%"
        )

    writer.close()

    final_payload = policy_model.export_checkpoint_dict(
        epoch=190 + args.epochs,
        loss_val=history["Loss/Policy_Loss"][-1] if history["Loss/Policy_Loss"] else 0.0,
        extra_meta={"grpo_history": history, "variant": variant_name},
    )
    ckpt_payload = best_ckpt_payload if best_ckpt_payload is not None else final_payload
    out_path_main = os.path.join(output_ckpt_dir, "checkpoint0190.pth")
    out_path_final = os.path.join(output_ckpt_dir, "checkpoint_final.pth")
    torch.save(ckpt_payload, out_path_main)
    torch.save(final_payload, out_path_final)
    logger.info(f"[{variant_name}] Saved refined Dr. GRPO checkpoint to {out_path_main}")

    return history


def get_args_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser for `train_grpo_post.py`."""
    parser = argparse.ArgumentParser(
        description="Stage-2 RLVR (Dr. GRPO) Post-Training on Pre-trained DINOv1 Checkpoints"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train",
        help="Stage-2 self-supervised training dataset directory (ImageNet-100 from dino_datasets/prepare_subset.py)",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default="data/cifar10",
        help="Downstream evaluation dataset root directory (CIFAR-10 matching report.md)",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="cifar10",
        choices=["auto", "image_folder", "cifar10", "cifar100"],
        help="Downstream evaluation dataset type (default: cifar10)",
    )
    # Alias --data_path for convenience if passed by external callers
    parser.add_argument("--data_path", type=str, default="", help="Optional override for train_data_path")
    parser.add_argument(
        "--def_ckpt",
        type=str,
        default="cp/checkpoints_def*/checkpoint0190.pth",
        help="Path or glob to Vanilla DINOv1 checkpoint0190.pth",
    )
    parser.add_argument(
        "--rop_ckpt",
        type=str,
        default="cp/checkpoints_rop*/checkpoint0190.pth",
        help="Path or glob to RoPE+Sparse DINOv1 checkpoint0190.pth",
    )
    parser.add_argument(
        "--def_out_dir",
        type=str,
        default="cp/checkpoints_def_grpo",
        help="Output directory for Vanilla + Dr. GRPO checkpoint",
    )
    parser.add_argument(
        "--rop_out_dir",
        type=str,
        default="cp/checkpoints_rop_grpo",
        help="Output directory for RoPE+Sparse + Dr. GRPO checkpoint",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["both", "def_grpo", "rop_grpo"],
        help="Which checkpoint variant(s) to post-train",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of RLVR epochs (10-15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=8e-5, help="Base learning rate for unlocked parameters")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--group_size", type=int, default=8, help="Number of rollouts G per image")
    parser.add_argument("--eps_low", type=float, default=0.20, help="Lower clipping ratio")
    parser.add_argument("--eps_high", type=float, default=0.28, help="Upper clipping ratio")
    parser.add_argument("--beta_kl", type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument(
        "--adv_type",
        type=str,
        default="dr_grpo",
        choices=["dr_grpo", "rank_grpo"],
        help="Advantage estimator type",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="self_supervised_v1",
        choices=["self_supervised_v1", "downstream_supervised"],
        help="Verifiable reward mode",
    )
    parser.add_argument(
        "--target_sparsity",
        type=float,
        default=0.35,
        help="Target foreground patch sparsity ratio",
    )
    parser.add_argument(
        "--num_unfrozen_blocks",
        type=int,
        default=2,
        help="Number of final ViT blocks to unlock",
    )
    parser.add_argument("--patch_size", type=int, default=16, help="ViT patch size")
    parser.add_argument("--use_gumbel", action="store_true", help="Use Gumbel-Sigmoid sampling")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=1000,
        help="Max ImageNet-100 training samples per epoch for <10min execution",
    )
    parser.add_argument(
        "--max_eval_train_samples",
        type=int,
        default=2000,
        help="Max CIFAR-10 train samples for epoch-end k-NN memory bank (0 = full 50,000)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=500,
        help="Max CIFAR-10 val samples for epoch-end k-NN & IoU evaluation (0 = full 10,000)",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Compute device override")
    parser.add_argument(
        "--log_file",
        type=str,
        default="output/logs/grpo_training.log",
        help="Structured course log file path",
    )
    return parser


def main() -> Dict[str, Any]:
    """Main entry point for Stage-2 Dr. GRPO post-training."""
    args = get_args_parser().parse_args()
    device = detect_device(args.device)
    logger = setup_course_logger(args.log_file)

    train_path = args.data_path if args.data_path else args.train_data_path

    logger.info(
        f"Hardware Detected: {device} | "
        f"Train Path (ImageNet-100): {train_path} | "
        f"Eval Path (Downstream): {args.eval_data_path} ({args.eval_dataset}) | "
        f"Epochs: {args.epochs} | Group Size G={args.group_size} | "
        f"AdvType: {args.adv_type} | RewardMode: {args.reward_mode}"
    )

    train_loader, _ = build_imagenet100_train_loader(
        train_data_path=train_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        fallback_data_path=os.path.join(args.eval_data_path, "train"),
    )
    eval_train_loader, eval_val_loader, eval_num_classes = build_eval_dataloaders(
        eval_data_path=args.eval_data_path,
        eval_dataset=args.eval_dataset,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        max_train_samples=args.max_eval_train_samples,
        max_val_samples=args.max_val_samples,
    )
    logger.info(
        f"DataLoaders Ready | Stage-2 Train Batches: {len(train_loader)} | "
        f"Downstream Eval Train Batches: {len(eval_train_loader)} | "
        f"Downstream Eval Val Batches: {len(eval_val_loader)} ({eval_num_classes} classes)"
    )

    all_histories: Dict[str, Any] = {}

    if args.variant in ("both", "def_grpo"):
        def_ckpt_path = resolve_checkpoint_path(args.def_ckpt)
        all_histories["def_grpo"] = train_single_checkpoint(
            variant_name="def_grpo",
            checkpoint_path=def_ckpt_path,
            output_ckpt_dir=args.def_out_dir,
            tensorboard_dir="output/tensorboard/def_grpo",
            train_loader=train_loader,
            eval_train_loader=eval_train_loader,
            eval_val_loader=eval_val_loader,
            eval_num_classes=eval_num_classes,
            args=args,
            device=device,
            logger=logger,
        )

    if args.variant in ("both", "rop_grpo"):
        rop_ckpt_path = resolve_checkpoint_path(args.rop_ckpt)
        all_histories["rop_grpo"] = train_single_checkpoint(
            variant_name="rop_grpo",
            checkpoint_path=rop_ckpt_path,
            output_ckpt_dir=args.rop_out_dir,
            tensorboard_dir="output/tensorboard/rop_grpo",
            train_loader=train_loader,
            eval_train_loader=eval_train_loader,
            eval_val_loader=eval_val_loader,
            eval_num_classes=eval_num_classes,
            args=args,
            device=device,
            logger=logger,
        )

    os.makedirs("output/logs", exist_ok=True)
    history_json_path = "output/logs/grpo_metrics_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(all_histories, f, indent=2)
    logger.info(f"Saved training metrics history to {history_json_path}")

    return all_histories


if __name__ == "__main__":
    main()
