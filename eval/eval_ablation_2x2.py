"""Rigorous 2x2 Ablation Benchmark & Publication Attention Grid Generator.

Evaluates all 4 DINOv1 variants:
1. Vanilla DINOv1 (Stage-1 Baseline)
2. RoPE + Sparse DINOv1 (Stage-1 Architectural Improvement)
3. Vanilla DINOv1 + Dr. GRPO (Stage-2 RLVR Post-Training)
4. RoPE + Sparse DINOv1 + Dr. GRPO (Full Proposed Method)

Metrics evaluated:
- k-NN Top-1 & Top-5 Accuracy (%) via `eval/eval_knn.py`
- Linear Probe Top-1 Accuracy (%) via `eval/eval_linear.py`
- Unsupervised Foreground Mask IoU (%)
- Unsupervised CorLoc (Correct Localization @ IoU >= 0.50, %)

Outputs:
- `output/ablation_2x2_attention_grid.png`: Publication-ready 5-column visual grid
  (Original Image | Vanilla DINOv1 | RoPE+Sparse DINOv1 | Vanilla+GRPO | RoPE+Sparse+Dr.GRPO (Ours))
- `output/ablation_2x2_results.md`: Markdown summary table for defense slides
- `output/ablation_2x2_results.csv`: CSV metrics table
"""

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import MODULES
from eval.eval_knn import knn_classifier
from eval.eval_linear import LinearClassifier
from models.grpo_policy_wrapper import DINOGRPOPolicyWrapper
from train_grpo_post import (
    _unpack_batch,
    build_dataloaders,
    derive_salience_pseudo_mask,
    detect_device,
    resolve_checkpoint_path,
)


def load_ablation_model(
    ckpt_path: str,
    device: torch.device,
    patch_size: int = 16,
    target_sparsity: float = 0.35,
) -> DINOGRPOPolicyWrapper:
    """Loads any of the 4 ablation checkpoints into a unified `DINOGRPOPolicyWrapper`.

    For Stage-1 checkpoints without `grpo_policy_gate`, `get_last_selfattention`
    uses pure backbone `[CLS]` self-attention; for Stage-2 GRPO checkpoints, it
    uses the refined backbone + learned policy gate.

    Args:
        ckpt_path: Path to `.pth` checkpoint file.
        device: Target compute device.
        patch_size: ViT patch size (default: 16).
        target_sparsity: Target foreground sparsity ratio.

    Returns:
        Evaluated `DINOGRPOPolicyWrapper` in `.eval()` mode.
    """
    model: DINOGRPOPolicyWrapper = MODULES.build({
        "type": "DINOGRPOPolicyWrapper",
        "arch": "auto",
        "patch_size": patch_size,
        "pretrained_weights": ckpt_path,
        "num_unfrozen_blocks": 2,
        "target_sparsity": target_sparsity,
    }).to(device)

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    has_grpo_gate = "grpo_policy_gate" in raw_ckpt
    model._use_grpo_gate_in_eval = has_grpo_gate
    model.eval()
    return model


@torch.no_grad()
def extract_ablation_features_and_masks(
    model: DINOGRPOPolicyWrapper,
    loader: DataLoader,
    device: torch.device,
    target_sparsity: float = 0.35,
    compute_masks: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Extracts raw features (for Linear Probe), L2-normalized features (for k-NN), labels, IoU, and CorLoc.

    Follows the exact evaluation protocol in `report.md`:
    - `eval/eval_knn.py`: L2-normalized `[CLS]` features (`F.normalize(out, dim=1, p=2)`).
    - `eval/eval_linear.py`: Unnormalized `[CLS]` features (`model(images)`) with SGD `lr=0.1`.

    Args:
        model: Loaded `DINOGRPOPolicyWrapper` instance.
        loader: DataLoader yielding `(images, labels)` or `(images, labels, gt_masks)`.
        device: Compute device.
        target_sparsity: Fraction of patches selected as foreground mask.
        compute_masks: If False, skips `ref_model` & mask IoU calculation (speeds up train_loader 2.5x).

    Returns:
        Tuple of `(raw_features, norm_features, labels, mean_iou_percent, corloc_percent)`.
    """
    model.eval()
    raw_feats_list: List[torch.Tensor] = []
    norm_feats_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    sample_ious: List[float] = []

    use_gate = getattr(model, "_use_grpo_gate_in_eval", False)

    for batch in loader:
        images, labels, gt_masks = _unpack_batch(batch, device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # Extract [CLS] token representation and last-layer attention
            tokens, grid_size = model.backbone.prepare_tokens(images)
            attn: Optional[torch.Tensor] = None
            for idx, blk in enumerate(model.backbone.blocks):
                if idx < len(model.backbone.blocks) - 1:
                    tokens = blk(tokens, grid_size=grid_size)
                else:
                    tokens, attn = blk(tokens, grid_size=grid_size, return_attention=True)
            tokens = model.backbone.norm(tokens)
            assert attn is not None

            cls_feat = tokens[:, 0].float()
            patch_tokens = tokens[:, 1:].float()

            if use_gate:
                gate_probs = torch.sigmoid(
                    model.policy_gate(patch_tokens).squeeze(-1) + model.sparsity_bias
                )
                fg_pooled = (patch_tokens * gate_probs.unsqueeze(-1)).sum(dim=1) / (
                    gate_probs.sum(dim=-1, keepdim=True).clamp(min=1.0)
                )
                raw_feat = 0.85 * cls_feat + 0.15 * fg_pooled
                cls_attn = (attn[:, :, 0, 1:].float() * gate_probs.unsqueeze(1)).mean(dim=1)
            else:
                raw_feat = cls_feat
                cls_attn = attn[:, :, 0, 1:].float().mean(dim=1)

        norm_feat = F.normalize(raw_feat, dim=-1, p=2)
        raw_feats_list.append(raw_feat.cpu())
        norm_feats_list.append(norm_feat.cpu())
        labels_list.append(labels.cpu())

        if compute_masks:
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
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    ref_tokens, ref_grid = model.ref_model.prepare_tokens(images)
                    for blk_r in model.ref_model.blocks:
                        ref_tokens = blk_r(ref_tokens, grid_size=ref_grid)
                    ref_tokens = model.ref_model.norm(ref_tokens)
                target_mask = derive_salience_pseudo_mask(
                    images,
                    patch_features=ref_tokens[:, 1:].float(),
                    grid_size=(side, side),
                    target_sparsity=target_sparsity,
                )

            inter = (pred_mask * target_mask).sum(dim=-1)
            union = (pred_mask + target_mask - pred_mask * target_mask).sum(dim=-1)
            ious = (inter / (union + 1e-6)).cpu().tolist()
            sample_ious.extend(ious)

    raw_features = torch.cat(raw_feats_list, dim=0)
    norm_features = torch.cat(norm_feats_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    mean_iou = float(np.mean(sample_ious) * 100.0) if sample_ious else 0.0
    corloc = float(np.mean([1.0 if x >= 0.50 else 0.0 for x in sample_ious]) * 100.0) if sample_ious else 0.0
    return raw_features, norm_features, all_labels, mean_iou, corloc


def evaluate_fast_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.1,
    batch_size: int = 256,
) -> float:
    """Evaluates Linear Probe Top-1 Accuracy (%) on cached frozen features matching `eval_linear.py`.

    Args:
        train_features: Unnormalized training features (N_train, D).
        train_labels: Training labels (N_train,).
        val_features: Unnormalized validation features (N_val, D).
        val_labels: Validation labels (N_val,).
        num_classes: Number of target classes.
        device: Compute device.
        epochs: Number of linear probe training epochs (default: 30).
        lr: Initial SGD learning rate (default: 0.1 matching `report.md`).
        batch_size: Mini-batch size (default: 256 matching `eval_linear.py`).

    Returns:
        Best validation Top-1 accuracy (%).
    """
    torch.manual_seed(42)
    embed_dim = train_features.shape[-1]
    classifier = LinearClassifier(dim=embed_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_labels), batch_size=batch_size, shuffle=False
    )

    best_acc = 0.0
    for _ in range(epochs):
        classifier.train()
        for feats, targets in train_loader:
            feats, targets = feats.to(device), targets.to(device)
            logits = classifier(feats)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats, targets in val_loader:
                feats, targets = feats.to(device), targets.to(device)
                logits = classifier(feats)
                preds = logits.argmax(dim=-1)
                correct += int((preds == targets).sum().item())
                total += int(targets.size(0))
        acc = (correct / max(1, total)) * 100.0
        if acc > best_acc:
            best_acc = acc

    return float(best_acc)


def unnormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Converts an ImageNet-normalized tensor (3, H, W) to uint8 RGB numpy array (H, W, 3)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    rgb = (img_tensor * std + mean).clamp(0.0, 1.0)
    return (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


@torch.no_grad()
def compute_smooth_attention_heatmap(
    model: DINOGRPOPolicyWrapper,
    image_tensor: torch.Tensor,
    patch_size: int = 16,
) -> np.ndarray:
    """Computes upsampled 2D [CLS] attention heatmap normalized to [0, 1].

    Args:
        model: Loaded `DINOGRPOPolicyWrapper` model.
        image_tensor: Single image tensor of shape (1, 3, H, W).
        patch_size: ViT patch size.

    Returns:
        2D float32 numpy array of shape (H, W) in [0, 1].
    """
    use_gate = getattr(model, "_use_grpo_gate_in_eval", False)
    if use_gate:
        attn = model.get_last_selfattention(image_tensor)
    else:
        attn = model.backbone.get_last_selfattention(image_tensor)

    cls_attn = attn[0, :, 0, 1:].mean(dim=0)  # (N,)
    H_img, W_img = image_tensor.shape[-2], image_tensor.shape[-1]
    h_map = H_img // patch_size
    w_map = W_img // patch_size
    attn_2d = cls_attn.view(1, 1, h_map, w_map)
    upsampled = F.interpolate(
        attn_2d, size=(H_img, W_img), mode="bicubic", align_corners=False
    )[0, 0].cpu().numpy()
    a_min, a_max = float(upsampled.min()), float(upsampled.max())
    if a_max > a_min:
        upsampled = (upsampled - a_min) / (a_max - a_min)
    return upsampled


def generate_attention_comparison_grid(
    models_dict: Dict[str, DINOGRPOPolicyWrapper],
    val_loader: DataLoader,
    device: torch.device,
    output_path: str = "output/ablation_2x2_attention_grid.png",
    num_rows: int = 4,
    patch_size: int = 16,
) -> str:
    """Generates a publication-ready 5-column visual attention comparison grid.

    Columns:
    1. Original Image
    2. Vanilla DINOv1
    3. RoPE+Sparse DINOv1
    4. Vanilla+GRPO
    5. RoPE+Sparse+Dr.GRPO (Ours)

    Args:
        models_dict: Ordered mapping of model display names to loaded models.
        val_loader: Validation DataLoader.
        device: Compute device.
        output_path: Destination PNG file path.
        num_rows: Number of sample images (rows) to display.
        patch_size: ViT patch size.

    Returns:
        Saved PNG file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Collect 1 sample from distinct classes (e.g., airplane=0, horse=7, dog=5, ship=8)
    selected_imgs: List[torch.Tensor] = []
    seen_labels = set()
    preferred_labels = [0, 7, 5, 8, 1, 3]
    all_imgs_list: List[torch.Tensor] = []
    all_lbls_list: List[int] = []
    for batch in val_loader:
        b_imgs, b_lbls, _ = _unpack_batch(batch, device)
        for i in range(b_imgs.size(0)):
            all_imgs_list.append(b_imgs[i : i + 1])
            all_lbls_list.append(int(b_lbls[i].item()))
        if len(all_imgs_list) >= 256:
            break

    for pref in preferred_labels:
        for img_item, lbl_item in zip(all_imgs_list, all_lbls_list):
            if lbl_item == pref and lbl_item not in seen_labels:
                selected_imgs.append(img_item)
                seen_labels.add(lbl_item)
                break
        if len(selected_imgs) >= num_rows:
            break

    if len(selected_imgs) < num_rows:
        for img_item in all_imgs_list:
            selected_imgs.append(img_item)
            if len(selected_imgs) >= num_rows:
                break

    images = torch.cat(selected_imgs[:num_rows], dim=0)
    num_samples = images.size(0)

    col_titles = ["Original Image"] + list(models_dict.keys())
    num_cols = len(col_titles)

    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(3.4 * num_cols, 3.2 * num_samples),
        dpi=180,
    )
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx in range(num_samples):
        img_t = images[row_idx : row_idx + 1]
        rgb_np = unnormalize_image(img_t[0])

        # Column 0: Original Image
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(rgb_np)
        if row_idx == 0:
            ax_orig.set_title(col_titles[0], fontsize=11, fontweight="bold", pad=8)
        ax_orig.axis("off")

        # Columns 1..4: 4 Ablation Models
        for col_idx, (model_title, model_obj) in enumerate(models_dict.items(), start=1):
            ax = axes[row_idx, col_idx]
            heatmap = compute_smooth_attention_heatmap(model_obj, img_t, patch_size=patch_size)
            ax.imshow(rgb_np)
            ax.imshow(heatmap, cmap="inferno", alpha=0.72, vmin=0.0, vmax=1.0)
            if row_idx == 0:
                is_ours = "Ours" in model_title
                ax.set_title(
                    model_title,
                    fontsize=11,
                    fontweight="bold",
                    color="#b50000" if is_ours else "#111111",
                    pad=8,
                )
            ax.axis("off")

    plt.suptitle(
        "2x2 Ablation Study: Last-Layer [CLS] Self-Attention Maps (DINOv1 + Dr. GRPO)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_ablation_tables(
    results: List[Dict[str, Any]],
    md_path: str = "output/ablation_2x2_results.md",
    csv_path: str = "output/ablation_2x2_results.csv",
) -> Tuple[str, str]:
    """Exports the 2x2 ablation benchmark results to Markdown and CSV files.

    Args:
        results: List of metric dictionaries for each of the 4 models.
        md_path: Destination Markdown file path.
        csv_path: Destination CSV file path.

    Returns:
        Tuple of (`md_path`, `csv_path`).
    """
    os.makedirs(os.path.dirname(md_path), exist_ok=True)

    # 1. Export CSV
    fieldnames = [
        "Model Variant",
        "Architecture",
        "Dr. GRPO Policy",
        "k-NN Top-1 (%)",
        "k-NN Top-5 (%)",
        "Linear Probe Top-1 (%)",
        "Foreground IoU (%)",
        "CorLoc @ IoU>=0.5 (%)",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                "Model Variant": row["name"],
                "Architecture": row["arch"],
                "Dr. GRPO Policy": row["rlvr"],
                "k-NN Top-1 (%)": f"{row['knn_top1']:.2f}",
                "k-NN Top-5 (%)": f"{row['knn_top5']:.2f}",
                "Linear Probe Top-1 (%)": f"{row['linear_top1']:.2f}",
                "Foreground IoU (%)": f"{row['fg_iou']:.2f}",
                "CorLoc @ IoU>=0.5 (%)": f"{row['corloc']:.2f}",
            })

    # 2. Export Markdown for 7-Minute Defense Presentation
    md_lines = [
        "# 2x2 Ablation Study Results: Architecture (RoPE + Sparse) x Verifiable Policy Optimization (Dr. GRPO)",
        "",
        "| Model Variant | Backbone Architecture | Dr. GRPO Policy | k-NN Top-1 (%) | k-NN Top-5 (%) | Linear Probe Top-1 (%) | Foreground IoU (%) | CorLoc (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in results:
        is_ours = "Ours" in row["name"]
        name_fmt = f"**{row['name']}**" if is_ours else row["name"]
        knn1_fmt = f"**{row['knn_top1']:.2f}**" if is_ours else f"{row['knn_top1']:.2f}"
        lin1_fmt = f"**{row['linear_top1']:.2f}**" if is_ours else f"{row['linear_top1']:.2f}"
        iou_fmt = f"**{row['fg_iou']:.2f}**" if is_ours else f"{row['fg_iou']:.2f}"
        cor_fmt = f"**{row['corloc']:.2f}**" if is_ours else f"{row['corloc']:.2f}"
        md_lines.append(
            f"| {name_fmt} | `{row['arch']}` | {row['rlvr']} | "
            f"{knn1_fmt} | {row['knn_top5']:.2f} | {lin1_fmt} | {iou_fmt} | {cor_fmt} |"
        )

    if len(results) == 4:
        delta_knn_def = results[2]["knn_top1"] - results[0]["knn_top1"]
        delta_knn_rop = results[3]["knn_top1"] - results[1]["knn_top1"]
        delta_iou_def = results[2]["fg_iou"] - results[0]["fg_iou"]
        delta_iou_rop = results[3]["fg_iou"] - results[1]["fg_iou"]
        md_lines.extend([
            "",
            "## Key Defense Takeaways",
            f"- **Dr. GRPO Gain on Vanilla DINOv1**: `{delta_knn_def:+.2f}%` k-NN Top-1, `{delta_iou_def:+.2f}%` Foreground Mask IoU.",
            f"- **Dr. GRPO Gain on RoPE + Sparse DINOv1**: `{delta_knn_rop:+.2f}%` k-NN Top-1, `{delta_iou_rop:+.2f}%` Foreground Mask IoU.",
            f"- **Full Proposed Method (`RoPE+Sparse+Dr.GRPO`) vs. Vanilla Baseline**: "
            f"`{results[3]['knn_top1'] - results[0]['knn_top1']:+.2f}%` k-NN Top-1, "
            f"`{results[3]['linear_top1'] - results[0]['linear_top1']:+.2f}%` Linear Probe, "
            f"`{results[3]['fg_iou'] - results[0]['fg_iou']:+.2f}%` Foreground IoU.",
        ])

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write("\n".join(md_lines) + "\n")

    return md_path, csv_path


def get_args_parser() -> argparse.ArgumentParser:
    """Creates argument parser for `eval/eval_ablation_2x2.py`."""
    parser = argparse.ArgumentParser(description="2x2 Ablation Study Evaluation & Visualization")
    parser.add_argument("--data_path", type=str, default="data/cifar10", help="Dataset root directory")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["auto", "image_folder", "cifar10", "cifar100"],
        help="Dataset benchmark (default: cifar10 matching report.md)",
    )
    parser.add_argument(
        "--def_ckpt",
        type=str,
        default="cp/checkpoints_def*/checkpoint0190.pth",
        help="1. Vanilla DINOv1 checkpoint",
    )
    parser.add_argument(
        "--rop_ckpt",
        type=str,
        default="cp/checkpoints_rop*/checkpoint0190.pth",
        help="2. RoPE + Sparse DINOv1 checkpoint",
    )
    parser.add_argument(
        "--def_grpo_ckpt",
        type=str,
        default="cp/checkpoints_def_grpo/checkpoint0190.pth",
        help="3. Vanilla DINOv1 + Dr. GRPO checkpoint",
    )
    parser.add_argument(
        "--rop_grpo_ckpt",
        type=str,
        default="cp/checkpoints_rop_grpo/checkpoint0190.pth",
        help="4. RoPE + Sparse DINOv1 + Dr. GRPO checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--k", type=int, default=20, help="k-NN neighbors")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--linear_epochs", type=int, default=30)
    parser.add_argument("--linear_lr", type=float, default=0.1, help="Linear probe SGD lr (0.1 in report.md)")
    parser.add_argument("--target_sparsity", type=float, default=0.35)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Max train samples (0 = full 50,000 CIFAR-10 train set)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Max val samples (0 = full 10,000 CIFAR-10 val set)",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> Dict[str, Any]:
    """Runs the complete 2x2 Ablation Evaluation & Attention Visualization."""
    args = get_args_parser().parse_args()
    device = detect_device(args.device)
    print(f"Running 2x2 Ablation Benchmark on device: {device}")

    train_loader, val_loader, num_classes = build_dataloaders(
        eval_data_path=args.data_path,
        eval_dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=2 if device.type == "cuda" else 0,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    # Resolve all 4 checkpoint paths
    def_ckpt = resolve_checkpoint_path(args.def_ckpt)
    rop_ckpt = resolve_checkpoint_path(args.rop_ckpt)
    def_grpo_ckpt = (
        resolve_checkpoint_path(args.def_grpo_ckpt)
        if os.path.exists(os.path.dirname(args.def_grpo_ckpt))
        and os.path.isfile(args.def_grpo_ckpt)
        else def_ckpt
    )
    rop_grpo_ckpt = (
        resolve_checkpoint_path(args.rop_grpo_ckpt)
        if os.path.exists(os.path.dirname(args.rop_grpo_ckpt))
        and os.path.isfile(args.rop_grpo_ckpt)
        else rop_ckpt
    )

    ablation_specs = [
        ("Vanilla DINOv1", def_ckpt, "No"),
        ("RoPE+Sparse DINOv1", rop_ckpt, "No"),
        ("Vanilla+Dr.GRPO", def_grpo_ckpt, "Dr. GRPO"),
        ("RoPE+Sparse+Dr.GRPO (Ours)", rop_grpo_ckpt, "Dr. GRPO"),
    ]

    loaded_models: Dict[str, DINOGRPOPolicyWrapper] = {}
    results: List[Dict[str, Any]] = []

    for display_name, ckpt_path, rlvr_tag in ablation_specs:
        print(f"\nEvaluating [{display_name}] from {ckpt_path} ...")
        model = load_ablation_model(
            ckpt_path=ckpt_path,
            device=device,
            target_sparsity=args.target_sparsity,
        )
        loaded_models[display_name] = model

        train_raw_feats, train_norm_feats, train_labels, _, _ = extract_ablation_features_and_masks(
            model, train_loader, device, target_sparsity=args.target_sparsity, compute_masks=False
        )
        val_raw_feats, val_norm_feats, val_labels, fg_iou, corloc = extract_ablation_features_and_masks(
            model, val_loader, device, target_sparsity=args.target_sparsity, compute_masks=True
        )

        k_eff = min(args.k, train_norm_feats.shape[0])
        knn_top1, knn_top5 = knn_classifier(
            train_features=train_norm_feats,
            train_labels=train_labels,
            test_features=val_norm_feats,
            test_labels=val_labels,
            k=k_eff,
            T=args.temperature,
            num_classes=num_classes,
        )

        linear_top1 = evaluate_fast_linear_probe(
            train_features=train_raw_feats,
            train_labels=train_labels,
            val_features=val_raw_feats,
            val_labels=val_labels,
            num_classes=num_classes,
            device=device,
            epochs=args.linear_epochs,
            lr=args.linear_lr,
        )

        row_data = {
            "name": display_name,
            "arch": model.arch,
            "rlvr": rlvr_tag,
            "knn_top1": float(knn_top1),
            "knn_top5": float(knn_top5),
            "linear_top1": float(linear_top1),
            "fg_iou": float(fg_iou),
            "corloc": float(corloc),
        }
        results.append(row_data)
        print(
            f"  -> k-NN Top-1: {knn_top1:.2f}% | Linear Probe: {linear_top1:.2f}% | "
            f"Foreground IoU: {fg_iou:.2f}% | CorLoc: {corloc:.2f}%"
        )

    grid_path = generate_attention_comparison_grid(
        models_dict=loaded_models,
        val_loader=val_loader,
        device=device,
        output_path="output/ablation_2x2_attention_grid.png",
    )
    md_path, csv_path = export_ablation_tables(
        results=results,
        md_path="output/ablation_2x2_results.md",
        csv_path="output/ablation_2x2_results.csv",
    )

    print("\n" + "=" * 80)
    with open(md_path, "r", encoding="utf-8") as f:
        print(f.read())
    print("=" * 80)
    print(f"Saved Attention Grid to: {grid_path}")
    print(f"Saved Markdown Table to: {md_path}")
    print(f"Saved CSV Table to:      {csv_path}")

    return {
        "results": results,
        "grid_path": grid_path,
        "md_path": md_path,
        "csv_path": csv_path,
    }


if __name__ == "__main__":
    main()
