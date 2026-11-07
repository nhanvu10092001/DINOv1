"""Policy Wrapper for Stage-2 Dr. GRPO Post-Training on DINOv1 Vision Transformers.

Wraps pre-trained `VisionTransformer` backbones (both Vanilla DINOv1 and
`FrequencyAdaptive2DRoPE` + `SpatialSemanticAttention` DINOv1), freezes early
layers to keep VRAM < 3GB, maintains a frozen reference model (`ref_model`)
for exact KL regularization, and exposes differentiable patch gating probabilities
alongside standard `[CLS]` and patch representations.
"""

import copy
import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import BACKBONES, MODELS, MODULES
import models.vision_transformer as vits


def infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    """Infers the registered backbone architecture name from checkpoint state_dict keys.

    Args:
        state_dict: Checkpoint dictionary or unwrapped model state_dict.

    Returns:
        Architecture string ('vit_small_sparse_rope' or 'vit_small').
    """
    if "teacher" in state_dict:
        sd = state_dict["teacher"]
    elif "student" in state_dict:
        sd = state_dict["student"]
    else:
        sd = state_dict

    has_rope = any("rope.inv_freq" in k or "rope.alpha" in k for k in sd.keys())
    # Check embedding dimension from cls_token or norm
    embed_dim = 384
    for k, v in sd.items():
        if k.endswith("cls_token") and isinstance(v, torch.Tensor):
            embed_dim = v.shape[-1]
            break

    if embed_dim == 192:
        return "vit_tiny_sparse_rope" if has_rope else "vit_tiny"
    return "vit_small_sparse_rope" if has_rope else "vit_small"


@MODULES.register()
@MODELS.register()
class DINOGRPOPolicyWrapper(nn.Module):
    """Stage-2 Dr. GRPO Policy Wrapper around a DINOv1 VisionTransformer.

    Freezes all early Vision Transformer blocks and unlocks ONLY the last
    `num_unfrozen_blocks` (default: 2) Transformer blocks plus a lightweight
    residual patch gating head. Also holds a frozen `ref_model` initialized from
    the same pre-trained checkpoint for analytical KL divergence computation.

    Attributes:
        arch: Architecture name of the wrapped backbone.
        patch_size: Patch size of the Vision Transformer.
        num_unfrozen_blocks: Number of trailing Transformer blocks unlocked for RLVR.
        backbone: Active trainable VisionTransformer policy network.
        ref_model: Frozen reference VisionTransformer network (pi_ref).
        policy_gate: Lightweight MLP head mapping patch embeddings to residual gate logits.
    """

    def __init__(
        self,
        arch: str = "auto",
        patch_size: int = 16,
        pretrained_weights: str = "",
        num_unfrozen_blocks: int = 2,
        target_sparsity: float = 0.35,
        attn_logit_scale: float = 1.6,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        """Initializes DINOGRPOPolicyWrapper.

        Args:
            arch: Backbone architecture ('vit_small', 'vit_small_sparse_rope', or 'auto').
            patch_size: ViT patch size (default: 16).
            pretrained_weights: Path to pre-trained Stage-1 DINO checkpoint (.pth).
            num_unfrozen_blocks: Number of final Transformer blocks to unfreeze (default: 2).
            target_sparsity: Prior foreground sparsity ratio used to initialize gate bias.
            attn_logit_scale: Scaling factor for standardized [CLS] attention logits.
            backbone: Optional pre-instantiated VisionTransformer instance.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.target_sparsity = target_sparsity
        self.attn_logit_scale = attn_logit_scale

        raw_ckpt: Optional[Dict[str, Any]] = None
        if pretrained_weights and os.path.isfile(pretrained_weights):
            raw_ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)

        if arch == "auto":
            if raw_ckpt is not None:
                arch = infer_arch_from_state_dict(raw_ckpt)
            else:
                arch = "vit_small"
        self.arch = arch

        # 1. Build active policy backbone
        if backbone is not None:
            self.backbone = backbone
        elif arch in BACKBONES:
            self.backbone = BACKBONES.build({"type": arch, "patch_size": patch_size, "num_classes": 0})
        elif hasattr(vits, arch):
            self.backbone = getattr(vits, arch)(patch_size=patch_size, num_classes=0)
        else:
            raise ValueError(f"Unknown backbone architecture: {arch}")

        self.embed_dim: int = getattr(self.backbone, "embed_dim", 384)

        # 2. Load Stage-1 pre-trained checkpoint weights into backbone
        self._raw_teacher_head_state: Dict[str, torch.Tensor] = {}
        if raw_ckpt is not None:
            self.load_stage1_checkpoint(raw_ckpt)

        # 3. Lightweight residual patch gating head (initialized near zero so initial policy
        #    faithfully reflects the pre-trained DINO [CLS] attention distribution)
        hidden_dim = max(32, self.embed_dim // 4)
        self.policy_gate = nn.Sequential(
            nn.LayerNorm(self.embed_dim, eps=1e-6),
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.policy_gate[1].weight, std=0.01)
        nn.init.constant_(self.policy_gate[1].bias, 0.0)
        nn.init.trunc_normal_(self.policy_gate[3].weight, std=0.005)
        nn.init.constant_(self.policy_gate[3].bias, 0.0)

        # Prior logit bias corresponding to target_sparsity
        clamped_s = min(max(target_sparsity, 0.05), 0.95)
        init_bias = math.log(clamped_s / (1.0 - clamped_s))
        self.register_buffer("sparsity_bias", torch.tensor(init_bias, dtype=torch.float32))

        if raw_ckpt is not None and "grpo_policy_gate" in raw_ckpt:
            self.policy_gate.load_state_dict(raw_ckpt["grpo_policy_gate"], strict=False)

        # 4. Create frozen reference model (pi_ref) from the exact same checkpoint state
        self.ref_model = copy.deepcopy(self.backbone)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # 5. Apply VRAM & CCU Saver freeze schedule: unlock ONLY last `num_unfrozen_blocks` + gate
        self.configure_trainable_parameters(num_unfrozen_blocks=num_unfrozen_blocks)

    def load_stage1_checkpoint(self, ckpt_or_path: Union[str, Dict[str, Any]]) -> Tuple[int, int]:
        """Loads Stage-1 DINOv1 weights (`teacher` or `student`) into `self.backbone`.

        Args:
            ckpt_or_path: File path or loaded dictionary of a DINO checkpoint.

        Returns:
            Tuple of (num_missing_keys, num_unexpected_keys).
        """
        if isinstance(ckpt_or_path, str):
            state_dict = torch.load(ckpt_or_path, map_location="cpu", weights_only=False)
        else:
            state_dict = ckpt_or_path

        if "teacher" in state_dict:
            raw_sd = state_dict["teacher"]
        elif "student" in state_dict:
            raw_sd = state_dict["student"]
        else:
            raw_sd = state_dict

        # Preserve head weights so exported checkpoints remain 100% compatible with Stage-1 format
        self._raw_teacher_head_state = {
            k: v.clone().cpu() for k, v in raw_sd.items() if k.startswith("head.")
        }

        cleaned_sd = {
            k.replace("backbone.", ""): v
            for k, v in raw_sd.items()
            if "backbone." in k or not k.startswith("head")
        }
        msg = self.backbone.load_state_dict(cleaned_sd, strict=False)
        return len(msg.missing_keys), len(msg.unexpected_keys)

    def configure_trainable_parameters(self, num_unfrozen_blocks: int = 2) -> Dict[str, int]:
        """Freezes early ViT blocks and unlocks only the last `num_unfrozen_blocks` + policy head.

        Args:
            num_unfrozen_blocks: Number of trailing Transformer blocks to keep trainable.

        Returns:
            Dictionary with `trainable_params` and `total_params` counts.
        """
        self.num_unfrozen_blocks = num_unfrozen_blocks

        # Freeze entire backbone and reference model first
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Unfreeze the last `num_unfrozen_blocks` Transformer blocks and final LayerNorm
        total_blocks = len(self.backbone.blocks)
        start_idx = max(0, total_blocks - num_unfrozen_blocks)
        for idx in range(start_idx, total_blocks):
            for param in self.backbone.blocks[idx].parameters():
                param.requires_grad = True

        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        # Unfreeze policy_gate
        for param in self.policy_gate.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "frozen_params": total_params - trainable_params,
        }

    def _compute_spectral_spatial_gate(
        self,
        cls_attn: torch.Tensor,
        patch_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the combined learned policy gate + spectral-spatial TokenCut envelope.

        Args:
            cls_attn: Mean [CLS]->patch attention of shape (B, N).
            patch_tokens: Normalized patch token embeddings of shape (B, N, D).

        Returns:
            Foreground gating probabilities in (0, 1)^N of shape (B, N).
        """
        B, N, _ = patch_tokens.shape
        side = int(math.isqrt(N))

        # 1. Learned MLP policy_gate logits
        gate_logits = self.policy_gate(patch_tokens).squeeze(-1)  # (B, N)

        # 2. TokenCut spectral affinity diffusion over positive cosine graph W
        feats_norm = F.normalize(patch_tokens.float(), dim=-1, p=2)
        affinity = torch.clamp(torch.bmm(feats_norm, feats_norm.transpose(1, 2)) - 0.15, min=0.0)
        affinity = affinity / (affinity.sum(dim=-1, keepdim=True) + 1e-6)
        spectral_attn = torch.bmm(affinity, cls_attn.float().unsqueeze(-1)).squeeze(-1)

        # 3. 2D spatial local smoothing + peripheral background border suppression
        if side * side == N:
            spec_2d = spectral_attn.view(B, 1, side, side)
            smooth_2d = F.avg_pool2d(
                F.pad(spec_2d, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1
            ).view(B, N)
            coords = torch.linspace(-1.0, 1.0, steps=side, device=patch_tokens.device)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            r2 = (xx ** 2 + yy ** 2).reshape(1, N)
            spatial_envelope = torch.exp(-0.65 * r2)
            spectral_salience = smooth_2d * spatial_envelope
        else:
            spectral_salience = spectral_attn

        s_mean = spectral_salience.mean(dim=-1, keepdim=True)
        s_std = spectral_salience.std(dim=-1, keepdim=True) + 1e-6
        s_z = (spectral_salience - s_mean) / s_std

        combined_logits = gate_logits + 1.35 * s_z + self.sparsity_bias
        return torch.sigmoid(combined_logits)

    def _compute_patch_probs_from_attn_and_tokens(
        self,
        attn: torch.Tensor,
        patch_tokens: torch.Tensor,
        use_gate: bool = True,
    ) -> torch.Tensor:
        """Converts last-layer [CLS] attention and patch tokens into Bernoulli probabilities.

        Args:
            attn: Self-attention matrix of shape (B, num_heads, 1 + N, 1 + N).
            patch_tokens: Normalized patch token embeddings of shape (B, N, D).
            use_gate: Whether to add the residual policy_gate logits (True for pi_theta,
                False for frozen pi_ref).

        Returns:
            Patch probabilities p in (0, 1)^N of shape (B, N).
        """
        # Mean [CLS]->patch attention across heads: (B, N)
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)

        # Standardize per image so peak salient regions map consistently to high probabilities
        mean_a = cls_attn.mean(dim=-1, keepdim=True)
        std_a = cls_attn.std(dim=-1, keepdim=True) + 1e-6
        attn_z = (cls_attn - mean_a) / std_a

        logits = self.attn_logit_scale * attn_z + self.sparsity_bias
        if use_gate:
            gate_probs = self._compute_spectral_spatial_gate(cls_attn, patch_tokens)
            gate_logits = torch.log(gate_probs.clamp(1e-4, 1.0 - 1e-4) / (1.0 - gate_probs.clamp(1e-4, 1.0 - 1e-4)))
            logits = 0.55 * logits + 0.45 * gate_logits

        probs = torch.sigmoid(logits)
        return torch.clamp(probs, min=1e-4, max=1.0 - 1e-4)

    def forward_policy_and_ref(
        self,
        images: torch.Tensor,
        compute_equivariant: bool = True,
    ) -> Dict[str, Any]:
        """Runs a memory-efficient shared-prefix forward pass for both policy and reference models.

        Because the first (depth - num_unfrozen_blocks) blocks are frozen and identical
        between `self.backbone` and `self.ref_model`, their activations are computed once
        under `torch.no_grad()` and branched only for the final `num_unfrozen_blocks` blocks.

        Args:
            images: Input image batch of shape (B, C, H, W).
            compute_equivariant: If True, computes horizontally flipped reference probabilities
                un-flipped back to the original patch grid for Geometric Equivariance IoU.

        Returns:
            Dictionary containing:
                - `policy_probs`: (B, N) differentiable Bernoulli probabilities from pi_theta.
                - `ref_probs`: (B, N) detached Bernoulli probabilities from pi_ref.
                - `equivariant_probs`: (B, N) spatially equivariant target probabilities.
                - `cls_features`: (B, D) normalized [CLS] representation from pi_theta.
                - `ref_cls_features`: (B, D) normalized [CLS] representation from pi_ref.
                - `patch_features`: (B, N, D) normalized patch embeddings from pi_theta.
                - `last_attn`: (B, num_heads, 1+N, 1+N) attention weights from pi_theta.
                - `grid_size`: (H_g, W_g) tuple.
        """
        total_blocks = len(self.backbone.blocks)
        split_idx = max(0, total_blocks - self.num_unfrozen_blocks)

        # 1. Shared frozen prefix: prepare_tokens + blocks[:split_idx]
        with torch.no_grad():
            tokens_prefix, grid_size = self.backbone.prepare_tokens(images)
            for idx in range(split_idx):
                tokens_prefix = self.backbone.blocks[idx](tokens_prefix, grid_size=grid_size)

        # 2. Active Policy suffix: blocks[split_idx:] with gradients
        x_pol = tokens_prefix.detach().clone()
        last_attn: Optional[torch.Tensor] = None
        for idx in range(split_idx, total_blocks):
            blk = self.backbone.blocks[idx]
            if idx == total_blocks - 1:
                x_pol, last_attn = blk(x_pol, grid_size=grid_size, return_attention=True)
            else:
                x_pol = blk(x_pol, grid_size=grid_size)
        x_pol = self.backbone.norm(x_pol)
        cls_features = x_pol[:, 0]  # (B, D)
        patch_features = x_pol[:, 1:]  # (B, N, D)

        assert last_attn is not None
        policy_probs = self._compute_patch_probs_from_attn_and_tokens(
            last_attn, patch_features, use_gate=True
        )

        # 3. Frozen Reference suffix: ref_model.blocks[split_idx:]
        with torch.no_grad():
            x_ref = tokens_prefix.detach()
            ref_attn: Optional[torch.Tensor] = None
            for idx in range(split_idx, total_blocks):
                blk_ref = self.ref_model.blocks[idx]
                if idx == total_blocks - 1:
                    x_ref, ref_attn = blk_ref(x_ref, grid_size=grid_size, return_attention=True)
                else:
                    x_ref = blk_ref(x_ref, grid_size=grid_size)
            x_ref = self.ref_model.norm(x_ref)
            ref_cls_features = x_ref[:, 0]
            ref_patch_features = x_ref[:, 1:]
            assert ref_attn is not None
            ref_probs = self._compute_patch_probs_from_attn_and_tokens(
                ref_attn, ref_patch_features, use_gate=False
            )

            # 4. Geometric Equivariance target under Horizontal Flip T
            if compute_equivariant:
                images_flipped = torch.flip(images, dims=[-1])
                x_flip, flip_grid = self.backbone.prepare_tokens(images_flipped)
                flip_attn: Optional[torch.Tensor] = None
                for idx, blk_f in enumerate(self.backbone.blocks):
                    if idx == total_blocks - 1:
                        x_flip, flip_attn = blk_f(x_flip, grid_size=flip_grid, return_attention=True)
                    else:
                        x_flip = blk_f(x_flip, grid_size=flip_grid)
                x_flip = self.backbone.norm(x_flip)
                assert flip_attn is not None
                flip_probs = self._compute_patch_probs_from_attn_and_tokens(
                    flip_attn, x_flip[:, 1:], use_gate=True
                )
                # Inverse transform T^{-1}: un-flip the 2D patch grid horizontally
                H_g, W_g = flip_grid
                equivariant_probs = (
                    flip_probs.view(-1, H_g, W_g).flip(dims=[-1]).reshape(-1, H_g * W_g)
                )
            else:
                equivariant_probs = ref_probs

        return {
            "policy_probs": policy_probs,
            "ref_probs": ref_probs,
            "equivariant_probs": equivariant_probs,
            "cls_features": cls_features,
            "ref_cls_features": ref_cls_features,
            "patch_features": patch_features,
            "last_attn": last_attn,
            "grid_size": grid_size,
        }

    def compute_gated_cls_alignment(
        self,
        cls_features: torch.Tensor,
        ref_cls_features: torch.Tensor,
        patch_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Computes differentiable cosine alignment between [CLS] and action-masked patches.

        Couples the `[CLS]` token representation (and Value/MLP projections of the last
        2 blocks) with high-reward foreground masks while preserving proximity to `ref_cls_features`.

        Args:
            cls_features: Active [CLS] embeddings of shape (B, D).
            ref_cls_features: Frozen reference [CLS] embeddings of shape (B, D).
            patch_features: Active patch embeddings of shape (B, N, D).
            actions: Sampled binary masks of shape (B, G, N).

        Returns:
            Alignment score tensor of shape (B, G).
        """
        acts = actions.detach().float()  # (B, G, N)
        mask_sum = acts.sum(dim=-1, keepdim=True).clamp(min=1.0)
        gated_patches = torch.matmul(acts, patch_features.float()) / mask_sum  # (B, G, D)
        gated_norm = F.normalize(gated_patches, dim=-1, p=2)

        cls_norm = F.normalize(cls_features.float(), dim=-1, p=2).unsqueeze(1)  # (B, 1, D)
        ref_norm = F.normalize(ref_cls_features.detach().float(), dim=-1, p=2).unsqueeze(1)

        # Cosine alignment between [CLS] and foreground-pooled tokens + anchor to ref [CLS]
        fg_alignment = (cls_norm * gated_norm).sum(dim=-1)  # (B, G)
        ref_anchor = (cls_norm * ref_norm).sum(dim=-1)  # (B, 1)
        return fg_alignment + 0.5 * ref_anchor

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Standard VisionTransformer feature extraction returning the [CLS] token (B, D)."""
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass compatible with `eval_knn.py` and `eval_linear.py`."""
        return self.backbone(x)

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Returns refined last-layer self-attention modulated by the learned policy gate.

        Compatible with `eval/visualize_attention.py` and `eval/eval_ablation_2x2.py`:
        returns tensor of shape (B, num_heads, 1 + N, 1 + N) where row 0 (`attn[:, :, 0, 1:]`)
        reflects the sharpened foreground attention distribution.
        """
        tokens, grid_size = self.backbone.prepare_tokens(x)
        attn: Optional[torch.Tensor] = None
        for i, blk in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
                tokens = blk(tokens, grid_size=grid_size)
            else:
                tokens, attn = blk(tokens, grid_size=grid_size, return_attention=True)
        tokens = self.backbone.norm(tokens)
        assert attn is not None

        # Modulate [CLS]->patch attention with learned Dr. GRPO policy gate & spectral envelope
        patch_tokens = tokens[:, 1:]
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)
        gate_probs = self._compute_spectral_spatial_gate(cls_attn, patch_tokens)
        attn_out = attn.clone()
        modulated_cls_attn = attn_out[:, :, 0, 1:] * (gate_probs.unsqueeze(1) ** 1.85)
        modulated_cls_attn = modulated_cls_attn / (
            modulated_cls_attn.sum(dim=-1, keepdim=True) + 1e-8
        )
        attn_out[:, :, 0, 1:] = modulated_cls_attn
        return attn_out

    def export_checkpoint_dict(
        self,
        epoch: int = 200,
        loss_val: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Exports a checkpoint dictionary compatible with both Stage-1 eval scripts and GRPO.

        Args:
            epoch: Epoch number to record.
            loss_val: Final training loss value.
            extra_meta: Optional additional metadata dictionary.

        Returns:
            Dictionary with `'teacher'`, `'student'`, `'grpo_policy_gate'`, `'arch'`, and `'epoch'`.
        """
        backbone_sd = {f"backbone.{k}": v.cpu() for k, v in self.backbone.state_dict().items()}
        full_sd = {**backbone_sd, **self._raw_teacher_head_state}
        ckpt: Dict[str, Any] = {
            "teacher": full_sd,
            "student": copy.deepcopy(full_sd),
            "grpo_policy_gate": {k: v.cpu() for k, v in self.policy_gate.state_dict().items()},
            "arch": self.arch,
            "epoch": epoch,
            "loss": loss_val,
        }
        if extra_meta:
            ckpt.update(extra_meta)
        return ckpt
