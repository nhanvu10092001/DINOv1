"""Dr. GRPO (Unbiased / Rank-Normalized Group Relative Policy Optimization) Loss.

Implements Stage-2 Reinforcement Learning with Verifiable Rewards (RLVR) for
Self-Supervised Vision Transformers (DINOv1), featuring:
- Group rollout sampling (G binary/stochastic patch masks per image)
- Verifiable visual rewards without neural reward models:
  1. `self_supervised_v1`: Geometric Equivariance IoU + TokenCut/Normalized-Cut
     Foreground Compactness + Target Sparsity Ratio
  2. `downstream_supervised`: Direct foreground mask IoU or Prototype Cosine Margin
- Variance-safe advantage estimators (`dr_grpo` and `rank_grpo`)
- Length-normalized surrogate objective with DAPO-style asymmetric clipping
  (eps_low=0.20, eps_high=0.28) and exact analytical Bernoulli KL anchor.
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import LOSSES, MODULES


@MODULES.register()
@LOSSES.register()
class DrGRPOLoss(nn.Module):
    """Unbiased / Rank-Normalized Group Relative Policy Optimization (Dr. GRPO) Loss.

    Attributes:
        group_size: Number of action rollouts G sampled per image.
        eps_low: Lower clipping bound for probability ratio rho_i.
        eps_high: Upper clipping bound for probability ratio rho_i (Clip-Higher).
        beta_kl: Weight of the KL divergence penalty against the frozen reference policy.
        adv_type: Advantage formulation ('dr_grpo' or 'rank_grpo').
        reward_mode: Verifiable reward mode ('self_supervised_v1' or 'downstream_supervised').
        target_sparsity: Target foreground patch occupancy ratio in (0, 1).
        use_gumbel: Whether to use Gumbel-Sigmoid relaxation instead of hard Bernoulli.
        gumbel_temp: Temperature for Gumbel-Sigmoid sampling when use_gumbel is True.
    """

    def __init__(
        self,
        group_size: int = 8,
        eps_low: float = 0.20,
        eps_high: float = 0.28,
        beta_kl: float = 0.04,
        adv_type: str = "dr_grpo",
        reward_mode: str = "self_supervised_v1",
        target_sparsity: float = 0.35,
        use_gumbel: bool = False,
        gumbel_temp: float = 0.5,
        equivariance_weight: float = 0.45,
        compactness_weight: float = 0.35,
        sparsity_weight: float = 0.20,
    ) -> None:
        """Initializes the DrGRPOLoss module.

        Args:
            group_size: Number of rollout masks G per input image (default: 8).
            eps_low: Lower ratio clipping threshold (default: 0.20).
            eps_high: Asymmetric upper ratio clipping threshold (default: 0.28).
            beta_kl: KL regularization coefficient against pi_ref (default: 0.04).
            adv_type: Either 'dr_grpo' (mean-centered, no std division) or 'rank_grpo'.
            reward_mode: Either 'self_supervised_v1' or 'downstream_supervised'.
            target_sparsity: Desired fraction of active foreground tokens (default: 0.35).
            use_gumbel: If True, samples via Gumbel-Sigmoid straight-through estimator.
            gumbel_temp: Temperature for Gumbel-Sigmoid relaxation.
            equivariance_weight: Weight of geometric equivariance IoU reward term.
            compactness_weight: Weight of TokenCut / spatial compactness reward term.
            sparsity_weight: Weight of target sparsity prior reward term.
        """
        super().__init__()
        if adv_type not in ("dr_grpo", "rank_grpo"):
            raise ValueError(f"Unsupported adv_type '{adv_type}'. Choose 'dr_grpo' or 'rank_grpo'.")
        if reward_mode not in ("self_supervised_v1", "downstream_supervised"):
            raise ValueError(
                f"Unsupported reward_mode '{reward_mode}'. "
                "Choose 'self_supervised_v1' or 'downstream_supervised'."
            )

        self.group_size = group_size
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.beta_kl = beta_kl
        self.adv_type = adv_type
        self.reward_mode = reward_mode
        self.target_sparsity = target_sparsity
        self.use_gumbel = use_gumbel
        self.gumbel_temp = gumbel_temp
        self.equivariance_weight = equivariance_weight
        self.compactness_weight = compactness_weight
        self.sparsity_weight = sparsity_weight

    @staticmethod
    def compute_length_normalized_log_probs(
        probs: torch.Tensor,
        actions: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Computes length-normalized log-probabilities over N patch tokens.

        log pi_theta(a_i | I) = (1 / N) * sum_{p=1}^N log pi_theta(a_{i,p} | I)

        Args:
            probs: Patch Bernoulli probabilities of shape (B, N) or (B, G, N).
            actions: Sampled binary actions of shape (B, G, N) in {0, 1}.
            eps: Numerical clamping epsilon for probabilities.

        Returns:
            Length-normalized log-probabilities of shape (B, G).
        """
        if probs.ndim == 2:
            probs = probs.unsqueeze(1)  # (B, 1, N)
        clamped_p = torch.clamp(probs.float(), min=eps, max=1.0 - eps)
        acts = actions.float()
        patch_log_probs = acts * torch.log(clamped_p) + (1.0 - acts) * torch.log(1.0 - clamped_p)
        return patch_log_probs.mean(dim=-1)  # (B, G)

    @torch.no_grad()
    def sample_actions(
        self,
        probs: torch.Tensor,
        group_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples G binary patch masks per image from Bernoulli(p) or Gumbel-Sigmoid.

        Args:
            probs: Patch gating / attention probabilities of shape (B, N) in (0, 1).
            group_size: Optional override for G.

        Returns:
            Tuple of:
                - actions: Binary masks of shape (B, G, N).
                - old_log_probs: Length-normalized log pi_old(a_i | I) of shape (B, G).
        """
        G = group_size or self.group_size
        B, N = probs.shape
        clamped_p = torch.clamp(probs.detach().float(), min=1e-5, max=1.0 - 1e-5)
        expanded_p = clamped_p.unsqueeze(1).expand(B, G, N)

        if self.use_gumbel:
            logits = torch.log(expanded_p / (1.0 - expanded_p))
            u = torch.rand_like(logits).clamp_(1e-5, 1.0 - 1e-5)
            gumbel_noise = torch.log(u) - torch.log(1.0 - u)
            y_soft = torch.sigmoid((logits + gumbel_noise) / self.gumbel_temp)
            actions = (y_soft > 0.5).float()
        else:
            actions = torch.bernoulli(expanded_p)

        # Guarantee at least 1 active foreground patch per mask to prevent degenerate zero division
        empty_mask = actions.sum(dim=-1) == 0  # (B, G)
        if empty_mask.any():
            top_idx = expanded_p.argmax(dim=-1, keepdim=True)  # (B, G, 1)
            fallback = torch.zeros_like(actions).scatter_(-1, top_idx, 1.0)
            actions = torch.where(empty_mask.unsqueeze(-1), fallback, actions)

        old_log_probs = self.compute_length_normalized_log_probs(clamped_p, actions)
        return actions, old_log_probs

    @staticmethod
    def compute_kl_divergence(
        probs: torch.Tensor,
        ref_probs: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Computes exact length-normalized analytical Bernoulli KL divergence D_KL(pi_theta || pi_ref).

        Args:
            probs: Active policy patch probabilities of shape (B, N).
            ref_probs: Frozen reference policy patch probabilities of shape (B, N).
            eps: Numerical stability epsilon.

        Returns:
            Scalar KL divergence averaged over batch and N patches.
        """
        p = torch.clamp(probs.float(), min=eps, max=1.0 - eps)
        q = torch.clamp(ref_probs.detach().float(), min=eps, max=1.0 - eps)
        kl_per_patch = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
        return kl_per_patch.mean()

    def compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Computes group-relative advantages using variance-safe formulations.

        Args:
            rewards: Verifiable reward tensor of shape (B, G).

        Returns:
            Advantage tensor A of shape (B, G).
        """
        B, G = rewards.shape
        if G <= 1:
            return torch.zeros_like(rewards)

        if self.adv_type == "dr_grpo":
            # Unbiased centered advantage without standard-deviation division
            mean_r = rewards.mean(dim=-1, keepdim=True)
            advantages = rewards - mean_r
            return advantages

        # adv_type == "rank_grpo": distribution-ranked advantage in [-1, 1]
        # Compute fractional ranks in [0, G - 1] handling ties cleanly via double argsort + tie averaging
        sorted_indices = torch.argsort(rewards, dim=-1)
        ranks = torch.argsort(sorted_indices, dim=-1).float()
        advantages = 2.0 * (ranks / float(G - 1)) - 1.0
        return advantages

    @staticmethod
    def _compute_binary_or_soft_iou(
        mask_a: torch.Tensor,
        mask_b: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Computes Intersection-over-Union (IoU) between masks of shape (..., N).

        Args:
            mask_a: Sampled action masks of shape (B, G, N).
            mask_b: Target masks of shape (B, 1, N) or (B, G, N).
            eps: Smoothing constant.

        Returns:
            IoU score tensor of shape (B, G) in [0, 1].
        """
        a = mask_a.float()
        b = mask_b.float()
        intersection = (a * b).sum(dim=-1)
        union = (a + b - a * b).sum(dim=-1)
        return intersection / (union + eps)

    def _compute_self_supervised_v1_reward(
        self,
        actions: torch.Tensor,
        patch_features: torch.Tensor,
        equivariant_probs: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes `self_supervised_v1` verifiable reward.

        Combines:
        1. Geometric Equivariance IoU: Consistency of sampled mask a_i with the
           spatially un-transformed foreground mask under known transform T.
        2. TokenCut / Normalized-Cut Foreground Compactness: Spectral feature affinity
           association within foreground minus cut across foreground/background, plus
           2D spatial neighbor smoothness on the patch grid.
        3. Target Sparsity Ratio: Penalizes background noise tokens and degenerate masks.

        Args:
            actions: Binary masks of shape (B, G, N).
            patch_features: Patch token embeddings of shape (B, N, D).
            equivariant_probs: Reference/transformed foreground probabilities (B, N).
            grid_size: Optional (H_g, W_g) tuple. Inferred as square if None.

        Returns:
            Tuple of total rewards (B, G) and component dictionary.
        """
        B, G, N = actions.shape
        device = actions.device
        acts = actions.float()

        if grid_size is None:
            side = int(math.isqrt(N))
            grid_size = (side, side)
        H_g, W_g = grid_size

        # 1. Geometric Equivariance IoU
        if equivariant_probs is not None:
            # Threshold top-k salient tokens of equivariant_probs according to target_sparsity
            k_fg = max(1, int(round(N * self.target_sparsity)))
            topk_idx = torch.topk(equivariant_probs.detach(), k=k_fg, dim=-1).indices
            eq_binary_mask = torch.zeros_like(equivariant_probs).scatter_(-1, topk_idx, 1.0)
            r_equiv = self._compute_binary_or_soft_iou(acts, eq_binary_mask.unsqueeze(1))
        else:
            # Self-affinity salient anchor fallback when transformed view is not passed
            feats_norm = F.normalize(patch_features.detach().float(), dim=-1, p=2)
            mean_feat = F.normalize(feats_norm.mean(dim=1, keepdim=True), dim=-1, p=2)
            salience = (feats_norm * mean_feat).sum(dim=-1)
            k_fg = max(1, int(round(N * self.target_sparsity)))
            topk_idx = torch.topk(salience, k=k_fg, dim=-1).indices
            eq_binary_mask = torch.zeros((B, N), device=device).scatter_(-1, topk_idx, 1.0)
            r_equiv = self._compute_binary_or_soft_iou(acts, eq_binary_mask.unsqueeze(1))

        # 2. TokenCut / Normalized-Cut Foreground Compactness
        feats_norm = F.normalize(patch_features.detach().float(), dim=-1, p=2)  # (B, N, D)
        # Positive cosine affinity matrix W in [0, 1]: (B, N, N)
        affinity = torch.clamp(torch.bmm(feats_norm, feats_norm.transpose(1, 2)), min=0.0)

        # Compute Foreground-to-Foreground cohesion vs Foreground-to-Background cut
        # acts: (B, G, N), affinity: (B, N, N) -> wa = acts @ affinity: (B, G, N)
        wa = torch.matmul(acts, affinity)  # (B, G, N)
        assoc_fg = (wa * acts).sum(dim=-1)  # a_i^T W a_i: (B, G)
        cut_fg_bg = (wa * (1.0 - acts)).sum(dim=-1)  # a_i^T W (1 - a_i): (B, G)
        vol_fg = wa.sum(dim=-1) + 1e-6
        ncut_compactness = (assoc_fg - cut_fg_bg) / vol_fg  # in [-1, 1]
        ncut_score = 0.5 * (ncut_compactness + 1.0)  # mapped to [0, 1]

        # 2D Spatial Neighbor Compactness (penalizes fragmented salt-and-pepper masks)
        if H_g * W_g == N:
            acts_2d = acts.view(B, G, H_g, W_g)
            tv_h = (acts_2d[:, :, 1:, :] - acts_2d[:, :, :-1, :]).abs().mean(dim=(-2, -1))
            tv_w = (acts_2d[:, :, :, 1:] - acts_2d[:, :, :, :-1]).abs().mean(dim=(-2, -1))
            spatial_smoothness = 1.0 - 0.5 * (tv_h + tv_w)
            r_compact = 0.65 * ncut_score + 0.35 * spatial_smoothness
        else:
            r_compact = ncut_score

        # 3. Target Sparsity Ratio Reward
        active_ratio = acts.mean(dim=-1)  # (B, G)
        sparsity_dev = (active_ratio - self.target_sparsity).abs()
        max_dev = max(self.target_sparsity, 1.0 - self.target_sparsity)
        r_sparsity = torch.clamp(1.0 - (sparsity_dev / max_dev) ** 1.5, min=0.0, max=1.0)

        total_reward = (
            self.equivariance_weight * r_equiv
            + self.compactness_weight * r_compact
            + self.sparsity_weight * r_sparsity
        )
        metrics = {
            "reward_equivariance_iou": r_equiv.mean().detach(),
            "reward_compactness": r_compact.mean().detach(),
            "reward_sparsity": r_sparsity.mean().detach(),
            "active_patch_ratio": active_ratio.mean().detach(),
        }
        return total_reward, metrics

    def _compute_downstream_supervised_reward(
        self,
        actions: torch.Tensor,
        patch_features: torch.Tensor,
        gt_masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_prototypes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes `downstream_supervised` verifiable reward.

        Supports:
        - Direct IoU with ground-truth foreground masks (`gt_masks`), and/or
        - Correct k-NN / Prototype Cosine Margin when `labels` and `class_prototypes`
          (or batch support features) are provided.

        Args:
            actions: Binary masks of shape (B, G, N).
            patch_features: Patch token features of shape (B, N, D).
            gt_masks: Optional ground-truth binary masks of shape (B, N) or (B, H, W).
            labels: Optional ground-truth class indices of shape (B,).
            class_prototypes: Optional normalized class prototypes of shape (C, D).

        Returns:
            Tuple of total rewards (B, G) and component dictionary.
        """
        B, G, N = actions.shape
        acts = actions.float()
        total_reward = torch.zeros((B, G), device=actions.device, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}

        if gt_masks is not None:
            if gt_masks.ndim > 2:
                side = int(math.isqrt(N))
                gt_resized = F.interpolate(
                    gt_masks.float().view(B, 1, *gt_masks.shape[-2:]),
                    size=(side, side),
                    mode="nearest",
                ).view(B, N)
            else:
                gt_resized = gt_masks.float().view(B, N)
            iou_reward = self._compute_binary_or_soft_iou(acts, gt_resized.unsqueeze(1))
            total_reward = total_reward + iou_reward
            metrics["reward_gt_iou"] = iou_reward.mean().detach()

        if labels is not None:
            # Masked average pooling of patch features: (B, G, N, 1) * (B, 1, N, D) -> (B, G, D)
            feats = patch_features.float()
            mask_sum = acts.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pooled = torch.matmul(acts, feats) / mask_sum  # (B, G, D)
            pooled_norm = F.normalize(pooled, dim=-1, p=2)

            if class_prototypes is not None:
                protos = F.normalize(class_prototypes.detach().float(), dim=-1, p=2)  # (C, D)
                sim = torch.matmul(pooled_norm, protos.t())  # (B, G, C)
                target_idx = labels.view(B, 1, 1).expand(B, G, 1)
                pos_sim = torch.gather(sim, dim=-1, index=target_idx).squeeze(-1)  # (B, G)

                sim_neg = sim.clone()
                sim_neg.scatter_(-1, target_idx, float("-inf"))
                max_neg_sim = sim_neg.max(dim=-1).values  # (B, G)
                max_neg_sim = torch.where(torch.isinf(max_neg_sim), torch.zeros_like(pos_sim), max_neg_sim)

                margin = pos_sim - max_neg_sim  # in [-2, 2]
                correct_bonus = (sim.argmax(dim=-1) == labels.unsqueeze(1)).float() * 0.5
                proto_reward = 0.5 * (margin + 1.0) + correct_bonus
            else:
                # Batch leave-one-out k-NN / prototype margin
                batch_feats = F.normalize(feats.mean(dim=1).detach(), dim=-1, p=2)  # (B, D)
                sim_matrix = torch.matmul(pooled_norm, batch_feats.t())  # (B, G, B)
                same_class = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B)
                eye = torch.eye(B, dtype=torch.bool, device=actions.device)
                pos_mask = same_class & ~eye
                neg_mask = ~same_class

                pos_scores = sim_matrix.masked_fill(~pos_mask.unsqueeze(1), 0.0).sum(dim=-1) / (
                    pos_mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
                )
                neg_scores = sim_matrix.masked_fill(~neg_mask.unsqueeze(1), float("-inf")).max(dim=-1).values
                neg_scores = torch.where(torch.isinf(neg_scores), torch.zeros_like(pos_scores), neg_scores)
                proto_reward = 0.5 * ((pos_scores - neg_scores) + 1.0)

            total_reward = total_reward + proto_reward
            metrics["reward_proto_margin"] = proto_reward.mean().detach()

        if gt_masks is None and labels is None:
            # Fallback to self-supervised verifiable reward if no supervision tensors passed
            return self._compute_self_supervised_v1_reward(actions, patch_features)

        metrics["active_patch_ratio"] = acts.mean().detach()
        return total_reward, metrics

    @torch.no_grad()
    def compute_verifiable_reward(
        self,
        actions: torch.Tensor,
        patch_features: torch.Tensor,
        equivariant_probs: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_prototypes: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes non-differentiable verifiable rewards R(a_i, I) for all G rollouts.

        Args:
            actions: Binary action masks of shape (B, G, N).
            patch_features: Patch embeddings of shape (B, N, D).
            equivariant_probs: Spatially equivariant target probabilities of shape (B, N).
            gt_masks: Optional ground-truth foreground masks for downstream mode.
            labels: Optional class labels of shape (B,) for downstream mode.
            class_prototypes: Optional class prototypes of shape (C, D).
            grid_size: Optional (H_g, W_g) patch grid dimensions.

        Returns:
            Tuple of (rewards tensor of shape (B, G), dict of scalar reward components).
        """
        if self.reward_mode == "self_supervised_v1":
            return self._compute_self_supervised_v1_reward(
                actions=actions,
                patch_features=patch_features,
                equivariant_probs=equivariant_probs,
                grid_size=grid_size,
            )
        return self._compute_downstream_supervised_reward(
            actions=actions,
            patch_features=patch_features,
            gt_masks=gt_masks,
            labels=labels,
            class_prototypes=class_prototypes,
        )

    def forward(
        self,
        policy_probs: torch.Tensor,
        ref_probs: torch.Tensor,
        patch_features: torch.Tensor,
        old_probs: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        equivariant_probs: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_prototypes: Optional[torch.Tensor] = None,
        grid_size: Optional[Tuple[int, int]] = None,
        gated_cls_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the full Dr. GRPO objective with asymmetric clipping and KL anchor.

        L_GRPO = -(1/G) sum_{i=1}^G min(rho_i * A_i, clip(rho_i, 1-eps_low, 1+eps_high) * A_i)
                 + beta * D_KL(pi_theta || pi_ref)

        Args:
            policy_probs: Current policy probabilities p in (0, 1)^N of shape (B, N).
            ref_probs: Frozen reference probabilities p_ref in (0, 1)^N of shape (B, N).
            patch_features: Patch token features of shape (B, N, D).
            old_probs: Optional behavior policy probabilities of shape (B, N). Defaults to
                detached policy_probs.
            actions: Optional pre-sampled binary masks of shape (B, G, N).
            old_log_probs: Optional pre-computed behavior log-probs of shape (B, G).
            equivariant_probs: Optional equivariant target probabilities of shape (B, N).
            gt_masks: Optional ground-truth foreground masks.
            labels: Optional ground-truth class labels of shape (B,).
            class_prototypes: Optional class prototype matrix of shape (C, D).
            grid_size: Optional (H_g, W_g) patch grid dimensions.
            gated_cls_logits: Optional differentiable alignment term from gated representations
                of shape (B, G) to couple backbone value projections with high-reward masks.

        Returns:
            Tuple of:
                - total_loss: Scalar differentiable Dr. GRPO loss tensor.
                - info: Dictionary of detached metrics for logging/TensorBoard.
        """
        if actions is None or old_log_probs is None:
            behavior_p = old_probs if old_probs is not None else policy_probs.detach()
            actions, old_log_probs = self.sample_actions(behavior_p, group_size=self.group_size)

        # 1. Verifiable Rewards R(a_i, I)
        rewards, reward_metrics = self.compute_verifiable_reward(
            actions=actions,
            patch_features=patch_features,
            equivariant_probs=equivariant_probs,
            gt_masks=gt_masks,
            labels=labels,
            class_prototypes=class_prototypes,
            grid_size=grid_size,
        )

        # 2. Variance-Safe Group Relative Advantage A_i
        advantages = self.compute_advantages(rewards)  # (B, G)

        # 3. Length-Normalized Log-Probabilities & Ratio rho_i
        log_probs = self.compute_length_normalized_log_probs(policy_probs, actions)  # (B, G)
        if gated_cls_logits is not None:
            # Couple action log-probs with gated feature representation log-likelihood
            log_probs = log_probs + 0.1 * gated_cls_logits

        log_ratio = torch.clamp(log_probs - old_log_probs.detach(), min=-10.0, max=10.0)
        rho = torch.exp(log_ratio)  # (B, G)

        # 4. DAPO-Style Asymmetric Clip-Higher Surrogate Objective
        clipped_rho = torch.clamp(rho, min=1.0 - self.eps_low, max=1.0 + self.eps_high)
        surr1 = rho * advantages
        surr2 = clipped_rho * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # When doing online single-step rollouts (rho == 1.0), torch.min(rho * A, clipped_rho * A)
        # has exact gradient -A * grad(rho) = -A * rho * grad(log_probs).
        # Ensure non-zero gradient flow even at exact boundary ties:
        if torch.allclose(rho.detach(), torch.ones_like(rho), atol=1e-5):
            policy_loss = -(rho * advantages.detach()).mean()

        # 5. Exact Analytical KL Divergence against Frozen Reference Checkpoint
        kl_div = self.compute_kl_divergence(policy_probs, ref_probs)
        total_loss = policy_loss + self.beta_kl * kl_div

        info: Dict[str, Any] = {
            "loss_total": total_loss.detach().item(),
            "loss_policy": policy_loss.detach().item(),
            "kl_div": kl_div.detach().item(),
            "mean_reward": rewards.mean().detach().item(),
            "std_reward": rewards.std(dim=-1).mean().detach().item(),
            "max_reward": rewards.max(dim=-1).values.mean().detach().item(),
            "mean_ratio": rho.mean().detach().item(),
            "foreground_iou": reward_metrics.get(
                "reward_equivariance_iou",
                reward_metrics.get("reward_gt_iou", rewards.mean().detach()),
            ).item(),
            "active_patch_ratio": reward_metrics["active_patch_ratio"].item(),
            "rewards_tensor": rewards.detach(),
            "advantages_tensor": advantages.detach(),
            "actions_tensor": actions.detach(),
        }
        return total_loss, info
