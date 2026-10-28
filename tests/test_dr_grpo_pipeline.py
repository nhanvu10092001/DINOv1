"""Unit and integration tests for Stage-2 Dr. GRPO Loss and Policy Wrapper."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import LOSSES, MODULES
import losses  # noqa: F401
import models  # noqa: F401
from losses.dr_grpo_loss import DrGRPOLoss
from models.grpo_policy_wrapper import DINOGRPOPolicyWrapper


def test_registry_registration() -> None:
    """Verifies DrGRPOLoss and DINOGRPOPolicyWrapper are registered in MODULES."""
    assert "DrGRPOLoss" in MODULES
    assert "DrGRPOLoss" in LOSSES
    assert "DINOGRPOPolicyWrapper" in MODULES


def test_dr_grpo_loss_modes_and_gradients() -> None:
    """Tests both reward modes and advantage formulations of DrGRPOLoss."""
    B, N, D, G = 4, 196, 192, 8
    ref_probs = torch.full((B, N), 0.35)
    patch_features = torch.randn(B, N, D)
    equivariant_probs = torch.sigmoid(torch.randn(B, N))
    gt_masks = (torch.rand(B, N) > 0.6).float()
    labels = torch.tensor([0, 1, 0, 1])
    prototypes = torch.randn(3, D)

    for adv_type in ("dr_grpo", "rank_grpo"):
        for reward_mode in ("self_supervised_v1", "downstream_supervised"):
            loss_fn: DrGRPOLoss = MODULES.build({
                "type": "DrGRPOLoss",
                "group_size": G,
                "adv_type": adv_type,
                "reward_mode": reward_mode,
            })
            probs = torch.sigmoid(torch.randn(B, N)).requires_grad_(True)
            loss, info = loss_fn(
                policy_probs=probs,
                ref_probs=ref_probs,
                patch_features=patch_features,
                equivariant_probs=equivariant_probs,
                gt_masks=gt_masks,
                labels=labels,
                class_prototypes=prototypes,
                grid_size=(14, 14),
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert probs.grad is not None
            assert torch.isfinite(probs.grad).all()
            assert info["rewards_tensor"].shape == (B, G)
            assert info["advantages_tensor"].shape == (B, G)

            if adv_type == "dr_grpo":
                # Centered advantage sum across group must be ~0
                assert torch.allclose(
                    info["advantages_tensor"].mean(dim=-1),
                    torch.zeros(B),
                    atol=1e-5,
                )
            else:
                # Rank advantage bounded in [-1, 1]
                assert info["advantages_tensor"].min() >= -1.0 - 1e-5
                assert info["advantages_tensor"].max() <= 1.0 + 1e-5


def test_policy_wrapper_freeze_and_forward() -> None:
    """Verifies DINOGRPOPolicyWrapper unlocks only the last 2 blocks + gate."""
    wrapper: DINOGRPOPolicyWrapper = MODULES.build({
        "type": "DINOGRPOPolicyWrapper",
        "arch": "vit_tiny_sparse_rope",
        "patch_size": 16,
        "num_unfrozen_blocks": 2,
    })

    # Early blocks must be frozen; last 2 blocks + norm + policy_gate must be trainable
    for idx in range(10):
        for p in wrapper.backbone.blocks[idx].parameters():
            assert not p.requires_grad
    for idx in range(10, 12):
        for p in wrapper.backbone.blocks[idx].parameters():
            assert p.requires_grad
    for p in wrapper.ref_model.parameters():
        assert not p.requires_grad

    x = torch.randn(2, 3, 224, 224)
    out = wrapper.forward_policy_and_ref(x, compute_equivariant=True)
    assert out["policy_probs"].shape == (2, 196)
    assert out["ref_probs"].shape == (2, 196)
    assert out["equivariant_probs"].shape == (2, 196)
    assert out["cls_features"].shape == (2, 192)
    assert out["patch_features"].shape == (2, 196, 192)

    attn = wrapper.get_last_selfattention(x)
    assert attn.shape == (2, 3, 197, 197)


if __name__ == "__main__":
    test_registry_registration()
    print("✓ test_registry_registration passed")
    test_dr_grpo_loss_modes_and_gradients()
    print("✓ test_dr_grpo_loss_modes_and_gradients passed")
    test_policy_wrapper_freeze_and_forward()
    print("✓ test_policy_wrapper_freeze_and_forward passed")
    print("All Dr. GRPO pipeline tests passed successfully!")

