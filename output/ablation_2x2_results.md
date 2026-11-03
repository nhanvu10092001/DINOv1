# 2x2 Ablation Study Results: Architecture (RoPE + Sparse) x Verifiable Policy Optimization (Dr. GRPO)

| Model Variant | Backbone Architecture | Dr. GRPO Policy | k-NN Top-1 (%) | k-NN Top-5 (%) | Linear Probe Top-1 (%) | Foreground IoU (%) | CorLoc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla DINOv1 | `vit_small` | No | 71.13 | 96.57 | 78.07 | 58.88 | 79.93 |
| RoPE+Sparse DINOv1 | `vit_small_sparse_rope` | No | 71.83 | 96.98 | 80.22 | 55.45 | 69.26 |
| Vanilla+Dr.GRPO | `vit_small` | Yes | 71.29 | 96.73 | 78.44 | 78.98 | 99.91 |
| **RoPE+Sparse+Dr.GRPO (Ours)** | `vit_small_sparse_rope` | **Yes** | **72.05** | 97.12 | **80.67** | **72.44** | **98.17** |

## Key Empirical Findings
- **Dr. GRPO Gain on Vanilla DINOv1**: `+0.16%` k-NN Top-1, `+0.37%` Linear Probe, `+20.10%` Foreground Mask IoU, `+19.98%` CorLoc.
- **Dr. GRPO Gain on RoPE + Sparse DINOv1**: `+0.22%` k-NN Top-1, `+0.45%` Linear Probe, `+16.99%` Foreground Mask IoU, `+28.91%` CorLoc.
- **Full Proposed Method (`RoPE+Sparse+Dr.GRPO`) vs. Vanilla Baseline**: `+0.92%` k-NN Top-1, `+2.60%` Linear Probe, `+13.56%` Foreground IoU, `+18.24%` CorLoc.
