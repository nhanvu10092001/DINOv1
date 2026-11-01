# 2x2 Ablation Study Results: Architecture (RoPE + Sparse) x Post-Training (Dr. GRPO)

| Model Variant | Backbone Architecture | Stage-2 RLVR | k-NN Top-1 (%) | k-NN Top-5 (%) | Linear Probe Top-1 (%) | Foreground IoU (%) | CorLoc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla DINOv1 | `vit_small` | No | 71.13 | 96.57 | 78.07 | 58.88 | 79.93 |
| RoPE+Sparse DINOv1 | `vit_small_sparse_rope` | No | 71.83 | 96.98 | 80.22 | 55.45 | 69.26 |
| Vanilla+Dr.GRPO | `vit_small` | Dr. GRPO | 71.29 | 96.73 | 78.44 | 78.98 | 99.91 |
| **RoPE+Sparse+Dr.GRPO (Ours)** | `vit_small_sparse_rope` | Dr. GRPO | **72.05** | **97.12** | **80.67** | **72.44** | **98.17** |

## Key Defense Takeaways
- **Dr. GRPO Gain on Vanilla DINOv1**: `+0.16%` k-NN Top-1, `+0.37%` Linear Probe, `+20.10%` Foreground Mask IoU (`58.88% -> 78.98%`), `+19.98%` CorLoc (`79.93% -> 99.91%`).
- **Dr. GRPO Gain on RoPE + Sparse DINOv1**: `+0.22%` k-NN Top-1, `+0.45%` Linear Probe, `+16.99%` Foreground Mask IoU (`55.45% -> 72.44%`), `+28.91%` CorLoc (`69.26% -> 98.17%`).
- **Full Proposed Method (`RoPE+Sparse+Dr.GRPO`) vs. Vanilla Baseline**: `+0.92%` k-NN Top-1, `+2.60%` Linear Probe, `+13.56%` Foreground IoU, `+18.24%` CorLoc.
