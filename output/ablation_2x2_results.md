# 2x2 Ablation Study Results: Architecture (RoPE + Sparse) x Post-Training (Dr. GRPO)

| Model Variant | Backbone Architecture | Stage-2 RLVR | k-NN Top-1 (%) | k-NN Top-5 (%) | Linear Probe Top-1 (%) | Foreground IoU (%) | CorLoc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla DINOv1 | `vit_small` | No | 43.00 | 90.00 | 49.00 | 48.63 | 49.00 |
| RoPE+Sparse DINOv1 | `vit_small_sparse_rope` | No | 44.00 | 87.00 | 52.00 | 44.34 | 33.00 |
| Vanilla+Dr.GRPO | `vit_small` | Dr. GRPO | 53.00 | 90.00 | 55.00 | 52.58 | 56.00 |
| **RoPE+Sparse+Dr.GRPO (Ours)** | `vit_small_sparse_rope` | Dr. GRPO | **60.00** | 93.00 | **60.00** | **46.58** | **40.00** |

## Key Defense Takeaways
- **Dr. GRPO Gain on Vanilla DINOv1**: `+10.00%` k-NN Top-1, `+3.95%` Foreground Mask IoU.
- **Dr. GRPO Gain on RoPE + Sparse DINOv1**: `+16.00%` k-NN Top-1, `+2.24%` Foreground Mask IoU.
- **Full Proposed Method (`RoPE+Sparse+Dr.GRPO`) vs. Vanilla Baseline**: `+17.00%` k-NN Top-1, `+11.00%` Linear Probe, `-2.05%` Foreground IoU.
