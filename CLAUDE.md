# CLAUDE.md — Dr. DINO Self-Supervised Learning & RLVR Framework

Comprehensive developer and AI-assistant reference for the **Dr. DINO** repository (*Verification Is All You Need: Emergent Object Discovery in Self-Supervised Vision Transformers via Dr. GRPO*), extending **DINOv1** (Caron et al., Meta AI, ICCV 2021) with a Modular Component Registry, Frequency-Adaptive 2D RoPE, Spatial-Semantic Sparse Attention ($S^3$), and Stage-2 Verifiable RLVR via Unbiased Dr. GRPO.

---

## 1. Project Overview

**Dr. DINO** is a modular PyTorch framework for self-supervised representation learning and label-free object discovery. It is engineered to run efficiently on single-GPU workstations (e.g., RTX 3070 8 GB / Tesla T4 16 GB VRAM) using PyTorch Automatic Mixed Precision (AMP), Gradient Accumulation, and Shared-Prefix RLVR execution.

### Core Architectural & Algorithmic Pillars:
1. **Modular Registry & Declarative Builder Engine (`core/`)**:
   - Decorator-based registration (`@REGISTRY.register()`) across `MODULES`, `BACKBONES`, `HEADS`, `MODELS`, `LOSSES`, `TRANSFORMS`, `OPTIMIZERS`, and `SCHEDULERS`.
   - Declarative YAML configuration driving full Stage-1 pre-training and evaluation pipelines.
2. **Frequency-Adaptive 2D Rotary Position Embedding (`models/rope.py`)**:
   - Partitions head dimension $d_k = 64$ into horizontal ($d_x = 32$) and vertical ($d_y = 32$) subspaces ($M = 16$ frequency bands per axis, $\omega_m = 10000^{-2m/d_x}$).
   - Spectral coordinate blending ($\alpha_m = 1 - \frac{m}{M-1}$): High frequencies ($m \to 0$) retain raw integer lattice steps $x_{\text{raw}}$ for sharp local edges, while low frequencies ($m \to M-1$) use scale-normalized coordinates $x_{\text{norm}} = x_{\text{raw}}(S_{\text{ref}}/W_g)$ for multi-crop scale invariance.
   - Leaves `[CLS]` unrotated ($\boldsymbol{\theta}_{\text{cls}} = \mathbf{0} \implies \mathbf{R}_{\Theta}(\text{cls}) = \mathbf{I}$).
   - Supported modes: `adaptive_2d` (default), `standard_2d`, `normalized_2d`, `learnable_2d`.
3. **Spatial-Semantic Sparse Softmax Attention ($S^3$, `models/sparse_attention.py`)**:
   - Prunes active attention keys per query from $197 \to \mathbf{26}$ (**86.8% reduction**, $13.2\%$ active density):
     - **Local Chebyshev Window $\mathcal{N}_{\text{local}}(i)$**: $3 \times 3$ neighborhood ($R=1$, 9 keys).
     - **Dynamic Semantic Top-$K$ $\mathcal{N}_{\text{semantic}}(i)$**: 16 highest dot-product keys outside the local window.
     - **Bidirectional `[CLS]` Conduit**: Global-to-local and local-to-global communication (1 key).
   - Enforces exact Softmax normalization over $\mathcal{S}_i$ by masking non-support logits to $-\infty$ prior to exponentiation.
4. **Stage-2 Verifiable RLVR via Unbiased Dr. GRPO (`models/grpo_policy_wrapper.py`, `losses/dr_grpo_loss.py`)**:
   - **Shared-Prefix Policy Wrapper (`DINOGRPOPolicyWrapper`)**: Freezes `patch_embed` and `blocks[0..9]` (`torch.no_grad()`), unlocking only `blocks[10..11]`, `norm`, and a residual MLP `policy_gate` ($384 \to 96 \to 1$), training **8.27%** of wrapper parameters ($3.59\text{M} / 43.37\text{M}$) in **< 3 GB VRAM**.
   - **6-Step Bernoulli Patch Policy**: Fuses per-image Z-scored `[CLS]` attention $\hat{a}_p$, learned residual gate logits $g_\phi(\mathbf{z}_p)$, TokenCut diffusion prior $\hat{s}_p$, and log-odds sparsity bias $b_{\text{sparsity}} = \log(0.35 / 0.65) \approx -0.619$.
   - **Tri-Objective Verifiable Rewards (`self_supervised_v1`)**:
     - $R_{\text{equiv}}$ ($w=0.45$): Horizontal-flip geometric equivariance IoU against $\operatorname{Top-}69\big(T^{-1}(\pi_\theta(T(I)))\big)$.
     - $R_{\text{compact}}$ ($w=0.35$): $0.65 R_{\text{ncut}} + 0.35 R_{\text{TV}}$ (TokenCut spectral normalized cut + 2D Total Variation smoothness).
     - $R_{\text{sparsity}}$ ($w=0.20$): Target foreground occupancy ratio prior peaking at $\rho_{\text{target}} = 0.35$.
   - **Surrogate Objective**: Unbiased mean-centered group advantage $A_i^{\text{Dr.GRPO}} = R^{(i)} - \bar{R}$, length-normalized importance ratio $\rho_i(\theta)$ across $N=196$ patches, DAPO asymmetric clipping ($\epsilon_{\text{low}}=0.20, \epsilon_{\text{high}}=0.28$), and analytical Bernoulli KL divergence ($\beta_{\text{KL}}=0.04$).

---

## 2. Repository Structure

```text
DINOv1/
├── configs/                   # Declarative YAML training configurations
│   ├── dry_run.yaml               # Rapid 2-epoch pipeline test with synthetic fallback
│   ├── vit_small_modular.yaml     # Baseline ViT-Small/16 configuration
│   ├── vit_small_200k.yaml        # Extended ViT-Small/16 schedule
│   └── vit_small_sparse_rope.yaml # Dr. DINO-Base (2D RoPE + S³ Sparse Attention)
├── core/                      # Core engine and component registry
│   ├── registry.py                # Registry class managing string-to-class mappings
│   └── builder.py                 # Factory methods: build_model, build_loss, build_optimizer, build_schedulers
├── dino_datasets/             # Data loading and multi-crop augmentations
│   ├── transforms.py              # DataAugmentationDINO (2 global 224² + 6 local 96² views)
│   └── prepare_subset.py          # Hugging Face ImageNet-100 downloader
├── eval/                      # Evaluation suite and visualization
│   ├── eval_utils.py              # Unified checkpoint loader and dataset factory
│   ├── eval_knn.py                # Weighted k-NN classifier evaluation
│   ├── eval_linear.py             # Linear Probing evaluation with feature caching
│   ├── eval_ablation_2x2.py       # Full 2x2 factorial ablation benchmark (k-NN, Linear, IoU, CorLoc)
│   └── visualize_attention.py     # Multi-head [CLS] self-attention map renderer
├── losses/                    # Self-supervised & RLVR objective functions
│   ├── dino_loss.py               # DINOLoss with Teacher Centering & Sharpening
│   └── dr_grpo_loss.py            # DrGRPOLoss with verifiable visual rewards & DAPO clipping
├── models/                    # Neural network architectures
│   ├── vision_transformer.py      # Standard & enhanced Vision Transformers (vit_tiny, vit_small)
│   ├── rope.py                    # FrequencyAdaptive2DRoPE module
│   ├── sparse_attention.py        # SpatialSemanticAttention (S³) & MultiScale2DWindowAttention
│   ├── dino_head.py               # DINOHead (MLP + L2 norm + WeightNorm Linear)
│   ├── multi_crop_wrapper.py      # MultiCropWrapper grouping crops by spatial resolution
│   └── grpo_policy_wrapper.py     # DINOGRPOPolicyWrapper for Stage-2 RLVR
├── notebooks/                 # Interactive research & visual walkthrough notebooks
│   ├── demo_01_stage1_rope_and_sparse_attention.ipynb
│   ├── demo_02_stage2_policy_rewards_and_dr_grpo.ipynb
│   ├── demo_03_end_to_end_2x2_ablation_showcase.ipynb
│   └── colab_grpo_master_pipeline.ipynb
├── output/                    # Exported empirical figures, attention grids, and JSON metrics
├── papers/                    # LaTeX source & compiled paper (dr_dino.tex, dr_dino.pdf)
├── slides/                    # LaTeX Beamer deck (defense_presentation.tex, defense_presentation.pdf)
├── tests/                     # Automated pytest suite
├── utils/                     # Schedulers, TokenCut spectral saliency, and logging utilities
├── train.py                   # Stage-1 DINO pre-training entry point
└── train_grpo_post.py         # Stage-2 RLVR Dr. GRPO post-training entry point
```

---

## 3. Essential Commands & Workflows

### 3.1. Environment & Unit Testing
```bash
# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .

# Run full automated test suite
pytest tests -v
```

### 3.2. Stage-1 Self-Supervised Pre-Training
```bash
# Dry run verification
python train.py --config configs/dry_run.yaml

# Train Dr. DINO-Base (2D RoPE + S³ Sparse Attention)
python train.py --config configs/vit_small_sparse_rope.yaml

# Train Vanilla DINOv1 Baseline
python train.py --config configs/vit_small_modular.yaml
```

### 3.3. Stage-2 RLVR Dr. GRPO Post-Training
```bash
python train_grpo_post.py \
    --train_data_path data/train \
    --eval_data_path data/cifar10 \
    --eval_dataset cifar10 \
    --variant both \
    --epochs 10 \
    --batch_size 64 \
    --group_size 8 \
    --adv_type dr_grpo \
    --reward_mode self_supervised_v1
```

### 3.4. Downstream Evaluation & $2 \times 2$ Ablation
```bash
# Full 2x2 Ablation Benchmark
python eval/eval_ablation_2x2.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --batch_size 128 \
    --max_train_samples 0 \
    --max_val_samples 0

# k-NN Classifier Evaluation
python eval/eval_knn.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights cp/checkpoints_rope_2/checkpoint0190.pth \
    --arch vit_small_sparse_rope \
    --k 10

# Linear Probing Evaluation with Feature Caching
python eval/eval_linear.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights cp/checkpoints_rope_2/checkpoint0190.pth \
    --arch vit_small_sparse_rope \
    --epochs 100 \
    --lr 0.1 \
    --cache_features
```
