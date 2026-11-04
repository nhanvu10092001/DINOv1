# Verification Is All You Need: Emergent Object Discovery in Self-Supervised Vision Transformers via Dr. GRPO

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Paper](https://img.shields.io/badge/Paper-dr__dino.pdf-darkred.svg)](papers/dr_dino.pdf)
[![Slides](https://img.shields.io/badge/Slides-Defense_Presentation.pdf-navy.svg)](slides/defense_presentation.pdf)

> **Dr. DINO** (**D**istillation with no labels **R**einforced via **Dr**.\ GRPO) is a unified self-supervised Vision Transformer framework extending the knowledge distillation paradigm of **DINOv1** (*Emerging Properties in Self-Supervised Vision Transformers* — Caron et al., Meta AI, ICCV 2021).  
> The framework unifies continuous relative geometry via **Frequency-Adaptive 2D Rotary Position Embeddings (2D RoPE)**, computation-pruned routing via **Spatial-Semantic Sparse Softmax Attention (S³ Attention)**, and emergent object contour discovery via **Stage-2 Unsupervised Reinforcement Learning with Verifiable Rewards (Dr. GRPO)** under a decoupled, industrial **Modular Registry & Builder** design.
> 
> *Authors: Pham Cung Le Nhan Vu, Nguyen Hoang Khang, Nguyen Hoai Nam (Master of Science in Data Science, VNUHCM -- University of Science).*

---

## 📌 Executive Summary & Key Innovations

Vision Transformers (ViTs) pre-trained with self-distillation natively exhibit semantic clustering and scene decomposition. However, canonical ViTs suffer from geometric distortion under multi-crop scaling and quadratic attention dilution over background clutter. This framework introduces three synergistic remedies:

```
                          [ Input Image (224x224 / 96x96) ]
                                          |
                        [ Patch Embedding (16x16 Convolution) ]
                                          |
    +-------------------------------------+-------------------------------------+
    |                                                                           |
    v                                                                           v
[ Innovation 1: Frequency-Adaptive 2D RoPE ]              [ Innovation 2: Spatial-Semantic Sparse Attn ]
- Continuous 2D relative displacement encoding            - Active key set: 26 tokens (86.8% reduction)
- High-freq: Raw coordinates (sharp local contours)       - 3x3 Local Spatial Window (Chebyshev R=1)
- Low-freq: Normalized coordinates (scale invariance)     - Semantic Top-16 keys via cosine affinity
- Preserved unrotated [CLS] token (theta = 0)             - Bidirectional [CLS] global communication
    |                                                                           |
    +-------------------------------------+-------------------------------------+
                                          |
                     [ 12-Block Vision Transformer Backbone ]
                                          |
                +-------------------------+-------------------------+
                | Stage 1: SSL Pretraining                          | Stage 2: RLVR Post-Training
                v                                                   v
   [ DINO Head & Centering/Sharpening ]              [ Innovation 3: Verifiable Dr. GRPO Policy ]
   - Multi-Crop Student-Teacher EMA Distillation     - Shared-prefix wrapper (only top-2 blocks unlocked)
   - Cross-Entropy against ImageNet-100 views         - Tri-objective verifiable rewards:
                                                       * Geometric Equivariance IoU (w=0.45)
                                                       * TokenCut Spectral & Spatial Compactness (w=0.35)
                                                       * Target Sparsity Ratio (w=0.20)
                                                     - Unbiased advantage & DAPO-style asymmetric clipping
```

1. **Frequency-Adaptive 2D Rotary Position Embedding (2D RoPE)**:
   - Eliminates learnable Absolute Positional Embedding (APE) tables warped by bicubic interpolation during multi-crop training ($224^2 \leftrightarrow 96^2$).
   - Decomposes head dimensions across spatial axes ($d_x = d_y = 32$) with 16 frequency bands $\omega_m$. Tensors Query and Key are rotated directly by orthogonal matrices $\mathbf{R}_\Theta$, naturally encoding continuous relative shifts $(\Delta x, \Delta y)$.
   - Frequency coordinate blending: High frequencies retain raw integer coordinates ($p_{\text{raw}}$) to capture fine-grained edges; low frequencies scale continuously ($p_{\text{norm}}$) relative to a $224 \times 224$ reference grid for scale invariance.
   - Preserves global semantic representation by leaving the `[CLS]` token unrotated ($\theta_{\text{cls}} = \mathbf{0} \Rightarrow \mathbf{R} = \mathbf{I}$).

2. **Spatial-Semantic Sparse Softmax Attention**:
   - Reduces active keys from 197 to **26 active tokens per query** (**86.8% reduction** in attention pairs):
     - **Local Window $3 \times 3$**: 9 spatial neighbors (Chebyshev radius $R=1$).
     - **Semantic Top-16**: 16 distant keys with the highest dot-product similarity outside the local window.
     - **Bidirectional `[CLS]` routing**: Global-to-local and local-to-global token communication.
   - **Strict Softmax Normalization**: Computes exact Softmax over the active candidate set $\mathcal{S}_i$, ensuring valid probability distributions and stable gradients.
   - **Zero Parameter Overhead**: Exactly preserves 21,665,664 backbone parameters and 9.20 theoretical GFLOPs.

3. **Stage-2 Unsupervised Reinforcement Learning via Dr. GRPO**:
   - **Shared-Prefix Policy Wrapper (`DINOGRPOPolicyWrapper`)**: Freezes the first 10 Transformer blocks and unlocks only the final 2 blocks, LayerNorm, and a lightweight gating head ($384 \to 96 \to 1$), training only **8.27% of parameters** (3.59M / 43.37M) with **< 3GB VRAM footprint**.
   - **Verifiable Visual Rewards (No Human Labels, No Neural Reward Model)**:
     - *Geometric Equivariance IoU* ($w_{\text{eq}} = 0.45$): Invariance under horizontal reflections $T^{-1}(\pi_\theta(T(I)))$.
     - *TokenCut Spectral & Spatial Compactness* ($w_{\text{comp}} = 0.35$): Normalized-Cut foreground affinity minus background cut, regularized by 2D Total Variation smoothness.
     - *Target Sparsity Prior* ($w_{\text{spar}} = 0.20$): Penalizes background leakage, targeting $\rho_{\text{target}} = 0.35$.
   - **Unbiased Group Relative Advantage**: Mean-centered advantage without variance inflation ($A^{(g)} = R^{(g)} - \bar{R}$), asymmetric DAPO-style clipping ($\epsilon_{\text{low}}=0.20, \epsilon_{\text{high}}=0.28$), and analytical Bernoulli KL regularization ($\beta_{\text{KL}}=0.04$) against a frozen Stage-1 reference model.

4. **Modular Architecture & Software Engineering**:
   - Industrial Registry pattern (`core/registry.py`) and declarative builder (`core/builder.py`) managing `BACKBONES`, `HEADS`, `MODELS`, `LOSSES`, `TRANSFORMS`, `OPTIMIZERS`, and `SCHEDULERS`.
   - Standalone evaluation tools with feature caching, comprehensive unit tests, and automatic mixed precision (AMP) enabling training on consumer GPUs (e.g., RTX 3070 8GB).

---

## 📊 Benchmark & Empirical Evaluation

Both baseline and proposed backbones were pre-trained for **190 epochs** on **ImageNet-100** using an **NVIDIA H100 GPU** and evaluated downstream on the complete **CIFAR-10** benchmark (50,000 train / 10,000 test images).

### 1. Comprehensive $2 \times 2$ Ablation Study (Architecture $\times$ Verifiable Policy)

| Model Variant | Backbone Architecture | Dr. GRPO Policy | $k$-NN Top-1 (%) | $k$-NN Top-5 (%) | Linear Probe Top-1 (%) | Foreground Mask IoU (%) | CorLoc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1. Vanilla DINOv1 | `vit_small` | No | 71.13% | 96.57% | 78.07% | 58.88% | 79.93% |
| 2. RoPE + Sparse DINOv1 | `vit_small_sparse_rope` | No | 71.83% | 96.98% | 80.22% | 55.45% | 69.26% |
| 3. Vanilla + Dr. GRPO | `vit_small` | Yes | 71.29% | 96.73% | 78.44% | **78.98%** | **99.91%** |
| **4. RoPE+Sparse + Dr. GRPO (Full Proposed)** | `vit_small_sparse_rope` | **Yes** | **72.05%** | **97.12%** | **80.67%** | **72.44%** | **98.17%** |

#### Key Empirical Insights:
- **Architecture Gains (Stage 1)**: 2D RoPE + Sparse Attention substantially improves linear separability and clustering, boosting Linear Probe by **+2.15%** ($78.07\% \to 80.22\%$) and $k$-NN Top-1 by **+0.70%**.
- **Policy Optimization Gains (Stage 2)**: Dr. GRPO dramatically improves unsupervised spatial localization, boosting Foreground Mask IoU by **+16.99% to +20.10%** and reaching **98.17%–99.91% CorLoc** (Correct Localization @ $\text{IoU} \ge 0.50$).
- **Overall Synergy**: The combined framework (**Variant 4**) achieves across-the-board superiority over Vanilla DINOv1: **+0.92%** $k$-NN Top-1, **+2.60%** Linear Probing, **+13.56%** Foreground IoU, and **+18.24%** CorLoc.

### 2. Standardized 190-Epoch CIFAR-10 Transfer Benchmark

| Evaluation Protocol | Metric | Vanilla ViT-Small | RoPE + Sparse ViT-Small | Improvement ($\Delta$) |
|---|:---:|:---:|:---:|:---:|
| **$k$-Nearest Neighbors ($k=10$)** | Top-1 Accuracy | 74.41% | **75.97%** | <span style="color:#00aa00">**+1.56%**</span> |
| **$k$-Nearest Neighbors ($k=10$)** | Top-5 Accuracy | 96.23% | **96.55%** | <span style="color:#00aa00">**+0.32%**</span> |
| **$k$-Nearest Neighbors ($k=20$)** | Top-1 Accuracy | 74.34% | **75.31%** | <span style="color:#00aa00">**+0.97%**</span> |
| **Linear Probing (Epoch 10)** | Top-1 Accuracy | 73.76% | **79.86%** | <span style="color:#00aa00">**+6.10%** (Fast Convergence)</span> |
| **Linear Probing (Epoch 30)** | Top-1 Accuracy | 80.85% | **83.63%** | <span style="color:#00aa00">**+2.78%**</span> |
| **Linear Probing (Best Checkpoint)** | Top-1 Accuracy | 81.46% | **83.98%** | <span style="color:#00aa00">**+2.52%**</span> |
| **Linear Probing (Best Checkpoint)** | Top-5 Accuracy | 99.31% | **99.44%** | <span style="color:#00aa00">**+0.13%**</span> |

### 3. Computational Complexity & Hardware Profiling

*Measured on an NVIDIA GeForce RTX 3070 8GB VRAM at $224 \times 224$ resolution:*

| Metric | Vanilla ViT-Small | RoPE + Sparse ViT-Small | Architectural Note |
|---|:---:|:---:|---|
| **Backbone Parameters** | **21,665,664** (~21.67M) | **21,665,664** (~21.67M) | Zero parameter increase |
| **Trainable Parameters** | 21,665,664 (100%) | 21,665,664 (100%) | Exactly identical |
| **DINO Projection Head Parameters** | 9,720,064 (~9.72M) | 9,720,064 (~9.72M) | Identical MLP projection |
| **Backbone FLOPs ($224 \times 224$)** | **9.20 GFLOPs** (4.60 GMACs) | **9.20 GFLOPs** (4.60 GMACs) | Measured via PyTorch FlopCounterMode |
| **Inference Latency** | 1.20 ms / image | 2.20 ms / image | Vector rotation arithmetic |
| **Inference Throughput** | 831.9 images/sec | 455.0 images/sec | PyTorch tensor indexing |

### 4. Qualitative Self-Attention Visualizations

| Class: Airplane (*output/attention_comparison.png*) | Class: Horse (*output/attention_comparison_horse.png*) |
| :---: | :---: |
| ![Airplane](output/attention_comparison.png) | ![Horse](output/attention_comparison_horse.png) |
| *Vanilla ViT disperses attention into sky background.*<br>**RoPE+Sparse sharply segments fuselage, wings, and rudder.** | *Vanilla ViT blurs boundaries into pasture turf.*<br>**RoPE+Sparse isolates dorsal curve, head, and limbs.** |

---

## 📂 Repository Layout

```
DINOv1/
├── configs/                   # Declarative YAML experiment configurations
│   ├── dry_run.yaml               # Rapid pipeline test with synthetic data
│   ├── vit_small_modular.yaml     # Standard ViT-Small modular config
│   ├── vit_small_200k.yaml        # Long-schedule ViT-Small config
│   └── vit_small_sparse_rope.yaml # ViT-Small with 2D RoPE & Sparse Attention
├── core/                      # Core architectural foundation
│   ├── registry.py                # Decorator-based module registry system
│   └── builder.py                 # Dynamic model, loss, and scheduler builder
├── dino_datasets/             # Data loading and augmentations
│   ├── transforms.py              # Multi-crop augmentation (2 global + N local)
│   └── prepare_subset.py          # Hugging Face ImageNet subset downloader
├── eval/                      # Evaluation suite and visualization
│   ├── eval_utils.py              # Shared model loader and evaluation dataloaders
│   ├── eval_knn.py                # k-Nearest Neighbors classifier
│   ├── eval_linear.py             # Linear Probing classifier with feature caching
│   ├── eval_ablation_2x2.py       # Full 2x2 ablation benchmark and attention grid
│   └── visualize_attention.py     # Multi-head [CLS] attention map renderer
├── losses/                    # Self-supervised loss functions
│   ├── dino_loss.py               # DINOLoss with Centering and Sharpening
│   └── dr_grpo_loss.py            # DrGRPOLoss with verifiable visual rewards
├── models/                    # Neural network components
│   ├── vision_transformer.py      # Standard & enhanced Vision Transformer backbones
│   ├── rope.py                    # FrequencyAdaptive2DRoPE implementation
│   ├── sparse_attention.py        # SpatialSemanticAttention implementation
│   ├── dino_head.py               # DINOHead (MLP + L2 norm + WeightNorm Linear)
│   ├── multi_crop_wrapper.py      # Resolution-grouped batch forwarding wrapper
│   └── grpo_policy_wrapper.py     # DINOGRPOPolicyWrapper for Stage-2 RLVR
├── utils/                     # Utility modules
│   ├── logging_utils.py           # Structured logger and FallbackSummaryWriter
│   ├── saliency.py                # Unsupervised TokenCut & spatial pseudo-mask generator
│   ├── metrics.py                 # AverageMeter for tracking statistics
│   └── schedulers.py              # Cosine annealing, LR, and EMA schedulers
├── tests/                     # Comprehensive automated unit test suite
├── slides/                    # Defense and presentation slides (defense_presentation.tex & .pdf)
├── papers/                    # Research paper (dr_dino.tex & compiled dr_dino.pdf)
├── train.py                   # Stage-1 DINO pre-training entry point
├── train_grpo_post.py         # Stage-2 RLVR Dr. GRPO post-training entry point
├── pyproject.toml             # Standard PEP 517/518 packaging configuration
├── requirements.txt           # Python dependency specification
└── report.md                  # Comprehensive empirical research report
```

---

## 🚀 Getting Started & Usage Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/DINOv1.git
cd DINOv1

# Install core dependencies
pip install -r requirements.txt

# Install repository in editable mode
pip install -e .
```

### 2. Dataset Preparation

```bash
# Download and prepare ImageNet-100 from Hugging Face Hub
python dino_datasets/prepare_subset.py --dataset clane9/imagenet-100 --output_dir data

# Download CIFAR-10 for downstream evaluation
python utils/download_cifar10.py
```

### 3. Stage-1 Self-Supervised Pre-Training

```bash
# Option 1: Quick verification dry-run
python train.py --config configs/dry_run.yaml

# Option 2: Train baseline ViT-Small via Registry builder
python train.py --config configs/vit_small_modular.yaml

# Option 3: Train ViT-Small with Frequency-Adaptive 2D RoPE & Sparse Attention
python train.py --config configs/vit_small_sparse_rope.yaml
```

*Memory Optimization:* Uses Automatic Mixed Precision (`torch.cuda.amp.autocast`) with Gradient Accumulation (accum=4, batch_per_gpu=32 $\to$ effective batch size 128) to run stably on 8GB VRAM GPUs.

### 4. Stage-2 Unsupervised RLVR (Dr. GRPO) Post-Training

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

### 5. Downstream Evaluation

#### k-Nearest Neighbors ($k$-NN) Evaluation
```bash
python eval/eval_knn.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights cp/checkpoints_rope_2/checkpoint0190.pth \
    --arch vit_small_sparse_rope \
    --k 20
```

#### Linear Probing Evaluation
```bash
python eval/eval_linear.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights cp/checkpoints_rope_2/checkpoint0190.pth \
    --arch vit_small_sparse_rope \
    --epochs 30 \
    --lr 0.1 \
    --cache_features
```

#### Full $2 \times 2$ Ablation Benchmark
```bash
python eval/eval_ablation_2x2.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --batch_size 128 \
    --max_train_samples 0 \
    --max_val_samples 0
```

#### Self-Attention Map Visualization
```bash
python eval/visualize_attention.py \
    --image_path output/sample_airplane.jpg \
    --pretrained_weights cp/checkpoints_rope_2/checkpoint0190.pth \
    --arch vit_small_sparse_rope \
    --output_dir output
```

---

## 📚 References & Citation

```bibtex
@article{dino2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}

@article{su2024roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Ahmed, Murtadha and Lu, Yu and Pan, Shengfeng and Bo, Wen and Liu, Yunfeng},
  journal={Neurocomputing},
  volume={568},
  pages={127063},
  year={2024}
}

@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Zhang, Mingchuan and Zhang, YK and Wu, Y and Guo, Daya},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```

---

## 📄 License

This repository is distributed under the [Apache 2.0 License](LICENSE).
