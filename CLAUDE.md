# CLAUDE.md — DINOv1 Self-Supervised Learning Framework

Comprehensive technical reference for the **DINOv1** repository (*Self-distillation with no labels* — Caron et al., Meta AI, ICCV 2021) augmented with a Modular Component Registry, Frequency-Adaptive 2D RoPE, Spatial-Semantic Sparse Attention, and Stage-2 RLVR Dr. GRPO.

---

## 1. Project Overview

This repository is a modular PyTorch research implementation of the self-supervised representation learning method **DINO** (Vision Transformer distillation without labels). It is engineered for single-GPU and workstation environments (e.g., RTX 3070 8GB VRAM) leveraging PyTorch Automatic Mixed Precision (AMP) and Gradient Accumulation.

### Core Architectural & Algorithmic Innovations:
1. **Modular Registry & Declarative Builder System**:
   - Component management via Decorator pattern (`@REGISTRY.register()`).
   - Global registries: `MODULES`, `BACKBONES`, `HEADS`, `MODELS`, `LOSSES`, `TRANSFORMS`, `OPTIMIZERS`, `SCHEDULERS`.
   - Complete pipeline instantiation (backbone, projection head, loss, optimizer, scheduler) driven by declarative YAML configurations.
2. **Frequency-Adaptive 2D Rotary Position Embedding (2D RoPE)**:
   - Extends RoPE to two-dimensional visual grids with frequency-aware coordinate adaptation.
   - Frequency interpolation: High frequencies (local details) preserve raw integer coordinates (`raw`), while low frequencies (global topology) scale continuously (`normalized`) relative to a reference resolution ($224 \times 224$).
   - Supported modes: `adaptive_2d`, `standard_2d`, `normalized_2d`, `learnable_2d`.
3. **Spatial-Semantic Sparse Softmax Attention**:
   - Retains exact Softmax probability normalization across 26 active tokens (86.8% reduction in attention pairs):
     - **Local Spatial Window**: 9 spatial neighbors in a $3 \times 3$ window (Chebyshev radius $R=1$).
     - **Global Semantic Top-$K$**: 16 keys with highest dot-product similarity outside the local window.
     - **Bidirectional `[CLS]` Token Routing**: Unrestricted two-way communication between `[CLS]` and all patch tokens.
4. **Stage-2 Unsupervised RLVR via Dr. GRPO**:
   - Shared-prefix policy wrapper (`DINOGRPOPolicyWrapper`) freezing the first 10 ViT blocks and unlocking only the final 2 blocks, LayerNorm, and a residual patch gating head (8.27% parameters, < 3GB VRAM).
   - Verifiable tri-objective visual rewards: horizontal-flip Geometric Equivariance IoU, spectral TokenCut foreground compactness, and target sparsity prior ($\rho_{\text{target}}=0.35$).
   - Unbiased group-relative advantage estimation, DAPO-style asymmetric clipping, and analytical Bernoulli KL regularization against a frozen Stage-1 reference checkpoint.

---

## 2. Directory Structure

```
DINOv1/
├── configs/                   # Declarative YAML training configurations
│   ├── dry_run.yaml               # Rapid pipeline test (2 epochs, synthetic data fallback)
│   ├── vit_small_modular.yaml     # Standard ViT-Small with Registry builder
│   ├── vit_small_200k.yaml        # Long-schedule ViT-Small configuration
│   └── vit_small_sparse_rope.yaml # ViT-Small integrating 2D RoPE + Sparse Attention
├── core/                      # Core engine and component registry
│   ├── __init__.py                # Exports Registry and Builder
│   ├── registry.py                # Registry class managing string-to-class mappings
│   └── builder.py                 # Factory methods: build_model, build_loss, build_optimizer, build_schedulers
├── dino_datasets/             # Data loading and augmentations
│   ├── __init__.py
│   ├── transforms.py              # DataAugmentationDINO (Multi-crop: 2 global + N local views)
│   └── prepare_subset.py          # Hugging Face ImageNet subset downloader
├── eval/                      # Evaluation suite and visualization
│   ├── __init__.py
│   ├── eval_utils.py              # Unified evaluation model loader and dataset factory
│   ├── eval_linear.py             # Linear Probing evaluation with feature caching
│   ├── eval_knn.py                # k-Nearest Neighbors classifier evaluation
│   ├── eval_ablation_2x2.py       # Full 2x2 ablation benchmark (Architecture x Dr. GRPO)
│   └── visualize_attention.py     # Multi-head [CLS] self-attention map renderer
├── losses/                    # Self-supervised objective functions
│   ├── __init__.py
│   ├── dino_loss.py               # DINOLoss with Teacher Centering & Sharpening
│   └── dr_grpo_loss.py            # DrGRPOLoss with verifiable visual rewards
├── models/                    # Neural network architectures
│   ├── __init__.py
│   ├── vision_transformer.py      # Standard & enhanced Vision Transformers (vit_tiny, vit_small)
│   ├── rope.py                    # FrequencyAdaptive2DRoPE module
│   ├── sparse_attention.py        # SpatialSemanticAttention & MultiScale2DWindowAttention
│   ├── dino_head.py               # DINOHead (MLP + L2 norm + WeightNorm Linear)
│   ├── multi_crop_wrapper.py      # MultiCropWrapper grouping crops by resolution
│   └── grpo_policy_wrapper.py     # DINOGRPOPolicyWrapper for Stage-2 RLVR
├── utils/                     # Computation and scheduling utilities
│   ├── __init__.py
│   ├── logging_utils.py           # Structured logger and FallbackSummaryWriter
│   ├── saliency.py                # TokenCut spectral & spatial pseudo-mask generator
│   ├── schedulers.py              # Cosine scheduler, LRScheduler, get_params_groups
│   └── metrics.py                 # AverageMeter for loss and timing tracking
├── tests/                     # Automated unit test suite (pytest)
├── papers/                    # Research paper drafts (LaTeX & compiled PDF)
├── slides/                    # Defense and presentation slides (LaTeX Beamer)
├── data/                      # Local dataset storage (train / val)
├── train.py                   # Stage-1 DINO pre-training execution script
├── train_grpo_post.py         # Stage-2 RLVR Dr. GRPO post-training execution script
├── pyproject.toml             # Standard PEP 517/518 package specification
├── requirements.txt           # Python dependency specification
├── pytest.ini                 # Pytest test discovery configuration
└── LICENSE                    # Apache 2.0 Open Source License
```

---

## 3. Component Deep Dive

### 3.1. Core: Registry & Builder (`core/`)

- **`Registry` (`core/registry.py`)**:
  - Maps string identifiers (`type`) to Python classes or functions.
  - Supports decorator usage: `@REGISTRY.register()`, `@REGISTRY.register("custom_name")`, or direct registration `REGISTRY.register(name, cls)`.
  - Method `build(cfg, **default_args)`: Extracts the `type` key from dictionary `cfg` and instantiates the target object.
  - Global registries:
    - `MODULES`: Fundamental blocks (`Block`, `PatchEmbed`, `Attention`, `SpatialSemanticAttention`, `FrequencyAdaptive2DRoPE`).
    - `BACKBONES`: Feature extractors (`VisionTransformer`, `vit_tiny`, `vit_small`, `vit_small_sparse_rope`).
    - `HEADS`: Projection heads (`DINOHead`).
    - `MODELS`: Wrapper architectures (`MultiCropWrapper`, `DINOGRPOPolicyWrapper`).
    - `LOSSES`: Objectives (`DINOLoss`, `DrGRPOLoss`).
    - `TRANSFORMS`: Data augmentations (`DataAugmentationDINO`, `GaussianBlur`, `Solarization`).
    - `OPTIMIZERS`: PyTorch optimizers (`AdamW`, `SGD`, `Adam`).
    - `SCHEDULERS`: Learning rate and momentum schedulers.

- **`Builder` (`core/builder.py`)**:
  - `build_backbone(backbone_cfg)`: Constructs backbone networks from `BACKBONES`.
  - `build_head(head_cfg, in_dim)`: Constructs projection heads from `HEADS`, automatically injecting `in_dim`.
  - `build_model(cfg, device)`:
    1. Instantiates identical backbones for **Student** and **Teacher**.
    2. Resolves `embed_dim`.
    3. Builds projection heads for Student and Teacher.
    4. Encapsulates in `MultiCropWrapper`.
    5. Synchronizes Teacher weights with Student and freezes Teacher parameters (`requires_grad = False`).
  - `build_loss(cfg, device)`: Instantiates `DINOLoss`.
  - `build_optimizer(model, cfg)`: Segregates parameters into weight decay groups (excluding bias and LayerNorm parameters) via `get_params_groups`.
  - `build_schedulers(optimizer, cfg, niter_per_ep)`: Constructs Cosine schedules for Learning Rate (with warmup), Weight Decay, and Teacher Momentum EMA ($0.996 \to 1.0$).

---

### 3.2. Models (`models/`)

#### Vision Transformer (`models/vision_transformer.py`)
- Standard ViT implementation with modular plug-and-play attention and positional embeddings:
  - `PatchEmbed`: Projects $C \times H \times W$ images via $16 \times 16$ Convolution into 1D token sequences.
  - `Block`: LayerNorm $\to$ Self-Attention (or SpatialSemanticAttention) $\to$ Residual $\to$ LayerNorm $\to$ MLP $\to$ Residual.
  - `interpolate_pos_encoding`: 2D Bicubic interpolation for static positional tables when image resolutions vary ($96 \times 96 \leftrightarrow 224 \times 224$).
- Pre-registered variants:
  - `vit_tiny`: `embed_dim=192`, `depth=12`, `num_heads=3`.
  - `vit_small`: `embed_dim=384`, `depth=12`, `num_heads=6`.
  - `vit_tiny_sparse_rope`: ViT-Tiny with 2D RoPE and Sparse Attention.
  - `vit_small_sparse_rope`: ViT-Small with 2D RoPE and Sparse Attention.

#### Frequency-Adaptive 2D RoPE (`models/rope.py`)
- Rotates Query and Key vectors directly, encoding continuous 2D relative spatial displacements:
  $$\text{coordinate}_k = \alpha_k \cdot p_{\text{raw}} + (1 - \alpha_k) \cdot p_{\text{norm}}$$
- Properties:
  - High frequencies: $\alpha_k \approx 1$ preserves raw pixel coordinates for sharp local contours.
  - Low frequencies: $\alpha_k \approx 0$ normalizes coordinates to reference grid ($224 \times 224$), preserving scale invariance across multi-crop views.
- Operating modes: `adaptive_2d`, `standard_2d`, `normalized_2d`, `learnable_2d`.

#### Spatial-Semantic Sparse Attention (`models/sparse_attention.py`)
- Combines local spatial proximity with semantic Top-$K$ routing:
  1. **Local Window ($3 \times 3$)**: 9 spatial neighbors (Chebyshev radius $R=1$).
  2. **Semantic Top-$K$**: 16 keys with highest dot-product similarity outside the local window.
  3. **Bidirectional `[CLS]` Routing**: Full communication between `[CLS]` and spatial tokens.
- Maintains exact Softmax probability normalization over active keys $\mathcal{S}_i$.

#### DINO Head (`models/dino_head.py`)
- Projects features from backbone space (`embed_dim`) to high-dimensional latent prototype space (`out_dim = 16384` or $65536$).
- 3-layer MLP with Hidden Dimension 2048, Bottleneck 256, L2 normalization, and `weight_norm(..., norm_last_layer=True)`.

#### MultiCropWrapper (`models/multi_crop_wrapper.py`)
- Processes lists of multi-crop tensors (2 global $224 \times 224$ and 6 local $96 \times 96$).
- Groups crops by spatial resolution to execute batched forward passes, maximizing GPU throughput.

---

### 3.3. Losses (`losses/`)

#### DINOLoss (`losses/dino_loss.py`)
- Computes cross-entropy between Student and Teacher probability distributions:
  $$\mathcal{L}_{\text{DINO}} = - \sum_{i \in \{1, 2\}} \sum_{j \ne i} P_t(x_i) \log P_s(x_j)$$
- **Sharpening**: Student uses fixed temperature $\tau_s = 0.1$; Teacher temperature $\tau_t$ warms up linearly ($0.04 \to 0.04$ or $0.07$) to avoid representation collapse.
- **Centering (EMA Center Update)**: Subtracts moving average center vector $C$ from Teacher logits to prevent dominance by single dimensions:
  $$C \leftarrow m \cdot C + (1 - m) \cdot \frac{1}{B} \sum_{i=1}^B g_t(x_i)$$

#### DrGRPOLoss (`losses/dr_grpo_loss.py`)
- Unbiased / Rank-Normalized Group Relative Policy Optimization loss for Stage-2 RLVR.
- Optimizes verifiable visual rewards (Geometric Equivariance IoU, TokenCut spectral compactness, target sparsity prior) with DAPO asymmetric clipping and analytical Bernoulli KL regularization.

---

## 4. Workflows & Reproduction Commands

### 4.1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### 4.2. Automated Testing

All unit tests are located in `tests/`:

```bash
# Run complete test suite
pytest tests -v

# Run individual module tests
pytest tests/test_rope2d.py -v
pytest tests/test_sparse_attention.py -v
pytest tests/test_builder.py -v
pytest tests/test_dr_grpo_pipeline.py -v
```

### 4.3. Stage-1 Pre-Training

```bash
# Dry run verification
python train.py --config configs/dry_run.yaml

# Train standard ViT-Small
python train.py --config configs/vit_small_modular.yaml

# Train ViT-Small with Frequency-Adaptive 2D RoPE & Sparse Attention
python train.py --config configs/vit_small_sparse_rope.yaml
```

### 4.4. Stage-2 RLVR Dr. GRPO Post-Training

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

### 4.5. Evaluation Suite

#### $k$-Nearest Neighbors ($k$-NN) Evaluation
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
