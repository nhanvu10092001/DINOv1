# CLAUDE.md — DINOv1 Self-Supervised Learning Framework

Tài liệu kỹ thuật toàn diện cho repository **DINOv1** (Self-distillation with no labels - Caron et al., 2021) kết hợp hệ thống Registry module hóa, Frequency-Adaptive 2D RoPE, Spatial-Semantic Sparse Attention và YOLO-style DAG parser.

---

## 1. Tổng quan dự án (Project Overview)

Repository này là bản triển khai hoàn chỉnh bằng PyTorch của phương pháp tự học tự giám sát (Self-Supervised Learning) **DINO** (Vision Transformer distillation without labels), được tối ưu hóa cho huấn luyện trên GPU đơn/máy trạm (như RTX 3070 8GB VRAM) với Mixed Precision (AMP) và Gradient Accumulation.

### Các cải tiến kiến trúc cốt lõi:
1. **Modular Registry & Dynamic Builder System**:
   - Quản lý linh hoạt các thành phần qua Decorator pattern (`@REGISTRY.register()`).
   - Các registry toàn cục: `MODULES`, `BACKBONES`, `HEADS`, `MODELS`, `LOSSES`, `TRANSFORMS`, `OPTIMIZERS`, `SCHEDULERS`.
   - Khởi tạo toàn bộ pipeline (backbone, head, loss, optimizer, scheduler) thông qua tệp cấu hình YAML.
2. **Frequency-Adaptive 2D Rotary Position Embedding (RoPE)**:
   - RoPE 2 chiều không gian cho Vision Transformer.
   - Nội suy thích ứng tần số: tần số cao (chi tiết cục bộ) giữ toạ độ tuyệt đối (`raw`), tần số thấp (cấu trúc toàn cục) chuẩn hóa theo độ phân giải tham chiếu (`normalized`).
   - Hỗ trợ các chế độ: `adaptive_2d`, `standard_2d`, `normalized_2d`, `learnable_2d`.
3. **Spatial-Semantic Sparse Softmax Attention**:
   - Cơ chế chú ý thưa duy trì phân phối Softmax chuẩn xác:
     - **Local Spatial Window**: Kết nối lân cận trong cửa sổ $K \times K$.
     - **Global Semantic Top-K**: Kết nối $K$ token có độ tương đồng cosine lớn nhất nằm ngoài cửa sổ cục bộ.
     - **Bidirectional CLS Token Routing**: Token `[CLS]` luôn chú ý tới tất cả token và ngược lại.
4. **YOLO-style Model DAG Parser**:
   - Cho phép định nghĩa kiến trúc Vision Transformer dạng danh sách lớp `[from, repeats, module, args]` tương tự cú pháp YOLO.
   - Hỗ trợ phân nhánh, skip-connections, trích xuất token tự động.

---

## 2. Cấu trúc thư mục (Directory Structure)

```
DINOv1/
├── checkpoints/               # Thư mục lưu checkpoint mô hình (.pth)
├── configs/                   # Các tệp cấu hình YAML huấn luyện
│   ├── dry_run.yaml               # Cấu hình kiểm thử nhanh (2 epochs, synthetic data)
│   ├── vit_tiny_yolo.yaml         # ViT-Tiny định nghĩa qua YOLO-style layer DAG
│   ├── vit_small_modular.yaml     # ViT-Small chuẩn với Registry builder
│   ├── vit_small_200k.yaml        # ViT-Small huấn luyện chu kỳ dài
│   └── vit_small_sparse_rope.yaml # ViT-Small tích hợp 2D RoPE + Sparse Attention
├── core/                      # Hệ thống lõi (Core Engine)
│   ├── __init__.py                # Export Registry và Builder
│   ├── registry.py                # Lớp Registry quản lý định danh và khởi tạo
│   └── builder.py                 # Hàm dựng: build_model, build_loss, build_optimizer, build_schedulers
├── dino_datasets/             # Xử lý dữ liệu & Augmentations
│   ├── __init__.py
│   ├── transforms.py              # DataAugmentationDINO (Multi-crop: 2 global + N local)
│   └── prepare_subset.py          # Script tải ImageNet subset từ Hugging Face Hub
├── eval/                      # Các công cụ đánh giá & trực quan hóa
│   ├── __init__.py
│   ├── eval_linear.py             # Linear Probing evaluation
│   ├── eval_knn.py                # k-Nearest Neighbors classifier evaluation
│   └── visualize_attention.py     # Trực quan hóa Attention Map của [CLS] token
├── losses/                    # Hàm mất mát
│   ├── __init__.py
│   └── dino_loss.py               # DINOLoss với Teacher Centering & Sharpening
├── models/                    # Kiến trúc mạng nơ-ron
│   ├── __init__.py                # Export các mô hình và lớp
│   ├── vision_transformer.py      # VisionTransformer chuẩn, ViT-Tiny, ViT-Small
│   ├── rope.py                    # FrequencyAdaptive2DRoPE
│   ├── sparse_attention.py        # SpatialSemanticAttention
│   ├── dino_head.py               # DINOHead (MLP 3 lớp + L2 norm + WeightNorm Linear)
│   ├── multi_crop_wrapper.py      # MultiCropWrapper gom nhóm crop theo độ phân giải
│   ├── yolo_layers.py             # Các lớp phụ trợ YOLO: PosEmbedAndClassToken, ExtractToken, LayerNorm1d, Concat
│   └── yolo_parser.py             # Trình phân tích cú pháp parse_model_layers và YOLOModel
├── utils/                     # Tiện ích tính toán & lịch học
│   ├── __init__.py
│   ├── schedulers.py              # Cosine scheduler, LRScheduler, get_params_groups (weight decay exclude)
│   └── metrics.py                 # AverageMeter theo dõi loss và thời gian
├── tests/                     # Toàn bộ 40 unit test kiểm thử tự động
│   ├── test_builder.py
│   ├── test_components_registration.py
│   ├── test_configs.py
│   ├── test_multi_crop_wrapper.py
│   ├── test_registry.py
│   ├── test_rope2d.py
│   ├── test_sparse_attention.py
│   ├── test_sparse_rope_training.py
│   ├── test_vit_sparse_rope.py
│   └── test_yolo_parser.py
├── papers/                    # Tài liệu nghiên cứu gốc (dino_v1_2104.14294.pdf)
├── data/                      # Thư mục dữ liệu cục bộ (train / val)
├── train.py                   # Script thực thi huấn luyện DINO chính
├── requirements.txt           # Danh sách thư viện phụ thuộc
├── pytest.ini                 # Cấu hình kiểm thử pytest
└── LICENSE                    # Giấy phép nguồn mở Apache 2.0
```

---

## 3. Chi tiết các thành phần (Component Deep Dive)

### 3.1. Core: Registry & Builder (`core/`)

- **`Registry` (`core/registry.py`)**:
  - Quản lý bảng ánh xạ giữa chuỗi định danh (`type`) và lớp/hàm Python.
  - Hỗ trợ decorator: `@REGISTRY.register()`, `@REGISTRY.register("custom_name")`, hoặc gọi trực tiếp `REGISTRY.register(name, cls)`.
  - Phương thức `build(cfg, **default_args)`: Lấy khóa `type` trong từ điển `cfg` và khởi tạo đối tượng với các tham số tương ứng.
  - Các biến toàn cục:
    - `MODULES`: Chứa các block, layer cơ bản (`Block`, `PatchEmbed`, `Attention`, `SpatialSemanticAttention`, `FrequencyAdaptive2DRoPE`, `PosEmbedAndClassToken`, v.v.).
    - `BACKBONES`: Chứa các kiến trúc trích xuất đặc trưng (`VisionTransformer`, `vit_tiny`, `vit_small`, `vit_tiny_sparse_rope`, `vit_small_sparse_rope`, `YOLOModel`).
    - `HEADS`: Chứa các projection head (`DINOHead`).
    - `MODELS`: Chứa các model wrapper (`MultiCropWrapper`).
    - `LOSSES`: Chứa hàm mất mát (`DINOLoss`).
    - `TRANSFORMS`: Chứa các phép biến đổi tăng cường (`DataAugmentationDINO`, `GaussianBlur`, `Solarization`).
    - `OPTIMIZERS`: PyTorch optimizers (`AdamW`, `SGD`, `Adam`).
    - `SCHEDULERS`: Bộ lập lịch học tập.

- **`Builder` (`core/builder.py`)**:
  - `build_backbone(backbone_cfg)`: Xây dựng mạng backbone từ `BACKBONES`.
  - `build_head(head_cfg, in_dim)`: Xây dựng projection head từ `HEADS`, tự động tiêm `in_dim` nếu cần.
  - `build_model(cfg, device)`:
    1. Tạo 2 backbone giống nhau cho **Student** và **Teacher**.
    2. Xác định `embed_dim` (tự động suy diễn qua `embed_dim` hoặc `_infer_embed_dim()`).
    3. Tạo 2 projection head cho Student và Teacher.
    4. Bọc trong `MultiCropWrapper`.
    5. Sao chép trọng số từ Student sang Teacher và đóng băng gradient của Teacher (`requires_grad = False`).
  - `build_loss(cfg, device)`: Khởi tạo `DINOLoss`.
  - `build_optimizer(model, cfg)`: Tách trọng số thành 2 nhóm (áp dụng weight decay cho ma trận trọng số, không áp dụng cho bias và LayerNorm) qua `get_params_groups`.
  - `build_schedulers(optimizer, cfg, niter_per_ep)`: Tạo lịch Cosine annealing cho Learning Rate (có warmup), Weight Decay, và Teacher Momentum EMA ($0.996 \rightarrow 1.0$).

---

### 3.2. Models (`models/`)

#### Vision Transformer (`models/vision_transformer.py`)
- Thực thi tiêu chuẩn kiến trúc ViT:
  - `PatchEmbed`: Chiếu ảnh $C \times H \times W$ qua Convolution $16 \times 16$, phẳng hóa thành sequence.
  - `Block`: LayerNorm $\rightarrow$ Multi-Head Attention (hoặc SpatialSemanticAttention) $\rightarrow$ Residual $\rightarrow$ LayerNorm $\rightarrow$ MLP $\rightarrow$ Residual.
  - `interpolate_pos_encoding`: Nội suy 2D Bicubic cho positional embedding khi kích thước ảnh thay đổi (ví dụ ảnh local crop $96 \times 96$ so với $224 \times 224$).
- Các biến thể có sẵn:
  - `vit_tiny`: `embed_dim=192`, `depth=12`, `num_heads=3`.
  - `vit_small`: `embed_dim=384`, `depth=12`, `num_heads=6`.
  - `vit_tiny_sparse_rope`: ViT-Tiny tích hợp RoPE và Sparse Attention.
  - `vit_small_sparse_rope`: ViT-Small tích hợp RoPE và Sparse Attention.

#### Frequency-Adaptive 2D RoPE (`models/rope.py`)
- Định vị vị trí quay 2D thay thế hoặc kết hợp với positional embedding tĩnh:
  $$\text{coordinate}_k = \alpha_k \cdot p_{\text{raw}} + (1 - \alpha_k) \cdot p_{\text{norm}}$$
- Trong đó:
  - Tần số cao (dải đầu): $\alpha_k \approx 1$ giữ toạ độ pixel tuyệt đối, bảo toàn cấu trúc cạnh, hoa văn chi tiết cục bộ.
  - Tần số thấp (dải sau): $\alpha_k \approx 0$ chuẩn hóa theo tỷ lệ lưới tham chiếu ($224 \times 224$), giúp mô hình khái quát hóa khi độ phân giải đầu vào biến đổi.
- Các chế độ hoạt động (`mode`):
  - `adaptive_2d`: Tự động nội suy tuyến tính $\alpha_k$ từ $1.0 \rightarrow 0.0$ theo dải tần số.
  - `standard_2d`: $\alpha_k = 1.0$ (chỉ dùng toạ độ raw).
  - `normalized_2d`: $\alpha_k = 0.0$ (chỉ dùng toạ độ normalized).
  - `learnable_2d`: $\alpha_k$ là tham số học được (`nn.Parameter` thông qua logit reparameterization).

#### Spatial-Semantic Sparse Attention (`models/sparse_attention.py`)
- Giải quyết bài toán độ phức tạp tính toán nhưng vẫn duy trì đặc tính Softmax toàn vẹn:
  1. **Local Window ($K_{\text{local}} \times K_{\text{local}}$)**: Mỗi patch chú ý đến các patch lân cận trong bán kính không gian.
  2. **Semantic Top-K ($K_{\text{semantic}}$)**: Chọn ra $K$ token có dot-product lớn nhất ngoài vùng lân cận để kết nối thông tin ngữ nghĩa đường dài.
  3. **CLS Token Routing**: Luôn duy trì kết nối 2 chiều giữa token `[CLS]` và toàn bộ các token khác.
- Các chế độ (`sparse_mode`): `sparse` (kết hợp cả local + semantic), `local_only`, `semantic_only`, `dense`.

#### DINO Head (`models/dino_head.py`)
- Chiếu đặc trưng từ không gian ViT (`embed_dim`) sang không gian phân phối nhãn tiềm ẩn lớn (`out_dim = 16384` hoặc $65536$).
- Cấu trúc: MLP 3 lớp với Hidden Dimension 2048, Bottleneck 256, chuẩn hóa L2 normalization, và lớp Linear cuối cùng được ràng buộc bằng `weight_norm(..., norm_last_layer=True)`.

#### MultiCropWrapper (`models/multi_crop_wrapper.py`)
- Nhận danh sách các crop ảnh có kích thước khác nhau (2 ảnh global $224 \times 224$ và 6 ảnh local $96 \times 96$).
- Gom nhóm (group) các crop có cùng độ phân giải không gian để thực hiện forward theo batch, tối ưu hóa tốc độ GPU và bộ nhớ.
- Hỗ trợ forward cả tensor đơn lẻ trong giai đoạn kiểm thử / inference.

#### YOLO-style Layer DAG Parser (`models/yolo_parser.py` & `models/yolo_layers.py`)
- Cho phép xây dựng Vision Transformer bằng cấu trúc danh sách trong YAML:
  ```yaml
  backbone:
    type: YOLOModel
    layers:
      - [-1, 1, PatchEmbed, [224, 16, 3, 192]]
      - [-1, 1, PosEmbedAndClassToken, [192, 196]]
      - [-1, 12, Block, [192, 3, 4.0, True]]
      - [-1, 1, LayerNorm1d, [192]]
      - [-1, 1, ExtractToken, [0]]
  ```

---

### 3.3. Losses & Huấn luyện tự giám sát (`losses/`)

#### DINOLoss (`losses/dino_loss.py`)
- Tính hàm mất mát Cross-Entropy giữa phân phối Student và phân phối Teacher:
  $$\mathcal{L} = - \sum_{i \in \{1, 2\}} \sum_{j \ne i} P_t(x_i) \log P_s(x_j)$$
- **Sharpening**:
  - Student phân phối với nhiệt độ cố định $\tau_s = 0.1$.
  - Teacher phân phối với nhiệt độ $\tau_t$ tăng tuyến tính (warmup từ $0.04 \rightarrow 0.04$ hoặc $0.07$) để tránh sụp đổ biểu diễn (representation collapse).
- **Centering (Center update via EMA)**:
  - Trừ đi vector tâm `center` của Teacher để tránh 1 chiều kích hoạt thống trị (collapse to uniform/constant):
    $$C \leftarrow m \cdot C + (1 - m) \cdot \frac{1}{B} \sum_{i=1}^B g_t(x_i)$$

---

### 3.4. Datasets & Data Augmentation (`dino_datasets/`)

- **Multi-crop Strategy**:
  - **2 Global Crops** ($224 \times 224$): Che phủ vùng diện tích lớn (tỷ lệ $0.4 \rightarrow 1.0$), bao gồm ColorJitter, Grayscale, Gaussian Blur và Solarization. Chỉ Teacher và Student được nhìn thấy 2 crop này.
  - **Local Crops** ($96 \times 96$, mặc định 6 crops): Che phủ vùng diện tích nhỏ (tỷ lệ $0.05 \rightarrow 0.4$), chỉ Student được nhìn thấy các crop này.
- **Tải và chuẩn bị dữ liệu (`dino_datasets/prepare_subset.py`)**:
  - Tải tập dữ liệu từ Hugging Face Hub (ví dụ `clane9/imagenet-100` hoặc `Maysee/tiny-imagenet`) và trích xuất thành cấu trúc thư mục chuẩn `ImageFolder`:
    ```
    data/
    ├── train/
    │   ├── class_0000/
    │   └── class_0001/
    └── validation/
    ```

---

## 4. Hướng dẫn sử dụng & Quy trình làm việc (Workflows & Commands)

### 4.1. Môi trường phát triển (Environment Setup)

Dự án sử dụng Python 3.12+ và môi trường ảo đặt tại `.venv`:

```bash
# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt các gói phụ thuộc nếu cần
pip install -r requirements.txt
```

### 4.2. Chạy kiểm thử tự động (Running Tests)

Tất cả 40 unit test đã được kiểm thử và vượt qua:

```bash
# Chạy toàn bộ test
.venv/bin/pytest tests -v

# Chạy kiểm thử riêng lẻ theo từng module
.venv/bin/pytest tests/test_rope2d.py -v
.venv/bin/pytest tests/test_sparse_attention.py -v
.venv/bin/pytest tests/test_builder.py -v
.venv/bin/pytest tests/test_configs.py -v
.venv/bin/pytest tests/test_sparse_rope_training.py -v
```

### 4.3. Chuẩn bị tập dữ liệu (Dataset Preparation)

```bash
# Tải và chuẩn bị ImageNet-100 từ Hugging Face Hub
python dino_datasets/prepare_subset.py --dataset clane9/imagenet-100 --output_dir data
```

*Lưu ý:* Nếu chưa tải dữ liệu, khi chạy `train.py`, hệ thống sẽ tự động tạo thư mục dummy `data/train/dummy` với 64 ảnh ngẫu nhiên để phục vụ kiểm thử chạy thông luồng (dry-run).

### 4.4. Huấn luyện mô hình (Training)

```bash
# 1. Chạy thử nghiệm nhanh (Dry run)
python train.py --config configs/dry_run.yaml

# 2. Huấn luyện ViT-Tiny định nghĩa qua YOLO DAG parser
python train.py --config configs/vit_tiny_yolo.yaml

# 3. Huấn luyện ViT-Small chuẩn qua Modular Registry
python train.py --config configs/vit_small_modular.yaml

# 4. Huấn luyện ViT-Small kết hợp Frequency-Adaptive 2D RoPE & Sparse Attention
python train.py --config configs/vit_small_sparse_rope.yaml
```

**Chiến lược tối ưu hóa bộ nhớ:**
- `batch_size_per_gpu: 32` kết hợp `gradient_accumulation_steps: 4` tạo thành Effective Batch Size = 128.
- Kích hoạt PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast`) giúp chạy ổn định trên các GPU có 8GB VRAM (như RTX 3070).

### 4.5. Đánh giá mô hình (Evaluation)

Hệ thống hỗ trợ cả định dạng `image_folder` (ImageNet / ImageNet-100) và các tập downstream chuẩn của DINO (`cifar10`, `cifar100`), đồng thời hỗ trợ cả kiến trúc chuẩn (`vit_small`, `vit_tiny`) lẫn biến thể tích hợp RoPE (`vit_small_sparse_rope`, `vit_tiny_sparse_rope`).

#### 1. Đánh giá k-Nearest Neighbors (k-NN Classifier)
Trích xuất đặc trưng của tập train và val từ Teacher backbone đã đóng băng, tính độ tương đồng cosine và dự đoán nhãn:
```bash
# Đánh giá ViT-Small chuẩn
python eval/eval_knn.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_df/checkpoints_default_1/checkpoint0040.pth \
    --arch vit_small \
    --k 20

# Đánh giá ViT-Small tích hợp RoPE & Sparse Attention
python eval/eval_knn.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_rope/checkpoints_rope_1/checkpoint0040.pth \
    --arch vit_small_sparse_rope \
    --k 20
```

#### 2. Đánh giá Linear Probing
Huấn luyện một lớp Linear Classifier trên đầu các biểu diễn đặc trưng đóng băng của backbone (hỗ trợ `--cache_features` giúp tăng tốc độ tối đa):
```bash
# Huấn luyện Linear Classifier trên ViT-Small chuẩn
python eval/eval_linear.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_df/checkpoints_default_1/checkpoint0040.pth \
    --arch vit_small \
    --epochs 30 \
    --lr 0.1 \
    --cache_features

# Huấn luyện Linear Classifier trên ViT-Small RoPE
python eval/eval_linear.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_rope/checkpoints_rope_1/checkpoint0040.pth \
    --arch vit_small_sparse_rope \
    --epochs 30 \
    --lr 0.1 \
    --cache_features
```

#### 3. Trực quan hóa Attention Map (Self-Attention Maps Visualization)
Trích xuất và vẽ bản đồ chú ý của `[CLS]` token trên các attention head khác nhau:
```bash
python eval/visualize_attention.py \
    --image_path output/sample_airplane.jpg \
    --pretrained_weights weight_rope/checkpoints_rope_1/checkpoint0040.pth \
    --arch vit_small_sparse_rope \
    --output_dir output
```
Kết quả ảnh trực quan hóa sẽ được lưu phân biệt theo kiến trúc: `output/attn_map_{arch}.png`.

---

## 5. Hướng dẫn mở rộng hệ thống (Extending the Codebase)

### Cách đăng ký thêm Backbone mới:
```python
from core.registry import BACKBONES
import torch.nn as nn

@BACKBONES.register("my_custom_vit")
def build_custom_vit(patch_size=16, **kwargs):
    # Khởi tạo mô hình tuỳ biến
    return MyCustomViT(...)
```

### Cách đăng ký thêm Hàm mất mát (Loss) mới:
```python
from core.registry import LOSSES
import torch.nn as nn

@LOSSES.register("MyContrastiveLoss")
class MyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        ...
```

Sau khi đăng ký, chỉ cần chỉ định tên định danh trong trường `type` của tệp YAML cấu hình, hàm `core.builder.build_model` hoặc `build_loss` sẽ tự động phân giải và khởi tạo mà không cần sửa đổi mã nguồn điều phối `train.py`.

---

## 6. Bảng quy chuẩn thông số cấu hình YAML (YAML Config Reference)

| Khóa cấu hình | Ý nghĩa & Giá trị khuyến nghị |
|---|---|
| `dataset.data_path` | Đường dẫn thư mục chứa dữ liệu (`data/train`) |
| `dataset.global_crops_scale` | Khoảng scale cho ảnh global, khuyến nghị `[0.4, 1.0]` |
| `dataset.local_crops_scale` | Khoảng scale cho ảnh local, khuyến nghị `[0.05, 0.4]` |
| `dataset.local_crops_number`| Số lượng ảnh crop cục bộ (mặc định: `6`) |
| `model.backbone.type` | Định danh backbone trong Registry (`vit_small`, `vit_small_sparse_rope`, `YOLOModel`) |
| `model.backbone.rope_cfg` | Cấu hình RoPE (`mode: adaptive_2d`, `base: 10000.0`, `ref_img_size: 224`) |
| `model.backbone.sparse_attn_cfg` | Cấu hình Sparse Attention (`k_local_window: 3`, `k_semantic: 16`) |
| `model.head.out_dim` | Số chiều vector đầu ra của Projection Head (mặc định: `16384`) |
| `train.batch_size_per_gpu` | Kích thước batch cục bộ trên mỗi GPU (`32` cho 8GB VRAM) |
| `train.gradient_accumulation_steps` | Số bước tích lũy gradient trước khi cập nhật trọng số |
| `train.momentum_teacher` | Hệ số momentum cập nhật trọng số Teacher qua EMA ($0.996 \rightarrow 1.0$) |
| `train.clip_grad` | Giới hạn chuẩn gradient tối đa (khuyến nghị: `3.0`) |
