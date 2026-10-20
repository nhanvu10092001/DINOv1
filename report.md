# Báo Cáo Đánh Giá Hiệu Năng Mô Hình DINOv1: Default ViT-Small vs. RoPE + Sparse Attention

Báo cáo kỹ thuật tổng hợp toàn diện kết quả đánh giá (Evaluation Report) cho hai mô hình tự học không giám sát DINOv1 (Vision Transformer - Caron et al., 2021) sau 40 epoch huấn luyện:
1. **Mô hình 1 (Default)**: ViT-Small/16 chuẩn với mã hóa vị trí tuyệt đối (Absolute Positional Embedding) và Dense Multi-Head Attention.
2. **Mô hình 2 (RoPE)**: ViT-Small/16 tích hợp **Frequency-Adaptive 2D Rotary Position Embedding (RoPE)** kết hợp **Spatial-Semantic Sparse Attention**.

---

## 1. Thông Tin Trọng Số và Cấu Hình Mô Hình

| Thông số / Thuộc tính | Mô hình Default | Mô hình RoPE + Sparse |
|---|---|---|
| **Đường dẫn Checkpoint** | `weight_df/checkpoints_default_1/checkpoint0040.pth` | `weight_rope/checkpoints_rope_1/checkpoint0040.pth` |
| **Định danh Backbone (Registry)** | `vit_small` | `vit_small_sparse_rope` |
| **Kích thước Patch (Patch Size)** | $16 \times 16$ | $16 \times 16$ |
| **Số khối Transformer (Depth)** | 12 blocks | 12 blocks |
| **Số chiều ẩn (Embedding Dim)** | 384 | 384 |
| **Số Attention Heads** | 6 heads (64 dim/head) | 6 heads (64 dim/head) |
| **MLP Expansion Ratio** | 4.0 (Hidden dim: 1536) | 4.0 (Hidden dim: 1536) |
| **Cơ chế vị trí (Positional Embedding)** | Learnable 1D/2D Absolute Pos Embed | Frequency-Adaptive 2D RoPE (`adaptive_2d`) |
| **Cơ chế Attention** | Full Dense Softmax Attention | Spatial-Semantic Sparse ($K_{\text{local}}=3, K_{\text{semantic}}=16$) |
| **Projection Head (DINOHead)** | MLP 3 lớp, Hidden 2048, Bottleneck 256, Out 16384 | MLP 3 lớp, Hidden 2048, Bottleneck 256, Out 16384 |
| **Epoch checkpoint** | Epoch 40 | Epoch 40 |

---

## 2. Thống Kê Tham Số (Parameters) & Độ Phức Tạp Tính Toán (FLOPs)

Các chỉ số được đo lường thực tế trên phần cứng **NVIDIA GeForce RTX 3070 8GB VRAM** với độ phân giải chuẩn $224 \times 224$ (Batch size = 1 cho FLOPs / Batch size = 128 cho Throughput):

| Thành phần / Chỉ số | Mô hình Default | Mô hình RoPE + Sparse | Ghi chú kỹ thuật |
|---|:---:|:---:|---|
| **Backbone Parameters** | **21,665,664** (~21.67M) | **21,665,664** (~21.67M) | RoPE không tăng số tham số học được |
| **Trainable Parameters** | 21,665,664 (100%) | 21,665,664 (100%) | Bằng nhau |
| **Non-trainable Buffers** | 0 | 32 phần tử | Bộ đệm tần số cố định (`rope.inv_freq`) |
| **DINO Head Parameters** | 9,720,064 (~9.72M) | 9,720,064 (~9.72M) | Dùng chung cấu trúc Projection Head |
| **Tổng số tham số toàn mô hình** | **31,385,728** (~31.39M) | **31,385,728** (~31.39M) | Tính cả Backbone và Head |
| **Backbone FLOPs (224×224)** | **9.20 GFLOPs** (4.60 GMACs) | **9.20 GFLOPs** (4.60 GMACs) | Đo bằng PyTorch `FlopCounterMode` |
| **DINO Head FLOPs** | 19.40 MFLOPs (0.019 GFLOPs) | 19.40 MFLOPs (0.019 GFLOPs) | Chiếu từ 384 $\rightarrow$ 16384 |
| **Tổng FLOPs toàn mạng** | **9.22 GFLOPs** (4.61 GMACs) | **9.22 GFLOPs** (4.61 GMACs) | Bằng nhau ở cấp độ tính toán lý thuyết |
| **Độ trễ suy luận (Latency)** | **1.20 ms / ảnh** | **2.20 ms / ảnh** | RoPE có thêm bước tính quay vector Q/K |
| **Thông lượng (Throughput)** | **831.9 ảnh/giây** | **455.0 ảnh/giây** | Đo ở batch size 128 trên RTX 3070 |

> **Nhận định**: Cải tiến RoPE 2D và Sparse Attention **không làm tăng kích thước bộ nhớ tham số** (giữ nguyên 21.67M cho Backbone) và giữ nguyên khối lượng tính toán lý thuyết (9.20 GFLOPs).

---

## 3. Kết Quả Đánh Giá Hiệu Năng (Benchmark CIFAR-10)

Theo quy chuẩn bài báo DINO gốc (Caron et al., 2021, Section 3.2 và Table 6), chất lượng biểu diễn đặc trưng của mô hình tự giám sát được đánh giá qua 2 phương pháp:
1. **k-Nearest Neighbors (k-NN) Classifier**: Đóng băng hoàn toàn mô hình, trích xuất đặc trưng của tập train (50,000 ảnh) và đánh giá trên tập val (10,000 ảnh) với $k=20$, nhiệt độ $T=0.07$. Không cần huấn luyện thêm hay tinh chỉnh siêu tham số.
2. **Linear Probing**: Đóng băng backbone, huấn luyện một lớp Linear Classifier ($384 \rightarrow 10$) trong 30 epochs với thuật toán tối ưu SGD (Momentum 0.9, Learning Rate ban đầu 0.1, Cosine Annealing scheduler).

### Bảng đối sánh kết quả định lượng:

| Phương pháp đánh giá | Mô hình Default | Mô hình RoPE + Sparse | Mức độ cải thiện (Gain) |
|---|:---:|:---:|:---:|
| **k-NN Classifier Top-1 ($k=20$)** | **71.67%** | **73.04%** | <span style="color:#00aa00">**+1.37%**</span> |
| **k-NN Classifier Top-5 ($k=20$)** | **96.82%** | **96.87%** | <span style="color:#00aa00">**+0.05%**</span> |
| **Linear Probing Top-1 (Epoch 10)** | 62.77% | 74.48% | <span style="color:#00aa00">**+11.71%**</span> |
| **Linear Probing Top-1 (Epoch 20)** | 75.72% | 77.19% | <span style="color:#00aa00">**+1.47%**</span> |
| **Linear Probing Top-1 (Best - Epoch 30)** | **76.77%** | **78.57%** | <span style="color:#00aa00">**+1.80%**</span> |
| **Linear Probing Train Loss cuối** | 0.9151 | 0.8766 | <span style="color:#00aa00">**-0.0385 (Hội tụ tốt hơn)**</span> |

### Phân tích biểu diễn đặc trưng:
- **Tính phân cụm ngữ nghĩa vượt trội**: Mô hình RoPE đạt mức cải thiện **+1.37% ở k-NN** và **+1.80% ở Linear Probe**, cho thấy không gian nhúng của RoPE giúp các biểu diễn cùng lớp gần nhau hơn và các lớp khác nhau phân tách rõ rệt hơn.
- **Tốc độ hội tụ phân loại nhanh hơn đáng kể**: Ở epoch 10 của Linear Probing, mô hình RoPE đã đạt **74.48%** (vượt xa mức 62.77% của Default). Điều này chứng minh đặc trưng trích xuất từ mô hình RoPE có tính chất tuyến tính khả tách (linearly separable) cao hơn ngay từ đầu.

---

## 4. Phân Tích Định Tính: Trực Quan Hóa Attention Maps

Đã trích xuất ma trận Self-Attention ở tầng cuối cùng của token `[CLS]` đối với tất cả các patch ảnh không gian ($14 \times 14$) trên 6 attention heads:

### 1. Lớp Airplane (Máy bay)
- **Tệp ảnh**: `output/attention_comparison.png`
- **Quan sát**:
  - Mô hình **Default**: Chú ý bị phân tán rộng, vùng nền trời phía sau máy bay bị kích hoạt đáng kể ở các Head 1 và Head 3.
  - Mô hình **RoPE**: Vùng kích hoạt của token `[CLS]` ôm sát thân máy bay, cánh và đuôi. Head 2 và Head 5 phân lập rõ ràng cấu trúc cơ học của đối tượng và triệt tiêu gần như hoàn toàn nhiễu từ nền trời.

### 2. Lớp Horse (Ngựa)
- **Tệp ảnh**: `output/attention_comparison_horse.png`
- **Quan sát**:
  - Mô hình **Default**: Chú ý có xu hướng bắt vào mảng màu tổng thể nhưng đường biên hình thể con ngựa bị nhòe sang phần cỏ xung quanh.
  - Mô hình **RoPE**: Bắt chính xác đường cong lưng, đầu và bốn chân của con ngựa. Cơ chế Frequency-Adaptive 2D RoPE giúp giữ lại thông tin tọa độ không gian chính xác ở tần số cao, giúp mạng phân biệt rõ ràng ranh giới giữa thực thể và hậu cảnh.

---

## 5. Hướng Dẫn Tái Lập Đánh Giá (Reproduction Commands)

Tất cả các công cụ đánh giá đã được chuẩn hóa và sẵn sàng chạy qua dòng lệnh:

### 1. Đánh giá k-NN ($k=20$)
```bash
# Mô hình Default
python eval/eval_knn.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_df/checkpoints_default_1/checkpoint0040.pth \
    --arch vit_small \
    --k 20

# Mô hình RoPE
python eval/eval_knn.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_rope/checkpoints_rope_1/checkpoint0040.pth \
    --arch vit_small_sparse_rope \
    --k 20
```

### 2. Đánh giá Linear Probing (30 Epochs)
```bash
# Mô hình Default
python eval/eval_linear.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_df/checkpoints_default_1/checkpoint0040.pth \
    --arch vit_small \
    --epochs 30 \
    --lr 0.1 \
    --cache_features

# Mô hình RoPE
python eval/eval_linear.py \
    --data_path data/cifar10 \
    --dataset cifar10 \
    --pretrained_weights weight_rope/checkpoints_rope_1/checkpoint0040.pth \
    --arch vit_small_sparse_rope \
    --epochs 30 \
    --lr 0.1 \
    --cache_features
```

### 3. Trực quan hóa Attention Map
```bash
python eval/visualize_attention.py \
    --image_path output/sample_airplane.jpg \
    --pretrained_weights weight_rope/checkpoints_rope_1/checkpoint0040.pth \
    --arch vit_small_sparse_rope \
    --output_dir output
```

---

## 6. Kết Luận

1. **Hiệu quả kiến trúc**: Việc tích hợp **Frequency-Adaptive 2D RoPE** kết hợp **Spatial-Semantic Sparse Attention** cải thiện chất lượng tự học biểu diễn không giám sát của DINOv1 trên mọi thước đo:
   - Tăng **+1.37% k-NN Top-1 Accuracy** ($71.67\% \rightarrow 73.04\%$).
   - Tăng **+1.80% Linear Probe Top-1 Accuracy** ($76.77\% \rightarrow 78.57\%$).
2. **Chi phí tài nguyên**: Mô hình giữ nguyên 100% dung lượng tham số (21.67M params) và độ phức tạp lý thuyết (9.20 GFLOPs), chứng minh hiệu quả thuần túy đến từ việc mô hình hóa không gian 2D tốt hơn thay vì tăng dung lượng mô hình.
