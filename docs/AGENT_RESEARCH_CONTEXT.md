# Research & Engineering Context: Multi-Scale 2D Sliding-Window Sparse Attention for Self-Supervised ViT (DINO)

> **Purpose of this document:** This file serves as a comprehensive context handoff for AI coding/research agents. It summarizes the user's graduation thesis topic, mathematical formulation, implementation direction in this repository (`DINOv1`), and the exact literature/reference papers being studied.

---

## 1. Core Research Direction & Problem Statement

### 1.1. The Problem Being Solved
- **Base Architecture:** Self-Supervised Vision Transformer (**DINOv1** with **Plain ViT**, e.g., `ViT-Small`, $H=6$ heads, $14 \times 14 = 196$ spatial patch tokens + $1$ `[CLS]` token $\Rightarrow N = 197$).
- **Inefficiency of Vanilla Full Attention:** Computing dense $\mathcal{O}(N^2)$ self-attention across all $H$ heads is computationally redundant because most heads in ViT naturally converge to attend to immediate local spatial neighborhoods.
- **Why Masking (`-inf`) + `torch.topk` Fails in Practice:** Implementing sparse/local attention via an $(N \times N)$ boolean mask filled with $-\infty$ (`masked_fill(-inf)`) combined with dynamic `torch.topk` selection still forces the GPU to compute the full $(N \times N)$ matrix multiplication $QK^\top$ and adds sorting/indexing overhead from `topk`. It does **not** reduce actual FLOPs or wall-clock latency.

### 1.2. Proposed Method (The User's Core Idea)
Instead of uniform full attention or $-\infty$ masking, we **explicitly partition the $H$ attention heads within each MHA layer into groups with different 2D sliding-window kernel sizes** (extracted directly via `F.unfold` / true 2D sliding window without computing the full $N \times N$ matrix), combined with a **bidirectional `[CLS]` global bridge** (and/or a small subset of dense global heads) and **2D Rotary Position Embedding (2D RoPE)**:

1. **Head-wise Multi-Scale 2D Sliding Windows (e.g., for `ViT-Small` with $H=6$ heads):**
   - **Config A (Pure Multi-Scale Local + `[CLS]` Bridge):**
     - Group 1 (2 heads): $3 \times 3$ sliding window ($k_1 = 3 \Rightarrow 9$ local neighbors).
     - Group 2 (2 heads): $5 \times 5$ sliding window ($k_2 = 5 \Rightarrow 25$ local neighbors).
     - Group 3 (2 heads): $7 \times 7$ sliding window ($k_3 = 7 \Rightarrow 49$ local neighbors).
   - **Config B (Hybrid Multi-Scale Local + Global Dense Heads):**
     - Group 1 (2 heads): $3 \times 3$ sliding window ($9$ local neighbors).
     - Group 2 (2 heads): $5 \times 5$ sliding window ($25$ local neighbors).
     - Group 3 (2 heads): Full Dense Attention ($197$ tokens) for global semantic routing.
   - *(Optional Layer-wise Scaling inspired by MSWA):* Shallower layers use smaller windows, deeper layers use larger windows or more global heads.

2. **Bidirectional `[CLS]` Token Global Bridge:**
   - The `[CLS]` query always attends to all $N$ tokens: $\bm{q}_{\mathrm{CLS}} \in \mathbb{R}^{1 \times d_k}$ attends to $\bm{K}_{\text{all}} \in \mathbb{R}^{N \times d_k}$ so `[CLS]` aggregates global image representation for the DINO self-distillation loss.
   - Every spatial patch query $\bm{q}_i$ ($i \in \{1, \dots, N_p\}$) in head group $g$ (window $k_g \times k_g$) attends to its $k_g^2$ unfolded 2D neighbors $\mathcal{N}_{k_g}(i)$ **concatenated with $\bm{k}_{\mathrm{CLS}}$**:
     $$\widetilde{\bm{K}}_g(i) = \Big[\, \bm{k}_{\mathrm{CLS}} \;\Big\Vert\; \{\bm{k}_j\}_{j \in \mathcal{N}_{k_g}(i)} \,\Big] \in \mathbb{R}^{(1 + k_g^2) \times d_k}$$
     $$\bm{y}_i^{(g)} = \mathrm{Softmax}\!\left(\frac{\bm{q}_i \widetilde{\bm{K}}_g(i)^\top}{\sqrt{d_k}}\right) \widetilde{\bm{V}}_g(i)$$
   - This allows spatial patches to exchange long-range context through the `[CLS]` token in $\mathcal{O}(N \cdot k_g^2)$ complexity, completely eliminating $-\infty$ masks and `topk` overhead.

3. **Complexity Reduction on `ViT-Small` ($N=197, H=6$):**
   - Vanilla Full Attention: $6 \times 197^2 = 232,854$ query-key pairs per layer.
   - Config A ($3\times 3, 5\times 5, 7\times 7$ + `[CLS]`): $34,898$ pairs (**$-85.0\%$** compute).
   - Config B ($3\times 3, 5\times 5$ + 2 Dense heads): $123,098$ pairs (**$-47.2\%$** compute).

---

## 2. Core Literature & Reference Papers Being Researched

The user is actively referencing the following papers to build the theoretical foundation and related work comparison:

### Group A: Theoretical Foundation (Why ViT Needs Local + Global Head Diversity & `[CLS]`)
1. **Raghu et al. (NeurIPS 2021) — *"Do Vision Transformers See Like Convolutional Neural Networks?"***
   - **Link / ID:** `arXiv:2108.08810`
   - **Key Takeaways Used:**
     - **Mean Attention Distance analysis:** In standard ViT (trained on large data), lower layers naturally split into heads with very small attention distances (local heads, acting like CNNs) and heads with large attention distances (global heads). When trained on smaller datasets without inductive bias, ViT fails to learn local attention in early layers and accuracy drops sharply.
     - **Role of `[CLS]` vs. Spatial Tokens:** ViT exhibits a phase transition around layer 5–6 where information flows into `[CLS]` via skip connections. Because global class aggregation is offloaded to `[CLS]`, spatial tokens in late layers preserve crisp 2D spatial localization (unlike CNNs which blur spatial info via global average pooling).
   - **How it supports user's method:** Directly justifies hard-coding multi-scale local sliding windows across heads in early/middle layers while preserving the `[CLS]` token bridge in Plain ViT.

2. **Child et al. (OpenAI, 2019) — *"Generating Long Sequences with Sparse Transformers"***
   - **Link / ID:** `arXiv:1904.10509`
   - **Key Takeaways Used:** Foundational formulation of factorized sparse attention ($S = \{A_1, \dots, A_n\}$) reducing $\mathcal{O}(N^2)$ to $\mathcal{O}(N\sqrt{N})$ or $\mathcal{O}(N \cdot k)$ via local window + global summary routing.

3. **Caron et al. (ICCV 2021) — *"Emerging Properties in Self-Supervised Vision Transformers (DINO)"***
   - **Link / ID:** `arXiv:2104.14294`
   - **Key Takeaways Used:** Self-distillation with no labels (`student` / EMA `teacher`), Multi-Crop training ($2 \times 224^2$ global views + $V \times 96^2$ local views), and emergent object segmentation boundaries in `[CLS]` attention maps of Plain ViT.

---

### Group B: Multi-Scale & Sliding Window Attention in NLP (Direct Inspiration for Head-Wise Window Allocation)
4. **Xu et al. (Jan 2025) — *"MSWA: Refining Local Attention with Multi-Scale Window Attention"***
   - **Link / ID:** `arXiv:2501.01039`
   - **Key Takeaways Used:**
     - Proposes **MSWA-h** (splitting heads within the same layer into 4 window size groups: $\{w_l/4, w_l/2, w_l, 2w_l\}$) and **MSWA-l** (increasing base window size $w_l$ from shallow layers to deep layers: $\frac{1}{3}W \to \frac{2}{3}W \to W \to \frac{4}{3}W$).
     - **Implicit Weighted Emphasis:** Because all 4 head groups cover the innermost $\frac{w_l}{4}$ region, 3 groups cover $\frac{w_l}{2}$, 2 groups cover $w_l$, and 1 group covers $2w_l$, overlapping multi-scale windows naturally create a smooth step-wise distance decay ($100\% \to 75\% \to 50\% \to 25\%$ head coverage) around the query token.
     - Achieves lower perplexity than uniform Sliding Window Attention (SWA) while using only **87.9%** of the total window budget.
   - **Difference vs. User's Method:** MSWA is strictly **1D causal attention for autoregressive LLMs**. It has no 2D spatial grid$(H \times W)$, no isotropic neighborhood $(k \times k)$, and no `[CLS]` global bridge.

5. **Fu et al. (Feb 2025) — *"Sliding Window Attention Training for Efficient Large Language Models (SWAT)"***
   - **Link / ID:** `arXiv:2502.18845`
   - **Key Takeaways Used:**
     - Explains **why vanilla Sliding Window Attention fails when trained from scratch**: standard `Softmax` suffers from **Attention Sink** (dumping probability mass onto the first token) and **Winner-Takes-Most** (sharpening onto a single token), which destroys multi-hop information compression across layers.
     - Proposes replacing `Softmax` with **`Sigmoid` activation + ALiBi + RoPE** so sliding windows retain dense multi-token representations across layers.
   - **Relevance to User's Method:** Explains why pure sliding windows lose global context and motivates either the **bidirectional `[CLS]` bridge** or testing **Sigmoid + 2D RoPE** inside 2D sliding windows.

6. **Ramapuram et al. (Apple, ICLR 2025) — *"Theory, Analysis, and Best Practices for Sigmoid Self-Attention (SigMa)"***
   - **Link / ID:** `arXiv:2409.04431`
   - **Key Takeaways Used:** Theoretical foundation for `Sigmoid` attention on Vision Transformers and LLMs with bias initialization $b = -\ln N$.

---

### Group C: Prior Vision Transformer Local / Multi-Scale Window Papers (For Related Work Comparison & Highlighting Novelty)
7. **Swin Transformer — Liu et al. (ICCV 2021)** (`arXiv:2103.14030`)
   - Non-overlapping windows + shifted windows; **all heads in a layer share the exact same window size $M \times M$**; hierarchical patch merging breaks Plain ViT compatibility with DINO.
8. **NAT (Neighborhood Attention Transformer) — Hassani et al. (CVPR 2023)** (`arXiv:2204.07143`)
   - True pixel-wise 2D sliding window attention (via `NATTEN`), avoiding Swin's shifted window boundary issues, **but all heads share the same uniform kernel size $k \times k$** (typically $7 \times 7$) and uses a hierarchical pyramid without a `[CLS]` token.
9. **DilateFormer — Jiao et al. (IEEE TMM 2023)** (`arXiv:2302.01791`)
   - Splits MHA heads into groups with **different dilation rates ($r \in \{1, 2, 3\}$)** while keeping the kernel size fixed at $3 \times 3$ (always 9 points).
   - *Limitation vs. User's Method:* Sparse dilated sampling ($r > 1$) causes **gridding/checkerboard artifacts** and misses contiguous object boundaries needed by DINO. Our method uses **contiguous multi-scale kernels ($3\times 3, 5\times 5, 7\times 7$)** + `[CLS]` bridge.
10. **DW-ViT (Dynamic Window ViT) — Ren et al. (CVPR 2022)** (`arXiv:2203.12856`)
    - Assigns different window sizes to head groups, but built on **Swin's non-overlapping partitioned windows** and hierarchical stages for supervised classification.
11. **VSA (Varied-Size Window Attention) — Zhang et al. (ECCV 2022)** (`arXiv:2204.08446`)
    - Uses an auxiliary regression sub-network to predict continuous window scale/offset per head using `grid_sample`. Adds parameter/interpolation overhead and does not address Plain ViT `[CLS]` self-supervised learning.
12. **StyleNAT — Walton, Hassani et al. (arXiv 2022)** (`arXiv:2211.05770`)
    - Introduces *Variadic Attention Heads* (assigning different neighborhood kernel sizes and dilations per head) inside a **StyleGAN generator** for image synthesis. Does not apply to discriminative/self-supervised Plain ViT, has no `[CLS]` global bridge, and no Multi-Crop 2D RoPE.

---

## 3. Summary of Novelty (What Makes the User's Thesis Unique)

While head-wise window splitting exists in 1D LLMs (**MSWA**, Jan 2025) and non-overlapping/GAN vision backbones (**DW-ViT**, **StyleNAT**), **no prior work has combined**:
1. **True 2D Multi-Kernel Sliding-Window Attention across MHA heads** ($3 \times 3, 5 \times 5, 7 \times 7$ via contiguous `Unfold`, zero $-\infty$ masking, zero `topk` sorting overhead) inside an **isotropic Plain ViT** architecture.
2. **Bidirectional `[CLS]` Token Global Bridge** (every local window head includes $\bm{k}_{\mathrm{CLS}}$ in its key set, and $\bm{q}_{\mathrm{CLS}}$ attends globally), preserving $\mathcal{O}(N \cdot k^2)$ complexity while preventing local window information isolation.
3. **Self-Supervised DINO Pre-training with Multi-Crop 2D RoPE**, preserving emergent object segmentation boundaries in `[CLS]` attention maps while cutting pair-wise attention compute by **47% to 85%**.

---

## 4. Workspace Files Reference (`DINOv1`)
- `models/sparse_attention.py` — Implementation of 2D RoPE and Multi-Scale / Sparse Attention modules.
- `models/vision_transformer.py` — Plain ViT backbone for DINO (`vit_tiny`, `vit_small`, `vit_base`).
- `docs/research_hypothesis_multiscale_attention.md` — Detailed Vietnamese research documentation & complexity math.
- `docs/bao_cao_de_xuat_phuong_phap.tex` & `docs/bao_cao_de_xuat_phuong_phap.pdf` — 1-page concise LaTeX proposal report for the thesis advisor.
