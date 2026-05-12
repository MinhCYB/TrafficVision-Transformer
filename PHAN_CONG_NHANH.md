# 🚀 TrafficVision-Transformer — Phân Công Nhanh (3 Người × 3 Ngày)

> **Bài toán:** Unsupervised Anomaly Detection | **Metric:** AUC-ROC | **Loss:** MSE reconstruction

---

## 📁 Cấu Trúc Thư Mục Hiện Tại

```
TrafficVision-Transformer/
│
├── compute_auc.py                   ← [P1] Đã tạo sẵn — test + push cuối Ngày 2
│
├── configs/
│   ├── train_config.py              ← [P1] ✅ Đã xong (đầy đủ args + model arch)
│   └── model_config.py
│
├── data/
│   └── dataset.py                   ← [P1] Fix bug shape dòng 56 + thêm docstring
│
├── engine/
│   ├── trainer.py                   ← ✅ Đã cleanup (bỏ acc1s/acc5s → auc)
│   └── checkpoint.py                ← ✅ Đã tạo (bỏ jax dependency)
│
├── models/
│   ├── vit_ste_tte.py               ← [P2] Thêm docstring + fix .cuda() → .to(device)
│   ├── vit_baseline_ca.py           ← [P3] Thêm docstring so sánh với STE-TTE
│   ├── vit_original.py
│   └── utils.py
│
├── scripts/
│   ├── train_ste_tte.py             ← [P2] ✅ Sẵn sàng chạy (import đã đúng)
│   └── train_baseline_ca.py         ← [P3] ✅ Sẵn sàng chạy (import đã đúng)
│
├── experiments_andt_ADrone_STE_TTE/          ← Output P2
│   └── checkpoints/ (best.pth, current.pth)
│
├── experiments_andt_ADrone_baseline_CA/      ← Output P3
│   └── checkpoints/ (best.pth, current.pth)
│
├── results/                         ← [P3]
│   ├── visualize.py                 ← ✅ Đã tạo sẵn, test Ngày 2, chạy Ngày 3
│   ├── comparison_table.md          ← Template — điền kết quả Ngày 3
│   └── roc_comparison.png           ← Output sau khi chạy visualize.py
│
├── training_log_ste_tte.txt         ← Auto-tạo khi P2 train
├── training_log_baseline.txt        ← Auto-tạo khi P3 train
└── README.md                        ← [P1] Viết Ngày 3
```

---

## 📅 Timeline 3 Ngày

### 🗓 Ngày 1 — Cleanup & Setup Code

| | **Người 1** | **Người 2** | **Người 3** |
|---|---|---|---|
| **Nhiệm vụ** | Fix data pipeline | Document + fix STE-TTE model | Document baseline CA |
| **Files** | `data/dataset.py` | `models/vit_ste_tte.py` | `models/vit_baseline_ca.py` |

**Người 1:**
- Fix bug trong `data/dataset.py` dòng 56:
  ```python
  # SAI (width/height bị hoán vị):
  batch_frames = np.zeros((..., self._resize_width, self._resize_height))
  # ĐÚNG:
  batch_frames = np.zeros((..., self._resize_height, self._resize_width))
  ```
- Thêm docstring vào class `DataLoader`

**Người 2:**
- Thêm docstring vào `VisionTransformer.forward()` trong `models/vit_ste_tte.py`
- Fix hardcode cuda:
  ```python
  # Đổi: x = x.float().cuda()
  # Thành: x = x.float().to(device)
  ```
- Kiểm tra decoder output shape đúng `(B, 3, 256, 256)`

**Người 3:**
- Thêm docstring đầu file `models/vit_baseline_ca.py` (giải thích khác biệt vs STE-TTE)
- Đọc + review `scripts/train_baseline_ca.py`

---

### 🗓 Ngày 2 — Training

| | **Người 1** | **Người 2** | **Người 3 (sáng)** | **Người 3 (chiều)** |
|---|---|---|---|---|
| **Nhiệm vụ** | ⚡ Push `compute_auc.py` → fix `dataset.py` | Train STE-TTE cả ngày | Train Baseline CA | Test `results/visualize.py` |
| **Output** | `compute_auc.py` trên git | `training_log_ste_tte.txt` | `training_log_baseline.txt` | `visualize.py` ready |

> ⚠️ **Người 1 phải push `compute_auc.py` trước cuối Ngày 2** — Người 2 và 3 cần nó vào Ngày 3!

**Lệnh train — Người 2:**
```bash
python scripts/train_ste_tte.py \
    --train 1 \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --model-arch b16 \
    --image-size 384 \
    --data-dir ../../UIT-ADrone
```

**Lệnh train — Người 3:**
```bash
python scripts/train_baseline_ca.py \
    --train 1 \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-4 \
    --model-arch b16 \
    --image-size 384 \
    --data-dir ../../UIT-ADrone
```

---

### 🗓 Ngày 3 — Inference, Kết Quả & Báo Cáo

| | **Người 1** | **Người 2** | **Người 3** |
|---|---|---|---|
| **Nhiệm vụ** | Viết README | Inference + tính AUC STE-TTE | Inference Baseline + tổng hợp + ROC |
| **Output** | `README.md` | AUC score STE-TTE | `comparison_table.md`, `roc_comparison.png` |

**Người 2 — Ngày 3:**
```bash
# 1. Inference
python scripts/train_ste_tte.py --train 0 --data-dir ../../UIT-ADrone

# 2. Tính AUC
python compute_auc.py \
    --scores-dir experiments_andt_ADrone_STE_TTE/ \
    --labels-dir ../../UIT-ADrone/test/test_frame_mask/ \
    --plot --save-plot results/roc_ste_tte.png
```

**Người 3 — Ngày 3:**
```bash
# 1. Inference
python scripts/train_baseline_ca.py --train 0 --data-dir ../../UIT-ADrone

# 2. Tính AUC
python compute_auc.py \
    --scores-dir experiments_andt_ADrone_baseline_CA/ \
    --labels-dir ../../UIT-ADrone/test/test_frame_mask/ \
    --plot --save-plot results/roc_baseline.png

# 3. Vẽ ROC comparison (sau khi cả 2 model đã xong)
python results/visualize.py

# 4. Điền kết quả vào results/comparison_table.md
```

---

## 🔗 Dependency Chain

```
P1: compute_auc.py ──(push cuối Ngày 2)──┐
                                          ├──► P2 Ngày 3 → AUC STE-TTE
                                          └──► P3 Ngày 3 → AUC Baseline
                                                      │
                                                      ▼
                                          P3: comparison_table.md
                                              roc_comparison.png
```

---

## ⚙️ Bảng File Ownership

| File | Người phụ trách | Trạng thái |
|---|---|---|
| `data/dataset.py` | **P1** | ⬜ Fix bug dòng 56 + docstring |
| `compute_auc.py` | **P1** | ⬜ Test + push Ngày 2 |
| `configs/train_config.py` | **P1** | ✅ Xong |
| `README.md` | **P1** | ⬜ Viết Ngày 3 |
| `models/vit_ste_tte.py` | **P2** | ⬜ Docstring + fix `.cuda()` |
| `scripts/train_ste_tte.py` | **P2** | ✅ Sẵn sàng chạy |
| `models/vit_baseline_ca.py` | **P3** | ⬜ Docstring |
| `scripts/train_baseline_ca.py` | **P3** | ✅ Sẵn sàng chạy |
| `results/visualize.py` | **P3** | ✅ Đã tạo, test Ngày 2 |
| `results/comparison_table.md` | **P3** | ⬜ Điền Ngày 3 |
| `engine/trainer.py` | — | ✅ Đã cleanup |
| `engine/checkpoint.py` | — | ✅ Đã tạo |

> **Lưu ý:** File `.npy` trong `experiments_andt_ADrone_STE_TTE/` và `experiments_andt_ADrone_baseline_CA/` — **ĐỪNG XÓA** sau khi test, cần để tính AUC và vẽ biểu đồ.
