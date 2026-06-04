# TrafficVision-Transformer

Dự án phát hiện bất thường giao thông không giám sát (**Unsupervised Traffic Anomaly Detection**) trên video drone, sử dụng kiến trúc Vision Transformer. Dataset: **UIT-ADrone**. Metric chính: **AUC-ROC frame-level**.

---

## Tổng quan

### Bài toán

Phát hiện hành vi bất thường (tai nạn, dừng đột ngột, người ngã, …) trong video giao thông quay từ drone ở góc nhìn trên cao. Mô hình học theo hướng **reconstruction-based**:

- **Train** chỉ trên video bình thường → mô hình học cách tái tạo cảnh bình thường.
- **Inference** dùng reconstruction error (MSE) làm anomaly score — frame nào lỗi tái tạo cao → bất thường.

### Hai mô hình

| Mô hình | File | Mô tả |
|---|---|---|
| **STE-TTE** | `models/vit_ste_tte.py` | Spatial-Temporal Encoder + Temporal Transformer Encoder — mô hình đề xuất chính |
| **Baseline CA** | `models/vit_baseline_ca.py` | Vision Transformer với Cross-Attention — mô hình so sánh |

Cả hai đều dùng kiến trúc **ViT-B/16** (`patch_size=16`, `emb_dim=768`, `num_heads=6`, `num_layers=2`, `image_size=384×384`, `num_frames=4`) với Decoder CNN để tái tạo ảnh.

### Cấu trúc thư mục

```
TrafficVision-Transformer/
├── configs/
│   ├── model_config.py          # Cấu hình ViT (b16, b32, l16, l32)
│   └── train_config.py          # Argument parser cho training
├── data/
│   ├── dataset.py               # DataLoader cho UIT-ADrone
│   ├── label_map.py             # Mapping nhãn thô → binary (0/1)
│   └── preprocess.py            # Chuyển đổi file label .npy
├── engine/
│   ├── trainer.py               # Training loop + validation + inference
│   └── checkpoint.py            # Save/load checkpoint .pth
├── models/
│   ├── vit_original.py          # ViT gốc (Encoder + Decoder)
│   ├── vit_ste_tte.py           # Model STE-TTE (đề xuất)
│   ├── vit_baseline_ca.py       # Model Baseline CA (so sánh)
│   └── utils.py                 # Tiện ích (MetricTracker, …)
├── scripts/
│   ├── train_ste_tte.py         # Script train/inference STE-TTE
│   └── train_baseline_ca.py     # Script train/inference Baseline CA
├── results/
│   ├── visualize.py             # Vẽ ROC curve so sánh 2 mô hình
│   └── comparison_table.md      # Bảng kết quả AUC
├── compute_auc.py               # Tính AUC-ROC từ file .npy anomaly scores
├── TrafficVision_Baseline_CA.ipynb   # Notebook Colab — Baseline CA
├── train_STE_TTE_colab.ipynb         # Notebook Colab — STE-TTE
└── requirements.txt
```

### Dataset — UIT-ADrone

Dataset video drone giao thông đô thị của UIT (Trường Đại học Công nghệ Thông tin, ĐHQG TP.HCM). Các scene quay từ drone ở độ cao 50–70m, góc 90°, vào buổi sáng.

```
UIT-ADrone/
├── train/
│   └── frames/
│       ├── scene_001/
│       │   ├── 000001.jpg
│       │   └── ...
│       └── scene_002/ ...
└── test/
    ├── frames/ ...
    └── test_frame_mask/         # Ground-truth label per-frame
        ├── scene_001.npy        # 0 = normal, 1 = anomaly
        └── ...
```

---

## Hướng dẫn chạy

Toàn bộ thực nghiệm chạy trên **Google Colab** (T4 GPU). Dataset đặt sẵn trên **Google Drive** tại `My Drive/Seminar/UIT-ADrone`.

> **Trước khi bắt đầu:** Vào **Runtime → Change runtime type → T4 GPU**.

### Train STE-TTE — `train_STE_TTE_colab.ipynb`

Mở notebook, sửa biến `DATA_DIR` ở **Bước 5** trỏ đúng đến thư mục `UIT-ADrone` trên Drive của bạn, sau đó chạy từng cell theo thứ tự từ trên xuống.

### Train Baseline CA — `TrafficVision_Baseline_CA.ipynb`

Mở notebook, sửa **duy nhất một biến** ở cell đầu tiên:

```python
SEMINAR_DIR = '/content/drive/MyDrive/Seminar'   # ← sửa thành thư mục của bạn
```

Tất cả path còn lại (`DATA_DIR`, `SAVE_DIR`, `RESULTS_DIR`) tự tính từ biến này. Sau đó chạy từng cell theo thứ tự từ trên xuống.

### Lưu ý

- Checkpoint được tự động lưu lên Drive nên không mất nếu session Colab bị ngắt. Resume bằng cách truyền `--checkpoint-path` vào cell Train.
- Colab T4 có ~15 GB VRAM. Nếu OOM, giảm `--batch-size` xuống 1.
- Không đóng tab trong khi train để tránh session bị thu hồi sớm.