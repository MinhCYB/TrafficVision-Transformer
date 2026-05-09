# 📅 Lịch phân công 3 người × 10 ngày — Project ASTT

## Vai trò

| Người | Ký hiệu | Vai trò chính |
|-------|---------|--------------|
| Người A | 🔵 A | **Data & Infrastructure** — cấu trúc thư mục, xử lý nhãn, data pipeline |
| Người B | 🟢 B | **Model & Training** — model files, engine train/eval, scripts |
| Người C | 🟡 C | **Experiment & Docs** — chạy thử nghiệm, đánh giá kết quả, README |

---

## Bảng tổng quan theo ngày

| Ngày | 🔵 A | 🟢 B | 🟡 C |
|------|------|------|------|
| **1** | Tạo cấu trúc thư mục mới | Nghiên cứu code model cũ | Đọc README, liệt kê dataset |
| **2** | Viết `data/label_map.py` | Tách `models/` từ root | Xác nhận 3 nhãn trong dataset |
| **3** | Viết `data/preprocess.py` | Tách `models/` (tiếp) | Viết script thống kê nhãn |
| **4** | Viết `data/dataset.py` | Viết `engine/trainer.py` | Kiểm thử `preprocess.py` |
| **5** | Viết `configs/` package | Viết `engine/evaluator.py` | Kiểm thử label mapping |
| **6** | Tích hợp data pipeline | Viết `scripts/train_*.py` | Hỗ trợ debug tích hợp |
| **7** | Unit test data module | Viết `scripts/train_*.py` (tiếp) | Chuẩn bị lệnh chạy experiment |
| **8** | Hỗ trợ fix bugs pipeline | Chạy training thử (dry run) | Theo dõi log training |
| **9** | Viết `requirements.txt` | Chạy evaluation + ghi kết quả | Phân tích kết quả |
| **10** | Review code toàn bộ | Fix bugs cuối + merge | Viết README mới + báo cáo |

---

## Chi tiết từng ngày

---

### 🗓️ Ngày 1 — Khởi động & lên kế hoạch

**🔵 A:**
- Tạo toàn bộ thư mục mới: `configs/`, `data/`, `models/`, `engine/`, `scripts/`, `outputs/`
- Tạo các file `__init__.py` placeholder
- Không xoá file cũ, để song song

**🟢 B:**
- Đọc kỹ 3 model files: `model_original.py`, `model_ste_tte.py`, `model_baseline_ca.py`
- Ghi chú các import dependency, interface giữa các file
- Lập danh sách thay đổi cần làm khi chuyển vào `models/`

**🟡 C:**
- Đọc `README.md` và dataset description
- Liệt kê 10 nhãn anomaly gốc với tên đầy đủ
- Xác nhận lại với nhóm: nhãn 2, 3, 9 là gì, có đúng file `.npy` không

**✅ Deliverable cuối ngày 1:** Thư mục mới tồn tại, danh sách nhãn được xác nhận

---

### 🗓️ Ngày 2 — Label mapping & Model tách

**🔵 A:**
- Viết `data/label_map.py` hoàn chỉnh:
  - Dict `LABEL_MAP = {3: 0, 2: 1, 9: 2}`
  - Hàm `remap_label_array(arr)`
  - Hàm `filter_frames_by_labels(arr)` → mask cho frame hợp lệ

**🟢 B:**
- Copy + refactor `model_original.py` → `models/vit_original.py`
  - Sửa import paths
  - Thêm docstring cho class/function chính

**🟡 C:**
- Kiểm tra thực tế file `.npy` ground truth trong `UIT-ADrone/test/test_frame_mask/`
- Xem phân phối nhãn: mỗi nhãn chiếm bao nhiêu frame?
- Báo lại cho A để điều chỉnh logic nếu cần

**✅ Deliverable:** `label_map.py` draft, `models/vit_original.py` xong

---

### 🗓️ Ngày 3 — Preprocessing & Model (tiếp)

**🔵 A:**
- Viết `data/preprocess.py`:
  - Hàm `convert_label_files(input_dir, output_dir)`
  - Tạo thư mục output `test_frame_mask_3cls/`
  - Log thống kê trước/sau chuyển đổi

**🟢 B:**
- Copy + refactor `model_ste_tte.py` → `models/vit_ste_tte.py`
- Copy + refactor `model_baseline_ca.py` → `models/vit_baseline_ca.py`
- Tạo `models/__init__.py` export cả 3 model

**🟡 C:**
- Viết script nhỏ `notebooks/explore_labels.ipynb` hoặc `.py`:
  - Đọc file `.npy` → vẽ histogram phân phối nhãn
  - Kiểm tra nhãn 2, 3, 9 có đủ data không
  - Ghi kết quả vào `outputs/label_stats.txt`

**✅ Deliverable:** `preprocess.py` draft, `models/` package hoàn chỉnh, thống kê nhãn

---

### 🗓️ Ngày 4 — Dataset class & Trainer

**🔵 A:**
- Viết `data/dataset.py` (từ `data_utils.py` cũ):
  - Giữ nguyên `DataLoader` class
  - Thêm parameter `label_map` tùy chọn
  - Test import từ `data/`

**🟢 B:**
- Viết `engine/trainer.py`:
  - Tách `train_epoch()`, `valid_epoch()`, `save_model()` từ các `train_val_*.py`
  - Nhận `config` và `save_path` làm tham số (không hardcode)

**🟡 C:**
- Chạy thử `data/preprocess.py` trên dataset thật
- Kiểm tra file `.npy` output có đúng format không
- Report bug nếu có

**✅ Deliverable:** `dataset.py` xong, `engine/trainer.py` draft

---

### 🗓️ Ngày 5 — Configs & Evaluator

**🔵 A:**
- Viết `configs/` package:
  - `configs/model_config.py`: ViT architecture configs (b16, b32, l16...)
  - `configs/train_config.py`: training hyperparameters
  - `configs/__init__.py`: export `get_train_config()`, `get_eval_config()`

**🟢 B:**
- Viết `engine/evaluator.py`:
  - Tách `test_all_scenes()` ra file riêng
  - Thêm param `label_map_path` để load nhãn mới
  - Tính AUC theo từng nhãn (per-class AUC)

**🟡 C:**
- Kiểm thử `data/label_map.py`:
  - Tạo array test cases: `[0, 0, 3, 2, 9, 5, 7]`
  - Xác nhận output đúng: `[0, 0, 1, 2, 3, -1, -1]`
  - Ghi test case vào `data/test_label_map.py`

**✅ Deliverable:** `configs/` xong, `engine/evaluator.py` draft, label_map được kiểm thử

---

### 🗓️ Ngày 6 — Tích hợp pipeline

**🔵 A:**
- Tích hợp toàn bộ data pipeline:
  - `configs/` → `data/dataset.py` → `engine/trainer.py` hoạt động cùng nhau
  - Test import: `from data.dataset import DataLoader` không lỗi
  - Fix path issues (relative → absolute, Windows/Linux)

**🟢 B:**
- Bắt đầu viết `scripts/train_STE_TTE.py`:
  - Import từ `models/`, `engine/`, `configs/`, `data/`
  - Không còn hardcode paths
  - Thêm `argparse` cho `--data-dir`, `--output-dir`, `--num-classes=3`

**🟡 C:**
- Hỗ trợ A debug tích hợp
- Test thử import toàn bộ module từ root:
  ```python
  from configs import get_train_config
  from data import DataLoader, LABEL_MAP
  from models import VisionTransformer
  ```
- Ghi lại lỗi gặp phải

**✅ Deliverable:** Pipeline import được, `scripts/train_STE_TTE.py` draft

---

### 🗓️ Ngày 7 — Scripts & Unit tests

**🔵 A:**
- Viết unit test cho `data/`:
  - `test_label_map.py`: kiểm thử remap function
  - `test_dataset.py`: kiểm thử DataLoader load ảnh
- Dùng `pytest` hoặc `assert` đơn giản

**🟢 B:**
- Viết thêm các scripts còn lại:
  - `scripts/train_UIT_ADrone.py`
  - `scripts/train_baseline_ca.py`
- Đảm bảo tất cả script dùng `--num-classes 3`

**🟡 C:**
- Chuẩn bị lệnh chạy experiment:
  ```bash
  python scripts/train_STE_TTE.py \
    --train 1 \
    --num-classes 3 \
    --data-dir ../../UIT-ADrone \
    --exp-name exp_3cls_v1
  ```
- Kiểm tra GPU available, CUDA version
- Chuẩn bị template ghi kết quả

**✅ Deliverable:** Scripts đầy đủ, unit tests pass, lệnh chạy sẵn sàng

---

### 🗓️ Ngày 8 — Dry run & Fix bugs

**🔵 A:**
- Hỗ trợ fix bug liên quan đến data pipeline khi chạy thật
- Đảm bảo `outputs/` structure tự tạo đúng:
  ```
  outputs/exp_3cls_v1/
  ├── checkpoints/
  ├── logs/
  └── predictions/
  ```

**🟢 B:**
- Chạy **dry run** (1-2 epoch) với model STE-TTE + 3 nhãn
- Ghi log: loss có giảm không, không crash
- Fix import errors nếu còn sót

**🟡 C:**
- Theo dõi log training, ghi lại:
  - Reconstruction loss theo epoch
  - Có warning/error không
- Bắt đầu viết bảng so sánh kết quả

**✅ Deliverable:** Training chạy được 2 epoch không crash

---

### 🗓️ Ngày 9 — Evaluation & Kết quả

**🔵 A:**
- Viết `requirements.txt`:
  ```
  torch>=1.10
  torchvision
  numpy
  opencv-python
  scikit-learn
  tqdm
  tensorboard
  ```
- Kiểm tra môi trường fresh install có hoạt động không

**🟢 B:**
- Chạy evaluation `test_all_scenes` với nhãn 3 class
- Tính AUC tổng và per-class AUC cho 3 nhãn
- Lưu kết quả vào `outputs/exp_3cls_v1/results.json`

**🟡 C:**
- Phân tích kết quả:
  - AUC của từng nhãn cao/thấp như thế nào?
  - So sánh với baseline (nếu có)
  - Vẽ ROC curve cho 3 nhãn

**✅ Deliverable:** Kết quả AUC 3 nhãn, `requirements.txt`, ROC curve

---

### 🗓️ Ngày 10 — Hoàn thiện & Tổng kết

**🔵 A:**
- Code review toàn bộ `data/`, `configs/`
- Xoá các file cũ không còn dùng (hoặc move vào `archive/`)
- Đảm bảo không còn hardcode path

**🟢 B:**
- Fix bugs cuối cùng
- Dọn dẹp `engine/`, `scripts/`, `models/`
- Đảm bảo mọi script có `--help` hoạt động

**🟡 C:**
- Viết lại `README.md`:
  - Mô tả 3 nhãn mới
  - Hướng dẫn cài đặt
  - Hướng dẫn train & eval
  - Kết quả thực nghiệm

**✅ Deliverable cuối:** Project hoàn chỉnh, README đầy đủ, kết quả có thể reproduce

---

## Phụ thuộc giữa các task (Critical Path)

```
Ngày 1 (A: cấu trúc) ──→ Ngày 2 (A: label_map) ──→ Ngày 3 (A: preprocess)
                                                              ↓
Ngày 1 (B: nghiên cứu) ──→ Ngày 2-3 (B: models/) ──→ Ngày 4 (B: trainer)
                                                              ↓
                                            Ngày 6 (A+B: Tích hợp) ──→ Ngày 8 (B: dry run) ──→ Ngày 9 (B: eval)
```

---

> [!NOTE]
> Lịch này giả sử dataset UIT-ADrone đã có sẵn. Nếu chưa có, Ngày 1-2 cần thêm bước download/setup dataset.
