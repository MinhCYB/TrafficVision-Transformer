import os
import sys

# Thêm thư mục gốc của dự án vào đường dẫn tìm kiếm của Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_train_config
from data.dataset import DataLoader
from torchvision import transforms

def main():
    print("--- BƯỚC 1: TẢI CẤU HÌNH ---")
    cfg = get_train_config()

    print("\n--- BƯỚC 2: KHỞI TẠO DATA PIPELINE ---")
    data_path = os.path.abspath(os.path.join(cfg.data_dir, "train", "frames"))
    print(f"Đường dẫn tìm kiếm dữ liệu: {data_path}")

    if not os.path.exists(data_path):
        print(f"\n[CẢNH BÁO] Thư mục '{data_path}' chưa tồn tại trên máy bạn.")
        print("=> Module import và Config đã thông suốt, nhưng bạn cần chép dataset vào đúng vị trí để test thực tế!")
        return

    try:
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = DataLoader(
            video_folder=data_path,
            transform=transform,
            resize_height=cfg.image_size,
            resize_width=cfg.image_size,
            time_step=cfg.num_frames
        )
        print(f"\n=> TÍCH HỢP THÀNH CÔNG! Pipeline đã sẵn sàng, load được {len(dataset)} samples.")
    except Exception as e:
        print(f"\n=> LỖI TÍCH HỢP: {e}")

if __name__ == '__main__':
    main()