import os
import glob
import numpy as np
from data.label_map import remap_label_array

def convert_label_files(input_dir, output_dir):
    """
    Đọc tất cả file .npy từ input_dir, map nhãn và lưu vào output_dir.
    In log thống kê trước và sau chuyển đổi.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    npy_files = glob.glob(os.path.join(input_dir, '*.npy'))
    if not npy_files:
        print(f"Không tìm thấy file .npy nào trong {input_dir}")
        return

    print(f"Đang xử lý {len(npy_files)} files từ {input_dir}...")
    
    total_original = {}
    total_converted = {}
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        
        # Load file gốc
        arr = np.load(file_path, allow_pickle=True)
        
        # Thống kê gốc
        unique, counts = np.unique(arr, return_counts=True)
        for u, c in zip(unique, counts):
            total_original[u] = total_original.get(u, 0) + c
            
        # Chuyển đổi nhãn
        new_arr = remap_label_array(arr)
        
        # Thống kê sau chuyển đổi
        new_unique, new_counts = np.unique(new_arr, return_counts=True)
        for u, c in zip(new_unique, new_counts):
            total_converted[u] = total_converted.get(u, 0) + c
            
        # Lưu file mới
        out_path = os.path.join(output_dir, filename)
        np.save(out_path, new_arr)
        
    # In Log thống kê
    print("\n--- THỐNG KÊ TRƯỚC KHI CHUYỂN ĐỔI ---")
    for k, v in sorted(total_original.items()):
        print(f"Nhãn cũ {k}: {v} frames")
        
    print("\n--- THỐNG KÊ SAU KHI CHUYỂN ĐỔI ---")
    for k, v in sorted(total_converted.items()):
        print(f"Nhãn mới {k}: {v} frames")
    
    print(f"\n=> Đã lưu thành công {len(npy_files)} files vào {output_dir}")

if __name__ == '__main__':
    # Test script khi chạy độc lập
    input_directory = "../../UIT-ADrone/test/test_frame_mask" 
    output_directory = "outputs/test_frame_mask_3cls" 
    
    convert_label_files(input_directory, output_directory)