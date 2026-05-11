import numpy as np
import sys
import os

# Trỏ đường dẫn để import được module data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.label_map import remap_label_array, filter_frames_by_labels

def test_remap_logic():
    print("--- CHẠY UNIT TEST: LABEL MAP ---")
    
    # Test case chuẩn theo yêu cầu từ Notion (Ngày 5 của C)
    input_arr = np.array([0, 0, 3, 2, 9, 5, 7])
    expected_arr = np.array([0, 0, 1, 2, 3, -1, -1])
    
    # Chạy hàm
    result = remap_label_array(input_arr)
    
    # Kiểm tra
    assert np.array_equal(result, expected_arr), f"Lỗi remap: Kỳ vọng {expected_arr}, nhưng code trả ra {result}"
    print("PASS: remap_label_array() map đúng 3 nhãn và loại bỏ nhãn rác.")

def test_filter_mask():
    input_arr = np.array([0, 0, 3, 2, 9, 5, 7])
    # 5 và 7 là nhãn rác -> False. Còn lại -> True
    expected_mask = np.array([True, True, True, True, True, False, False])
    
    result = filter_frames_by_labels(input_arr)
    
    assert np.array_equal(result, expected_mask), f"Lỗi mask: {result}"
    print("PASS: filter_frames_by_labels() tạo mask boolean chuẩn xác.")

if __name__ == "__main__":
    test_remap_logic()
    test_filter_mask()
    print("\nTẤT CẢ UNIT TEST ĐỀU THÀNH CÔNG!")