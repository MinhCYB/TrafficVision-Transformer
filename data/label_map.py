import numpy as np

# Label mapping dictionary
LABEL_MAP = {
    0: 0,
    3: 1,
    2: 2,
    9: 3
}

def remap_label_array(arr):
    """
    Map các nhãn anomaly gốc sang format 3 class mới.
    Các nhãn không nằm trong LABEL_MAP sẽ bị gán thành -1.
    """
    arr = np.asarray(arr)
    result = np.full_like(arr, -1)
    
    for old_label, new_label in LABEL_MAP.items():
        result[arr == old_label] = new_label
        
    return result

def filter_frames_by_labels(arr):
    """
    Tạo mask boolean cho các frame hợp lệ.
    Trả về True cho các frame có nhãn >= 0, False cho các frame bị loại (-1).
    """
    remapped = remap_label_array(arr)
    return remapped != -1