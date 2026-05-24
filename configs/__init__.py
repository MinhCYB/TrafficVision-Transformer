import os
from .train_config import get_train_config, get_eval_config
from . import model_config

def print_config(config):
    message = '\n' + '='*20 + ' CONFIGURATION ' + '='*20 + '\n'
    for k, v in sorted(vars(config).items()):
        message += f'{k:>25}: {v:<30}\n'
    message += '='*55 + '\n'
    print(message)

def setup_output_dirs(exp_name):
    """Tự động tạo cấu trúc thư mục lưu trữ cho experiment"""
    base_dir = os.path.join("outputs", exp_name)
    dirs_to_make = [
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "predictions")
    ]
    for d in dirs_to_make:
        os.makedirs(d, exist_ok=True)
    print(f"[*] Đã tự động khởi tạo cấu trúc thư mục tại: {base_dir}")

def get_train_config():
    config = get_train_args()
    config = getattr(model_config, f"get_{config.model_arch}_config")(config)
    
    # Gọi hàm tự động tạo thư mục output dựa theo tên experiment
    setup_output_dirs(config.exp_name)
    
    print_config(config)
    return config

def get_eval_config():
    config = get_eval_args()
    config = getattr(model_config, f"get_{config.model_arch}_config")(config)
    print_config(config)
    return config
