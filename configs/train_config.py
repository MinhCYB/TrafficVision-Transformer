import argparse

def get_train_args():
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")
    parser.add_argument("--exp-name", type=str, default="exp_3cls_v1", help="Tên experiment")
    parser.add_argument("--train", type=int, default=1, help="1: Train, 0: Eval")
    parser.add_argument("--n-gpu", type=int, default=1, help="Số lượng GPU")
    parser.add_argument("--model-arch", type=str, default="b16", choices=['b16', 'b32', 'l16', 'l32'])
    parser.add_argument("--checkpoint-path", type=str, default="", help="Đường dẫn file .pth")
    parser.add_argument("--image-size", type=int, default=256, help="Kích thước ảnh đầu vào")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=4, help="Số lượng frames")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help='Weight decay')
    parser.add_argument("--data-dir", type=str, default='../../UIT-ADrone', help='Thư mục dataset')
    parser.add_argument("--num-classes", type=int, default=3, help="Số lượng class (3 nhãn mới)")
    
    # parse_known_args() giúp không bị crash nếu có tham số lạ truyền vào
    config, _ = parser.parse_known_args()
    return config

def get_eval_args():
    parser = argparse.ArgumentParser("Visual Transformer Evaluation")
    parser.add_argument("--model-arch", type=str, default="b16")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--data-dir", type=str, default='../../UIT-ADrone')
    parser.add_argument("--num-classes", type=int, default=3)
    
    config, _ = parser.parse_known_args()
    return config