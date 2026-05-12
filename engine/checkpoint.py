"""
Checkpoint utilities — chỉ hỗ trợ .pth (bỏ jax dependency)
"""
import os
import torch


def load_checkpoint(path):
    """Load model weights từ file .pth checkpoint."""
    if not path.endswith('.pth'):
        raise ValueError(f"Only .pth checkpoints supported, got: {path}")
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint['state_dict']


def save_checkpoint(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    """Save model checkpoint vào thư mục."""
    os.makedirs(save_dir, exist_ok=True)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    torch.save(state, os.path.join(save_dir, 'current.pth'))
    if best:
        torch.save(state, os.path.join(save_dir, 'best.pth'))
        print(f"  → Best checkpoint saved to {save_dir}/best.pth")
