"""
Train / Evaluate — STE-TTE Model on UIT-ADrone Dataset
=======================================================
Người phụ trách: Người 1 (cleanup) + Người 2 (chạy thực nghiệm)

Usage:
    # Training
    python scripts/train_ste_tte.py --train 1 --epochs 5 --batch-size 8 --lr 1e-4 --model-arch b16 --image-size 384

    # Inference (lưu anomaly scores .npy)
    python scripts/train_ste_tte.py --train 0
"""

import os
import torch
import torch.nn as nn
import numpy as np
from models.vit_ste_tte import VisionTransformer
import torchvision.transforms as transforms
from configs.train_config import get_train_config
from models.utils import setup_device, MetricTracker, TensorboardWriter
from data.dataset import DataLoader
from sklearn.metrics import roc_auc_score
import torch.utils.data as data
import glob
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Anomaly score: higher MSE = more anomalous
loss_func_mse = nn.MSELoss(reduction='mean')

SAVE_PATH = 'experiments_andt_ADrone_STE_TTE/'


def normalize_scores(scores):
    """Normalize anomaly scores to [0,1] for fair threshold comparison."""
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def train_epoch(epoch, model, data_loader, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()
    average_loss = []

    for batch_idx, batch_data in enumerate(data_loader):
        batch_data_256 = batch_data['256'].to(device)
        batch_data_std = batch_data['standard'].to(device)

        optimizer.zero_grad()
        batch_pred = model(batch_data_std[:, :4])
        loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        average_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Reconstruction Loss: {:.4f}"
                  .format(epoch, batch_idx, len(data_loader), np.mean(average_loss)))
                # AUTO SAVE mỗi 500 batch
        if batch_idx % 500 == 0 and batch_idx > 0:
            save_path = os.path.join(
                SAVE_PATH,
                'checkpoints',
                f'batch_{batch_idx}.pth'
            )

            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)

            print(f"\n✅ Auto-saved checkpoint: {save_path}")

    return metrics.result()


def valid_epoch(epoch, model, data_loader, metrics, device=torch.device('cpu')):
    """
    Validate using MSE reconstruction loss as anomaly score.
    Higher MSE → frame is more anomalous → used to compute AUC-ROC.
    """
    metrics.reset()
    losses = []

    val_label_path = '../../UIT-ADrone/test/test_frame_mask/DJI_0073.npy'
    new_label = np.load(val_label_path)

    with torch.no_grad():
        for batch_data in data_loader:
            batch_data_256 = batch_data['256'].to(device)
            batch_data_std = batch_data['standard'].to(device)
            batch_pred = model(batch_data_std[:, :4])
            loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred)
            losses.append(loss.item())

    mean_loss = np.mean(losses)
    frame_auc = roc_auc_score(y_true=new_label[:len(losses)], y_score=losses)

    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', mean_loss)
    metrics.update('auc', frame_auc)

    print("Val Epoch {:03d} | Loss: {:.4f} | AUC: {:.4f}".format(epoch, mean_loss, frame_auc))
    return metrics.result()


def test_all_scenes(model, test_path, config, device=None):
    """
    Load best checkpoint và chạy inference trên toàn bộ test scenes.
    Saves .npy anomaly score files vào SAVE_PATH để dùng với compute_auc.py.
    """
    path_ckpt = os.path.join(SAVE_PATH, 'checkpoints/best.pth')
    checkpoint = torch.load(path_ckpt)
    print('Path checkpoint:', path_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    path_scenes = sorted(glob.glob(os.path.join(test_path, 'frames/*')))
    path_labels = os.path.join(test_path, 'test_frame_mask/')
    list_np_labels = []
    losses = []

    for idx_video, path_scene in enumerate(path_scenes):
        print('-' * 80)
        scene_name = os.path.basename(path_scene)
        print(f'Video {idx_video + 1}: {scene_name}')
        losses_curr_video = []

        test_dataset = DataLoader(
            path_scene, transforms.Compose([transforms.ToTensor()]),
            resize_height=config.image_size, resize_width=config.image_size
        )
        test_batch = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
        np_label = np.load(os.path.join(path_labels, scene_name + '.npy'), allow_pickle=True)

        with torch.no_grad():
            for batch_data in tqdm(test_batch, desc=f'Evaluating {scene_name}'):
                batch_data_256 = batch_data['256'].to(device)
                batch_data_std = batch_data['standard'].to(device)
                batch_pred = model(batch_data_std[:, :4])
                loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred)
                losses.append(loss.item())
                losses_curr_video.append(loss.item())

        list_np_labels.append(np_label[len(np_label) - len(losses_curr_video):])
        np.save(os.path.join(SAVE_PATH, scene_name + '.npy'), np.array(losses_curr_video))

    list_np_labels = np.concatenate(list_np_labels)

    # Thresholding + binary prediction
    threshold = np.mean(losses) + np.std(losses)
    binary_pred = (np.array(losses) > threshold).astype(int)
    print(f"\nThreshold: {threshold:.4f}")
    print(f"Detected anomalies: {binary_pred.sum()} / {len(binary_pred)} frames")

    frame_auc = roc_auc_score(y_true=list_np_labels, y_score=losses)
    print("Final AUC-ROC: {:.4f} | Mean Loss: {:.4f}".format(frame_auc, np.mean(losses)))
    return frame_auc


def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
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
        print(f"  → Best model saved (AUC improved)")


def main():
    config = get_train_config()
    device, device_ids = setup_device(config.n_gpu)
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    metric_names = ['loss', 'auc']
    train_metrics = MetricTracker(*metric_names, writer=writer)
    valid_metrics = MetricTracker(*metric_names, writer=writer)

    # Create model
    print("Creating model: STE-TTE (Spatial-Temporal Encoder + Temporal Encoder)")
    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate,
        num_frames=config.num_frames
    )
    model = model.to(device)

    ckpt_dir = os.path.join(SAVE_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if bool(config.train):
        # --- Data loading ---
        train_folder = os.path.join(config.data_dir, 'train/frames/')
        train_dataset = DataLoader(
            train_folder, transforms.Compose([transforms.ToTensor()]),
            resize_height=config.image_size, resize_width=config.image_size
        )
        print(f'Train size: {len(train_dataset)} samples')
        train_batch = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers, drop_last=True)

        # Validation: dùng 1 video làm val trong lúc training
        test_folder = os.path.join(config.data_dir, 'test/frames/DJI_0073/')
        test_dataset = DataLoader(
            test_folder, transforms.Compose([transforms.ToTensor()]),
            resize_height=config.image_size, resize_width=config.image_size
        )
        print(f'Val size: {len(test_dataset)} samples')
        test_batch = data.DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, num_workers=config.num_workers, drop_last=True)

        # Optimizer + scheduler
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.lr, weight_decay=config.wd, momentum=0.9
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.lr,
            pct_start=config.warmup_steps / config.train_steps,
            total_steps=config.train_steps
        )

        # Training loop
        print("Start training...")
        best_auc = 0.0
        log = {'val_auc': 0}
        log_file = open('training_log_ste_tte.txt', 'w')

        for epoch in range(1, config.epochs + 1):
            log['epoch'] = epoch

            model.train()
            result = train_epoch(epoch, model, train_batch, optimizer, lr_scheduler, train_metrics, device)
            log.update(result)

            model.eval()
            result = valid_epoch(epoch, model, test_batch, valid_metrics, device)
            log.update(**{'val_' + k: v for k, v in result.items()})

            best = False
            if log['val_auc'] > best_auc:
                best_auc = log['val_auc']
                best = True

            save_model(ckpt_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
            log_file.write(str(log) + '\n')
            log_file.flush()

        log_file.close()
        print(f"\nTraining done. Best AUC: {best_auc:.4f}")

    else:
        print('Testing on all scenes...')
        test_folder = os.path.join(config.data_dir, 'test/')
        test_all_scenes(model, test_folder, config, device=device)


if __name__ == '__main__':
    main()
