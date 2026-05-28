"""
Train / Evaluate — Baseline CA Model on UIT-ADrone Dataset
Người phụ trách: Người 3 (cleanup + chạy thực nghiệm)

Usage:
    python scripts/train_baseline_ca.py --train 1 --epochs 5 --batch-size 8 --lr 1e-4
    python scripts/train_baseline_ca.py --train 0
"""

import os
from itertools import islice
import torch
import torch.nn as nn
import numpy as np
from models.vit_baseline_ca import VisionTransformer
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
SAVE_PATH = os.environ.get('SAVE_PATH', 'experiments_andt_ADrone_baseline_CA/')


def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def train_epoch(epoch, model, data_loader, optimizer, lr_scheduler, metrics, ckpt_dir, current_start_batch=0, device=torch.device('cpu'), config=None):
    metrics.reset()
    average_loss = []
    start = current_start_batch if current_start_batch > 0 else 0
    for batch_idx, batch_data in enumerate(islice(data_loader, start, None), start=start):
        if batch_idx >= config.train_steps:
            break
        batch_data_256 = batch_data['256'].to(device)
        batch_data_std = batch_data['standard'].to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data_std[:, :4])
        loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred)
        loss.backward()
        optimizer.step()
        if lr_scheduler.last_epoch < config.train_steps:
            lr_scheduler.step()
        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        average_loss.append(loss.item())
        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f}"
                  .format(epoch, batch_idx, len(data_loader), np.mean(average_loss)))
        # Lưu resume checkpoint mỗi 500 batch
        if batch_idx % 500 == 0 and batch_idx > 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'batch': batch_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(ckpt_dir, 'resume.pth'))
            print(f'Đã lưu resume.pth tại epoch {epoch}, batch {batch_idx}')
    return metrics.result()


def valid_epoch(epoch, model, data_loader, metrics, config, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    label_path = os.path.join(config.data_dir, 'test/test_frame_mask/DJI_0073.npy')
    new_label = np.load(label_path)
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
    path_ckpt = os.path.join(config.checkpoint_dir, 'best.pth')
    checkpoint = torch.load(path_ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    path_scenes = sorted(glob.glob(os.path.join(test_path, 'frames/*')))
    path_labels = os.path.join(test_path, 'test_frame_mask/')
    list_np_labels, losses = [], []
    for idx_video, path_scene in enumerate(path_scenes):
        scene_name = os.path.basename(path_scene)
        losses_curr_video = []
        test_dataset = DataLoader(path_scene, transforms.Compose([transforms.ToTensor()]),
                                  resize_height=config.image_size, resize_width=config.image_size)
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
    threshold = np.mean(losses) + np.std(losses)
    print(f"Threshold: {threshold:.4f}")
    frame_auc = roc_auc_score(y_true=list_np_labels, y_score=losses)
    print("Final AUC-ROC: {:.4f}".format(frame_auc))
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


def main():
    config = get_train_config()
    device, device_ids = setup_device(config.n_gpu)
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)
    metric_names = ['loss', 'auc']
    train_metrics = MetricTracker(*metric_names, writer=writer)
    valid_metrics = MetricTracker(*metric_names, writer=writer)

    print("Creating model: Baseline CA (Cross-Attention)")
    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim, mlp_dim=config.mlp_dim,
        num_heads=config.num_heads, num_layers=config.num_layers,
        num_classes=config.num_classes, attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate, num_frames=config.num_frames
    )
    model = model.to(device)
    ckpt_dir = os.path.join(SAVE_PATH, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if bool(config.train):
        train_folder = os.path.join(config.data_dir, 'train/frames/')
        train_dataset = DataLoader(train_folder, transforms.Compose([transforms.ToTensor()]),
                                   resize_height=config.image_size, resize_width=config.image_size)
        train_batch = data.DataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True, num_workers=config.num_workers, drop_last=True)
        test_folder = os.path.join(config.data_dir, 'test/frames/DJI_0073/')
        test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor()]),
                                  resize_height=config.image_size, resize_width=config.image_size)
        test_batch = data.DataLoader(test_dataset, batch_size=1,
                                     shuffle=False, num_workers=config.num_workers, drop_last=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.wd, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=config.lr,
            pct_start=config.warmup_steps / config.train_steps,
            total_steps=config.train_steps)

        best_auc = 0.0
        log = {'val_auc': 0}
        log_file = open('training_log_baseline.txt', 'w')

        # Load resume checkpoint nếu có
        resume_path = os.path.join(ckpt_dir, 'resume.pth')
        start_epoch = 1
        start_batch = 0
        if os.path.exists(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            start_epoch = ckpt['epoch']
            start_batch = ckpt['batch']
            print(f'Resume từ epoch {start_epoch}, batch {start_batch}')

        for epoch in range(1, config.epochs + 1):
            if epoch < start_epoch:
                continue
            current_start_batch = start_batch if epoch == start_epoch else 0
            log['epoch'] = epoch
            model.train()
            result = train_epoch(epoch, model, train_batch, optimizer, lr_scheduler,
                                 train_metrics, ckpt_dir, current_start_batch, device, config)
            log.update(result)
            model.eval()
            result = valid_epoch(epoch, model, test_batch, valid_metrics, config, device)
            log.update(**{'val_' + k: v for k, v in result.items()})
            best = log['val_auc'] > best_auc
            if best:
                best_auc = log['val_auc']
            save_model(ckpt_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
            log_file.write(str(log) + '\n')
            log_file.flush()

        log_file.close()
        print(f"\nTraining done. Best AUC: {best_auc:.4f}")
    else:
        test_folder = os.path.join(config.data_dir, 'test/')
        test_all_scenes(model, test_folder, config, device=device)


if __name__ == '__main__':
    main()
