# === FILE: engine/trainer.py ===
import os
import torch
import torch.nn as nn
import numpy as np
# from models.vit_ste_tte import VisionTransformer
# from model_efficientnet import VisionTransformer
import torchvision.transforms as transforms
# from configs.train_config import get_train_config
# from checkpoint import load_checkpoint
# from data_loaders import *
from models.utils import setup_device, accuracy, MetricTracker, TensorboardWriter

# Anomaly score: higher MSE = more anomalous
loss_func_mse = nn.MSELoss(reduction='mean')
# from data_utils import DataLoader
from sklearn.metrics import *
import torch.utils.data as data
import pdb
import glob
from torch.autograd import Variable
import argparse
# import neptune


def save_resume_checkpoint(state, filepath):
    """Lưu checkpoint resume giữa epoch."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, config, device=torch.device('cpu'), start_batch=0):
    """Train 1 epoch, hỗ trợ resume từ batch chỉ định."""
    metrics.reset()
    average_loss = []
    # Vòng lặp training
    for batch_idx, (batch_data) in enumerate(data_loader):
        # Bỏ qua các batch đã train khi resume
        if batch_idx <= start_batch and start_batch > 0:
            continue

        batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
        optimizer.zero_grad()
        batch_pred = model(batch_data[:,:4])
        loss = loss_func_mse(batch_data_256[:,4].float(), batch_pred)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        metrics.writer.set_step((epoch - 1) * len(data_loader) + batch_idx)
        metrics.update('loss', loss.item())
        average_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Reconstruction Loss: {:.4f}"
                    .format(epoch, batch_idx, len(data_loader), np.mean(average_loss)))

        # Lưu checkpoint giữa epoch mỗi 500 batch
        if batch_idx % 500 == 0 and batch_idx > 0:
            save_resume_checkpoint({
                'epoch': epoch,
                'batch': batch_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
            }, filepath=os.path.join(config.checkpoint_dir, 'resume.pth'))

    return metrics.result()

def valid_epoch(epoch, model, data_loader, criterion, metrics, config, device=torch.device('cpu')):
    """
    Validate using MSE as anomaly score.
    Higher MSE → more anomalous → higher AUC-ROC = better model.
    """
    metrics.reset()
    losses = []
    label_path = os.path.join(config.data_dir, 'test/test_frame_mask/DJI_0073.npy')
    new_label = np.load(label_path)

    with torch.no_grad():
        for batch_idx, (batch_data) in enumerate(data_loader):
            batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
            batch_pred = model(batch_data[:, :4])
            loss = loss_func_mse(batch_data_256[:, 4].float(), batch_pred)
            losses.append(loss.item())

    mean_loss = np.mean(losses)
    frame_auc = roc_auc_score(y_true=new_label[:len(losses)], y_score=losses)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', mean_loss)
    metrics.update('auc', frame_auc)
    print("Val Epoch {:03d} | Loss: {:.4f} | AUC: {:.4f}".format(epoch, mean_loss, frame_auc))
    return metrics.result()

from tqdm import tqdm
save_path = 'experiments_andt_ADrone_STE_TTE/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def save_model(save_dir, epoch, model, optimizer, lr_scheduler, device_ids, best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict() if len(device_ids) <= 1 else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    filename = str('./' + save_path + 'checkpoints/' + 'current.pth')
    torch.save(state, filename)

    if best:
        filename = str('./' + save_path + 'checkpoints/' + 'best.pth')
        torch.save(state, filename)


def train(model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, train_metrics, valid_metrics, config, device, device_ids):
    """Vòng lặp training chính, hỗ trợ resume từ checkpoint."""
    # Kiểm tra resume checkpoint
    resume_path = os.path.join(config.checkpoint_dir, 'resume.pth')
    start_epoch = 1
    start_batch = 0
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        start_batch = ckpt['batch']
        print(f'Resume từ epoch {start_epoch}, batch {start_batch}')

    best_auc = 0.0
    for epoch in range(1, config.epochs + 1):
        # Bỏ qua các epoch đã train xong
        if epoch < start_epoch:
            continue

        model.train()
        # Chỉ truyền start_batch cho epoch đang resume, các epoch sau bắt đầu từ 0
        current_start_batch = start_batch if epoch == start_epoch else 0
        result = train_epoch(epoch, model, train_loader, criterion, optimizer, lr_scheduler,
                             train_metrics, config, device, start_batch=current_start_batch)

        model.eval()
        val_result = valid_epoch(epoch, model, valid_loader, criterion, valid_metrics, config, device)

        # Lưu checkpoint cuối epoch
        best = val_result.get('auc', 0) > best_auc
        if best:
            best_auc = val_result.get('auc', 0)
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

        # Xóa resume checkpoint sau khi hoàn thành epoch
        if os.path.exists(resume_path):
            os.remove(resume_path)

    print(f'Training hoàn tất. Best AUC: {best_auc:.4f}')