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
# from data_utils import DataLoader
from sklearn.metrics import *
import torch.utils.data as data
import pdb
import glob
from torch.autograd import Variable
import argparse
# import neptune

def train_epoch(epoch, model, data_loader, criterion, optimizer, lr_scheduler, metrics, device=torch.device('cpu')):
    metrics.reset()
    average_loss = []
    # training loop
    for batch_idx, (batch_data) in enumerate(data_loader):
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
    return metrics.result()

def valid_epoch(epoch, model, data_loader, criterion, metrics, device=torch.device('cpu')):
    metrics.reset()
    losses = []
    acc1s = []
    acc5s = []
    # validation loop
    new_label = np.load('../../UIT-ADrone/test/test_frame_mask/DJI_0073.npy')

    with torch.no_grad():
        for batch_idx, (batch_data) in enumerate(data_loader):
            batch_data_256, batch_data = batch_data['256'].to(device), batch_data['standard'].to(device)
            # batch_target = batch_target.to(device)
            batch_pred = model(batch_data[:,:4])
            loss = loss_func_mse(batch_data_256[:,4].float(), batch_pred)
            losses.append(loss.item())

    loss = np.mean(losses)
    frame_auc = roc_auc_score(y_true=new_label[:len(losses)], y_score=losses)
    acc1 = np.mean(acc1s)
    acc5 = np.mean(acc5s)
    metrics.writer.set_step(epoch, 'valid')
    metrics.update('loss', loss)
    metrics.update('acc1', frame_auc)
    # metrics.update('acc5', acc5)
    print("Test Epoch: {:03d}), AUC@1: {:.2f}".format(epoch, frame_auc))
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