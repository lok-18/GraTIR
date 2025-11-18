from PIL import Image
import numpy as np
from torch.autograd import Variable
# from model_gnn import backbone
from model.model_gnn_sgg import backbone  
from dataset import fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from loss import fusion_loss, SGGFinetuneLoss
import torch
from torch.utils.data import DataLoader
import warnings
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB
from model.device_utils import get_device, to_device, ensure_same_device

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import random

from collections import OrderedDict

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def seed_everything(seed=1118):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(num=0, logger=None):
    seed_everything()
    lr_start = 0.001
    model_path = './checkpoint'
    model_path = os.path.join(model_path)
    os.makedirs(model_path, exist_ok=True)
    
    device = get_device()
    
    train_model = backbone(use_transformer=True)
    train_model = to_device(train_model)
    train_model.train()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr_start)

    # MSRS
    ir_path = '/home/lijiawei/local/dataset/MSRS/Infrared/train/MSRS/'
    vis_path = '/home/lijiawei/local/dataset/MSRS/Visible/train/MSRS/'

    # M3FD
    # ir_path = '/home/lijiawei/local/dataset/M3FD_only_test/M3FD_Detection/ir_L/'
    # vis_path = '/home/lijiawei/local/dataset/M3FD_only_test/M3FD_Detection/vis/'

    train_dataset = fusion_dataset(ir_path=ir_path, vis_path=vis_path)
    print("The length of training dataset:{}".format(train_dataset.length))
    
    fusion_batch_size = 4
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=fusion_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    train_loss = fusion_loss()

    epoch = 10

    st = glob_st = time.time()
    logger.info('Train start!')

    for epo in range(0, epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epo
        for it, (image_vis_sgg, image_ir_sgg, image_vis, image_ir, name) in enumerate(train_loader):
            train_model.train()
            
            image_vis = to_device(image_vis)
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = to_device(image_ir)
            image_vis_sgg = to_device(image_vis_sgg)
            image_ir_sgg = to_device(image_ir_sgg)

            fused_image = train_model(image_vis_sgg, image_ir_sgg, image_vis_ycrcb, image_ir)
            optimizer.zero_grad()

            loss_total, loss_in, loss_grad, loss_ssim = train_loss(image_vis_ycrcb, image_ir, fused_image)
            loss_total = loss_total
            loss_total.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join(
                [
                    'step: {it}/{max_it}',
                    'loss_total: {loss_total:.4f}',
                    'loss_in: {loss_in:.4f}',
                    'loss_grad: {loss_grad:.4f}',
                    'loss_ssim: {loss_ssim:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]
            ).format(
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_ssim=loss_ssim.item(),
                    time=t_intv,
                    eta=eta,
                )
            logger.info(msg)
            st = ed

    logger.info('Finetune start!')
    
    lr_start = 0.0001
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr_start)
    finetune_loss = SGGFinetuneLoss()
    
    finetune_batch_size = 2
    finetune_loader = DataLoader(
        dataset=train_dataset,
        batch_size=finetune_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    finetune_loader.n_iter = len(finetune_loader)
    
    epoch = 5  

    for epo in range(0, epoch):
        lr_start = 0.0001
        lr_decay = 0.75
        lr_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epo
            
        for it, (image_vis_sgg, image_ir_sgg, image_vis, image_ir, name) in enumerate(finetune_loader):
            train_model.train()
            
            image_vis = to_device(image_vis)
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = to_device(image_ir)
            image_vis_sgg = to_device(image_vis_sgg)
            image_ir_sgg = to_device(image_ir_sgg)

            fused_image = train_model(image_vis_sgg, image_ir_sgg, image_vis_ycrcb, image_ir)
            
            loss_total = finetune_loss(image_ir, image_vis, fused_image, train_model)
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = finetune_loader.n_iter * epo + it + 1
            eta = int((finetune_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            
            msg = ', '.join([
                'step: {it}/{max_it}',
                'loss_total: {loss_total:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=now_it,
                max_it=finetune_loader.n_iter * epoch,
                loss_total=loss_total.item(),
                time=t_intv,
                eta=eta,
            )
            logger.info(msg)
            st = ed


    finetuned_model_file = os.path.join(model_path, 'model.pth')
    torch.save(train_model.state_dict(), finetuned_model_file)
    logger.info("Final finetuned model save as: {}".format(finetuned_model_file))
    logger.info('\n')

if __name__ == "__main__":
    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    train(0, logger)
    print("Train and finetune finish!")





