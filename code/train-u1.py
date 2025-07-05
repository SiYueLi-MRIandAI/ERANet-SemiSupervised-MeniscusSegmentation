import os
import sys
import time
import logging
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

# Custom imports
from utils.optim import DecayLR
from utils.metrics import diceCoeffv2
from utils.losses import (
    SoftDiceLoss,
    weight_self_pro_softmax_mse_loss
)
from networks.unet_loss import UNet_pro
from dataloaders.oai_meniscus import OAIMeniscusERA


def get_args():
    parser = argparse.ArgumentParser(description='Semi-supervised Meniscus Segmentation')
    parser.add_argument('--exp', type=str, default='experiment', help='Experiment name')
    parser.add_argument('--oai_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--csv_train_lb', type=str, default='train_lb.csv')
    parser.add_argument('--csv_train_unlb', type=str, default='train_unlb.csv')
    parser.add_argument('--csv_vali', type=str, default='val.csv')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--base_lr', type=float, default=1e-2)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=float, default=10.0)
    parser.add_argument('--gpu_num', type=str, default='0')
    return parser.parse_args()


def set_deterministic(seed=42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_current_consistency_weight(epoch, max_weight, rampup_length):
    if rampup_length == 0:
        return max_weight
    else:
        current = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return max_weight * float(np.exp(-5.0 * phase * phase))


def one_hot(masks):
    one_hot_list = []
    for class_id in range(5):
        class_map = (masks == class_id).float()
        one_hot_list.append(class_map)
    return one_hot_list


def create_model(device='cuda', ema=False):
    model = UNet_pro(in_chns=1, class_num=5).to(device)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def main():
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    snapshot_path = os.path.join("../model", args.exp)
    log_path = os.path.join(snapshot_path, 'log')
    os.makedirs(snapshot_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.deterministic:
        set_deterministic(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device=device)
    ema_model = create_model(device=device, ema=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        ema_model = nn.DataParallel(ema_model)

    logging.info(f"GPUs used: {torch.cuda.device_count()}")

    # Load data
    classes_oai = ['lateral_meniscus', 'lateral_tibial_cartilage', 'medial_meniscus', 'medial_tibial_cartilage']
    num_classes = len(classes_oai) + 1
    train_lb = OAIMeniscus_ERA(base_dir=args.oai_path, classes=classes_oai, csv_name=args.csv_train_lb, test_flag=False)
    train_unlb = OAIMeniscus(base_dir=args.oai_path, classes=classes_oai, csv_name=args.csv_train_unlb, test_flag=False)
    vali = OAIMeniscus(base_dir=args.oai_path, classes=classes_oai, csv_name=args.csv_vali, test_flag=True)

    trainloader_lb = DataLoader(train_lb, batch_size=args.batch_size, shuffle=True)
    trainloader_unlb = DataLoader(train_unlb, batch_size=args.batch_size, shuffle=True)
    valiloader = DataLoader(vali, batch_size=1, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, DecayLR(args.epochs, 0, args.decay_epochs).step)

    criterion = SoftDiceLoss(num_classes=num_classes)
    self_proloss = weight_self_pro_softmax_mse_loss

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        unlb_iter = iter(trainloader_unlb)
        lb_iter = iter(trainloader_lb)

        for _ in range(len(trainloader_unlb)):
            try:
                lb_batch = next(lb_iter)
            except:
                lb_iter = iter(trainloader_lb)
                lb_batch = next(lb_iter)

            ulb_batch = next(unlb_iter)
            images_lb, labels_lb = lb_batch['image'].to(device), lb_batch['label'].to(device)
            images_ulb = ulb_batch['image'].to(device)

            out_lb, sp_lb, cp_lb, entropy_lb = model(images_lb)
            out_ulb, sp_ulb, cp_ulb, entropy_ulb = model(images_ulb)

            label_ce = F.cross_entropy(out_lb, labels_lb.squeeze(1).long())
            label_dice = criterion(F.softmax(out_lb, dim=1), labels_lb)
            supervised_loss = label_ce + label_dice

            with torch.no_grad():
                ema_out, _, _, _ = ema_model(images_ulb)
                ema_mask = torch.argmax(F.softmax(ema_out, dim=1), dim=1)

            unsup_loss = F.cross_entropy(out_ulb, ema_mask)
            consistency_weight = get_current_consistency_weight(epoch, args.consistency, args.consistency_rampup)
            cons_loss = torch.mean(self_proloss(sp_ulb, out_ulb, entropy_ulb))

            total_loss = supervised_loss + unsup_loss + consistency_weight * cons_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema_variables(model, ema_model, args.ema_decay, global_step)
            global_step += 1

        scheduler.step()
        logging.info(f"Epoch {epoch+1}/{args.epochs} completed. Supervised Loss: {supervised_loss.item():.4f}")


if __name__ == '__main__':
    main()
