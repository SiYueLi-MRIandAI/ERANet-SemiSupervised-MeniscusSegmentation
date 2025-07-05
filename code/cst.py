import os
import sys
import time
import logging
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from networks.unet import UNet
from utils.losses import SoftDiceLoss
from utils.metrics import diceCoeffv2
from utils.optim import DecayLR
from dataloaders.oai_meniscus import OAIMeniscus, OAIMeniscusERA


def parse_args():
    parser = argparse.ArgumentParser(description='Self-Training with Pre-split Reliable/Unreliable Sets (U3 + U4)')
    parser.add_argument('--oai_path', type=str, default='./data')
    parser.add_argument('--csv_train_lb', type=str)
    parser.add_argument('--csv_vali', type=str)
    parser.add_argument('--csv_reliable', type=str)
    parser.add_argument('--csv_unreliable', type=str)
    parser.add_argument('--exp', type=str, default='U4_from_csv_split')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--gpu_num', type=str, default='0')
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def create_model():
    return UNet(n_channels=1, n_classes=5).cuda()


def train_model(model, dataloader, epochs, lr, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, DecayLR(epochs, 0, 0).step)
    criterion = SoftDiceLoss(num_classes=5)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in dataloader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            loss = criterion(F.softmax(out, dim=1), y) + F.cross_entropy(out, y.squeeze(1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.4f}")
        scheduler.step()
    return model


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    set_seed()

    snapshot_path = os.path.join("../model", args.exp)
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("Step 1: Load labeled + reliable (for U3 training)...")
    ds_lb = OAIMeniscusERA(args.oai_path, ['lateral_meniscus','lateral_tibial_cartilage','medial_meniscus','medial_tibial_cartilage'], args.csv_train_lb)
    ds_reliable = OAIMeniscusERA(args.oai_path, ['lateral_meniscus','lateral_tibial_cartilage','medial_meniscus','medial_tibial_cartilage'], args.csv_reliable)
    dl_u3 = DataLoader(torch.utils.data.ConcatDataset([ds_lb, ds_reliable]), batch_size=args.batch_size, shuffle=True)

    logging.info("Step 2: Train U3 on labeled + reliable pseudo-labeled data")
    U3 = create_model()
    U3 = train_model(U3, dl_u3, args.epochs, args.base_lr, device=torch.device("cuda"))

    logging.info("Step 3: Use U3 to relabel unreliable samples")
    ds_unreliable = OAIMeniscusERA(args.oai_path, ['lateral_meniscus','lateral_tibial_cartilage','medial_meniscus','medial_tibial_cartilage'], args.csv_unreliable, test_flag=True)
    dl_unreliable = DataLoader(ds_unreliable, batch_size=1, shuffle=False)

    updated_pairs = []
    U3.eval()
    with torch.no_grad():
        for batch in dl_unreliable:
            img = batch['image'].cuda()
            pred = torch.argmax(F.softmax(U3(img), dim=1), dim=1)
            updated_pairs.append({'image': img.squeeze(0).cpu(), 'label': pred.squeeze(0).cpu()})

    class PseudoDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return {'image': self.examples[idx]['image'], 'label': self.examples[idx]['label']}

    ds_updated_unrel = PseudoDataset(updated_pairs)

    logging.info("Step 4: Train U4 on labeled + reliable + updated-unreliable pseudo-labeled data")
    full_trainset = torch.utils.data.ConcatDataset([ds_lb, ds_reliable, ds_updated_unrel])
    dl_u4 = DataLoader(full_trainset, batch_size=args.batch_size, shuffle=True)
    U4 = create_model()
    train_model(U4, dl_u4, args.epochs, args.base_lr, device=torch.device("cuda"))

    logging.info("Self-training complete. U4 ready.")


if __name__ == '__main__':
    main()
