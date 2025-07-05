import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from networks.unet_loss import UNet_pro
from utils.metrics import diceCoeffv2
from dataloaders.oai_meniscus import OAIMeniscus


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 5 checkpoints and split unlabeled data')
    parser.add_argument('--oai_path', type=str, default='')
    parser.add_argument('--csv_vali', type=str, default='.csv')
    parser.add_argument('--gpu_num', type=str, default='0')
    parser.add_argument('--save_dir', type=str, default='../results_5/PseudoLabel')
    parser.add_argument('--dice_threshold', type=float, default=0.8)
    return parser.parse_args()


def one_hot(masks):
    return [(masks == i).float() for i in range(5)]


def load_model(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path)
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    return model


def evaluate_and_filter(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
    os.makedirs(args.save_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.save_dir, 'testing_results.txt'),
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    device = torch.device('cuda')
    model = UNet_pro(in_chns=1, class_num=5).to(device)
    model.eval()

    checkpoint_paths = [
        './pretrained/one_labeled_epoch_20.pth',
        './pretrained/one_labeled_epoch_40.pth',
        './pretrained/one_labeled_epoch_60.pth',
        './pretrained/one_labeled_epoch_80.pth',
        './pretrained/one_labeled_epoch_100.pth'
    ]

    dataset = OAIMeniscus(
        base_dir=args.oai_path,
        classes=['lateral_meniscus','lateral_tibial_cartilage','medial_meniscus','medial_tibial_cartilage'],
        csv_name=args.csv_vali,
        test_flag=True
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    below_threshold_names = []

    for i, batch in enumerate(dataloader):
        image = batch['image'].to(device)
        npy_name = batch['name'][0]

        predictions = []
        for ckpt in checkpoint_paths:
            model = load_model(ckpt, model)
            with torch.no_grad():
                pred = model(image)
            predictions.append(pred)

        # Reference pseudo mask from last checkpoint
        soft = F.softmax(predictions[0], dim=1)
        pseudo_mask = torch.argmax(soft, dim=1, keepdim=True)
        ref_masks = one_hot(pseudo_mask)

        for pred in predictions[1:]:
            soft_pred = F.softmax(pred, dim=1)
            mask_pred = torch.argmax(soft_pred, dim=1, keepdim=True)
            masks = one_hot(mask_pred)

            dices = [diceCoeffv2(m.cuda(), r.cuda()).item() for m, r in zip(masks, ref_masks)]
            mean_dice = np.mean(dices)

            logging.info(f"{npy_name}: Dice = {mean_dice:.4f} | Class-wise: {['{:.4f}'.format(d) for d in dices]}")

            if mean_dice < args.dice_threshold:
                below_threshold_names.append(npy_name)

    below_threshold_names = sorted(set(below_threshold_names))
    df = pd.DataFrame({'Npy Names': below_threshold_names})
    excel_file = os.path.join(args.save_dir, f'unreliable_pseudo_labels_thresh_{args.dice_threshold}.xlsx')
    df.to_excel(excel_file, index=False)
    print(f"Saved to: {excel_file}")


if __name__ == '__main__':
    args = parse_args()
    evaluate_and_filter(args)
