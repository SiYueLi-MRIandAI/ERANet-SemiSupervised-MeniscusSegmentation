import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, mode_upsampling=1):
        super(UpBlock, self).__init__()
        self.mode_upsampling = mode_upsampling
        if mode_upsampling == 0:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        else:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            mode = ['bilinear', 'nearest', 'bicubic'][mode_upsampling - 1]
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=(mode == 'bilinear' or mode == 'bicubic'))
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.mode_upsampling != 0:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        ft_chns = params['feature_chns']
        self.in_conv = ConvBlock(params['in_chns'], ft_chns[0], params['dropout'][0])
        self.down1 = DownBlock(ft_chns[0], ft_chns[1], params['dropout'][1])
        self.down2 = DownBlock(ft_chns[1], ft_chns[2], params['dropout'][2])
        self.down3 = DownBlock(ft_chns[2], ft_chns[3], params['dropout'][3])
        self.down4 = DownBlock(ft_chns[3], ft_chns[4], params['dropout'][4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


def masked_average_pooling(feature, mask):
    mask = F.interpolate(mask, size=feature.shape[-2:], mode='bilinear', align_corners=True)
    return torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)


def batch_prototype(feature, mask):
    B, C = mask.shape[0], mask.shape[1]
    batch_pro = torch.zeros(B, C, feature.shape[1], device=feature.device)
    for i in range(C):
        batch_pro[:, i, :] = masked_average_pooling(feature, mask[:, i:i + 1])
    return batch_pro


def similarity_calculation(feature, batchpro):
    B, C, H, W = feature.shape
    feature = F.normalize(feature.view(B, C, -1).transpose(1, 2), dim=2).reshape(-1, C)
    batchpro = F.normalize(batchpro.reshape(-1, C), dim=1)
    sim = torch.mm(feature, batchpro.T).view(B, H * W, B, C).permute(0, 1, 2, 3)
    return sim


def self_similarity(sim):
    B, N, _, C = sim.shape
    return torch.stack([sim[i, :, i, :] for i in range(B)], dim=0)


def other_similarity(sim):
    sim = torch.exp(sim)
    for i in range(sim.size(2)):
        sim[i, :, i, :] = 0
    sim_sum = sim.sum(dim=2)
    return sim_sum / (sim_sum.sum(dim=2, keepdim=True) + 1e-5)


def agreement_weight(sim):
    score_map = torch.argmax(sim, dim=3)
    one_hot = F.one_hot(score_map, num_classes=sim.shape[3]).float()
    avg_onehot = F.normalize(one_hot.sum(2), p=1.0, dim=2)
    entropy = -torch.sum(avg_onehot * torch.log(avg_onehot + 1e-6), dim=2) / np.log(sim.shape[3])
    return 1 - entropy


class Decoder_pro(nn.Module):
    def __init__(self, params):
        super(Decoder_pro, self).__init__()
        ft_chns = params['feature_chns']
        up_type = params['up_type']
        self.up1 = UpBlock(ft_chns[4], ft_chns[3], ft_chns[3], 0.0, up_type)
        self.up2 = UpBlock(ft_chns[3], ft_chns[2], ft_chns[2], 0.0, up_type)
        self.up3 = UpBlock(ft_chns[2], ft_chns[1], ft_chns[1], 0.0, up_type)
        self.up4 = UpBlock(ft_chns[1], ft_chns[0], ft_chns[0], 0.0, up_type)
        self.out_conv = nn.Conv2d(ft_chns[0], params['class_num'], kernel_size=3, padding=1)

    def forward(self, features):
        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        out = self.out_conv(x)
        mask = torch.softmax(out, dim=1)
        proto = batch_prototype(x, mask)
        sim_map = similarity_calculation(x, proto)
        return out, self_similarity(sim_map), other_similarity(sim_map), agreement_weight(sim_map)


class UNet_pro(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet_pro, self).__init__()
        params = {
            'in_chns': in_chns,
            'feature_chns': [64, 128, 256, 512, 1024],
            'dropout': [0.0] * 5,
            'class_num': class_num,
            'up_type': 1,
        }
        self.encoder = Encoder(params)
        self.decoder = Decoder_pro(params)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
