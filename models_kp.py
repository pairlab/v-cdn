import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from data import denormalize, normalize
from utils import load_data


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, lim=[-1., 1., -1., 1.], temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height))

        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints


class KeyPointPredictor(nn.Module):
    def __init__(self, args, lim=[-1., 1., -1., 1.]):
        super(KeyPointPredictor, self).__init__()

        nf = args.nf_hidden_kp
        k = args.n_kp
        norm_layer = args.norm_layer

        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(3, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # fesrcat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
            # feat size (n_kp) x 16 x 16
        ]

        self.model = nn.Sequential(*sequence)
        self.integrater = SpatialSoftmax(
            height=args.height//4, width=args.width//4, channel=k, lim=lim)

    def integrate(self, heatmap):
        return self.integrater(heatmap)

    def forward(self, img):
        heatmap = self.model(img)
        return self.integrate(heatmap)


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()

        nf = args.nf_hidden_kp
        norm_layer = args.norm_layer

        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(3, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, img):
        return self.model(img)


class Refiner(nn.Module):
    def __init__(self, args):
        super(Refiner, self).__init__()

        nf = args.nf_hidden_kp
        k = args.n_kp
        norm_layer = args.norm_layer

        sequence = [
            # input is (nf * 4) x 16 x 16
            nn.ConvTranspose2d(nf * 4, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, feat):
        return self.model(feat)


class KeyPointNet(nn.Module):
    def __init__(self, args, use_gpu=True):
        super(KeyPointNet, self).__init__()

        self.args = args
        self.use_gpu = use_gpu

        # visual feature extractor
        self.feature_extractor = FeatureExtractor(args)

        # key point predictor
        self.keypoint_predictor = KeyPointPredictor(args, lim=args.lim)

        # map the feature back to the image
        self.refiner = Refiner(args)

        lim = args.lim
        x = np.linspace(lim[0], lim[1], args.width // 4)
        y = np.linspace(lim[2], lim[3], args.height // 4)
        z = np.linspace(-1., 1., args.n_kp)

        if use_gpu:
            self.x = Variable(torch.FloatTensor(x)).cuda()
            self.y = Variable(torch.FloatTensor(y)).cuda()
            self.z = Variable(torch.FloatTensor(z)).cuda()
        else:
            self.x = Variable(torch.FloatTensor(x))
            self.y = Variable(torch.FloatTensor(y))
            self.z = Variable(torch.FloatTensor(z))

    def extract_feature(self, img):
        # img: B x 3 x H x W
        # ret: B x (nf * 4) x (H / 4) x (W / 4)
        return self.feature_extractor(img)

    def predict_keypoint(self, img):
        # img: B x 3 x H x W
        # ret: B x n_kp x 2
        return self.keypoint_predictor(img)

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        # keypoint: B x n_kp x 2
        # heatpmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (H / 4) x (W / 4)
        height = self.args.height // 4
        width = self.args.width // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def transport(self, src_feat, des_feat, src_hmap, des_hmap, des_feat_hmap=None):
        # src_feat: B x (nf * 4) x (H / 4) x (W / 4)
        # des_feat: B x (nf * 4) x (H / 4) x (W / 4)
        # src_hmap: B x n_kp x (H / 4) x (W / 4)
        # des_hmap: B x n_kp x (H / 4) x (W / 4)
        # des_feat_hmap = des_hmap * des_feat: B x (nf * 4) x (H / 4) * (W / 4)
        # mixed_feat: B x (nf * 4) x (H / 4) x (W / 4)
        src_hmap = torch.sum(src_hmap, 1, keepdim=True)
        des_hmap = torch.sum(des_hmap, 1, keepdim=True)
        src_digged = src_feat * (1. - src_hmap) * (1. - des_hmap)

        # print(src_digged.size())
        # print(des_hmap.size())
        # print(des_feat.size())
        if des_feat_hmap is None:
            mixed_feat = src_digged + des_hmap * des_feat
        else:
            mixed_feat = src_digged + des_feat_hmap

        return mixed_feat

    def refine(self, mixed_feat):
        # mixed_feat: B x (nf * 4) x (H / 4) x (W / 4)
        # ret: B x 3 x H x W
        return self.refiner(mixed_feat)

    def kp_feat(self, feat, hmap):
        # feat: B x (nf * 4) x (H / 4) x (W / 4)
        # hmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (nf * 4)
        B, nf, H, W = feat.size()
        n_kp = hmap.size(1)

        p = feat.view(B, 1, nf, H, W) * hmap.view(B, n_kp, 1, H, W)
        kp_feat = torch.sum(p, (3, 4))
        return kp_feat

    def forward(self, src, des):
        # src: B x 3 x H x W
        # des: B x 3 x H x W
        # des_pred: B x 3 x H x W
        cat = torch.cat([src, des], 0)
        feat = self.extract_feature(cat)
        kp = self.predict_keypoint(cat)
        B = kp.size(0)

        src_feat, des_feat = feat[:B//2], feat[B//2:]
        src_kp, des_kp = kp[:B//2], kp[B//2:]

        src_hmap = self.keypoint_to_heatmap(src_kp, self.args.inv_std)
        des_hmap = self.keypoint_to_heatmap(des_kp, self.args.inv_std)

        src_kp_feat = self.kp_feat(src_feat, src_hmap)
        des_kp_feat = self.kp_feat(des_feat, des_hmap)

        mixed_feat = self.transport(src_feat, des_feat, src_hmap, des_hmap)
        des_pred = self.refine(mixed_feat)

        return des_pred, src_kp_feat, des_kp_feat

