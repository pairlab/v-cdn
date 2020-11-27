import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from config import gen_args
from data import PhysicsDataset, load_data, resize_and_crop, pil_loader
from models_kp import KeyPointNet
from utils import count_parameters, Tee, AverageMeter, to_np, to_var, store_data

from data import normalize, denormalize

args = gen_args()

use_gpu = torch.cuda.is_available()

'''
model
'''

model_kp = KeyPointNet(args, use_gpu=use_gpu)

# print model #params
print("model #params: %d" % count_parameters(model_kp))

if args.stage == 'kp':
    if args.eval_kp_epoch == -1:
        model_path = os.path.join(args.outf_kp, 'net_best.pth')
    else:
        model_path = os.path.join(
            args.outf_kp, 'net_kp_epoch_%d_iter_%d.pth' % (args.eval_kp_epoch, args.eval_kp_iter))

    print("Loading saved ckp from %s" % model_path)
    model_kp.load_state_dict(torch.load(model_path))
    model_kp.eval()

    if use_gpu:
        model_kp.cuda()

criterionMSE = nn.MSELoss()


'''
data
'''
data_dir = os.path.join(args.dataf, args.eval_set)
data_store_dir = os.path.join(args.dataf + '_nKp_%d' % args.n_kp, args.eval_set)
if args.store_result:
    os.system('mkdir -p ' + data_store_dir)

loader = pil_loader

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


'''
store results
'''
os.system('mkdir -p ' + args.evalf)

log_path = os.path.join(args.evalf, 'log.txt')
tee = Tee(log_path, 'w')


def evaluate(roll_idx, video=True, image=True):

    eval_path = os.path.join(args.evalf, str(roll_idx))

    n_split = 4
    split = 4

    if image:
        os.system('mkdir -p ' + eval_path)
        print('Save images to %s' % eval_path)

    if video:
        video_path = eval_path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % video_path)
        frame_rate = 25 if args.env in ['Ball'] else 60
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (
            400 * n_split + split * (n_split - 1), 400))

    # load images
    imgs = []
    suffix = '.png' if args.env in ['Ball'] else '.jpg'
    for i in range(args.eval_st_idx, args.eval_ed_idx):
        img_path = os.path.join(data_dir, str(roll_idx), 'fig_%d%s' % (i, suffix))
        img = loader(img_path)

        img = resize_and_crop('valid', img, args.scale_size, args.crop_size)
        img = trans_to_tensor(img).unsqueeze(0).cuda()
        imgs.append(img)

    imgs = torch.cat(imgs, 0)


    '''
    model prediction
    '''

    loss_rec_acc = 0.
    loss_kp_acc = 0.
    for i in range(args.eval_ed_idx - args.eval_st_idx):

        if args.stage == 'kp':
            img = imgs[i:i+1]

            if i == 0:
                src = img.clone()

            with torch.set_grad_enabled(False):
                # reconstruct the target image using the source image
                img_pred, _, _ = model_kp(src, img)
                # predict the position of the keypoints
                keypoint = model_kp.predict_keypoint(img)
                # transform the keypoints to the heatmap
                heatmap = model_kp.keypoint_to_heatmap(keypoint, inv_std=args.inv_std)

            if args.store_result == 1:
                timesteps = args.eval_ed_idx - args.eval_st_idx
                if i == 0:
                    store_kp_result = np.zeros((timesteps, args.n_kp, 2))

                store_kp_result[i] = to_np(keypoint[0])

                if i == timesteps - 1:
                    store_data(['keypoints'], [store_kp_result], os.path.join(data_store_dir, '%d.h5' % roll_idx))

        if args.store_demo == 1:
            # transform the numpy
            img_pred = to_np(torch.clamp(img_pred, -1., 1.))[0].transpose(1, 2, 0)[:, :, ::-1]
            img_pred = (img_pred * 0.5 + 0.5) * 255.
            img_pred = cv2.resize(img_pred, (400, 400))

            lim = args.lim
            keypoint = to_np(keypoint)[0] - [lim[0], lim[2]]
            keypoint *= 400 / 2.
            keypoint = np.round(keypoint).astype(np.int)

            heatmap = to_np(heatmap)[0].transpose((1, 2, 0))
            heatmap = np.sum(heatmap, 2)

            # cv2.imshow('heatmap', heatmap)
            # cv2.waitKey(0)

            heatmap = np.clip(heatmap * 255., 0., 255.)
            heatmap = cv2.resize(heatmap, (400, 400), interpolation=cv2.INTER_NEAREST)
            heatmap = np.expand_dims(heatmap, -1)

            # generate the visualization
            img_path = os.path.join(data_dir, str(roll_idx), 'fig_%d%s' % (i + args.eval_st_idx, suffix))
            img = cv2.imread(img_path)
            img = cv2.resize(img, (400, 400)).astype(np.float)
            img_overlay = img.copy()
            kp_map = np.zeros((img.shape[0], img.shape[1], 3))

            c = [(255, 105, 65), (0, 69, 255), (50, 205, 50), (0, 165, 255), (238, 130, 238),
                 (128, 128, 128), (30, 105, 210), (147, 20, 255), (205, 90, 106), (0, 215, 255)]

            if args.env in ['Ball']:
                for j in range(keypoint.shape[0]):
                    cv2.circle(kp_map, (keypoint[j, 0], keypoint[j, 1]), 12, c[j], -1)
                    cv2.circle(kp_map, (keypoint[j, 0], keypoint[j, 1]), 12, (255, 255, 255), 1)
                    cv2.circle(img_overlay, (keypoint[j, 0], keypoint[j, 1]), 12, c[j], -1)
                    cv2.circle(img_overlay, (keypoint[j, 0], keypoint[j, 1]), 12, (255, 255, 255), 1)
            elif args.env in ['Cloth']:
                for j in range(keypoint.shape[0]):
                    cv2.circle(kp_map, (keypoint[j, 0], keypoint[j, 1]), 8, c[j], -1)
                    cv2.circle(kp_map, (keypoint[j, 0], keypoint[j, 1]), 8, (255, 255, 255), 1)
                    cv2.circle(img_overlay, (keypoint[j, 0], keypoint[j, 1]), 8, c[j], -1)
                    cv2.circle(img_overlay, (keypoint[j, 0], keypoint[j, 1]), 8, (255, 255, 255), 1)

            merge = np.zeros((img.shape[0], img.shape[1] * n_split + split * (n_split - 1), 3)) * 255.

            if args.stage == 'kp':
                merge[:, :img.shape[1]] = img
                merge[:, img.shape[1] + 4 : img.shape[1] * 2 + 4] = img_overlay
                merge[:, img.shape[1] * 2 + 8 : img.shape[1] * 3 + 8] = heatmap
                merge[:, img.shape[1] * 3 + 12 : img.shape[1] * 4 + 12] = img_pred

            merge = merge.astype(np.uint8)

            if image:
                cv2.imwrite(os.path.join(eval_path, 'fig_%d.png' % i), merge)

            if video:
                out.write(merge)

    if video:
        out.release()


ls_rollout_idx = np.arange(args.store_st_idx, args.store_ed_idx)
bar = ProgressBar()
for roll_idx in bar(ls_rollout_idx):
    if args.store_demo == 1:
        evaluate(roll_idx, video=True, image=True)
    else:
        evaluate(roll_idx, video=False, image=False)

