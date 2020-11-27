import os
import random

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
from data import PhysicsDataset, load_data
from models_kp import KeyPointNet
from utils import rand_int, count_parameters, Tee, AverageMeter, get_lr, set_seed

args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.outf_kp)
os.system('mkdir -p ' + args.dataf)

if args.stage == 'kp':
    tee = Tee(os.path.join(args.outf_kp, 'train.log'), 'w')
else:
    raise AssertionError("Unsupported stage %s" % args.stage)

print(args)

# generate data
trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = PhysicsDataset(args, phase=phase, trans_to_tensor=trans_to_tensor)

    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])

args.stat = datasets['train'].stat

use_gpu = torch.cuda.is_available()


'''
define model for keypoint detection
'''
model_kp = KeyPointNet(args, use_gpu=use_gpu)
print("model_kp #params: %d" % count_parameters(model_kp))

if args.stage == 'kp':
    if args.kp_epoch >= 0:
        model_kp_path = os.path.join(
            args.outf_kp, 'net_kp_epoch_%d_iter_%d.pth' % (args.kp_epoch, args.kp_iter))
        print("Loading saved ckp from %s" % model_kp_path)
        model_kp.load_state_dict(torch.load(model_kp_path))

# criterion
criterionMSE = nn.MSELoss()

# optimizer
if args.stage == 'kp':
    params = model_kp.parameters()
else:
    raise AssertionError('Unknown stage %s' % args.stage)

optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)

if use_gpu:
    model_kp = model_kp.cuda()
    criterionMSE = criterionMSE.cuda()

if args.stage == 'kp':
    st_epoch = args.kp_epoch if args.kp_epoch > 0 else 0
    log_fout = open(os.path.join(args.outf_kp, 'log_st_epoch_%d.txt' % st_epoch), 'w')

best_valid_loss = np.inf

for epoch in range(st_epoch, args.n_epoch):
    phases = ['train', 'valid'] if args.eval == 0 else ['valid']

    for phase in phases:
        model_kp.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_rec = AverageMeter()

        bar = ProgressBar(max_value=data_n_batches[phase])
        loader = dataloaders[phase]

        for i, data in bar(enumerate(loader)):

            if use_gpu:
                if isinstance(data, list):
                    data = [d.cuda() for d in data]
                else:
                    data = data.cuda()

            with torch.set_grad_enabled(phase == 'train'):
                if args.stage == 'kp':
                    src, des = data

                    des_pred, src_kp_feat, des_kp_feat = model_kp(src, des)

                    # reconstruction loss
                    loss_rec = criterionMSE(des_pred, des) * 10.
                    loss = loss_rec

                    meter_loss.update(loss.item(), src.size(0))

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % args.log_per_iter == 0:
                log = '%s [%d/%d][%d/%d] Loss: %.6f (%.6f), LR: %.6f' % (
                    phase, epoch, args.n_epoch, i, data_n_batches[phase],
                    loss.item(), meter_loss.avg,
                    get_lr(optimizer))

                print()
                print(log)
                log_fout.write(log + '\n')
                log_fout.flush()

            if phase == 'train' and i % args.ckp_per_iter == 0:
                if args.stage == 'kp':
                    torch.save(model_kp.state_dict(), '%s/net_kp_epoch_%d_iter_%d.pth' % (args.outf_kp, epoch, i))

        log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss)
        print(log)
        log_fout.write(log + '\n')
        log_fout.flush()

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg

                if args.stage == 'kp':
                    torch.save(model_kp.state_dict(), '%s/net_best.pth' % (args.outf_kp))

log_fout.close()
