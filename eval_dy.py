import os
import time
import random
import itertools
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'Times New Roman'
plt.rcParams["font.size"] = 12

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
from data import PhysicsDataset, load_data, store_data, resize_and_crop, pil_loader
from models_kp import KeyPointNet
from models_dy import DynaNetGNN, HLoss
from utils import count_parameters, Tee, AverageMeter, to_np, to_var, norm, set_seed

from data import normalize, denormalize

args = gen_args()

use_gpu = torch.cuda.is_available()

set_seed(args.random_seed)


# used for cnn encoder, minimum input observation length
min_res = args.min_res


'''
model
'''
model_kp = KeyPointNet(args, use_gpu=use_gpu)

# print model #params
print("model #params: %d" % count_parameters(model_kp))

model_kp_path = os.path.join(
    args.outf_kp, 'net_kp_epoch_%d_iter_%d.pth' % (args.eval_kp_epoch, args.eval_kp_iter))
print("Loading saved ckp from %s" % model_kp_path)
model_kp.load_state_dict(torch.load(model_kp_path))
model_kp.eval()



if args.stage == 'dy':

    if args.dy_model == 'mlp':
        model_dy = DynaNetMLP(args, use_gpu=use_gpu)
    elif args.dy_model == 'gnn':
        model_dy = DynaNetGNN(args, use_gpu=use_gpu)

    # print model #params
    print("model #params: %d" % count_parameters(model_dy))

    if args.eval_dy_epoch == -1:
        model_kp_path = os.path.join(args.outf_kp, 'net_best_kp.pth')
        model_dy_path = os.path.join(args.outf_dy, 'net_best_dy.pth')
    else:
        model_dy_path = os.path.join(
            args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.eval_dy_epoch, args.eval_dy_iter))

    print("Loading saved ckp from %s" % model_dy_path)
    model_dy.load_state_dict(torch.load(model_dy_path))
    model_dy.eval()

if use_gpu:
    model_kp.cuda()
    model_dy.cuda()


criterionMSE = nn.MSELoss()
criterionH = HLoss()


'''
data
'''
data_dir = os.path.join(args.dataf, args.eval_set)

if args.env in ['Ball']:
    data_names = ['attrs', 'states', 'actions', 'rels']
elif args.env in ['Cloth']:
    data_names = ['states', 'actions', 'scene_params']

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



def draw_graph(keypoint, edge_type, lim, c, file_name):
    # draw pred confidence
    fig, ax = plt.subplots(1)
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[2], lim[3])

    height = 400.

    for j in range(keypoint.shape[0]):
        x, y = keypoint[j, 0], keypoint[j, 1]
        x = x / height * 2
        x -= lim[1]
        y = y / height * 2
        y -= lim[3]
        y = -y

        if args.vis_edge == 1:
            for k in range(keypoint.shape[0]):
                if k == j:
                    continue
                xx, yy = keypoint[k, 0], keypoint[k, 1]
                xx = xx / height * 2
                xx -= lim[1]
                yy = yy / height * 2
                yy -= lim[3]
                yy = -yy


                edge_type_cur = edge_type[j, k]
                if edge_type_cur < args.edge_st_idx:
                    continue

                dist = norm(np.array([x - xx, y - yy]))

                direct = np.array([x - xx, y - yy]) / dist
                ax.arrow(xx + direct[0] * 0.05, yy + direct[1] * 0.05,
                         x - xx - direct[0] * 0.15, y - yy - direct[1] * 0.15,
                         fc=c[edge_type_cur], ec='w', width=0.02, head_width=0.06, head_length=0.06, alpha=0.5)

        ax.scatter(x, y, c=c[j], s=150)

    ax.set_aspect('equal')
    plt.tight_layout()

    # plt.show()
    plt.savefig(file_name)

    plt.close()



def evaluate(roll_idx, video=True, image=True):
    fwd_loss_mse_cur = []

    eval_path = os.path.join(args.evalf, str(roll_idx))

    split = 4
    if args.env in ['Ball', 'Cloth']:
        n_split_w = 3
        n_split_h = 1

    n_kp = args.n_kp

    if image:
        os.system('mkdir -p ' + eval_path)
        print('Save images to %s' % eval_path)

    if video:
        video_path = eval_path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % video_path)

        width_raw = 400
        height_raw = 400
        out = cv2.VideoWriter(video_path, fourcc, 10, (
            width_raw * n_split_w + split * (n_split_w - 1),
            height_raw * n_split_h + split * (n_split_h - 1)))


    # load images
    fig_suffix = '.png' if args.env == 'Ball' else '.jpg'
    imgs = []
    for i in range(args.eval_st_idx, args.eval_ed_idx):
        img_path = os.path.join(data_dir, str(roll_idx), 'fig_%d%s' % (i * args.frame_offset, fig_suffix))
        img = loader(img_path)

        img = resize_and_crop('valid', img, args.scale_size, args.crop_size)
        img = trans_to_tensor(img).unsqueeze(0).cuda()
        imgs.append(img)

    imgs = torch.cat(imgs, 0)

    # load action
    if args.env in ['Ball']:
        data_path = os.path.join(data_dir, str(roll_idx) + '.h5')
        data = load_data(data_names, data_path)

        actions = data[data_names.index('actions')] / 600.
        actions = torch.FloatTensor(actions).cuda()
        actions_id = actions[args.identify_st_idx:args.identify_ed_idx]

    elif args.env in ['Cloth']:
        data_path = os.path.join(data_dir, str(roll_idx) + '.h5')
        data = load_data(data_names, data_path)

        states = data[data_names.index('states')][::args.frame_offset]
        actions_raw = data[data_names.index('actions')][::args.frame_offset]
        scene_params = data[data_names.index('scene_params')]
        stiffness = scene_params[15]
        ctrl_idx = scene_params[7:15].astype(np.int)

        actions = np.zeros((states.shape[0], 6))
        actions[:, :3] = states[
            np.arange(actions.shape[0]),
            ctrl_idx[actions_raw[:, 0, 0].astype(np.int)],
            :3] / 0.5   # normalize
        actions[:, 3:] = actions_raw[:, 0, 1:] / 0.03   # normalize

        actions = torch.FloatTensor(actions)[:, None, :].repeat(1, args.n_kp, 1)
        actions = actions.cuda()
        actions_id = actions[args.identify_st_idx:args.identify_ed_idx]


    '''
    model prediction
    '''

    ### metadata

    metadata_path = os.path.join(data_dir, str(roll_idx) + '.h5')
    metadata = load_data(data_names, metadata_path)
    if args.env in ['Ball']:
        # graph_gt
        edge_type = metadata[data_names.index('rels')][0, :, 0].astype(np.int)
        edge_attr = metadata[data_names.index('rels')][0, :, 1:]
        edge_type_gt = np.zeros((args.n_kp, args.n_kp, args.edge_type_num))
        edge_attr_gt = np.zeros((args.n_kp, args.n_kp, edge_attr.shape[1]))
        cnt = 0
        # print(edge_type)
        # print(edge_attr)
        for x in range(args.n_kp):
            for y in range(x):
                edge_type_gt[x, y, edge_type[cnt]] = 1.
                edge_type_gt[y, x, edge_type[cnt]] = 1.
                edge_attr_gt[x, y] = edge_attr[cnt]
                edge_attr_gt[y, x] = edge_attr[cnt]
                cnt += 1

        graph_gt_ret = edge_type_gt, edge_attr_gt
        edge_type_gt = torch.FloatTensor(edge_type_gt).cuda()
        edge_attr_gt = torch.FloatTensor(edge_attr_gt).cuda()

        graph_gt = edge_type_gt, edge_attr_gt

        # kps_gt
        kps = metadata[1][args.eval_st_idx:args.eval_ed_idx, :, :2] / 80.
        kps[:, :, 1] *= -1
        kps = torch.FloatTensor(kps).cuda()
        kps_id = metadata[1][args.identify_st_idx:args.identify_ed_idx, :, :2] / 80.
        kps_id = torch.FloatTensor(kps_id).cuda()
        kps_id[:, :, 1] *= -1

        kps_gt = kps
        kps_gt_id = kps_id

        kps = None
        kps_id = None

    elif args.env in ['Cloth']:
        kps = None
        kps_id = None

    '''
    data for identification
    '''
    imgs_id = []
    for i in range(args.identify_st_idx, args.identify_ed_idx):
        img_path = os.path.join(data_dir, str(roll_idx), 'fig_%d%s' % (i * args.frame_offset, fig_suffix))
        img = loader(img_path)

        img = resize_and_crop('valid', img, args.scale_size, args.crop_size)
        img = trans_to_tensor(img).unsqueeze(0).cuda()
        imgs_id.append(img)

    imgs_id = torch.cat(imgs_id, 0)


    ### Evaluate the performance on graph discovery

    with torch.set_grad_enabled(False):
        # extract features for prediction
        feats = model_kp.extract_feature(imgs)
        kps = model_kp.predict_keypoint(imgs)
        hmaps = model_kp.keypoint_to_heatmap(kps, inv_std=args.inv_std)

        # extract features for graph identification
        # feats_id = model_kp.extract_feature(imgs_id)
        kps_id = model_kp.predict_keypoint(imgs_id)
        # hmaps_id = model_kp.keypoint_to_heatmap(kps_id, inv_std=args.inv_std)

        '''
        print(kps_id[0])
        print(kps_gt_id[0])
        '''

        # permute the keypoints to make the calculation of edge accuracy correct
        if args.env in ['Ball']:
            permu_node_list = list(itertools.permutations(np.arange(args.n_kp)))

            permu_node_error = np.inf
            permu_node_idx = None
            for ii in permu_node_list:
                p = np.array(ii)
                kps_permuted = kps[:, p]

                error = torch.mean((kps_permuted - kps_gt)**2).item()
                if error < permu_node_error:
                    permu_node_error = error
                    permu_node_idx = p

            print('selected node permu', permu_node_idx)

            kps = kps[:, permu_node_idx]
            kps_id = kps_id[:, permu_node_idx]


        graphs = []
        for i in range(min_res, kps_id.size(0) + 1):
            edge_type_distribution = 0
            edge_attr_distribution = []

            if args.baseline == 1:
                graph = model_dy.init_graph(kps_id[:i].unsqueeze(0), use_gpu=True, hard=True)
            else:
                graph = model_dy.graph_inference(
                    kps_id[:i].unsqueeze(0),
                    actions_id[:i].unsqueeze(0) if actions_id is not None else None,
                    hard=True, env=args.env)
            graphs.append(graph)    # append the inferred graph

        # edge_type_logits = graph[3][:, :, :, -1].view(-1, args.edge_type_num)
        edge_type_logits = graphs[-1][3].view(-1, args.edge_type_num)
        loss_H = -criterionH(edge_type_logits, args.prior)

        edge_attr, edge_type_logits = graphs[-1][1], graphs[-1][3]
        graph_pred_ret = to_np(edge_attr[0]), to_np(edge_type_logits[0])


        if args.env in ['Ball']:
            # record the inferred graph over different observation length
            idx_gt = torch.argmax(edge_type_gt, dim=2)
            idx_pred = torch.argmax(edge_type_logits[0], dim=2)
            assert idx_gt.size() == torch.Size([n_kp, n_kp])
            assert idx_pred.size() == torch.Size([n_kp, n_kp])

            idx_gt = to_np(idx_gt)
            idx_pred = to_np(idx_pred)

            permu_edge_list = list(itertools.permutations(np.arange(args.edge_type_num)))
            permu_edge_acc = 0.
            permu_edge_idx = None
            for ii in permu_edge_list:
                p = np.array(ii)
                idx_mapped = p[idx_gt]
                acc = np.logical_and(idx_mapped == idx_pred, np.logical_not(np.eye(n_kp)))
                acc = np.sum(acc) / (n_kp * (n_kp - 1))

                if acc > permu_edge_acc:
                    permu_edge_acc = acc
                    permu_edge_idx = p

            if args.env in ['Ball']:
                # permu_edge_idx = np.array([0, 2, 1])
                permu_edge_idx = np.array([0, 1, 2])

            print('selected edge premu', permu_edge_idx)

            # record the edge type accuracy over time
            acc_over_time = np.zeros(len(graphs))
            ent_over_time = np.zeros(len(graphs))
            for i in range(len(graphs)):
                edge_type_logits_cur = graphs[i][3][0]

                # accuracy
                idx_pred = torch.argmax(edge_type_logits_cur, dim=2)
                assert idx_pred.size() == torch.Size([n_kp, n_kp])

                idx_pred = to_np(idx_pred)

                idx_mapped = permu_edge_idx[idx_gt]
                tmp = np.logical_and(idx_mapped == idx_pred, np.logical_not(np.eye(n_kp)))
                acc_over_time[i] = np.sum(tmp) / (n_kp * (n_kp - 1))

                # entropy
                ent = F.softmax(edge_type_logits_cur, dim=2) * F.log_softmax(edge_type_logits_cur, dim=2)
                ent = -ent.sum(2)
                ent = ent.mean().item()
                ent_over_time[i] = ent

            print("Edge accuracy over different observation length:")
            print(acc_over_time)
            print("Entropy on edge distribution over different observation length:")
            print(ent_over_time)

            # record the edge param correlation over time
            cor_over_time_raw = []
            for i in range(len(graphs)):
                edge_attr_np = to_np(graphs[i][1][0])
                edge_attr_gt_np = graph_gt_ret[1]
                # print(edge_attr_np.shape, edge_attr_gt_np.shape)

                if args.env in ['Ball']:
                    idx_rel = np.argmax(graph_gt_ret[0], axis=2)
                    idx_empty = np.logical_and(idx_rel == 0, np.logical_not(np.eye(n_kp)))
                    idx_spring = np.logical_and(idx_rel == 1, np.logical_not(np.eye(n_kp)))
                    idx_rod = np.logical_and(idx_rel == 2, np.logical_not(np.eye(n_kp)))

                    cor_over_time_raw.append([
                        [edge_attr_np[idx_empty], edge_attr_gt_np[idx_empty]],
                        [edge_attr_np[idx_spring], edge_attr_gt_np[idx_spring]],
                        [edge_attr_np[idx_rod], edge_attr_gt_np[idx_rod]]])

            over_time_results = acc_over_time, ent_over_time, cor_over_time_raw

        else:
            # record the entropy over edge type over time
            ent_over_time = np.zeros(len(graphs))
            for i in range(len(graphs)):
                edge_type_logits_cur = graphs[i][3][0]

                # entropy
                ent = F.softmax(edge_type_logits_cur, dim=2) * F.log_softmax(edge_type_logits_cur, dim=2)
                ent = -ent.sum(2)
                ent = ent.mean().item()
                ent_over_time[i] = ent

            print("Entropy on edge distribution over different observation length:")
            print(ent_over_time)

            over_time_results = ent_over_time

    ### Evaluate the performance on forward prediction

    # the current keypoint state
    eps = 5e-2
    kp_cur = kps[:args.n_his].view(1, args.n_his, args.n_kp, 2)
    covar_gt = torch.FloatTensor(np.array([eps, 0., 0., eps])).cuda()
    covar_gt = covar_gt.view(1, 1, 1, 4).repeat(1, args.n_his, args.n_kp, 1)
    kp_cur = torch.cat([kp_cur, covar_gt], 3)
    # kp_cur = kps[:args.n_his].view(1, args.n_his, args.n_kp, 2)

    loss_kp_acc = 0.
    n_roll = args.eval_ed_idx - args.eval_st_idx - args.n_his

    for i in range(args.eval_ed_idx - args.eval_st_idx):

        if args.stage == 'dy':

            if i >= args.n_his:

                with torch.set_grad_enabled(False):
                    # predict the feat and hmap at the next time step
                    if actions is not None:
                        action_cur = actions[i-args.n_his+args.eval_st_idx:i+args.eval_st_idx].unsqueeze(0)
                    else:
                        action_cur = None

                    kp_pred = model_dy.dynam_prediction(kp_cur, graph, action_cur, env=args.env)

                    mean_pred, covar_pred = kp_pred[:, :, :2], kp_pred[:, :, 2:].view(1, n_kp, 2, 2)

                # compare with the ground truth
                kp_des = kps[i:i+1]

                loss_kp = criterionMSE(mean_pred, kp_des) * args.lam_kp

                fwd_loss_mse_cur.append(F.mse_loss(mean_pred, kp_des).item())

                # print(loss_rec.item(), loss_kp.item())

                loss_kp_acc += loss_kp.item()

                if i == args.n_his or i % 1 == 0:
                    print("step %d, kp: %.6f (%.6f), H: %.6f" % (
                        i, loss_kp.item(), loss_kp_acc / (i - args.n_his + 1), loss_H.item()))

                # update feat_cur and hmap_cur
                kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

                # img_pred & heatmap
                keypoint = mean_pred
                keypoint_covar = covar_pred
                keypoint_gt = kp_des

            else:
                kp_cur_t = kps[i:i+1]

                keypoint = kp_cur_t
                keypoint_covar = covar_gt[:, -1].view(1, n_kp, 2, 2)
                keypoint_gt = kp_cur_t



        # generate the visualization
        img_path = os.path.join(data_dir, str(roll_idx), 'fig_%d%s' % (
            (i + args.eval_st_idx) * args.frame_offset, fig_suffix))
        img = cv2.imread(img_path).astype(np.float)
        img = cv2.resize(img, (400, 400))
        overlay_gt = img.copy()
        overlay_pred = img.copy()

        c = [(255, 105, 65), (0, 69, 255), (50, 205, 50), (0, 165, 255), (238, 130, 238),
             (128, 128, 128), (30, 105, 210), (147, 20, 255), (205, 90, 106), (0, 215, 255)]

        # draw prediction
        lim = args.lim
        keypoint = to_np(keypoint)[0] - [lim[0], lim[2]]
        keypoint *= 400 / 2.
        keypoint = np.round(keypoint).astype(np.int)
        keypoint_covar = to_np(keypoint_covar[0])

        if args.env in ['Ball']:
            for j in range(keypoint.shape[0]):
                cv2.circle(overlay_pred, (keypoint[j, 0], keypoint[j, 1]), 8, c[j], -1)
                cv2.circle(overlay_pred, (keypoint[j, 0], keypoint[j, 1]), 8, (255, 255, 255), 1)
        elif args.env in ['Cloth']:
            for j in range(keypoint.shape[0]):
                cv2.circle(overlay_pred, (keypoint[j, 0], keypoint[j, 1]), 8, c[j], -1)
                cv2.circle(overlay_pred, (keypoint[j, 0], keypoint[j, 1]), 8, (255, 255, 255), 1)


        # draw gt
        keypoint_gt = to_np(keypoint_gt)[0] - [lim[0], lim[2]]
        keypoint_gt *= 400 / 2.
        keypoint_gt = np.round(keypoint_gt).astype(np.int)

        if args.env in ['Ball']:
            for j in range(keypoint.shape[0]):
                cv2.circle(overlay_gt, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 8, c[j], -1)
                cv2.circle(overlay_gt, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 8, (255, 255, 255), 1)
                # cv2.circle(overlay_pred, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 4, c[j], -1)
                # cv2.circle(overlay_pred, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 4, (255, 255, 255), 1)
        elif args.env in ['Cloth']:
            for j in range(keypoint.shape[0]):
                cv2.circle(overlay_gt, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 8, c[j], -1)
                cv2.circle(overlay_gt, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 8, (255, 255, 255), 1)
                # cv2.circle(overlay_pred, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 8, c[j], -1)
                # cv2.circle(overlay_pred, (keypoint_gt[j, 0], keypoint_gt[j, 1]), 8, (255, 255, 255), 1)



        if image:
            # draw predicted graph
            c = ['royalblue', 'orangered', 'limegreen', 'orange', 'violet',
                 'gray', 'chocolate', 'deeppink', 'slateblue', 'gold']

            file_name=os.path.join(eval_path, 'graph_pred_%d.png' % i)
            draw_graph(
                keypoint,
                edge_type=np.argmax(to_np(
                    edge_type_logits.view(args.n_kp, args.n_kp, args.edge_type_num)), -1),
                lim=lim, c=c,
                file_name=file_name)

            img_graph_pred = cv2.imread(file_name)[28:28+400, 119:119+400]


            # draw ground truth graph
            if args.env in ['Ball']:
                file_name = os.path.join(eval_path, 'graph_gt_%d.png' % i)
                draw_graph(
                    keypoint_gt,
                    edge_type=np.argmax(to_np(edge_type_gt), -1),
                    lim=lim, c=c,
                    file_name=file_name)

                img_graph_gt = cv2.imread(file_name)[28:28+400, 119:119+400]


        if image or video:
            img_h = img_graph_pred.shape[0]
            img_w = img_graph_pred.shape[1]

            merge = np.zeros((
                img_h * n_split_h + split * (n_split_h - 1),
                img_w * n_split_w + split * (n_split_w - 1), 3)) * 255.

            if args.env in ['Ball']:
                overlay_pred = cv2.resize(overlay_pred, (img_w, img_h))
                overlay_gt = cv2.resize(overlay_gt, (img_w, img_h))
                merge[:, :img_w] = img_graph_gt
                merge[:, img_w + split:img_w * 2 + split] = img_graph_pred
                merge[:, img_w * 2 + split * 2:] = overlay_pred

            elif args.env in ['Cloth']:
                merge[:, :img_w] = img_graph_pred
                merge[:, img_w + split:img_w * 2 + split] = overlay_pred
                merge[:, img_w * 2 + split * 2:, :] = overlay_gt

            merge = merge.astype(np.uint8)

        if image:
            cv2.imwrite(os.path.join(eval_path, 'fig_%d.png' % i), merge)

        if video:
            out.write(merge)

    if video:
        out.release()

    print("kp: %.6f" % (loss_kp_acc / n_roll))

    if args.env in ['Ball']:
        return graph_gt_ret, graph_pred_ret, over_time_results, np.array(fwd_loss_mse_cur)
    elif args.env in ['Cloth']:
        return graph_pred_ret, over_time_results, np.array(fwd_loss_mse_cur)


if args.store_demo == 1:
    ls_rollout_idx = np.arange(10)
else:
    ls_rollout_idx = np.arange(200)

bar = ProgressBar()




### visualize the results

edge_acc_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))
edge_ent_over_time_record = np.zeros(
    (len(ls_rollout_idx), args.identify_ed_idx - args.identify_st_idx - min_res + 1))
edge_cor_over_time_raw_record = []


fwd_loss_mse = []

for roll_idx in bar(ls_rollout_idx):
    print()
    print("Eval # %d / %d" % (roll_idx, ls_rollout_idx[-1]))

    if args.env in ['Ball']:
        graph_gt, graph_pred, over_time_results, fwd_loss_mse_cur = evaluate(
            roll_idx, video=args.store_demo, image=args.store_demo)
    elif args.env in ['Cloth']:
        gt_pred, over_time_results, fwd_loss_mse_cur = evaluate(
            roll_idx, video=args.store_demo, image=args.store_demo)

    fwd_loss_mse.append(fwd_loss_mse_cur)

    if args.env in ['Ball']:
        edge_acc_over_time_record[roll_idx] = over_time_results[0]
        edge_ent_over_time_record[roll_idx] = over_time_results[1]
        edge_cor_over_time_raw_record.append(over_time_results[2])
    elif args.env in ['Cloth']:
        edge_ent_over_time_record[roll_idx] = over_time_results

fwd_loss_mse = np.array(fwd_loss_mse)
print()
print('MSE on forward prediction', fwd_loss_mse.shape)
for i in range(fwd_loss_mse.shape[1]):
    print('Step:', i, 'mean: %.6f' % np.mean(fwd_loss_mse[:, i]), 'std: %.6f' % np.std(fwd_loss_mse[:, i]))



def plot_data_mean(ax, data, color, label):
    m, lo, hi = np.mean(data, 0), \
            np.mean(data, 0) - np.std(data, 0), \
            np.mean(data, 0) + np.std(data, 0)
    T = len(m)
    x = np.arange(min_res, min_res + T)
    ax.plot(x, m, '-', color=color, alpha=0.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def plot_data_median(ax, data, color, label):
    m, lo, hi = np.median(data, 0), \
            np.quantile(data, 0.25, 0), \
            np.quantile(data, 0.75, 0)
    T = len(m)
    x = np.arange(min_res, min_res + T)
    ax.plot(x, m, '-', color=color, alpha=0.8, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


# plot edge accuracy over time
if args.env in ['Ball']:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    plot_data_median(ax, edge_acc_over_time_record, color='b', label='Acc')

    # plt.legend(loc='best', fontsize=12)
    plt.xlabel('# of observation frames', fontsize=15)
    plt.ylabel('Accuracy on edge type', fontsize=15)
    plt.xlim([min_res, args.identify_ed_idx - args.identify_st_idx])
    plt.ylim([0.6, 1])
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(args.evalf, 'acc.png'))
    plt.savefig(os.path.join(args.evalf, 'acc.pdf'))
    plt.show()


# plot edge entropy over time
if args.env in ['Ball']:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
    plot_data_median(ax, edge_ent_over_time_record, color='b', label='Entropy')

    # plt.legend(loc='best', fontsize=12)
    plt.xlabel('# of observation frames', fontsize=15)
    plt.ylabel('Entropy on edge type', fontsize=15)
    plt.xlim([min_res, args.identify_ed_idx - args.identify_st_idx])
    plt.ylim([0.23, 0.34])
    plt.yticks(np.arange(0.24, 0.35, 0.02))
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(args.evalf, 'ent.png'))
    plt.savefig(os.path.join(args.evalf, 'ent.pdf'))
    plt.show()


# plot edge attr correlation over time
if args.env in ['Ball']:
    edge_cor_over_time_record = []
    for idx_rel in range(args.edge_st_idx, len(edge_cor_over_time_raw_record[0][0])):
        edge_cor_over_time_cur = np.zeros(
            (args.identify_ed_idx - args.identify_st_idx - min_res + 1))

        for i in range(len(edge_cor_over_time_raw_record[0])):
            edge_attr_gt = []
            edge_attr_pred = []

            for j in range(len(edge_cor_over_time_raw_record)):
                edge_attr_gt.append(edge_cor_over_time_raw_record[j][i][idx_rel][1])
                edge_attr_pred.append(edge_cor_over_time_raw_record[j][i][idx_rel][0])

            edge_attr_gt = np.concatenate(edge_attr_gt).reshape(-1)
            edge_attr_pred = np.concatenate(edge_attr_pred).reshape(-1)
            edge_cor_over_time_cur[i] = np.corrcoef(edge_attr_gt, edge_attr_pred)[0, 1]

        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        # plot_data_median(ax, edge_cor_over_time_record, color='b', label='Cor')
        plt.plot(np.arange(min_res, args.identify_ed_idx - args.identify_st_idx + 1),
                 np.abs(edge_cor_over_time_cur))

        plt.xlabel('# of observation frames', fontsize=15)
        plt.ylabel('Correlation on edge attr (Abs)', fontsize=15)
        plt.xlim([min_res, args.identify_ed_idx - args.identify_st_idx])
        plt.ylim([0.8, 0.95])
        plt.yticks(np.arange(0.8, 1.0, 0.05))
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(args.evalf, 'cor_%d.png' % idx_rel))
        plt.savefig(os.path.join(args.evalf, 'cor_%d.pdf' % idx_rel))
        plt.show()

        edge_cor_over_time_record.append(edge_cor_over_time_cur)

# plot the scatter plot on attr at the last step
if args.env in ['Ball']:
    for idx_rel in range(args.edge_st_idx, len(edge_cor_over_time_raw_record[0][0])):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)

        attr_pred = []
        attr_gt = []
        for i in range(len(edge_cor_over_time_raw_record)):
            attr_pred.append(edge_cor_over_time_raw_record[i][-1][idx_rel][0])
            attr_gt.append(edge_cor_over_time_raw_record[i][-1][idx_rel][1])

        attr_pred = np.concatenate(attr_pred, 0).reshape(-1)
        attr_gt = np.concatenate(attr_gt, 0).reshape(-1)

        if idx_rel == 1:
            idx = np.logical_and(attr_pred < 4.5, attr_gt >= 20)
            attr_gt = attr_gt[idx]
            attr_pred = attr_pred[idx]
        elif idx_rel == 2:
            idx = attr_gt <= 130
            attr_gt = attr_gt[idx]
            attr_pred = attr_pred[idx]

        from scipy import stats
        slope, intercept, r_value, p_value, std_err = \
                stats.linregress(attr_gt, attr_pred)
        # print(slope, intercept, r_value, p_value, std_err)

        plt.scatter(attr_gt, attr_pred, c='r', s=4)
        if idx_rel == 1:
            plt.xticks(np.arange(20, 121, 20))
        elif idx_rel == 2:
            plt.xticks(np.arange(30, 131, 20))
        plt.xlabel('Ground truth hidden confounder')
        plt.ylabel('Predicted edge parameter')
        plt.tight_layout(pad=0.8)
        plt.savefig(os.path.join(args.evalf, 'cor_raw_%d.png' % idx_rel))
        plt.savefig(os.path.join(args.evalf, 'cor_raw_%d.pdf' % idx_rel))
        plt.show()


# store data for plotting
if args.env in ['Ball']:
    # edge_acc_over_time: n_roll x n_timestep

    record_names = ['edge_acc_over_time', 'edge_cor_over_time', 'fwd_loss_mse']
    if args.baseline == 1:
        record_path = os.path.join(args.evalf, 'rec_%d_baseline.h5' % args.n_kp)
    else:
        record_path = os.path.join(args.evalf, 'rec_%d.h5' % args.n_kp)

    store_data(
        record_names,
        [edge_acc_over_time_record, edge_cor_over_time_record, fwd_loss_mse],
        record_path)

    print()
    print('Edge Accuracy')
    print('%.2f%%, std: %.6f' % (
        np.mean(edge_acc_over_time_record[:, -1]) * 100.,
        np.std(edge_acc_over_time_record[:, -1])))

    print()
    print('Correlation on Attributes')
    for i in range(len(edge_cor_over_time_record)):
        print('#%d:' % i, edge_cor_over_time_record[i][-1])

elif args.env in ['Cloth']:

    record_names = ['fwd_loss_mse']
    if args.baseline == 1:
        record_path = os.path.join(args.evalf, 'rec_%d_baseline.h5' % args.n_kp)
    else:
        record_path = os.path.join(args.evalf, 'rec_%d.h5' % args.n_kp)

    store_data(record_names, [fwd_loss_mse], record_path)


