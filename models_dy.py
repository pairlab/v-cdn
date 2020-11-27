import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model_modules import GRUNet, CNNet, PropNet
from models_kp import SpatialSoftmax
from data import denormalize, normalize
from utils import load_data, count_parameters


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, prior=None):
        if prior is None:
            b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
            b = -b.sum(1)
            b = b.mean()
        else:
            b = F.softmax(x, dim=1)
            b = b * (F.log_softmax(x, dim=1) - torch.log(prior).view(-1, x.size(1)))
            b = -b.sum(1)
            b = b.mean()
        return b


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.5, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    B, categorical_dim = logits.size()
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, categorical_dim)


class DynaNetGNN(nn.Module):

    def __init__(self, args, use_gpu=True, drop_prob=0.2):
        super(DynaNetGNN, self).__init__()

        self.propnet_selfloop = False
        self.mask_remove_self_loop = torch.FloatTensor(
            np.ones((args.n_kp, args.n_kp)) - np.eye(args.n_kp)).cuda().view(1, args.n_kp, args.n_kp, 1)

        self.args = args
        nf = args.nf_hidden_dy * 4

        self.ratio = (args.height // 64) * (args.width // 64)

        # infer the graph
        self.model_infer_encode = PropNet(
            node_dim_in=2,
            edge_dim_in=0,
            nf_hidden=nf * 3,
            node_dim_out=nf,
            edge_dim_out=nf,
            edge_type_num=1,
            pstep=1,
            batch_norm=1)

        if args.en_model == 'gru':
            self.model_infer_node_agg = GRUNet(
                nf + 2 + args.action_dim, nf * 4, nf,
                drop_prob=drop_prob)
            self.model_infer_edge_agg = GRUNet(
                nf + 4 + args.action_dim * 2, nf * 4, nf,
                drop_prob=drop_prob)

        elif args.en_model == 'cnn':
            self.model_infer_node_agg = CNNet(
                7 if args.env == 'Ball' else 3,
                nf + 2 + args.action_dim, nf * 4, nf)
            self.model_infer_edge_agg = CNNet(
                7 if args.env == 'Ball' else 3,
                nf + 4 + args.action_dim * 2, nf * 4, nf)

        self.model_infer_affi_matx = PropNet(
            node_dim_in=nf,
            edge_dim_in=nf,
            nf_hidden=nf * 3,
            node_dim_out=0,
            edge_dim_out=args.edge_type_num,
            edge_type_num=1,
            pstep=2,
            batch_norm=1)

        self.model_infer_graph_attr = PropNet(
            node_dim_in=nf,
            edge_dim_in=nf,
            nf_hidden=nf * 3,
            node_dim_out=args.node_attr_dim,
            edge_dim_out=args.edge_attr_dim,
            edge_type_num=args.edge_type_num,
            pstep=1,
            batch_norm=1)

        # dynamics modeling
        self.model_dynam_encode = PropNet(
            node_dim_in=args.node_attr_dim + 6,
            edge_dim_in=args.edge_attr_dim + 12,
            nf_hidden=nf * 3,
            node_dim_out=nf,
            edge_dim_out=nf,
            edge_type_num=args.edge_type_num,
            pstep=1,
            batch_norm=1)

        self.model_dynam_node_forward = GRUNet(
            nf + 6 + args.node_attr_dim + args.action_dim, nf * 2, nf,
            drop_prob=drop_prob)
        self.model_dynam_edge_forward = GRUNet(
            nf + 12 + args.edge_attr_dim + args.action_dim * 2, nf * 2, nf,
            drop_prob=drop_prob)

        self.model_dynam_decode = PropNet(
            node_dim_in=nf + args.node_attr_dim + args.action_dim + 6,
            edge_dim_in=nf + args.edge_attr_dim + args.action_dim * 2 + 12,
            nf_hidden=nf * 3,
            node_dim_out=5,
            edge_dim_out=1,
            edge_type_num=args.edge_type_num,
            pstep=1,
            batch_norm=0)

        print('model_infer_encode #params', count_parameters(self.model_infer_encode))
        print('model_infer_node_agg #params', count_parameters(self.model_infer_node_agg))
        print('model_infer_edge_agg #params', count_parameters(self.model_infer_edge_agg))
        print('model_infer_affi_matx #params', count_parameters(self.model_infer_affi_matx))
        print('model_infer_graph_attr #params', count_parameters(self.model_infer_graph_attr))
        print('model_dynam_encode #params', count_parameters(self.model_dynam_encode))
        print('model_dynam_node_forward #params', count_parameters(self.model_dynam_node_forward))
        print('model_dynam_edge_forward #params', count_parameters(self.model_dynam_edge_forward))
        print('model_dynam_decode #params', count_parameters(self.model_dynam_decode))

        # integration tools
        self.integrater = SpatialSoftmax(
            height=args.height//4, width=args.width//4, channel=args.n_kp, lim=args.lim)

        # for generating gaussian heatmap
        lim = args.lim
        x = np.linspace(lim[0], lim[1], args.width // 4)
        y = np.linspace(lim[2], lim[3], args.height // 4)

        if use_gpu:
            self.x = Variable(torch.FloatTensor(x)).cuda()
            self.y = Variable(torch.FloatTensor(y)).cuda()
        else:
            self.x = Variable(torch.FloatTensor(x))
            self.y = Variable(torch.FloatTensor(y))

        self.graph = [None, None, None]

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)

    def integrate(self, heatmap):
        return self.integrater(heatmap)

    def init_graph(self, kp, use_gpu=False, hard=False):
        # randomly generated graph
        # kp: B x T x n_kp x (2 + 4)
        #
        # node_attr: B x n_kp x node_attr_dim
        # edge_attr: B x n_kp x n_kp x edge_attr_dim
        # edge_type: B x n_kp x n_kp x edge_type_num
        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        args = self.args
        B, T, n_kp, _ = kp.size()

        node_attr = torch.FloatTensor(np.zeros((B, n_kp, args.node_attr_dim)))
        edge_attr = torch.FloatTensor(np.zeros((B, n_kp, n_kp, args.edge_attr_dim)))

        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        edge_type_logits = args.prior[None, None, None, :].repeat(B, n_kp, n_kp, 1)
        edge_type_logits = torch.log(edge_type_logits).view(B * n_kp * n_kp, args.edge_type_num)

        # edge_type: B x n_kp x n_kp x edge_type_num
        edge_type = gumbel_softmax(edge_type_logits, hard=hard).view(B, n_kp, n_kp, args.edge_type_num)
        edge_type_logits = edge_type_logits.view(B, n_kp, n_kp, args.edge_type_num)

        if use_gpu:
            node_attr = node_attr.cuda()
            edge_attr = edge_attr.cuda()
            edge_type = edge_type.cuda()
            edge_type_logits = edge_type_logits.cuda()

        graph = [node_attr, edge_attr, edge_type, edge_type_logits]
        return graph

    def graph_inference(self, kp, action=None, hard=False, env=None):
        # update the belief over the structure of the graph
        # kp: B x T x n_kp x (2 + 4)
        # action:
        #   ToyFullAct, BallAct, BallFullAct, BallFullActFull: B x T x n_kp x action_dim
        #   Fluid: B x T x action_dim
        args = self.args
        B, T, n_kp, _ = kp.size()
        nf = self.args.nf_hidden_dy * 4

        # node_enc: B x T x n_kp x (2 + 4)
        node_enc = kp.contiguous()

        # node_rep: B x T x N x nf
        # edge_rep: B x T x (N * N) x nf
        node_rep, edge_rep = self.model_infer_encode(
            node_enc.view(B * T, n_kp, 2), None)
        node_rep = node_rep.view(B, T, n_kp, nf)
        edge_rep = edge_rep.view(B, T, n_kp * n_kp, nf)

        kp_t = kp.transpose(1, 2).contiguous().view(B, n_kp, T, 2)
        kp_t_r = kp_t[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)
        kp_t_s = kp_t[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1)

        node_rep = node_rep.transpose(1, 2).contiguous().view(B * n_kp, T, nf)
        edge_rep = edge_rep.transpose(1, 2).contiguous().view(B * n_kp * n_kp, T, nf)

        node_rep = torch.cat([
            node_rep, kp_t.view(B * n_kp, T, 2)], 2)
        edge_rep = torch.cat([
            edge_rep, kp_t_r.view(B * n_kp**2, T, 2), kp_t_s.view(B * n_kp**2, T, 2)], 2)

        if action is not None:
            action_dim = self.args.action_dim

            action_t = action.transpose(1, 2).contiguous().view(B, n_kp, T, action_dim)
            action_t_r = action_t[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)
            action_t_s = action_t[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1)

            # print('node_rep', node_rep.size(), 'edge_rep', edge_rep.size())
            # print('action_t', action_t.size(), 'action_t_r', action_t_r.size(), 'action_t_s', action_t_s.size())

            node_rep = torch.cat([
                node_rep, action_t.view(B * n_kp, T, action_dim)], 2)
            edge_rep = torch.cat([
                edge_rep,
                action_t_r.view(B * n_kp**2, T, action_dim),
                action_t_s.view(B * n_kp**2, T, action_dim)], 2)

        # node_rep: (B * n_kp) x T x (nf + 2 + action_dim)
        # edge_rep: (B * n_kp * n_kp) x T x (nf + 4 + action_dim)
        # node_rep_agg: (B * n_kp) x nf
        # edge_rep_agg: (B * n_kp * n_kp) x nf
        node_rep_agg = self.model_infer_node_agg(node_rep).view(B, n_kp, nf)
        edge_rep_agg = self.model_infer_edge_agg(edge_rep).view(B, n_kp, n_kp, nf)

        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        edge_type_logits = self.model_infer_affi_matx(node_rep_agg, edge_rep_agg, ignore_node=True)

        if args.edge_share:
            edge_type_logits = (edge_type_logits + torch.transpose(edge_type_logits, 1, 2)) / 2.

        # edge_type: B x n_kp x n_kp x edge_type_num
        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        edge_type = gumbel_softmax(edge_type_logits.view(B * n_kp * n_kp, args.edge_type_num), hard=hard)
        edge_type = edge_type.view(B, n_kp, n_kp, args.edge_type_num)

        if self.propnet_selfloop == False:
            edge_type = edge_type * self.mask_remove_self_loop

        # node_attr: B x n_kp x node_attr_dim
        # edge_attr: B x n_kp x n_kp x edge_attr_dim
        node_attr, edge_attr = self.model_infer_graph_attr(node_rep_agg, edge_rep_agg, edge_type)

        if args.edge_share:
            edge_attr = (edge_attr + torch.transpose(edge_attr, 1, 2)) / 2.

        # node_attr: B x n_kp x node_attr_dim
        # edge_attr: B x n_kp x n_kp x edge_attr_dim
        # edge_type: B x n_kp x n_kp x edge_type_num
        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        self.graph = [node_attr, edge_attr, edge_type, edge_type_logits]

        return self.graph

    def dynam_prediction(self, kp, graph, action=None, eps=5e-2, env=None):
        # kp: B x n_his x n_kp x (2 + 4)
        # action:
        #   ToyFullAct, BallAct, BallFullAct, BallFullActFull: B x n_his x n_kp x action_dim
        #   Fluid: B x n_his x action_dim
        args = self.args
        nf = args.nf_hidden_dy * 4
        action_dim = args.action_dim
        node_attr_dim = args.node_attr_dim
        edge_attr_dim = args.edge_attr_dim
        edge_type_num = args.edge_type_num

        B, n_his, n_kp, _ = kp.size()

        # node_attr: B x n_kp x node_attr_dim
        # edge_attr: B x n_kp x n_kp x edge_attr_dim
        # edge_type: B x n_kp x n_kp x edge_type_num
        # edge_type_logits: B x n_kp x n_kp x edge_type_num
        node_attr, edge_attr, edge_type, edge_type_logits = graph

        # node_enc: B x n_his x n_kp x nf
        # edge_enc: B x n_his x (n_kp * n_kp) x nf
        node_enc = torch.cat([kp, node_attr.view(B, 1, n_kp, node_attr_dim).repeat(1, n_his, 1, 1)], 3)
        edge_enc = torch.cat([
            torch.cat([kp[:, :, :, None, :].repeat(1, 1, 1, n_kp, 1),
                       kp[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)], 4),
            edge_attr.view(B, 1, n_kp, n_kp, edge_attr_dim).repeat(1, n_his, 1, 1, 1)], 4)

        node_enc, edge_enc = self.model_dynam_encode(
            node_enc.view(B * n_his, n_kp, node_attr_dim + 6),
            edge_enc.view(B * n_his, n_kp, n_kp, edge_attr_dim + 12),
            edge_type[:, None, :, :, :].repeat(1, n_his, 1, 1, 1).view(B * n_his, n_kp, n_kp, edge_type_num),
            start_idx=args.edge_st_idx)

        node_enc = node_enc.view(B, n_his, n_kp, nf)
        edge_enc = edge_enc.view(B, n_his, n_kp * n_kp, nf)

        # node_enc: B x n_kp x n_his x nf
        # edge_enc: B x (n_kp * n_kp) x n_his x nf
        node_enc = node_enc.transpose(1, 2).contiguous().view(B, n_kp, n_his, nf)
        edge_enc = edge_enc.transpose(1, 2).contiguous().view(B, n_kp * n_kp, n_his, nf)

        # node_enc: B x n_kp x n_his x (nf + node_attr_dim + action_dim)
        # kp_node: B x n_kp x n_his x 6
        kp_node = kp.transpose(1, 2).contiguous().view(B, n_kp, n_his, 6)

        node_enc = torch.cat([
            kp_node, node_enc, node_attr.view(B, n_kp, 1, node_attr_dim).repeat(1, 1, n_his, 1)], 3)

        # edge_enc: B x (n_kp * n_kp) x n_his x (nf + edge_attr_dim + action_dim)
        # kp_edge: B x (n_kp * n_kp) x n_his x (2 + 2)
        kp_edge = torch.cat([
            kp_node[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1),
            kp_node[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1)], 4)
        kp_edge = kp_edge.view(B, n_kp**2, n_his, 12)

        edge_enc = torch.cat([
            kp_edge, edge_enc, edge_attr.view(B, n_kp**2, 1, edge_attr_dim).repeat(1, 1, n_his, 1)], 3)

        # append action
        if action is not None:
            action_t = action.transpose(1, 2).contiguous()
            action_t_r = action_t[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1).view(B, n_kp**2, n_his, action_dim)
            action_t_s = action_t[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1).view(B, n_kp**2, n_his, action_dim)
            # print('node_enc', node_enc.size(), 'edge_enc', edge_enc.size())
            # print('action_t', action_t.size(), 'action_t_r', action_t_r.size(), 'action_t_s', action_t_s.size())
            node_enc = torch.cat([node_enc, action_t], 3)
            edge_enc = torch.cat([edge_enc, action_t_r, action_t_s], 3)

        # node_enc: B x n_kp x nf
        # edge_enc: B x n_kp x n_kp x nf
        node_enc = self.model_dynam_node_forward(
            node_enc.view(B * n_kp, n_his, -1)).view(B, n_kp, nf)
        edge_enc = self.model_dynam_edge_forward(
            edge_enc.view(B * n_kp**2, n_his, -1)).view(B, n_kp, n_kp, nf)

        # kp_pred: B x n_kp x (2 + 3)
        node_enc = torch.cat([node_enc, node_attr, kp_node[:, :, -1]], 2)
        edge_enc = torch.cat([edge_enc, edge_attr, kp_edge[:, :, -1].view(B, n_kp, n_kp, 12)], 3)

        if action is not None:
            # print('node_enc', node_enc.size(), 'edge_enc', edge_enc.size(), 'action', action.size())
            action_r = action[:, :, :, None, :].repeat(1, 1, 1, n_kp, 1)
            action_s = action[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)
            node_enc = torch.cat([node_enc, action[:, -1]], 2)
            edge_enc = torch.cat([edge_enc, action_r[:, -1], action_s[:, -1]], 3)

        kp_pred = self.model_dynam_decode(
            node_enc, edge_enc, edge_type,
            start_idx=args.edge_st_idx, ignore_edge=True)

        # kp_pred: B x n_kp x (2 + 4)
        kp_pred = torch.cat([
            kp[:, -1, :, :2] + kp_pred[:, :, :2],               # mean
            F.relu(kp_pred[:, :, 2:3]) + args.gauss_std,        # covar (0, 0), need to > 0
            torch.zeros(B, n_kp, 1).cuda(),                     # covar (0, 1)
            kp_pred[:, :, 3:4],                                 # covar (1, 0)
            F.relu(kp_pred[:, :, 4:5]) + args.gauss_std],       # covar (1, 1), need to > 0
            dim=2)

        return kp_pred

    def forward(self, feat, hmap, action=None):
        pass


