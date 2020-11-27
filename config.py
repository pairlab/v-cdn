import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./50.)
parser.add_argument('--nf_hidden_kp', type=int, default=16)
parser.add_argument('--nf_hidden_dy', type=int, default=16)
parser.add_argument('--norm_layer', default='Batch', help='Batch|Instance')

parser.add_argument('--n_ball', type=int, default=0, help="option for ablating on the number of balls")
parser.add_argument('--n_kp', type=int, default=0, help="the number of keypoint")
parser.add_argument('--inv_std', type=float, default=10., help='the inverse of std of gaussian mask')

parser.add_argument('--stage', default='kp', help='kp|dy')
parser.add_argument('--outf', default='train')
parser.add_argument('--dataf', default='data')

parser.add_argument('--baseline', type=int, default=0, help="whether to use the baseline model - no inference module")

'''
train
'''
parser.add_argument('--random_seed', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)

parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0, help="whether to generate new data")
parser.add_argument('--train_valid_ratio', type=float, default=0.95, help="percentage of training data")

parser.add_argument('--log_per_iter', type=int, default=100, help="print log every x iterations")
parser.add_argument('--ckp_per_iter', type=int, default=5000, help="save checkpoint every x iterations")

parser.add_argument('--kp_epoch', type=int, default=-1)
parser.add_argument('--kp_iter', type=int, default=-1)
parser.add_argument('--dy_epoch', type=int, default=-1)
parser.add_argument('--dy_iter', type=int, default=-1)

parser.add_argument('--height_raw', type=int, default=0)
parser.add_argument('--width_raw', type=int, default=0)
parser.add_argument('--height', type=int, default=0)
parser.add_argument('--width', type=int, default=0)
parser.add_argument('--scale_size', type=int, default=0)
parser.add_argument('--crop_size', type=int, default=0)

parser.add_argument('--eval', type=int, default=0)

# for dynamics prediction
parser.add_argument('--min_res', type=int, default=0, help="minimal observation for the inference module")
parser.add_argument('--lam_kp', type=float, default=1.)
parser.add_argument('--gauss_std', type=float, default=5e-2)

parser.add_argument('--dy_model', default='gnn', help='the model for dynamics prediction - gnn|mlp')
parser.add_argument('--en_model', default='cnn', help='the model for encoding - gru|cnn|tra')
parser.add_argument('--n_his', type=int, default=5, help='number of frames used as input')
parser.add_argument('--n_identify', type=int, default=0, help='number of frames used for graph identification')
parser.add_argument('--n_roll', type=int, default=5, help='number of rollout steps for training')

parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--node_attr_dim', type=int, default=0)
parser.add_argument('--edge_attr_dim', type=int, default=0)
parser.add_argument('--edge_type_num', type=int, default=0)
parser.add_argument('--edge_st_idx', type=int, default=0, help="whether to exclude the first edge type")
parser.add_argument('--edge_share', type=int, default=0,
                    help="whether forcing the info being the same for both directions")

parser.add_argument('--preload_kp', type=int, default=1, help="whether to load saved predicted keypoints")

'''
eval
'''
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval_kp_epoch', type=int, default=-1)
parser.add_argument('--eval_kp_iter', type=int, default=-1)
parser.add_argument('--identify_st_idx', type=int, default=-1)
parser.add_argument('--identify_ed_idx', type=int, default=-1)
parser.add_argument('--eval_dy_epoch', type=int, default=-1)
parser.add_argument('--eval_dy_iter', type=int, default=-1)

parser.add_argument('--eval_set', default='valid', help='train|valid')
parser.add_argument('--eval_st_idx', type=int, default=0)
parser.add_argument('--eval_ed_idx', type=int, default=0)

parser.add_argument('--vis_edge', type=int, default=1)
parser.add_argument('--store_demo', type=int, default=1)
parser.add_argument('--store_result', type=int, default=0)
parser.add_argument('--store_st_idx', type=int, default=0)
parser.add_argument('--store_ed_idx', type=int, default=0)

'''
model
'''
# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# action:
parser.add_argument('--action_dim', type=int, default=0)

# relation:
parser.add_argument('--relation_dim', type=int, default=0)



def gen_args():
    args = parser.parse_args()

    if args.env == 'Ball':
        args.data_names = ['attrs', 'states', 'actions', 'rels']

        args.n_rollout = 5000
        args.frame_offset = 1
        args.time_step = 500
        args.train_valid_ratio = 0.95

        # radius
        args.attr_dim = 1
        # x, y, xdot, ydot
        args.state_dim = 4
        # ddx, ddy
        args.action_dim = 2
        # none, spring, rod
        args.relation_dim = 3

        # size of the latent causal graph
        args.node_attr_dim = 0
        args.edge_attr_dim = 1
        args.edge_type_num = 3

        args.height_raw = 110
        args.width_raw = 110
        args.height = 64
        args.width = 64
        args.scale_size = 64
        args.crop_size = 64

        args.lim = [-1., 1., -1., 1.]

        args.prior = torch.FloatTensor(
            np.array([0.4, 0.3, 0.3])).cuda()

    elif args.env == 'Cloth':
        args.data_names = ['states', 'actions', 'scene_params']

        args.n_rollout = 2000
        if args.stage == 'dy':
            args.frame_offset = 5
        else:
            args.frame_offset = 1
        args.time_step = 300 // args.frame_offset
        args.train_valid_ratio = 0.9

        # x, y, z, xdot, ydot, zdot
        args.state_dim = 6
        # x, y, z, dx, dy, dz
        args.action_dim = 6

        # size of the latent causal graph
        args.node_attr_dim = 0
        args.edge_attr_dim = 1
        args.edge_type_num = 2

        args.height_raw = 400
        args.width_raw = 400
        args.height = 64
        args.width = 64
        args.scale_size = 64
        args.crop_size = 64

        args.lim = [-1., 1., -1., 1.]
        args.prior = torch.FloatTensor(np.array([0.85, 0.15])).cuda()

    else:
        raise AssertionError("Unsupported env %s" % args.env)


    # path to data
    args.dataf = 'data/' + args.dataf + '_' + args.env

    # path to train
    dump_prefix = 'dump_{}/'.format(args.env)

    args.outf_kp = dump_prefix + args.outf
    args.outf_kp += '_' + args.env + '_kp'
    args.outf_kp += '_nkp_' + str(args.n_kp) + '_invStd_' + str(int(args.inv_std))

    if args.stage == 'dy':
        args.outf_dy = dump_prefix + args.outf
        args.outf_dy += '_' + args.env + '_dy'
        args.outf_dy += '_nkp_' + str(args.n_kp) + '_invStd_' + str(int(args.inv_std))
        args.outf_dy += '_%s' % args.en_model
        args.outf_dy += '_%s' % args.dy_model
        args.outf_dy += '_nId_' + str(args.n_identify)
        args.outf_dy += '_nHis_' + str(args.n_his)

        if args.edge_st_idx > 0:
            args.outf_dy += '_noEdge0'
        if args.edge_share == 1:
            args.outf_dy += '_edgeShare'

        if args.baseline == 1:
            args.outf_dy += '_baseline'

    # path to eval
    args.evalf = dump_prefix + args.evalf
    args.evalf += '_' + args.env
    args.evalf += '_' + args.stage

    args.evalf += '_' + str(args.eval_set)

    args.evalf += '_nkp_' + str(args.n_kp)
    args.evalf += '_invStd_' + str(int(args.inv_std))

    if args.eval_kp_epoch > -1:
        args.evalf += '_kpEpoch_' + str(args.eval_kp_epoch)
        args.evalf += '_kpIter_' + str(args.eval_kp_iter)
    else:
        args.evalf += '_kpEpoch_best'

    if args.stage == 'dy':
        args.evalf += '_%s' % args.en_model
        args.evalf += '_%s' % args.dy_model
        args.evalf += '_nId_' + str(args.n_identify)
        args.evalf += '_nHis_' + str(args.n_his)

        if args.edge_st_idx > 0:
            args.evalf += '_noEdge0'
        if args.edge_share == 1:
            args.evalf += '_edgeShare'

        if args.eval_dy_epoch > -1:
            args.evalf += '_dyEpoch_' + str(args.eval_dy_epoch)
            args.evalf += '_dyIter_' + str(args.eval_dy_iter)
        else:
            args.evalf += '_dyEpoch_best'

        if args.baseline == 1:
            args.evalf += '_baseline'

    return args

