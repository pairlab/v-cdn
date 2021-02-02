import multiprocessing as mp
import os
import time

from PIL import Image

import cv2
import numpy as np
import imageio
import scipy.misc
import torch
from progressbar import ProgressBar
from torch.autograd import Variable
from torch.utils.data import Dataset

from physics_engine import BallEngine, ClothEngine

from utils import rand_float, rand_int
from utils import init_stat, combine_stat, load_data, store_data



def gen_Ball(info):
    seed, data_dir, data_names = info['seed'], info['data_dir'], info['data_names']
    time_step, dt, n_ball = info['time_step'], info['dt'], info['n_ball']
    file_name = info['file_name']


    os.system('mkdir -p ' + data_dir)

    np.random.seed(seed)

    attr_dim = 1
    state_dim = 4
    action_dim = 2

    engine = BallEngine(dt, state_dim, action_dim=2)
    engine.init(n_ball)

    n_obj = engine.num_obj
    attrs_all = np.zeros((time_step, n_obj, attr_dim))
    states_all = np.zeros((time_step, n_obj, state_dim))
    actions_all = np.zeros((time_step, n_obj, action_dim))
    rel_attrs_all = np.zeros((time_step, engine.param_dim, 2))

    act = np.zeros((n_obj, 2))
    for j in range(time_step):
        state = engine.get_state()

        vel_dim = state_dim // 2
        pos = state[:, :vel_dim]
        vel = state[:, vel_dim:]

        if j > 0:
            vel = (pos - states_all[j - 1, :, :vel_dim]) / dt

        attrs = np.zeros((n_obj, attr_dim))
        attrs[:] = engine.radius

        attrs_all[j] = attrs
        states_all[j, :, :vel_dim] = pos
        states_all[j, :, vel_dim:] = vel
        rel_attrs_all[j] = engine.param

        # apply zero action
        engine.step(act)

        actions_all[j] = act.copy()

    datas = [attrs_all, states_all, actions_all, rel_attrs_all]
    store_data(data_names, datas, os.path.join(data_dir, '%s.h5' % file_name))
    '''
    engine.render(states_all, actions_all, engine.get_param(), video=False, image=True,
                  path=data_dir, draw_edge=False, verbose=False)
    '''



info = {
    'seed': 41,
    'data_dir': 'demo_video',
    'data_names': ['attr', 'state', 'action', 'rel_attr'],
    'time_step': 500,
    'dt': 1. / 50.,
    'n_ball': 5,
    'file_name': '5_balls_w_relation_multipleType'
}


gen_Ball(info)

