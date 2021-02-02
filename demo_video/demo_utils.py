import os
import cv2
import h5py
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Polygon



def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)



def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data



def render_BallEnv(states, actions, param, video=True, image=False, path=None, draw_edge=True,
           lim=(-80, 80, -80, 80), verbose=True, st_idx=0, image_prefix='fig'):
    # states: time_step x n_ball x 4
    # actions: time_step x 2

    radius = 6

    if video:
        video_path = path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        if verbose:
            print('Saving video as %s ...' % video_path)
        out = cv2.VideoWriter(video_path, fourcc, 25, (110, 110))

    if image:
        image_path = path
        if verbose:
            print('Saving images to %s ...' % image_path)
        command = 'mkdir -p %s' % image_path
        os.system(command)

    c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'black', 'crimson']

    time_step = states.shape[0]
    n_ball = states.shape[1]

    for i in range(time_step):
        fig, ax = plt.subplots(1, dpi=100)
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
        # plt.axis('off')

        fig.set_size_inches(1.5, 1.5)

        if draw_edge:
            # draw force
            for x in range(n_ball):
                F = actions[i, x]

                normF = norm(F)
                Fx = F / normF * normF * 0.05
                st = states[i, x, :2] + F / normF * 12.
                ax.arrow(st[0], st[1], Fx[0], Fx[1], fc='Orange', ec='Orange', width=3., head_width=15., head_length=15.)

            # draw edge
            cnt = 0
            for x in range(n_ball):
                for y in range(x):
                    rel_type = int(param[cnt, 0]); cnt += 1
                    if rel_type == 0:
                        continue

                    plt.plot([states[i, x, 0], states[i, y, 0]],
                             [states[i, x, 1], states[i, y, 1]],
                             '-', color=c[rel_type], lw=1, alpha=0.5)

        circles = []
        circles_color = []
        for j in range(n_ball):
            circle = Circle((states[i, j, 0], states[i, j, 1]), radius=radius)
            circles.append(circle)
            circles_color.append(c[j % len(c)])

        pc = PatchCollection(circles, facecolor=circles_color, linewidth=0)
        ax.add_collection(pc)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.tight_layout()

        if video or image:
            fig.canvas.draw()
            frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame[21:-19, 21:-19]

        if video:
            out.write(frame)
            if i == time_step - 1:
                for _ in range(5):
                    out.write(frame)

        if image:
            cv2.imwrite(os.path.join(image_path, '%s_%s.png' % (image_prefix, i + st_idx)), frame)

        plt.close()

    if video:
        out.release()
