import os
import cv2
import numpy as np
import pymunk
from pymunk.vec2d import Vec2d
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle, Polygon

from utils import rand_float, rand_int, calc_dis, norm


class Engine(object):

    def __init__(self, dt, state_dim, action_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.param_dim = None
        self.state = None
        self.action = None
        self.param = None

    def init(self):
        pass

    def get_param(self):
        return self.param.copy()

    def set_param(self, param):
        self.param = param.copy()

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def get_scene(self):
        return self.state.copy(), self.param.copy()

    def set_scene(self, state, param):
        self.state = state.copy()
        self.param = param.copy()

    def get_action(self):
        return self.action.copy()

    def set_action(self, action):
        self.action = action.copy()

    def d(self, state, t, param):
        # time derivative
        pass

    def step(self):
        pass

    def render(self, state, param):
        pass

    def clean(self):
        pass


class BallEngine(Engine):

    def __init__(self, dt, state_dim, action_dim):

        # state_dim = 4
        # action_dim = 2
        # param_dim = n_ball * (n_ball - 1)

        # param [relation_type, coefficient]
        # relation_type
        # 0 - no relation
        # 1 - spring (DampedSpring)
        # 2 - string (SlideJoint)
        # 3 - rod (PinJoint)

        super(BallEngine, self).__init__(dt, state_dim, action_dim)

        self.init()

    def add_segments(self, p_range=(-80, 80, -80, 80)):
        a = pymunk.Segment(self.space.static_body, (p_range[0], p_range[2]), (p_range[0], p_range[3]), 1)
        b = pymunk.Segment(self.space.static_body, (p_range[0], p_range[2]), (p_range[1], p_range[2]), 1)
        c = pymunk.Segment(self.space.static_body, (p_range[1], p_range[3]), (p_range[0], p_range[3]), 1)
        d = pymunk.Segment(self.space.static_body, (p_range[1], p_range[3]), (p_range[1], p_range[2]), 1)
        a.friction = 1; a.elasticity = 1
        b.friction = 1; b.elasticity = 1
        c.friction = 1; c.elasticity = 1
        d.friction = 1; d.elasticity = 1
        self.space.add(a); self.space.add(b)
        self.space.add(c); self.space.add(d)

    def add_balls(self, center=(0., 0.), p_range=(-60, 60)):
        inertia = pymunk.moment_for_circle(self.mass, 0, self.radius, (0, 0))
        for i in range(self.n_ball):
            while True:
                x = rand_float(p_range[0], p_range[1])
                y = rand_float(p_range[0], p_range[1])
                flag = True
                for j in range(i):
                    if calc_dis([x, y], self.balls[j].position) < 30:
                        flag = False
                if flag:
                    break
            body = pymunk.Body(self.mass, inertia)
            body.position = Vec2d((x, y))
            shape = pymunk.Circle(body, 0., (0, 0))
            shape.elasticity = 1
            self.space.add(body, shape)
            self.balls.append(body)

    def add_rels(self, param_load=None):
        param = np.zeros((self.n_ball * (self.n_ball - 1) // 2, 2))
        self.param_dim = param.shape[0]

        if param_load is not None:
            print("Load param for init env")

        cnt = 0
        rels_idx = []
        for i in range(self.n_ball):
            for j in range(i):
                rel_type = rand_int(0, self.n_rel_type) if param_load is None else param_load[cnt, 0]
                param[cnt, 0] = rel_type

                rels_idx.append([i, j])

                pos_i = self.balls[i].position
                pos_j = self.balls[j].position

                if rel_type == 0:
                    # no relation
                    pass

                elif rel_type == 1:
                    # spring
                    rest_length = rand_float(20, 120) if param_load is None else param_load[cnt, 1]
                    param[cnt, 1] = rest_length
                    c = pymunk.DampedSpring(
                        self.balls[i], self.balls[j], (0, 0), (0, 0),
                        rest_length=rest_length, stiffness=20, damping=0.)
                    self.space.add(c)

                elif rel_type == 2:
                    # string
                    rest_length = calc_dis(pos_i, pos_j) if param_load is None else param_load[cnt, 1]
                    param[cnt, 1] = rest_length
                    c = pymunk.SlideJoint(
                        self.balls[i], self.balls[j], (0, 0), (0, 0),
                        rest_length - 5, rest_length + 5)
                    self.space.add(c)

                else:
                    raise AssertionError("Unknown relation type")

                cnt += 1

        if param_load is not None:
            assert((param == param_load).all())

        self.rels_idx = rels_idx
        self.param = param

    def add_impulse(self, p_range=(-200, 200)):
        for i in range(self.n_ball):
            impulse = (rand_float(p_range[0], p_range[1]), rand_float(p_range[0], p_range[1]))
            self.balls[i].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def add_boundary_impulse(self, p_range=(-75, 75, -75, 75)):
        f_scale = 5e2
        eps = 2
        for i in range(self.n_ball):
            impulse = np.zeros(2)
            p = np.array([self.balls[i].position[0], self.balls[i].position[1]])

            d = min(20, max(eps, p[0] - p_range[0]))
            impulse[0] += f_scale / d
            d = max(-20, min(-eps, p[0] - p_range[1]))
            impulse[0] += f_scale / d
            d = min(20, max(eps, p[1] - p_range[2]))
            impulse[1] += f_scale / d
            d = max(-20, min(-eps, p[1] - p_range[3]))
            impulse[1] += f_scale / d

            self.balls[i].apply_impulse_at_local_point(impulse=impulse, point=(0, 0))

    def init(self, n_ball=5, init_impulse=True, param_load=None):
        self.space = pymunk.Space()
        self.space.gravity = (0., 0.)

        self.n_rel_type = 3
        self.n_ball = n_ball
        self.mass = 1.
        self.radius = 6
        self.balls = []
        # self.add_segments()
        self.add_balls()
        self.add_rels(param_load)

        if init_impulse:
            self.add_impulse()

        self.state_prv = None

    @property
    def num_obj(self):
        return self.n_ball

    def get_state(self):
        state = np.zeros((self.n_ball, 4))
        for i in range(self.n_ball):
            ball = self.balls[i]
            state[i] = np.array([ball.position[0], ball.position[1], ball.velocity[0], ball.velocity[1]])

        vel_dim = self.state_dim // 2
        if self.state_prv is None:
            state[:, vel_dim:] = 0
        else:
            state[:, vel_dim:] = (state[:, :vel_dim] - self.state_prv[:, :vel_dim]) / self.dt

        return state

    def add_action(self, action):
        if action is None:
            return
        for i in range(self.n_ball):
            self.balls[i].apply_force_at_local_point(force=action[i], point=(0, 0))

    def step(self, action=None):
        self.state_prv = self.get_state()
        self.add_action(action)
        self.add_boundary_impulse()
        self.space.step(self.dt)

    def render(self, states, actions, param, video=True, image=False, path=None, draw_edge=True,
               lim=(-80, 80, -80, 80), verbose=True, st_idx=0, image_prefix='fig'):
        # states: time_step x n_ball x 4
        # actions: time_step x 2

        # lim = (lim[0] - self.radius, lim[1] + self.radius, lim[2] - self.radius, lim[3] + self.radius)

        if video:
            video_path = path + '.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            if verbose:
                print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 25, (110, 110))

        if image:
            image_path = path
            if verbose:
                print('Save images to %s' % image_path)
            command = 'mkdir -p %s' % image_path
            os.system(command)

        c = ['royalblue', 'tomato', 'limegreen', 'orange', 'violet', 'chocolate', 'black', 'crimson']

        time_step = states.shape[0]
        n_ball = states.shape[1]

        for i in range(time_step):
            fig, ax = plt.subplots(1)
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
                circle = Circle((states[i, j, 0], states[i, j, 1]), radius=self.radius)
                circles.append(circle)
                circles_color.append(c[j % len(c)])

            pc = PatchCollection(circles, facecolor=circles_color, linewidth=0, alpha=0.5)
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



class ClothEngine(Engine):

    def __init__(self, dt, state_dim, action_dim):

        # state_dim = 6
        # action_dim = 2

        super(ClothEngine, self).__init__(dt, state_dim, action_dim)

    def init(self, pyflex, scene_params=None):
        self.pyflex = pyflex

        if scene_params is None:
            # offset
            radius = 0.05
            offset_x = -1.
            offset_y = 0.06
            offset_z = -1.

            # fabrics
            fabric_type = rand_int(0, 3) # 0: Cloth, 1: shirt, 2: pants

            if fabric_type == 0:
                # parameters of the shape
                dimx = rand_int(25, 35)    # dimx, width
                dimy = rand_int(25, 35)    # dimy, height
                dimz = 0
                # the actuated points
                ctrl_idx = np.array([
                    0, dimx // 2, dimx - 1,
                    dimy // 2 * dimx,
                    dimy // 2 * dimx + dimx - 1,
                    (dimy - 1) * dimx,
                    (dimy - 1) * dimx + dimx // 2,
                    (dimy - 1) * dimx + dimx - 1])

                offset_x = -dimx * radius / 2.
                offset_y = 0.06
                offset_z = -dimy * radius / 2.

            elif fabric_type == 1:
                # parameters of the shape
                dimx = rand_int(16, 25)     # width of the body
                dimy = rand_int(30, 35)     # height of the body
                dimz = 7                    # size of the sleeves
                # the actuated points
                ctrl_idx = np.array([
                    dimx * dimy,
                    dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4,
                    dimx * dimy + (1 + dimz) * dimz // 2 + dimz * (dimz - 1),
                    dimx * dimy + dimz * (dimz + dimz // 2) + (1 + dimz) * (dimz + 1) // 4 + \
                        (1 + dimz) * dimz // 2 + dimz * dimz - 1,
                    dimy // 2 * dimx,
                    dimy // 2 * dimx + dimx - 1,
                    (dimy - 1) * dimx,
                    dimy * dimx - 1])

                offset_x = -(dimx + dimz * 4) * radius / 2.
                offset_y = 0.06
                offset_z = -dimy * radius / 2.

            elif fabric_type == 2:
                # parameters of the shape
                dimx = rand_int(9, 13) * 2 # width of the pants
                dimy = rand_int(6, 11)      # height of the top part
                dimz = rand_int(24, 31)     # height of the leg
                # the actuated points
                ctrl_idx = np.array([
                    0, dimx - 1,
                    (dimy - 1) * dimx,
                    (dimy - 1) * dimx + dimx - 1,
                    dimx * dimy + dimz // 2 * (dimx - 4) // 2,
                    dimx * dimy + (dimz - 1) * (dimx - 4) // 2,
                    dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
                        dimz // 2 * (dimx - 4) // 2 + (dimx - 4) // 2 - 1,
                    dimx * dimy + dimz * (dimx - 4) // 2 + 3 + \
                        dimz * (dimx - 4) // 2 - 1])

                offset_x = -dimx * radius / 2.
                offset_y = 0.06
                offset_z = -(dimy + dimz) * radius / 2.


            # physical param
            stiffness = rand_float(0.4, 1.0)
            stretchStiffness = stiffness
            bendStiffness = stiffness
            shearStiffness = stiffness

            dynamicFriction = 0.6
            staticFriction = 1.0
            particleFriction = 0.6

            invMass = 1.0

            # other parameters
            windStrength = 0.0
            draw_mesh = 1.

            # set up environment
            self.scene_params = np.array([
                offset_x, offset_y, offset_z,
                fabric_type, dimx, dimy, dimz,
                ctrl_idx[0], ctrl_idx[1], ctrl_idx[2], ctrl_idx[3],
                ctrl_idx[4], ctrl_idx[5], ctrl_idx[6], ctrl_idx[7],
                stretchStiffness, bendStiffness, shearStiffness,
                dynamicFriction, staticFriction, particleFriction,
                invMass, windStrength, draw_mesh])

        else:
            self.scene_params = scene_params

        scene_idx = 15
        self.pyflex.set_scene(scene_idx, self.scene_params, 0)

        # set up camera pose
        camPos = np.array([0., 3.5, 0.])
        camAngle = np.array([0., -90./180. * np.pi, 0.])
        pyflex.set_camPos(camPos)
        pyflex.set_camAngle(camAngle)

        self.n_particles = self.pyflex.get_n_particles()

        # let the cloth drop
        action_zero = np.zeros(4)
        for i in range(5):
            pyflex.step(action_zero)

    def get_state(self):
        state = np.zeros((self.n_particles, self.state_dim))
        state[:, :3] = self.pyflex.get_positions().reshape(-1, 4)[:, :3]
        return state

    def set_action(self, action):
        self.action = action.copy()

    def get_action(self):
        return self.action.copy()

    def get_param(self):
        return self.scene_params

    def step(self, capture=False, path=None):
        self.pyflex.step(self.action, capture=capture, path=path)


