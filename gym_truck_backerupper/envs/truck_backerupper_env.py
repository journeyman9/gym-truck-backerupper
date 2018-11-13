import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import scipy.integrate as spi
import dubins
from gym_truck_backerupper.envs.DubinsPark import DubinsPark
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import time

try:
    import dubins
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install Dubins by running 'pip3 install Dubins')".format(e))

class TruckBackerUpperEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        ''' '''
        self.min_x = -200.0
        self.min_y = -200.0
        self.max_x = 200.0
        self.max_y = 200.0

        self.min_psi_1 = np.radians(0)
        self.min_psi_2 = np.radians(0)
        self.max_psi_1 = np.radians(360)
        self.max_psi_2 = np.radians(360)
        
        self.max_hitch = np.radians(90)
        self.max_steer = np.radians(45)
        self.goal_position = [0, 0, 0]

        self.min_action = -1.0
        self.max_action = 1.0

        self.low_state = np.array([self.min_psi_1, self.min_psi_2, self.min_y])
        self.high_state = np.array([self.max_psi_1, self.max_psi_2, self.max_y])

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)

        self.path_planner = DubinsPark(13.716, .05)
        
        self.t0 = 0.0
        self.t_final = 80.0
        self.dt = .001
        self.num_steps = int((self.t_final - self.t0)/self.dt) + 1
        
        self.L1 = 5.7336
        self.L2 = 12.192
        self.h = -0.2286
        self.v = 25.0      
        self.u = 0
        
        self.seed()
        self.reset() 
        
        self.solver = spi.ode(self.kinematic_model).set_integrator('dopri5')
        self.solver.set_initial_value(self.ICs, self.t0)
        self.solver.set_f_params(self.u)
        
        self.goal = False
        self.jackknife = False

        self.H_c = self.L2 / 3
        self.H_t = self.L2 / 3
        
        self.fig, self.ax = plt.subplots(1, 1)
        self.tractor, = self.ax.plot([], [], lw=2)
        
        self.anim = animation.FuncAnimation(self.fig, self.animate,
                                            init_func=self.init_anim,
                                            frames=range(self.num_steps), 
                                            blit=False,
                                            repeat=False)

        self.DCM = lambda ang: np.array([[np.cos(ang), -np.sin(ang), 0], 
                                         [np.sin(ang), np.cos(ang),  0],
                                         [     0     ,      0     ,  1]])

        self.center = lambda x, y: np.array([[1, 0, x],
                                             [0, 1, y],
                                             [0, 0, 1]])
        self.sim_i = 1


    def kinematic_model(self, t, x, u):
        n = len(x)
        xd = np.zeros((n, 1))
        xd[0] = (self.v / self.L1) * np.tan(u)

        self.theta = x[0] - x[1]

        xd[1] = (self.v / self.L2) * np.sin(self.theta) -\
                (self.h / self.L2) * xd[0] * np.cos(self.theta)
        
        vt = self.v * np.cos(self.theta) + self.h * xd[0] * np.sin(self.theta)

        xd[2] = self.v * np.cos(x[0])
        xd[3] = self.v * np.sin(x[0])

        xd[4] = vt * np.cos(x[1])
        xd[5] = vt * np.sin(x[1])

        return xd

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        ''' '''
        done = False
        self.solver.set_f_params(a)
        self.solver.integrate(self.solver.t + self.dt)

        self.t = self.solver.t
        self.psi_1 = self.solver.y[0]
        self.psi_2 = self.solver.y[1]
        self.x1 = self.solver.y[2]
        self.y1 = self.solver.y[3]
        self.x2 = self.solver.y[4]
        self.y2 = self.solver.y[5]

        self.s = np.array([self.solver.y[0], self.solver.y[1], self.solver.y[2], 
                          self.solver.y[3], self.solver.y[4], self.solver.y[5]])

        if self.theta > self.max_hitch:
            self.jackknife = True
        elif self.theta < -self.max_hitch:
            self.jackknife = True
        else:
            self.jackknife = False

        self.d_goal =  self.path_planner.distance(self.s[4:6], self.qg[0:2])
        self.psi_goal = self.path_planner.safe_minus(self.s[1], self.qg[2])
        self.goal = bool(self.d_goal <= 0.15 and self.psi_goal <= 0.1)

        if self.goal or self.jackknife or self.sim_i >= self.num_steps:
            done = True

        r = 0
        if self.goal:
            r += 100
        elif self.jackknife:
            r -= 10
        elif self.sim_i >= self.num_steps:
            r -= 10
        else:
            r -= 1

        self.sim_i += 1

        return self.s, r, done, {}

    def reset(self):
        ''' '''
        self.x_start = np.random.randint(self.min_x, self.max_x)
        self.y_start = np.random.randint(self.min_y, self.max_y)
        self.psi_start = np.radians(np.random.randint(
                                    np.degrees(self.min_psi_2), 
                                    np.degrees(self.max_psi_2)))

        self.x_goal = np.random.randint(self.min_x, self.max_x)
        self.y_goal = np.random.randint(self.min_y, self.max_y)
        self.psi_goal = np.radians(np.random.randint(
                                   np.degrees(self.min_psi_2),
                                   np.degrees(self.max_psi_2)))

        self.q0 = [self.x_start, self.y_start, self.psi_start]
        self.qg = [self.x_goal, self.y_goal, self.psi_goal]
        self.track_vector = self.path_planner.generate(self.q0, self.qg)

        if self.v < 0:
            self.track_vector[0, 4] += np.pi
            self.track_vector[:, 3] *= -1

        self.y_IC = 0
        self.psi_2_IC = np.radians(0) + self.track_vector[0, 4]
        self.hitch_IC = np.radians(0)
        self.psi_1_IC = self.hitch_IC + self.psi_2_IC
        self.curv_IC = self.track_vector[0, 3]
        
        self.trailerIC = [self.track_vector[0, 0] - 
                          self.y_IC*np.sin(self.psi_2_IC),
                          self.track_vector[0, 1] + 
                          self.y_IC*np.cos(self.psi_2_IC)]
        self.tractorIC = [self.trailerIC[0] + self.L2*np.cos(self.psi_2_IC) + 
                          self.h*np.cos(self.psi_1_IC),
                          self.trailerIC[1] + self.L2*np.sin(self.psi_2_IC) + 
                          self.h*np.sin(self.psi_1_IC)]
        self.ICs = [self.psi_1_IC, self.psi_2_IC,
                    self.tractorIC[0], self.tractorIC[1],
                    self.trailerIC[0], self.trailerIC[1]]

        self.psi_1 = self.psi_1_IC.copy()
        self.psi_2 = self.psi_2_IC.copy()
        self.x1 = self.tractorIC[0].copy()
        self.y1 = self.tractorIC[1].copy()
        self.x2 = self.trailerIC[0].copy()
        self.y2 = self.trailerIC[1].copy()

        self.s = np.array(self.ICs.copy())
        return np.array(self.s)

    def init_anim(self):
        ''' '''
        self.tractor.set_data([], [])
        return self.tractor,

    def animate(self, f):
        ''' '''
        x_trac = [self.x1+self.L1, self.x1, self.x1, self.x1+self.L1, 
                  self.x1+self.L1]
        y_trac = [self.y1+self.H_c/2, self.y1+self.H_c/2, 
                  self.y1-self.H_c/2, self.y1-self.H_c/2, 
                  self.y1+self.H_c/2]

        corners_trac = np.zeros((5, 3))
        for j in range(len(x_trac)):
            corners_trac[j, 0:3] = self.center(self.x1, self.y1).dot(
                                   self.DCM(self.psi_2)).dot(
                                   self.center(-self.x1, -self.y1)).dot(
                                   np.array([x_trac[j], y_trac[j], 1]).T)
        self.tractor.set_data(corners_trac[:, 0], corners_trac[:, 1])
        #self.tractor.set_data(x_trac, y_trac)
        self.ax.set_xlim(self.x2-25, self.x2+25)
        self.ax.set_ylim(self.y2-25, self.y2+25)
        return self.tractor,

    def render(self, mode='human'):
        ''' '''
        plt.pause(np.finfo(np.float32).eps)

    def close(self):
        plt.close()
