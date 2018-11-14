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
        self.min_x = -40.0
        self.min_y = -40.0
        self.max_x = 40.0
        self.max_y = 40.0

        self.min_psi_1 = np.radians(0)
        self.min_psi_2 = np.radians(0)
        self.max_psi_1 = np.radians(360)
        self.max_psi_2 = np.radians(360)
        
        self.max_hitch = np.radians(90)
        self.max_steer = np.radians(45)

        self.min_action = -1.0
        self.max_action = 1.0

        self.low_state = np.array([self.min_psi_1, self.min_psi_2, self.min_y])
        self.high_state = np.array([self.max_psi_1, self.max_psi_2, self.max_y])

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)
        self.turning_radius = 13.716
        self.res = .05
        self.path_planner = DubinsPark(self.turning_radius, self.res)
        
        self.t0 = 0.0
        self.t_final = 80.0
        self.dt = .028
        self.num_steps = int((self.t_final - self.t0)/self.dt) + 1
        
        self.L1 = 5.7336
        self.L2 = 12.192
        self.h = -0.2286
        self.v = 25.0      
        self.u = 0
        
        self.seed()
        self.reset() 
         
        self.goal = False
        self.jackknife = False

        self.H_c = self.L2 / 3
        self.H_t = self.L2 / 3
        
        self.fig, self.ax = plt.subplots(1, 1)

        self.ax.set_xlim(self.min_x-2*self.turning_radius, self.max_x+2*self.turning_radius)
        self.ax.set_ylim(self.min_y-2*self.turning_radius, self.max_y+2*self.turning_radius)

        '''plotcols = ["blue", "green", "blue", "green", "red"]
        self.lines = []
        for index in range(2):
            lobj = self.ax.plot([], [], ls="-", marker="", lw=2, color=plotcols[index])[0]
            self.lines.append(lobj)

        self.points = []
        for index in range(5):
            pobj = self.ax.plot([], [], ls="", marker='*', color=plotcols[index])[0]
            self.points.append(pobj)

        self.dubins_dash, = self.ax.plot([], [], ls="--", marker="", lw=1, color="red")
 
        self.anim = animation.FuncAnimation(self.fig, self.animate,
                                            init_func=self.init_anim,
                                            frames=None, 
                                            Gblit=False,
                                            repeat=False)'''

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

        self.t[self.sim_i] = self.solver.t
        self.psi_1[self.sim_i] = self.solver.y[0]
        self.psi_2[self.sim_i] = self.solver.y[1]
        self.x1[self.sim_i] = self.solver.y[2]
        self.y1[self.sim_i] = self.solver.y[3]
        self.x2[self.sim_i] = self.solver.y[4]
        self.y2[self.sim_i] = self.solver.y[5]

        print(self.x2[self.sim_i], self.y2[self.sim_i], self.psi_2[self.sim_i])

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

        if self.goal or self.jackknife or self.sim_i >= self.num_steps-1:
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
        #print('step: x2: {}, y2: {}'.format(self.x2[self.sim_i], self.y2[self.sim_i]))
        print('step: ', self.sim_i)
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
            self.track_vector[0, 3] += np.pi
            self.track_vector[:, 4] *= -1

        self.y_IC = 0
        self.psi_2_IC = np.radians(0) + self.track_vector[0, 3]
        self.hitch_IC = np.radians(0)
        self.psi_1_IC = self.hitch_IC + self.psi_2_IC
        self.curv_IC = self.track_vector[0, 4]

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

        self.solver = spi.ode(self.kinematic_model).set_integrator('dopri5')
        self.solver.set_initial_value(self.ICs, self.t0)
        self.solver.set_f_params(self.u)

        print(self.q0)
        print(self.trailerIC[0], self.trailerIC[1])
        print(self.psi_2_IC)

        self.t = np.zeros((self.num_steps, 1))
        self.psi_1 = np.zeros((self.num_steps, 1))
        self.psi_2 = np.zeros((self.num_steps, 1))
        self.x1 = np.zeros((self.num_steps, 1))
        self.y1 = np.zeros((self.num_steps, 1))
        self.x2 = np.zeros((self.num_steps, 1))
        self.y2 = np.zeros((self.num_steps, 1))
        
        self.psi_1[0] = self.psi_1_IC.copy()
        self.psi_2[0] = self.psi_2_IC.copy()
        self.x1[0] = self.tractorIC[0].copy()
        self.y1[0] = self.tractorIC[1].copy()
        self.x2[0] = self.trailerIC[0].copy()
        self.y2[0] = self.trailerIC[1].copy()

        self.s = np.array(self.ICs.copy())
        return np.array(self.s)

    def gen_f(self):
        while True:
            yield self.sim_i

    def init_anim(self):
        ''' '''
        self.dubins_dash.set_data(self.track_vector[:, 0], self.track_vector[:, 1])
        return self.dubins_dash

    def animate(self, f):
        ''' '''
        print('animate: ', f)
        x_trail = [self.x2[f]+self.L2, self.x2[f], self.x2[f], 
                  self.x2[f]+self.L2, self.x2[f]+self.L2]
        y_trail = [self.y2[f]+self.H_t/2, self.y2[f]+self.H_t/2, 
                  self.y2[f]-self.H_t/2, self.y2[f]-self.H_t/2, 
                  self.y2[f]+self.H_t/2]

        corners_trail = np.zeros((5, 3))
        for j in range(len(x_trail)):
            corners_trail[j, 0:3] = self.center(self.x2[f], self.y2[f]).dot(
                                    self.DCM(self.psi_2[f])).dot(
                                    self.center(-self.x2[f], -self.y2[f])).dot(
                                    np.array([x_trail[j], y_trail[j], 1]).T)


        x_trac = [self.x1[f]+self.L1, self.x1[f], self.x1[f], 
                  self.x1[f]+self.L1, self.x1[f]+self.L1]
        y_trac = [self.y1[f]+self.H_c/2, self.y1[f]+self.H_c/2, 
                  self.y1[f]-self.H_c/2, self.y1[f]-self.H_c/2, 
                  self.y1[f]+self.H_c/2]

        corners_trac = np.zeros((5, 3))
        for j in range(len(x_trac)):
            corners_trac[j, 0:3] = self.center(self.x1[f], self.y1[f]).dot(
                                   self.DCM(self.psi_1[f])).dot(
                                   self.center(-self.x1[f], -self.y1[f])).dot(
                                   np.array([x_trac[j], y_trac[j], 1]).T)


        xlist = [corners_trail[:, 0], corners_trac[:, 0],
                 self.x2[f], self.x1[f]]
        ylist = [corners_trail[:, 1], corners_trac[:, 1], 
                self.y2[f], self.y1[f]]
        
        for lnum, line, in enumerate(self.lines):
            line.set_data(xlist[lnum], ylist[lnum])

        hitch_trac = self.center(self.x1[f], self.y1[f]).dot(
                     self.DCM(self.psi_1[f])).dot(
                     self.center(-self.x1[f], -self.y1[f])).dot(
                     np.array([self.x1[f]-self.h, self.y1[f], 1]).T)


        hitch_trail = self.center(self.x2[f], self.y2[f]).dot(
                      self.DCM(self.psi_2[f])).dot(
                      self.center(-self.x2[f], -self.y2[f])).dot(
                      np.array([self.x2[f]+self.L2, self.y2[f], 1]).T)
        
        xlist = [self.x2[f], self.x1[f], hitch_trail[0], hitch_trac[0],
                 self.track_vector[0, 0]]
        ylist = [self.y2[f], self.y1[f], hitch_trail[1], hitch_trac[1],
                 self.track_vector[0, 1]]

        for pnum, point in enumerate(self.points):
            point.set_data(xlist[pnum], ylist[pnum])

        #print('animate: x2: {}, y2: {}'.format(self.x2[f-1], self.y2[f-1]))
        #self.ax.set_xlim(self.x2[f]-25, self.x2[f]+25)
        #self.ax.set_ylim(self.y2[f]-25, self.y2[f]+25)
        return self.lines, self.points, self.dubins_dash

    def render(self, mode='human'):
        ''' '''
        f = self.sim_i - 1
        x_trail = [self.x2[f]+self.L2, self.x2[f], self.x2[f], 
                  self.x2[f]+self.L2, self.x2[f]+self.L2]
        y_trail = [self.y2[f]+self.H_t/2, self.y2[f]+self.H_t/2, 
                  self.y2[f]-self.H_t/2, self.y2[f]-self.H_t/2, 
                  self.y2[f]+self.H_t/2]

        corners_trail = np.zeros((5, 3))
        for j in range(len(x_trail)):
            corners_trail[j, 0:3] = self.center(self.x2[f], self.y2[f]).dot(
                                    self.DCM(self.psi_2[f])).dot(
                                    self.center(-self.x2[f], -self.y2[f])).dot(
                                    np.array([x_trail[j], y_trail[j], 1]).T)


        x_trac = [self.x1[f]+self.L1, self.x1[f], self.x1[f], 
                  self.x1[f]+self.L1, self.x1[f]+self.L1]
        y_trac = [self.y1[f]+self.H_c/2, self.y1[f]+self.H_c/2, 
                  self.y1[f]-self.H_c/2, self.y1[f]-self.H_c/2, 
                  self.y1[f]+self.H_c/2]

        corners_trac = np.zeros((5, 3))
        for j in range(len(x_trac)):
            corners_trac[j, 0:3] = self.center(self.x1[f], self.y1[f]).dot(
                                   self.DCM(self.psi_1[f])).dot(
                                   self.center(-self.x1[f], -self.y1[f])).dot(
                                   np.array([x_trac[j], y_trac[j], 1]).T)

        hitch_trac = self.center(self.x1[f], self.y1[f]).dot(
                     self.DCM(self.psi_1[f])).dot(
                     self.center(-self.x1[f], -self.y1[f])).dot(
                     np.array([self.x1[f]-self.h, self.y1[f], 1]).T)


        hitch_trail = self.center(self.x2[f], self.y2[f]).dot(
                      self.DCM(self.psi_2[f])).dot(
                      self.center(-self.x2[f], -self.y2[f])).dot(
                      np.array([self.x2[f]+self.L2, self.y2[f], 1]).T)
        
        self.ax.clear()
        self.ax.plot(corners_trail[:, 0], corners_trail[:, 1], 'b')
        self.ax.plot(corners_trac[:, 0], corners_trac[:, 1], 'g')
        self.ax.plot(self.x2[f], self.y2[f], 'b*')
        self.ax.plot(self.x1[f], self.y1[f], 'g*')
        self.ax.plot(hitch_trail[0], hitch_trail[1], 'b*')
        self.ax.plot(hitch_trac[0], hitch_trac[1], 'g*')
        self.ax.plot(self.qg[0], self.qg[1], 'r*')
        self.ax.plot(self.track_vector[:, 0], self.track_vector[:, 1], '--r')
        self.ax.set_xlim(self.min_x-2*self.turning_radius, self.max_x+2*self.turning_radius)
        self.ax.set_ylim(self.min_y-2*self.turning_radius, self.max_y+2*self.turning_radius)
        plt.pause(np.finfo(np.float32).eps)
        print('Render: ', f)
        #print('Render: x2: {}, y2: {}'.format(self.x2[self.sim_i-1], self.y2[self.sim_i-1]))

    def close(self):
        plt.close()
