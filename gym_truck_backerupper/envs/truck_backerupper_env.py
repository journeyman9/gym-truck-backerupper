import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import scipy.integrate as spi
from gym_truck_backerupper.envs.DubinsPark import DubinsPark
import matplotlib.pyplot as plt
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

        self.min_psi_1 = np.radians(0.0)
        self.min_psi_2 = np.radians(0.0)
        self.max_psi_1 = np.radians(360.0)
        self.max_psi_2 = np.radians(360.0)
 
        self.max_hitch = np.radians(90.0)

        self.min_action = -np.radians(45.0)
        self.max_action = np.radians(45.0)

        self.low_state = np.array([self.min_psi_1, self.min_psi_2, self.min_y])
        self.high_state = np.array([self.max_psi_1, self.max_psi_2, self.max_y])

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)
        self.manual = False
        self.offset = False
        self.rendering = False

        self.turning_radius = 13.716
        self.res = .05
        self.path_planner = DubinsPark(self.turning_radius, self.res)
        
        self.t0 = 0.0
        self.t_final = 80.0
        self.dt = .010
        self.num_steps = int((self.t_final - self.t0)/self.dt) + 1
        self.sim_i = 1
        
        self.L1 = 5.7336
        self.L2 = 12.192
        self.h = -0.2286
        self.v = -25.0
        self.u = 0.0

        self.last_index = 0
        self.last_c_index = 0
        self.look_ahead = 0

        self.DCM = lambda ang: np.array([[np.cos(ang), np.sin(ang),  0], 
                                        [-np.sin(ang), np.cos(ang),  0],
                                        [     0     ,      0     ,   1]]) 
        self.seed()
         
        self.goal = False
        self.jackknife = False
        self.out_of_bounds = False
        self.times_up = False
        self.min_d = self.max_x - self.min_x
        self.min_psi = self.max_psi_1.copy()

        self.H_c = self.L2 / 3.0
        self.H_t = self.L2 / 3.0
        
        self.fig, self.ax = plt.subplots(1, 1)

        self.DCM_g = lambda ang: np.array([[np.cos(ang), -np.sin(ang), 0], 
                                           [np.sin(ang), np.cos(ang),  0],
                                           [     0     ,      0     ,  1]])

        self.center = lambda x, y: np.array([[1, 0, x],
                                             [0, 1, y],
                                             [0, 0, 1]])

    def kinematic_model(self, t, x, u):
        n = len(x)
        xd = np.zeros((n))
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

    def reset(self):
        ''' '''
        if self.manual == False: 
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

        if (max(self.track_vector[:, 0]) >= self.max_x - self.L2 or
            min(self.track_vector[:, 0]) <= self.min_x + self.L2 or
            max(self.track_vector[:, 1]) >= self.max_y - self.L2 or
            min(self.track_vector[:, 1]) <= self.min_y + self.L2):
            print('Dubins spawned out of bounds, respawning new track')
            self.bound_reset()
        else: 
            if self.v < 0:
                print('Going Backwards!')
                self.track_vector[:, 3] += np.pi
                self.q0[2] += np.pi
                self.qg[2] += np.pi
                self.track_vector[:, 2] *= -1

            if self.offset == False:
                self.y_IC = 0
                self.psi_2_IC = np.radians(0) + self.track_vector[0, 3]
                self.hitch_IC = np.radians(0)
            else:
                self.y_IC = self.y_IC_mod
                self.psi_2_IC = np.radians(self.psi_2_IC_mod) + self.track_vector[0, 3]
                self.hitch_IC = np.radians(self.hitch_IC_mod)
            
            self.psi_1_IC = self.hitch_IC + self.psi_2_IC
            self.curv_IC = self.track_vector[0, 4]

            self.trailerIC = [self.track_vector[0, 0] - 
                              self.y_IC*np.sin(self.track_vector[0, 3]),
                              self.track_vector[0, 1] + 
                              self.y_IC*np.cos(self.track_vector[0, 3])]
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

            self.t = np.zeros((self.num_steps))
            self.psi_1 = np.zeros((self.num_steps))
            self.psi_2 = np.zeros((self.num_steps))
            self.x1 = np.zeros((self.num_steps))
            self.y1 = np.zeros((self.num_steps))
            self.x2 = np.zeros((self.num_steps))
            self.y2 = np.zeros((self.num_steps))
            
            self.psi_1[0] = self.psi_1_IC.copy()
            self.psi_2[0] = self.psi_2_IC.copy()
            self.x1[0] = self.tractorIC[0].copy()
            self.y1[0] = self.tractorIC[1].copy()
            self.x2[0] = self.trailerIC[0].copy()
            self.y2[0] = self.trailerIC[1].copy()

            self.r_x2 = np.zeros((self.num_steps))
            self.r_y2 = np.zeros((self.num_steps))
            self.r_psi_2 = np.zeros((self.num_steps))
            self.r_psi_1 = np.zeros((self.num_steps))

            self.psi_2_e = np.zeros((self.num_steps))
            self.psi_1_e = np.zeros((self.num_steps))
            self.x2_e = np.zeros((self.num_steps))
            self.y2_e = np.zeros((self.num_steps))

            self.error = np.zeros((self.num_steps, 3))
            self.curv = np.zeros((self.num_steps))

            self.s = self.get_error(0)
        return self.s

    def bound_reset(self):
        ''' '''
        self.reset()

    def manual_course(self, q0, qg):
        self.q0 = np.array([q0[0], q0[1], np.radians(q0[2])])
        self.qg = np.array([qg[0], qg[1], np.radians(qg[2])])
        self.manual = True
        print('Manual Track inputted')

    def manual_offset(self, y_IC, psi_2_IC, hitch_IC):
        self.y_IC_mod = y_IC
        self.psi_2_IC_mod = psi_2_IC
        self.hitch_IC_mod = hitch_IC
        self.offset = True
        print('Manual offset inputted')

    def step(self, a):
        ''' '''
        done = False
        if a > self.max_action:
            a = self.max_action
        elif a < self.min_action:
            a = self.min_action

        self.solver.set_f_params(a)
        self.solver.integrate(self.solver.t + self.dt)

        self.t[self.sim_i] = self.solver.t
        self.psi_1[self.sim_i] = self.solver.y[0]
        self.psi_2[self.sim_i] = self.solver.y[1]
        self.x1[self.sim_i] = self.solver.y[2]
        self.y1[self.sim_i] = self.solver.y[3]
        self.x2[self.sim_i] = self.solver.y[4]
        self.y2[self.sim_i] = self.solver.y[5]

        if self.theta > self.max_hitch:
            self.jackknife = True
            done = True
            print('Jackknife')
        elif self.theta < -self.max_hitch:
            self.jackknife = True
            done = True
            print('Jackknife')
        else:
            self.jackknife = False

        if (self.x1[self.sim_i] >= self.max_x or
            self.x1[self.sim_i] <= self.min_x or
            self.x2[self.sim_i] >= self.max_x or
            self.x2[self.sim_i] <= self.min_y or
            self.y1[self.sim_i] >= self.max_y or
            self.y1[self.sim_i] <= self.min_y or
            self.y2[self.sim_i] >= self.max_y or
            self.y2[self.sim_i] <= self.min_y):
            self.out_of_bounds = True
            print('Out of Bounds')
            done = True

        d_goal =  self.path_planner.distance(self.qg[0:2], 
                                             [self.x2[self.sim_i], 
                                              self.y2[self.sim_i]])
        psi_goal = self.path_planner.safe_minus(self.qg[2], 
                                                self.psi_2[self.sim_i])
        if d_goal < self.min_d:
            self.min_d = d_goal
            self.min_psi = psi_goal
        self.goal = bool(d_goal <= 0.15 and abs(psi_goal) <= 0.1 and self.sim_i > 10)

        if self.goal:
            done = True
            print('GOAL')

        if self.sim_i+1 >= self.num_steps:
            self.times_up = True
            done = True
            print('Times Up')

        self.s = self.get_error(self.sim_i)

        r = 0
        if self.goal:
            r += 100
        elif self.jackknife:
            r -= 10
        elif self.sim_i >= self.num_steps:
            r -= 10
        else:
            r -= 1
         
        if done:
            print('d = {:.3f} m and psi = {:.3f} degrees'.format(self.min_d, 
                                         np.degrees(self.min_psi)))
            if self.rendering:
                plt.show()

        self.sim_i += 1
        return self.s, r, done, {}

    def render(self, mode='human'):
        ''' '''
        self.rendering = True
        f = self.sim_i - 1
        x_trail = [self.x2[f]+self.L2, self.x2[f], self.x2[f], 
                  self.x2[f]+self.L2, self.x2[f]+self.L2]
        y_trail = [self.y2[f]+self.H_t/2, self.y2[f]+self.H_t/2, 
                  self.y2[f]-self.H_t/2, self.y2[f]-self.H_t/2, 
                  self.y2[f]+self.H_t/2]

        corners_trail = np.zeros((5, 3))
        for j in range(len(x_trail)):
            corners_trail[j, 0:3] = self.center(self.x2[f], self.y2[f]).dot(
                                    self.DCM_g(self.psi_2[f])).dot(
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
                                   self.DCM_g(self.psi_1[f])).dot(
                                   self.center(-self.x1[f], -self.y1[f])).dot(
                                   np.array([x_trac[j], y_trac[j], 1]).T)

        hitch_trac = self.center(self.x1[f], self.y1[f]).dot(
                     self.DCM_g(self.psi_1[f])).dot(
                     self.center(-self.x1[f], -self.y1[f])).dot(
                     np.array([self.x1[f]-self.h, self.y1[f], 1]).T)


        hitch_trail = self.center(self.x2[f], self.y2[f]).dot(
                      self.DCM_g(self.psi_2[f])).dot(
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
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        plt.pause(np.finfo(np.float32).eps)
        
    def close(self):
        ''' '''
        plt.close()

    def lookahead(self, look_ahead):
        self.look_ahead = look_ahead

    def get_closest_index(self, x, y, last_index, track_vector, look_ahead):
        ''' '''
        min_index = last_index
        min_dist = self.path_planner.distance([track_vector[last_index, 0],
                                               track_vector[last_index, 1]], 
                                               [x, y])
        search_forward = True

        ## search behind
        search_behind = 10
        if last_index > search_behind:
            last_index = last_index - search_behind

        for i in range(last_index, last_index + search_behind - 1):
            cur_dist = self.path_planner.distance([track_vector[i, 0],
                                                   track_vector[i, 1]],
                                                   [x, y])
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_index = i
                search_forward = False

            ## search ahead
            if search_forward:
                search_ahead = 10
                if min_index > len(track_vector) - search_ahead:
                    search_length = len(track_vector)
                else:
                    search_length = min_index + search_ahead

                for i in range(min_index, search_length):
                    cur_dist = self.path_planner.distance([track_vector[i, 0],
                                                           track_vector[i, 1]],
                                                           [x, y])
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        min_index = i

            ## implement look_ahead
            if look_ahead:
                if min_index >= len(track_vector) - look_ahead:
                    pass
                else:
                    min_index = min_index + look_ahead
        return min_index

    def get_error(self, i):
        ''' '''
        self.last_index = self.get_closest_index(self.x2[i], self.y2[i],
                                                 self.last_index,
                                                 self.track_vector,
                                                 self.look_ahead)
        
        self.last_c_index = self.get_closest_index(self.x1[i], self.y1[i],
                                                 self.last_c_index,
                                                 self.track_vector,
                                                 self.look_ahead)
        self.r_x2[i] = self.track_vector[self.last_index, 0]
        self.r_y2[i] = self.track_vector[self.last_index, 1]
        self.r_psi_2[i] = self.track_vector[self.last_index, 3]
        self.r_psi_1[i] = self.track_vector[self.last_c_index, 3]

        self.psi_2_e[i] = self.path_planner.safe_minus(self.r_psi_2[i],
                                                       self.psi_2[i])
        self.psi_1_e[i] = self.path_planner.safe_minus(self.r_psi_1[i],
                                                       self.psi_1[i])
        self.x2_e[i] = self.r_x2[i] - self.x2[i]
        self.y2_e[i] = self.r_y2[i] - self.y2[i]

        self.error[i, 0:3] = self.DCM(self.psi_2[i]).dot(
                                               np.array([self.x2_e[i], 
                                               self.y2_e[i],
                                               self.psi_2[i]]).T)
        self.curv[i] = self.track_vector[self.last_c_index, 2]
        return np.array([self.psi_1_e[i], self.psi_2_e[i], self.error[i, 1], 
                         self.L1 * self.curv[i]])
