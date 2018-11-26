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
        
        self.turning_radius = 13.716
        self.res = .05
        self.path_planner = DubinsPark(self.turning_radius, self.res)

        self.max_curv = 1.0 / self.turning_radius
        self.min_curv = -1.0 / self.turning_radius
 
        self.max_hitch = np.radians(90.0)

        self.min_action = -np.radians(45.0)
        self.max_action = np.radians(45.0)

        self.low_state = np.array([self.min_psi_1, self.min_psi_2, self.min_y,
                                  self.min_curv])
        self.high_state = np.array([self.max_psi_1, self.max_psi_2, self.max_y,
                                   self.max_curv])

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)
        self.manual_track = False
        self.offset = False
        self.rendering = False
        
        self.t0 = 0.0
        self.t_final = 160.0
        self.dt = .010
        self.num_steps = int((self.t_final - self.t0)/self.dt) + 1
        self.sim_i = 1
        
        self.L1 = 5.7336
        self.L2 = 12.192
        self.h = -0.2286
        self.v = -1.12
        self.u = 0.0

        self.look_ahead = 0

        self.DCM = lambda ang: np.array([[np.cos(ang), np.sin(ang),  0], 
                                        [-np.sin(ang), np.cos(ang),  0],
                                        [     0     ,      0     ,   1]]) 
        self.seed()
         
        self.H_c = self.L2 / 3.0
        self.H_t = self.L2 / 3.0
        self.trail = 2.0 
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        
        self.DCM_g = lambda ang: np.array([[np.cos(ang), -np.sin(ang), 0], 
                                           [np.sin(ang), np.cos(ang),  0],
                                           [     0     ,      0     ,  1]])

        self.center = lambda x, y: np.array([[1, 0, x],
                                             [0, 1, y],
                                             [0, 0, 1]])

    def kinematic_model(self, t, x, u):
        self.u = u
        n = len(x)
        xd = np.zeros((n))
        xd[0] = (self.v / self.L1) * np.tan(self.u)

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
        self.u = 0
        self.goal_side = 1
        self.fin = False
        self.sim_i = 1
        self.last_index = 0
        self.last_c_index = 0
        self.goal = False
        self.jackknife = False
        self.out_of_bounds = False
        self.times_up = False
        self.min_d = self.max_x - self.min_x
        self.min_psi = self.max_psi_1.copy()

        if self.manual_track == False: 
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
            #print('Dubins spawned out of bounds, respawning new track')
            self.bound_reset()
        else: 
            if self.v < 0:
                #print('Going Backwards!')
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

            ## normal to goal loading dock
            self.dx = 100.0 * (self.track_vector[-1, 0] - self.track_vector[-2, 0])
            self.dy = 100.0 * (self.track_vector[-1, 1] - self.track_vector[-2, 1])
            self.dock_x = np.linspace(-self.dy, self.dy, 5) + self.qg[0]
            self.dock_y = np.linspace(self.dx, -self.dx, 5) + self.qg[1]
            
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
        self.manual_track = True
        print('Manual Track inputted')

    def manual_offset(self, y_IC, psi_2_IC, hitch_IC):
        self.y_IC_mod = y_IC
        self.psi_2_IC_mod = psi_2_IC
        self.hitch_IC_mod = hitch_IC
        self.offset = True
        print('Manual offset inputted')

    def manual_velocity(self, v):
        self.v = v
        print('Manual velocity inputted')

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
        elif self.theta < -self.max_hitch:
            self.jackknife = True
            done = True
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
            done = True

        t_x, t_y, _ = np.array([self.x2[self.sim_i], self.y2[self.sim_i], 1]).T + \
                               self.DCM_g(self.psi_2[self.sim_i]).dot(
                               np.array([-self.trail, 0, 1]).T)
        d_goal =  self.path_planner.distance(self.qg[0:2], 
                                             [t_x, 
                                              t_y])
        psi_goal = self.path_planner.safe_minus(self.qg[2], 
                                                self.psi_2[self.sim_i])
        if d_goal < self.min_d:
            self.min_d = d_goal.copy()
            self.min_psi = psi_goal.copy()
         
        if self.min_d < 3:
            self.goal_side = ((self.dock_y[0] - self.dock_y[-1]) * \
                             (self.track_vector[-100, 0] - self.dock_x[0]) + \
                             (self.dock_x[-1] - self.dock_x[0]) * \
                             (self.track_vector[-100, 1] - self.dock_y[0])) * \
                             ((self.dock_y[0] - self.dock_y[-1]) * \
                             (t_x - self.dock_x[0]) + \
                             (self.dock_x[-1] - self.dock_x[0]) * \
                             (t_y - self.dock_y[0]))
            if self.goal_side < 0 and self.sim_i > 3500:
                done = True
                self.fin = True

        self.goal = bool(self.min_d <= 0.15 and abs(self.min_psi) <= 0.1 
                         and self.fin)       

        self.s = self.get_error(self.sim_i)

        self.sim_i += 1
        if self.sim_i >= self.num_steps:
            self.times_up = True
            done = True

        r = -1 + self.goal * 100 - self.jackknife * 10 - \
            self.out_of_bounds * 10 - self.times_up * 10 + \
            bool(abs(self.s[0]) < 0.362) * 1 + bool(abs(self.s[1]) < 0.102) * 1 + \
            bool(abs(self.s[2]) < 0.724) * 1
        return self.s, r, done, {'goal' : self.goal, 'jackknife': self.jackknife,
                                 'out_of_bounds' : self.out_of_bounds,
                                 'times_up' : self.times_up, 'fin' : self.fin,
                                 'min_d' : self.min_d, 
                                 'min_psi' : np.degrees(self.min_psi),
                                 't' : self.t[self.sim_i-1]}

    def render(self, mode='human'):
        ''' '''
        self.rendering = True
        f = self.sim_i - 1
        
        ## Steering tyres
        r_tyre = 2.286 / 2
        t_tyre = r_tyre / 3
        fl_x, fl_y, _ = np.array([self.x1[f], self.y1[f], 1]).T + \
                     self.DCM_g(self.psi_1[f]).dot(
                     np.array([self.L1, self.H_c/2-t_tyre/2, 1]).T)


        fr_x, fr_y, _ = np.array([self.x1[f], self.y1[f], 1]).T + \
                     self.DCM_g(self.psi_1[f]).dot(
                     np.array([self.L1, -self.H_c/2+t_tyre/2, 1]).T)


        fl_x_steer = [fl_x+r_tyre,  fl_x-r_tyre, fl_x-r_tyre, fl_x+r_tyre, fl_x+r_tyre]
        fl_y_steer = [fl_y+t_tyre, fl_y+t_tyre, fl_y-t_tyre, fl_y-t_tyre, fl_y+t_tyre]
        corners_fl_steer = np.zeros((5, 3))
        for j in range(len(fl_x_steer)):
            corners_fl_steer[j, 0:3] = self.center(fl_x, fl_y).dot(
                                    self.DCM_g(self.psi_1[f]+self.u)).dot(
                                    self.center(-fl_x, -fl_y)).dot(
                                    np.array([fl_x_steer[j], fl_y_steer[j], 1]).T)

        fr_x_steer = [fr_x+r_tyre,  fr_x-r_tyre, fr_x-r_tyre, fr_x+r_tyre, fr_x+r_tyre]
        fr_y_steer = [fr_y+t_tyre, fr_y+t_tyre, fr_y-t_tyre, fr_y-t_tyre, fr_y+t_tyre]
        corners_fr_steer = np.zeros((5, 3))
        for j in range(len(fr_x_steer)):
            corners_fr_steer[j, 0:3] = self.center(fr_x, fr_y).dot(
                                    self.DCM_g(self.psi_1[f]+self.u)).dot(
                                    self.center(-fr_x, -fr_y)).dot(
                                    np.array([fr_x_steer[j], fr_y_steer[j], 1]).T)

        ## Tractor Rear tyres
        rl_x, rl_y, _ = np.array([self.x1[f], self.y1[f], 1]).T + \
                     self.DCM_g(self.psi_1[f]).dot(
                     np.array([0, self.H_c/2-t_tyre/2, 1]).T)


        rr_x, rr_y, _ = np.array([self.x1[f], self.y1[f], 1]).T + \
                     self.DCM_g(self.psi_1[f]).dot(
                     np.array([0, -self.H_c/2+t_tyre/2, 1]).T)


        rl_x_drive = [rl_x+r_tyre,  rl_x-r_tyre, rl_x-r_tyre, rl_x+r_tyre, rl_x+r_tyre]
        rl_y_drive = [rl_y+t_tyre, rl_y+t_tyre, rl_y-t_tyre, rl_y-t_tyre, rl_y+t_tyre]
        corners_rl_drive = np.zeros((5, 3))
        for j in range(len(rl_x_drive)):
            corners_rl_drive[j, 0:3] = self.center(rl_x, rl_y).dot(
                                    self.DCM_g(self.psi_1[f])).dot(
                                    self.center(-rl_x, -rl_y)).dot(
                                    np.array([rl_x_drive[j], rl_y_drive[j], 1]).T)

        rr_x_drive = [rr_x+r_tyre,  rr_x-r_tyre, rr_x-r_tyre, rr_x+r_tyre, rr_x+r_tyre]
        rr_y_drive = [rr_y+t_tyre, rr_y+t_tyre, rr_y-t_tyre, rr_y-t_tyre, rr_y+t_tyre]
        corners_rr_drive = np.zeros((5, 3))
        for j in range(len(rr_x_drive)):
            corners_rr_drive[j, 0:3] = self.center(rr_x, rr_y).dot(
                                    self.DCM_g(self.psi_1[f])).dot(
                                    self.center(-rr_x, -rr_y)).dot(
                                    np.array([rr_x_drive[j], rr_y_drive[j], 1]).T)

        ## Trailer tyres
        t_rl_x, t_rl_y, _ = np.array([self.x2[f], self.y2[f], 1]).T + \
                            self.DCM_g(self.psi_2[f]).dot(
                            np.array([0, self.H_t/2-t_tyre/2, 1]).T)


        t_rr_x, t_rr_y, _ = np.array([self.x2[f], self.y2[f], 1]).T + \
                            self.DCM_g(self.psi_2[f]).dot(
                            np.array([0, -self.H_t/2+t_tyre/2, 1]).T)


        rl_x_trailer = [t_rl_x+r_tyre,  t_rl_x-r_tyre, t_rl_x-r_tyre, t_rl_x+r_tyre, t_rl_x+r_tyre]
        rl_y_trailer = [t_rl_y+t_tyre, t_rl_y+t_tyre, t_rl_y-t_tyre, t_rl_y-t_tyre, t_rl_y+t_tyre]
        corners_rl_trailer = np.zeros((5, 3))
        for j in range(len(rl_x_trailer)):
            corners_rl_trailer[j, 0:3] = self.center(t_rl_x, t_rl_y).dot(
                                    self.DCM_g(self.psi_2[f])).dot(
                                    self.center(-t_rl_x, -t_rl_y)).dot(
                                    np.array([rl_x_trailer[j], rl_y_trailer[j], 1]).T)

        rr_x_trailer = [t_rr_x+r_tyre,  t_rr_x-r_tyre, t_rr_x-r_tyre, t_rr_x+r_tyre, t_rr_x+r_tyre]
        rr_y_trailer = [t_rr_y+t_tyre, t_rr_y+t_tyre, t_rr_y-t_tyre, t_rr_y-t_tyre, t_rr_y+t_tyre]
        corners_rr_trailer = np.zeros((5, 3))
        for j in range(len(rr_x_trailer)):
            corners_rr_trailer[j, 0:3] = self.center(t_rr_x, t_rr_y).dot(
                                    self.DCM_g(self.psi_2[f])).dot(
                                    self.center(-t_rr_x, -t_rr_y)).dot(
                                    np.array([rr_x_trailer[j], rr_y_trailer[j], 1]).T)
        ## Cab
        cab = 1.5
        x_trac = [self.x1[f]+self.L1+cab, self.x1[f]-cab, self.x1[f]-cab, 
                  self.x1[f]+self.L1+cab, self.x1[f]+self.L1+cab]
        y_trac = [self.y1[f]+self.H_c/2, self.y1[f]+self.H_c/2, 
                  self.y1[f]-self.H_c/2, self.y1[f]-self.H_c/2, 
                  self.y1[f]+self.H_c/2]

        corners_trac = np.zeros((5, 3))
        for j in range(len(x_trac)):
            corners_trac[j, 0:3] = self.center(self.x1[f], self.y1[f]).dot(
                                   self.DCM_g(self.psi_1[f])).dot(
                                   self.center(-self.x1[f], -self.y1[f])).dot(
                                   np.array([x_trac[j], y_trac[j], 1]).T)
        ## Trailer
        x_trail = [self.x2[f]+self.L2+self.trail, self.x2[f]-self.trail, self.x2[f]-self.trail, 
                  self.x2[f]+self.L2+self.trail, self.x2[f]+self.L2+self.trail]
        y_trail = [self.y2[f]+self.H_t/2, self.y2[f]+self.H_t/2, 
                  self.y2[f]-self.H_t/2, self.y2[f]-self.H_t/2, 
                  self.y2[f]+self.H_t/2]

        corners_trail = np.zeros((5, 3))
        for j in range(len(x_trail)):
            corners_trail[j, 0:3] = self.center(self.x2[f], self.y2[f]).dot(
                                    self.DCM_g(self.psi_2[f])).dot(
                                    self.center(-self.x2[f], -self.y2[f])).dot(
                                    np.array([x_trail[j], y_trail[j], 1]).T)
        ## Points
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
        self.ax.plot(hitch_trac[0], hitch_trac[1], 'g.')
        self.ax.plot(hitch_trail[0], hitch_trail[1], 'b.')
        self.ax.plot(self.qg[0], self.qg[1], 'r*')
        self.ax.plot(self.track_vector[:, 0], self.track_vector[:, 1], '--r')
        self.ax.plot(self.dock_x, self.dock_y, '--k')

        self.ax.plot(corners_fl_steer[:, 0], corners_fl_steer[:, 1], 'k')
        self.ax.plot(corners_fr_steer[:, 0], corners_fr_steer[:, 1], 'k')
        self.ax.plot(corners_rl_drive[:, 0], corners_rl_drive[:, 1], 'k')
        self.ax.plot(corners_rr_drive[:, 0], corners_rr_drive[:, 1], 'k')
        self.ax.plot(corners_rl_trailer[:, 0], corners_rl_trailer[:, 1], 'k')
        self.ax.plot(corners_rr_trailer[:, 0], corners_rr_trailer[:, 1], 'k')


        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        plt.pause(np.finfo(np.float32).eps)
        
    def close(self):
        ''' '''
        plt.close('all')
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))

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
