import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import scipy.integrate as spi
import dubins
from gym_truck_backerupper.envs.kinematic_model import kinematic_model
from gym_truck_backerupper.envs.DubinsPark import DubinsPark

try:
    import dubins
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install Dubins by running 'pip3 install Dubins')".format(e))

class TruckBackerUpperEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 30}

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
        
        self.max_hitch = 90
        self.max_steer = 45
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
        
        self.solver = spi.ode(kinematic_model).set_integrator('dopri5')
        self.solver.set_initial_value(self.ICs, self.t0)
        self.solver.set_f_params(self.u)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        ''' '''
        #return obs, reward, done, info

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
        self.ICs = [self.psi_1_IC, self.psi_2_IC, self.y_IC, self.curv_IC]

        self.state = np.array(self.ICs.copy())
        return np.array(self.state)

    def render(self, mode='human', close=False):
        ''' '''

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

