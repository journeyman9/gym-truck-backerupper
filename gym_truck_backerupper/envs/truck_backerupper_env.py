import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import scipy.integrate as spi
import dubins
from kinematic_model import kinematic_model

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
        
        self.max_hitch = 90
        self.max_steer = 45
        self.goal_position = [0, 0, 0]

        self.min_action = -1.0
        self.max_action = 1.0

        self.low_state = np.array([-4 * np.pi, -4 * np.pi, -200])
        self.high_state = np.array([4 * np.pi, 4 * np.pi, 200])

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action, shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state,
                                            high=self.high_state)
        
        self.seed()
        self.reset()

        self.t0 = 0.0
        self.t_final = 80.0
        self.dt = .001
        self.num_steps = int((t_final - t0)/dt) + 1
        
        self.L1 = 5.7336
        self.L2 = 12.192
        self.h = -0.2286
        self.v = 25.0      
        self.u = 0
        
        path_planner = DubinsPark()
        solver = spi.ode(kinematic_model).set_integrator('dopri5')

    def seed(self):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        ''' '''
        #return obs, reward, done, info

    def reset(self):
        ''' '''
        #return self._get_obs()

    def render(self, mode='human', close=False):
        ''' '''

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
