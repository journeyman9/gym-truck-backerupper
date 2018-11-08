import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import scipy.integrate as spi
import dubins

try:
    import dubins
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install Dubins by running 'pip3 install Dubins')".format(e))

class TruckBackerUpperEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        ''' '''

    def step(self, a):
        ''' '''
        #return obs, reward, done, info

    def reset(self):
        ''' '''
        #return self._get_obs()

    def render(self, mode='human', close=False):
        ''' '''
