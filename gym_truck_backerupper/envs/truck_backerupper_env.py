import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

try:
    import dubins
except ImportError as e:
    raise error.DependencyNotInstalled("{}. >>> pip install dubins".format(e))

import logging
logger = logging.getLogger(__name__)

class TruckBackerUpperEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ''' '''

    def step(self, action):
        ''' '''
        return True

    def get_reward(self):
        ''' '''

    def reset(self):
        ''' '''

    def render(self, mode='human', close=False):
        ''' '''
