import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

class KukaCustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Connect to the UI for visualization
        p.connect(p.GUI)

        # Reset the 3D OpenGL debug visualizer camera distance (between eye and camera
        # target position), camera yaw and pitch and camera target position
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # Define the action space
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        # Define the observation space
        self.observation_space = spaces.Box(np.array([-1] * 5), np.array([1] * 5))


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass

