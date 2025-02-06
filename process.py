import gymnasium
import fruit_picker_gym
import torch.nn as nn

import matplotlib.pyplot as plt

import numpy as np
import cv2
import stable_baselines3

from typing import Any

# Make the observation space to be image-based using the gym wrappers
class ProcessFrame84(gymnasium.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.env = env
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(90,90,1), dtype=np.uint8)

    def observation(self, obs: Any = None) -> np.ndarray:
        """
        This method overwrites the base method from the Gym ObservationWrapper.
        It takes in the actual observation from the environment and returns the
        modified observation
        :param obs: <Any> Actual observation
        :return: <np.ndarray> Modified observation. Processed image in this case
        """
        observation = self.env.render()
        return self.process(observation)

    @staticmethod
    def process(frame: np.ndarray) -> np.ndarray:
        """
        Static method within the ProcessFrame84 class for scaling down
        and turning the image into black and white.
        :param frame: <np.ndarray> Image from the gym environment's render method
        :return: <np.ndarray> Processed image
        """
        if frame.size == 720 * 960 * 3:
            img = np.reshape(frame, [720, 960, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img, None, alpha=0, beta=1,
                            norm_type=cv2.NORM_MINMAX)

        resized_screen = cv2.resize(
            img, (90, 90), interpolation=cv2.INTER_AREA)
        y_t = resized_screen
        y_t = np.reshape(y_t, [90, 90, 1])
        return y_t.astype(np.uint8)

# Make the image compatible with the PyTorch wrapper
class ImageToPyTorch(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        This method overwrites the base method from the Gym ObservationWrapper.
        It takes in the actual observation from the environment and returns the
        modified observation
        :param observation: <Any> Actual observation
        :return: <np.ndarray> Modified observation
        """
        return np.moveaxis(observation, 2, 0)


