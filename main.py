import gymnasium
import fruit_picker_gym
import torch.nn as nn

import matplotlib.pyplot as plt

from stable_baselines3 import PPO  # Or any other RL algorithm

from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.env_checker import check_env

from process import ProcessFrame84
from process import ImageToPyTorch
#
# import logging
#
# logging.basicConfig(filename="fruit_picker_trainer.log",
#                     format='%(asctime)s %(message)s',
#                     filemode='w')
#
# # Creating an object
# logger = logging.getLogger()
#
# # Setting the threshold of logger to DEBUG
# logger.setLevel(logging.DEBUG)


env = gymnasium.make('fruitpicker-env-01', max_episode_steps=2048)
env = ProcessFrame84(env)
env = ImageToPyTorch(env)
obs_ = env.reset()
check_env(env, warn=True)

model = PPO("CnnPolicy", env, batch_size=128, verbose=1, device="cuda", policy_kwargs=dict(normalize_images=False), n_steps=2048)

model.learn(total_timesteps=1000000)

# Save the trained model
model.save("fruit_picker_model")