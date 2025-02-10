import gymnasium
import fruit_picker_gym

from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env

from pydantic import BaseModel
from pydantic import Field

from model_params import ModelParameters
from pre_process_config import PreProcessConfig
from process import ProcessFrame84
from process import ImageToPyTorch
from process import NormalizeObservation
from process import NormalizeReward

class Trainer(BaseModel):

    model_parameters: ModelParameters = Field(default=None, description="Parameters to train the robot using PPO")
    pre_processing_config: PreProcessConfig = Field(default=None, description="Config for the pre-processing steps")


    @property
    def train(self) -> str:
        try:
            env = gymnasium.make('fruitpicker-env-01', max_episode_steps=self.model_parameters.max_episode_steps)
            if self.pre_processing_config.use_image_observation:
                env = ProcessFrame84(env)
                env = ImageToPyTorch(env)
                if self.pre_processing_config.apply_observation_normalization:
                    env = NormalizeObservation(env)
                if self.pre_processing_config.apply_reward_normalization:
                    env = NormalizeReward(env)
                check_env(env, warn=True)
                obs_ = env.reset()

            model = PPO(self.model_parameters.policy, env, batch_size=self.model_parameters.batch_size, verbose=1,
                        device="cuda", policy_kwargs=dict(normalize_images=False), n_steps=self.model_parameters.n_steps)

            model.learn(total_timesteps=self.model_parameters.total_time_steps)

            # Save the trained model
            model.save(self.model_parameters.model_name)
            return f"Model trained successfully and saved as {self.model_parameters.model_name}"
        except Exception as e:
            raise Exception(f"Model training failed: {e}")


    def infer(self) -> None:
        pass




