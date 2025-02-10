# import gymnasium
# import fruit_picker_gym
#
# from stable_baselines3 import PPO
#
# from stable_baselines3.common.env_checker import check_env
#
# from process import ProcessFrame84
# from process import ImageToPyTorch
# from process import NormalizeObservation
# from process import NormalizeReward
#
# def main() -> None:
#
#
#     env = gymnasium.make('fruitpicker-env-01', max_episode_steps=2048)
#     env = ProcessFrame84(env)
#     env = ImageToPyTorch(env)
#     # env = NormalizeObservation(env)
#     env = NormalizeReward(env)
#     check_env(env, warn=True)
#     obs_ = env.reset()
#
#
#     model = PPO("CnnPolicy", env, batch_size=64, verbose=1, device="cuda", policy_kwargs=dict(normalize_images=False), n_steps=4096)
#
#     model.learn(total_timesteps=2000000)
#
#     # Save the trained model
#     model.save("fruit_picker_model")
#
# if __name__ == '__main__':
#     main()


import argparse
import json
import os

from trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description='Processing configuration for training the fruit picker robot')
    parser.add_argument('-c', '--config_json',
                        type=str, help='path to the json file containing the configuration parameters',
                        required=True)
    args = vars(parser.parse_args())
    json_path: str = args['config_json']

    print(f"Args = {args}")
    print(f"json path = {json_path}")

    if not os.path.exists(json_path):
        raise Exception(f"Invalid path for the config json: {json_path}")

    if not ".json" in json_path:
        raise Exception(f"Entered path {json_path} is not a json. Only json files are accepted")

    with open(json_path, 'r') as file:
        config_json = json.load(file)

    try:
        robot_trainer = Trainer.model_validate_json(json.dumps(config_json))
        success_message = robot_trainer.train
        print(f"{success_message}")
    except Exception as e:
        raise Exception(f"Operation failed: {e}")


if __name__ == '__main__':
    main()