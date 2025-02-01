from gym.envs.registration import register

register(
    id='fruitpicker-env-01',
    entry_point='fruit_picker_gym.envs:FruitPickerEnv',
)