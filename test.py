import gymnasium
import fruit_picker_gym
env = gymnasium.make('fruitpicker-env-01')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()