import gym
#import mujoco_py
env = gym.make('HumanoidStandup-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

