

import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import concurrent.futures
LR = 2.55673e-05
epoch = 20
timestep = 1e7
bsize = 32

PPO_Path = os.path.join('Training','Saved Models', 'PPOHumanStand{}{}{}{}'.format(timestep,LR,bsize,epoch))
model= PPO.load(PPO_Path)
env = gym.make('HumanoidStandup-v2')
episodes = 10
for episode in range(1,episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs) # By doing this, rather than taking a random action, the model is used to take actions
        obs, reward, done, info = env.step(action)
        score += reward
        
env.close()