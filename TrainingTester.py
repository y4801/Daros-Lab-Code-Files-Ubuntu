import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('HumanoidStandup-v2')
log_path = os.path.join('Training','Logs')

env = DummyVecEnv([lambda:env])

PPO_Path = os.path.join('Training','Saved Models','PPOHumanStand10M5')
#Load the model:
model = PPO.load(PPO_Path,env=env)

#Run the model in the environment and get rewards
env = gym.make('HumanoidStandup-v2')
episodes = 10
for episode in range(1,episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        #env.render()
        action, _ = model.predict(obs) # By doing this, rather than taking a random action, the model is used to take actions
        obs, reward, done, info = env.step(action)
        score += reward
    #print('Episode:{} Score:{}'.format(episode, score))
    
    rewardArray = []
    rewardArray.append(score)
    
    env.close()

meanReward = sum(rewardArray)/len(rewardArray)
print('Mean Reward for 10 Episodes: {}'.format(meanReward))

