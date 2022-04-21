import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

#env = gym.make('HumanoidStandup-v2', n_envs=4)
env = make_vec_env('HumanoidStandup-v2', n_envs=4)
log_path = os.path.join('Training','Logs')

#env = DummyVecEnv([lambda:env])
model = PPO('MlpPolicy',env,learning_rate=0.003,batch_size=64,n_epochs=10, verbose=1,tensorboard_log=log_path)
#Train for 10 mil timesteps divided by 4 because we have 4 envs running simultaneously?
model.learn(total_timesteps=2.5E6)

# Saves the model 
PPO_Path = os.path.join('Training','Saved Models', 'PPOHumanStand10M9')
model.save(PPO_Path)

#Evaluate the model and get average reward for 10 episodes:

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
