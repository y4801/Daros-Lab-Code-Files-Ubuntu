import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
# Should run training at a fixed learning rate for 4 different epochs
# if this works, code up a script that runs this for a different learning rates and different 
# Epochs. So a nested loop

Epoch = [10,15,20,30]
meanReward = []
for i in range(len(Epoch)):
    numEpoch = Epoch[i]
    env = gym.make('HumanoidStandup-v2') 
    # to see if it's more efficient.
    #env = make_vec_env('HumanoidStandup-v2', n_envs=24)
    log_path = os.path.join('Training','Logs')

    env = DummyVecEnv([lambda:env])
    model = PPO('MlpPolicy',env,learning_rate=0.0003,n_epochs=numEpoch, verbose=1,tensorboard_log=log_path)
    #Train for 10 mil timesteps divided by 4 because we have 4 envs running simultaneously?
    model.learn(total_timesteps=20E6)
    j = 15
    # Saves the model 
    PPO_Path = os.path.join('Training','Saved Models', 'PPOHumanStand10M{}'.format(j))
    model.save(PPO_Path)
    j = j+1

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
    meanReward.append(sum(rewardArray)/len(rewardArray))
    MR = meanReward[i]
for j in range(len(meanReward)):
    print('{}'.format(meanReward[j]))