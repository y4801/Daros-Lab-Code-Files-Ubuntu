import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import concurrent.futures

log_path = os.path.join('Training','Logs')

def HumanoidStandup(bsize):
    rewardArray = []
    env = gym.make('HumanoidStandup-v2')
    env = DummyVecEnv([lambda:env])
    model = PPO('MlpPolicy',env,learning_rate=0.0003,batch_size=bsize,n_epochs=11, verbose=1,tensorboard_log=log_path)
    
    model.learn(total_timesteps=20E6)

    # Saves the model 
    PPO_Path = os.path.join('Training','Saved Models', 'PPOHumanStand20M')
    #model.save(PPO_Path)

    #Evaluate the model and get average reward for 10 episodes:

    #Run the model in the environment and get rewards
    env = gym.make('HumanoidStandup-v2')
    episodes = 100
    for episode in range(1,episodes+1):
        obs = env.reset()
        done = False
        score = 0
        
        while not done:
            #env.render()
            action, _ = model.predict(obs) # By doing this, rather than taking a random action, the model is used to take actions
            obs, reward, done, info = env.step(action)
            score += reward
            rewardArray.append(score)
        #print('Episode:{} Score:{}'.format(episode, score))
    meanReward = sum(rewardArray)/len(rewardArray)
    return meanReward

#Reward = []


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #results = [executor.submit(HumanoidStandup,bsize[i]) for i in range(len(bsize))]
        bsize = [128,256,512]
        results = executor.map(HumanoidStandup.bsize)
        for result in results:
            print(result)
        #for f in concurrent.futures.as_completed(results):
            #Reward.append(f.result())
    
#for j in range(len(Reward)):
    #print(f'{round(Reward[j],2)}')