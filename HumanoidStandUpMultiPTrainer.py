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
    LR = 2.55673e-05
    epoch = 20
    model = PPO('MlpPolicy',env,learning_rate=LR,batch_size=bsize,n_epochs=epoch,n_steps=512,gamma=0.99,ent_coef=3.62109e-06,clip_range=0.3,max_grad_norm=0.7,vf_coef=0.430793,verbose=1,gae_lambda=0.9,tensorboard_log=log_path)
    timestep = 1e7
    model.learn(total_timesteps=timestep)
    ts = timestep/1e6 # In the millions

    # Saves the model 
    PPO_Path = os.path.join('Training','Saved Models', 'PPOHumanStand{}{}{}{}'.format(ts,LR,bsize,epoch))
    model.save(PPO_Path)

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

Reward = []
#bsize = [128]

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        #results = [executor.submit(HumanoidStandup,bsize[i]) for i in range(len(bsize))]
        bsize = [32]
        results = executor.map(HumanoidStandup,bsize)
        for result in results:
            print(result)
            
    
#for j in range(len(Reward)):
    #print(f'{round(Reward[j],2)}')


