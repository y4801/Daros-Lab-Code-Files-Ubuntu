import gym
env = gym.make('CartPole-v0')#Choose the environment we want top work with

# Below line prints out the cart position, velocity, pole angle and pole tip velocity.
print("Observation space:", env.observation_space)
#Below line prints out the action taken on the cart, whether it moves left or right
print("Action space:", env.action_space)
class Agent(): #We are creating an agent class that uses info from the environment to make custom actions
    def __init__(self,env):
        self.action_size = env.action_space.n
        print("Action size:", self.action_size)
    def get_action(self,state):#method that chooses action from the available actions
        
        pole_angle = state[2]
        cart_position = state[0]

        action = 0 if pole_angle < 0 & cart_position > 0 else 1
        return action

agent = Agent(env)
state = env.reset() #Resets the environment, also returns the initial state

for _ in range(1000):#Basically acting over a set number of time steps i.e. 200 in this case
 
    action = agent.get_action(state)#Uses our action choosing method
    state, reward, done, info = env.step(action)#apply the action to the env
    env.render()#Render out what happens with the action



