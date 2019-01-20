import gym
import mujoco_py
import torch
from numpy import array
import numpy as np
import logging
import datetime

np.set_printoptions(threshold=np.nan)


# initialize
model = 'HumanoidStandup-v2'
env = gym.make(model)

# useful variables
action_size = len(env.action_space.high)
observation_size = len(env.observation_space.high)

# Record the best parameters. Useful for long parameters.
def record_best(reward, params):
    timestr = datetime.datetime.now()
    f = open('logs/best_{}'.format(model), 'w')
    f.write("{}|{}".format(repr(params),reward))
    f.close()

def read_best():
    f = open('./logs/best_{}'.format(model), 'r')
    fdata = f.read().split("|")
    return [eval(fdata[0]), float(fdata[1])]
    
# Run one episode and return the reward.
def run_episode(env, parameters, frames=200):
    observation = env.reset()
    for _ in range(frames):
        env.render()
        
        # Calculate the action based on the weights.
        action = [0 for i in range(action_size)]
        for i in range(action_size):
            output = np.matmul(parameters[i], observation)/8
            action[i] = output
             
        observation, reward, done, info = env.step(action)
        #if done:
        #    print("DONE")
        #    break
        
    return reward

def get_random_params():
    return np.random.rand(action_size,observation_size)*2-1

new_params = input("New params? y/n ").lower().startswith("y")


noise_scaling = 0.1

parameters = get_random_params()
bestreward = 0
if not new_params:
    best = read_best()
    parameters = best[0]
    bestreward = best[1]

while True:
    new_params = parameters + get_random_params()*noise_scaling
    reward = np.abs(run_episode(env,parameters,1000))
    
    if reward > bestreward:
        print(reward)
        #print(new_params)

        record_best(reward, new_params)
        
        bestreward = reward
        parameters = new_params

