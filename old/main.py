import gym
import mujoco_py
import torch as pt
from numpy import array
import numpy as np
import logging
import datetime
import sys
import math
import random

# Setup
np.set_printoptions(threshold=np.nan)
do_render = True
frames = 200
if len(sys.argv)==3:
    print("this happened")
    do_render = sys.argv[2].startswith("r")
    frames = int(sys.argv[1])
else:
    print("usage: python *.py frames (r)ender")
    sys.exit(1)
pt.set_default_tensor_type('torch.cuda.FloatTensor')
cuda = pt.device('cuda')

# Initialize
model = 'Ant-v2'
env = gym.make(model)
action_size = len(env.action_space.high)
observation_size = len(env.observation_space.high)

# Record the best parameters. Useful for long parameters.
def record_best(reward, params):
    timestr = datetime.datetime.now()
    f = open('logs/best_{}'.format(model), 'w')
    f.write("{}|{}".format(repr(params),reward))
    f.close()

def read_best():
    try:
        f = open('./logs/best_{}'.format(model), 'r')
        fdata = f.read().split("|")
        return [eval(fdata[0]), float(fdata[1])]
    except:
        return [0,0]

# Normalize a float between -1 and 1
def sigmoid(gamma):
  if gamma < 0:
    return 1 - 1/(1 + math.exp(gamma))*2-1
  else:
    return 1/(1 + math.exp(-gamma))*2-1
    
# Run one episode and return the reward.
def run_episode(env, parameters, frames):
    observation = env.reset()

    sum_reward = 0
    max_reward = 0
    
    for z in range(frames):
        if do_render: env.render()
        
        # Calculate action
        obs = pt.tensor([observation]).cuda()
        pars = pt.transpose(pt.tensor(parameters).cuda(), 0, 1)

        action = [sigmoid(x) for x in pt.mm(obs, pars).cuda()[0]]    

        # Perform action
        observation, reward, done, info = env.step(action)

        #print(reward)
        #input()
        
        # Update values
        if reward > max_reward: max_reward = reward
        sum_reward += reward

        #if done:
        #    print("DONE")
        #    break


    # Return distance
    return sum_reward
        
    # Return distance/time == speed
    #return abs(sum_reward)/frames



def get_random_params():
    return np.random.rand(action_size,observation_size)*2-1


new_params = input("New params? ").lower().startswith("y")


noise_scaling = 0.01

total_record = read_best()[1]

parameters = get_random_params()
bestreward = 0
if not new_params:
    best = read_best()
    parameters = best[0]
    bestreward = best[1]
    
ep = 0
while True:
    ep += 1
    
    new_params = parameters + get_random_params()*noise_scaling

    series_reward = 0
    n = 1
    for x in range(n):
        reward = np.abs(run_episode(env,parameters,frames))
        series_reward += reward
    series_reward /= n
    reward = series_reward
        
    #print("{}: {}, scaling {}".format(ep, reward, noise_scaling))

    if ep%100==0:
        print(ep)
        noise_scaling = 0.1
    
    if reward > bestreward:
        noise_scaling = 0.1
        print('new best: ',reward)
        if reward > total_record:
            record_best(reward, new_params)
        bestreward = reward
        parameters = new_params
    else:
        noise_scaling += 0.1
        pass
