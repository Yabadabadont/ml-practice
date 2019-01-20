import gym
import mujoco_py
import torch
from numpy import array
import numpy as np
import logging
import datetime
import sys
import math
import random

np.set_printoptions(threshold=np.nan)

do_render = True
if len(sys.argv)==3:
    do_render = False

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
    try:
        f = open('./logs/best_{}'.format(model), 'r')
        fdata = f.read().split("|")
        return [eval(fdata[0]), float(fdata[1])]
    except:
        return [0,0]
    
# Run one episode and return the reward.
def run_episode(env, parameters, frames=100):
    total_reward = 0
    observation = env.reset()
    for _ in range(frames):
        if do_render: env.render()
        
        # Calculate the action based on the weights.
        action = [0 for i in range(action_size)]
        for i in range(action_size):
            output = np.matmul(parameters[i], observation)/8
            action[i] = output
             
        observation, reward, done, info = env.step(action)

        if reward > total_reward: total_reward = reward

        #if done:
        
        #    print("DONE")
        #    break
        
    return total_reward

def get_random_params():
    return np.random.rand(action_size,observation_size)*2-1

new_params = input("New params? y/n ").lower().startswith("y")


base_scaling = 0.1
noise_scaling = 0

total_record = read_best()[1]

parameters = get_random_params()
bestreward = 0
if not new_params:
    best = read_best()
    parameters = best[0]
    bestreward = best[1]

frame_count = 200
if len(sys.argv) >= 2:
    frame_count = int(sys.argv[1])

ep = 0
scaling = 1
while True:
    ep += 1

    #curious = random.random() < noise_scaling

    #if curious:
    #    print("hmmm")
    
    new_params = [parameters + get_random_params()*noise_scaling,get_random_params()][0]#curious]

    series_reward = 0
    n = 1
    for x in range(n):
        reward = np.abs(run_episode(env,parameters,frame_count))
        series_reward += reward
    series_reward /= n
    reward = series_reward
        
    print("{}: {}, scaling {}".format(ep, reward, noise_scaling))
    
    if reward > bestreward:
        #scaling = 1
        print('new best: ',reward)
        if reward > total_record:
            record_best(reward, new_params)
        bestreward = reward
        parameters = new_params
    else:
        #scaling += 1
        noise_scaling = 0.1#1/10*math.log(scaling)

        #if noise_scaling > 0.5:
        #    scaling = 0
