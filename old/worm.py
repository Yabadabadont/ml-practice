import gym
import mujoco_py
import torch
import numpy as np

env = gym.make('Swimmer-v2')

action_size = len(env.action_space.high)
observation_size = len(env.observation_space.high)

print("action space: ", env.action_space)
print("high: ", env.action_space.high)
print("low: ", env.action_space.low)

print("observation space: ", env.observation_space)
print("high: ", env.observation_space.high)
print("low: ", env.observation_space.low)

input("Press enter to begin")

def run_episode(env, parameters):
    observation = env.reset()
    #totalreward = 0

    testmax = -10

    
    for _ in range(200):
        env.render()
        
        
        # action calc
        action = [0 for i in env.action_space.high]
        for i in range(len(env.action_space.high)):
            output = np.matmul(parameters[i], observation)/8
            #print("output: ", output)
            action[i] = output
            
            if np.abs(output) > testmax: testmax = np.abs(output)
            
        print(action)

            
        observation, reward, done, info = env.step(action)

        #if done:
        #    print("DONE")
        #    break

    do_render = False
    #print(testmax)
    #input()
    return reward

noise_scaling = 0.01

# best worm params of last session
#3.81196390848482
#[[-0.01992764 -1.14601906 -0.52076286  0.98520917  0.88103359 -0.82097795
#   0.58067774 -0.01042194]
# [-0.26960252 -0.14309877  0.72557209 -0.83686253  1.00713725  0.34495342
#  -0.68305433 -0.35418443]]
#

parameters = np.random.rand(action_size,observation_size)*2-1


#bestparams = None
bestreward = 0
while True:
    new_params = parameters + (np.random.rand(action_size,observation_size) * 2 - 1)*noise_scaling
    reward = np.abs(run_episode(env,parameters))
    
    if reward > bestreward:
        print(reward)
        print(new_params)
        bestreward = reward
        parameters = new_params

