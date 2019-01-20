import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
import gym
import random

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(4, 2) # 4 in, 2 hidden, 1 out
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(x)
        return x

def main():
    env = gym.make('CartPole-v1')
    policy_net = PolicyNet()
    
    net = PolicyNet()

    while True:
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        
        for t in range(200):
            env.render()

            probs = net(state)
            m = Bernoulli(probs)
            action = m.sample()
            action = int(action.item())

            print(action)
            
            state, reward, done, _ = env.step(action)

            state = torch.from_numpy(state).float()
            state = Variable(state)
    
if __name__ == '__main__':
    main()
