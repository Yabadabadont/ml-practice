#https://keras.io/layers/writing-your-own-keras-layers/

# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Layer
from keras.optimizers import Adam
import keras.backend as K

EPISODES = 1000

class Gate:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.in_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.out_size, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

class Mode:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.in_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.out_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

class Agent:
    def __init__(self, state_size, action_size, num_modes):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Set up the gate & mode array
        self.gate = Gate(self.state_size, num_modes)
        self.modes = [Mode(self.state_size, self.action_size) for n in range(num_modes)]

    def act(self, input):
        mixture = self.gate.model.predict(input)
        print(mixture[0])
        result = None
        for i in range(len(self.modes)):
            mode_result = self.modes[i].model.predict(input)
            weight = mixture[0][i]
            if result is None:
                result = mode_result * weight
            else:
                result += mode_result * weight

        # Return the mixture so it can be stored with remember
        return np.argmax(result[0]), mixture

    def remember(self, state, action, reward, next_state, done, mixture):
        self.memory.append((state, action, reward, next_state, done, mixture))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        # Train gate
        for state, action, reward, next_state, done, mixture in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                        np.amax(self.gate.model.predict(next_state)[0])
            target_f = self.gate.model.predict(state)
            target_f[0][action] = target
            self.gate.model.fit(state, target_f, epochs=1, verbose=0, class_weight=mixture[0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

            


        # Train modes
        for i in range(len(self.modes)):
            for state, action, reward, next_state, done, mixture in minibatch:
                target = reward
                if not done:
                    target = reward + self.gamma * \
                        np.amax(self.modes[i].model.predict(next_state)[0])
                target_f = self.modes[i].model.predict(state)
                target_f[0][action] = target
                K.set_value(self.modes[i].model.optimizer.lr, self.learning_rate*mixture[0][i])
                self.modes[i].model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size, 2)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action, mixture = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done, mixture)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, EPISODES, time, 0))  
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")

