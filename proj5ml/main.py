import numpy as np
from tensorflow import keras

class Game:
    def __init__(self):
        self.turn = 1
        self.state = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        
    def step(self, x, y):
        # Flip the turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # Update the game state
        self.state[x][y] = self.turn

        # Determine the reward
        reward = 0
        winner = self.check()
        if winner:
            if winner == self.turn:
                reward = 1
            else:
                reward = -1

        # Return the new state & reward
        return self.state.reshape(9,), reward
            
    def check(self):
        # Check for win states on the game board, returns winner of 0
        slices = []
        for i in range(3): # Rows & cols
            slices.append(game.state[:, i])
            slices.append(game.state[i, :])
        slices.append(game.state.diagonal()) # Diagonal
        slices.append(np.fliplr(game.state).diagonal()) # Reverse diagonal
        for s in slices:
            if np.unique(s).size == 1:
                return s[0]
        return 0

class Agent:
    def __init__(self):
        pass
    
game = Game()

moves = [[0,0], [0,1], [0,2], [1,0], [2, 2], [2,0], [1,1]]
for move in moves:
    state, reward = game.step(*move)
    if reward == 1:
        print(str(game.turn) + " wins")

