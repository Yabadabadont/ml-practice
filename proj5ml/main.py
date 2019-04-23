import numpy as np
from tensorflow import keras
import random
import numpy as np
from allyourbase import BaseConvert as base

class Game:
    def __init__(self):
        self.turn = 1
        self.moves = 0
        self.val2char = {1:"X", 2:"O", 0:" "}
        self.state = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        self.isdone = False
        self.action_space = []
        for x in range(3):
            for y in range(3):
                self.action_space.append([x, y])
        
        self.state_space = []
        for i in range(3**9):
            self.state_space.append(self.num2state(i))

    def state2num(self, state):
        flat_state = state.reshape(9,)
        string_state = ""
        for i in flat_state:
            string_state = str(i) + string_state
        base10state = base(3, 10).convert(string_state)
        print(base10state)
        input("base 10 before")
        print(int(base10state))
        input("int base 10 before")
        return int(base10state)
        
    def num2state(self, num):
        trinary = base(10, 3).encode(num)
        state = [0,0,0,0,0,0,0,0,0]
        tristring = str(trinary)
        for n in range(len(tristring)):
            state[len(state)-n-1] = int(tristring[len(tristring)-n-1])
        square_state = [state[0:3], state[3:6], state[6:9]]
        return square_state

    def valid_moves(self):
        possible_moves = []
        for x in range(len(game.state)):
            for y in range(len(game.state[0])):
                if game.state[x][y] == 0:
                    possible_moves.append([x, y])
        return possible_moves
        
    def step(self, x, y):
        self.moves += 1
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
        if self.moves == 9:
            self.isdone = True

        # Return the new state & reward
        return [x, y], self.state.reshape(9,), reward

    def action2num(self,action):
        return action[0]*3 + action[1]

    def num2action(self,num):
        x = num//3
        y = num-x
        return [x, y]
        
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

    def render(self):
        print("")
        for x in self.state:
            x = [self.val2char[i] for i in x]
            out = "{}|{}|{}".format(*x)
            print(out)
    
class RandomAgent:
    def __init__(self):
        pass

    def pick_move(self, game):
        return random.choice(game.valid_moves())
        
class Agent:
    def __init__(self):

        # In a q table, the columns are the actions and the rows are the states.
        # There are maximum 9 actions, and for each square there are 3 possible states, empty, X, or O.
        # So there are 3^9 possible board positions
        self.qtable = np.zeros((9,3**9,1))
        self.epsilon = 1
        self.lr = 0.01
        self.gamma = 0.8
        self.laststate = None
        self.lastaction = None
        
    def update_q(self, action, state, reward):
        self.qtable[self.lastaction, self.laststate] = \
            self.qtable[self.lastaction, self.laststate] + \
            self.lr*(reward + self.gamma*(max(self.qtable[action, state]- \
                                            self.qtable[self.lastaction, self.laststate])))
    
    def pick_move(self, game):
        # explore
        self.laststate = game.state
            
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(game.valid_moves())
        else:
            print(self.qtable.shape)
            print(np.array(game.state).shape)
            print(game.state2num(game.state))
            print("returning this")

            
            num = self.qtable[:, game.state2num(game.state)]
            print(num)
            print("this is num: " + str(int(max(num))))

            num = int(max(num))
            
            y = int(num)//3
            x = int(num)-y

            print("the move is", x, y)
            
            
            return [x,y]
        
    
game = Game()
ragent = RandomAgent()
agent = Agent()

while(1):
    if game.turn == 1:
        action, state, reward = game.step(*agent.pick_move(game))
        agent.update_q(game.action2num(action), game.state2num(state), reward)
    else:
        game.step(*ragent.pick_move(game))
    game.render()
    game.lastaction = game.action2num(action)
    if game.isdone:
        print("cats game")
        break
    if reward != 0:
        print(str(game.turn) + " wins")
        break
