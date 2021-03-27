
import tkinter as tk
import random 
import time
import numpy as np
import _pickle as cPickle
from tqdm import tqdm

class Agent:
    """  Class that defines agent and its possible actions """

    def __init__(self, symbol, state, load_trainer = None):
        """
        symbol (string)
        state (Board)
        load_trainer (string) - Path to saved trainer
        """

        self.symbol = symbol
        self.current_state = state

        self.actions = self.current_state.getAvailablePos()
        self.action_history = []

        if load_trainer is None:
            self.trainer = Trainer(self)
        else:
            print("load_trainer: "+str(load_trainer))
            self.trainer = self.loadTrainer(load_trainer)

    def getPossibleActions(self):
        """ Get possible actions """
        self.updatePossibleActions()
        return self.actions

    def updatePossibleActions(self):
        """ Update possible actions """
        self.actions = self.current_state.getAvailablePos()
    
    def performAction(self, action, state = None, updateQ = False):
        """ 
            Make move from agent, updates the state and possible actions.
            Also updates Q at the same time.                        
        """
        if action.shape != (2,):
            print("Wrong shape " + str(action))

        if state == None:
            state = self.current_state

        # Read action
        x = action[0]
        y = action[1]
        # print("x"+str(x))
        # print("y"+str(y))

        # Update Q as part of Q-learning in the Trainer class
        if updateQ is True:
            self.trainer.updateQ(state, action)

        # Make move
        state.setPosition(x, y, self.symbol)
        self.action_history.append(action)

        # Update possible actions
        self.updatePossibleActions()
    
    def performRandomAction(self, updateQ=True):
        """ Perform random actions, important for exploration of state-pairs """
        
        self.updatePossibleActions()
        random_idx = np.random.choice(self.actions.shape[0])
        action = self.actions[random_idx]

        self.performAction(action, updateQ=updateQ)

        return action

    def revertLastAction(self, state = None):
        """ Make move from agent, updates the state and possible actions  """

        if state == None:
            state = self.current_state

        # Get last action
        last_action = self.action_history.pop()
        x = last_action[0]
        y = last_action[1]
        
        # Set to zero
        state.setPosition(x, y, 0) 

        # Update possible actions
        self.updatePossibleActions()

    def getActionHash(self, action):
        """ Get hash key of action """
        action_hash = 10*(action[0]+1) + (action[1]+1)
        return action_hash

    def getActionHashFromState(self, action = None, state = None):
        """ Get hash key of actions in a given state, also returns the hash key of that state """

        if state is None:
            state = self.current_state

        if not action is None:
            self.performAction(action, state=state)

        next_state_hash = state.getStateHash()
        next_actions_hash = []
        for a in self.actions:
            next_actions_hash.append(self.getActionHash(a))

        if not action is None:
            self.revertLastAction(state=state)

        return next_state_hash, next_actions_hash

    def rewardFunction(self, state, action):
        """ Returns positive value actions turns into win, else zero """

        # Perform action
        self.performAction(action, state=state)

        # Check winner
        winner = state.checkWinner()
        missed_blocking_move = state.checkWinPossible(self.symbol)
        if winner == self.symbol:
                reward = 1
        elif missed_blocking_move is True:
                reward = -1
        else:
            reward = 0
        # Revert action
        self.revertLastAction(state=state)

        return reward

    def assignState(self, state):
        """ Assign a state (Board) to the agent"""
        self.current_state = state
        self.updatePossibleActions()

    def getBestAction(self):
        """ Get best move from the Trainer that has the largest expected reward """

        self.updatePossibleActions()

        # Get hash key for state and actions
        state_hash, actions_hash = self.getActionHashFromState()

        # Return best move (if all are equally good, then it picks one at random)
        return self.trainer.getBestAction(state_hash, actions_hash, self.actions)
    
    
    def loadTrainer(self, save_path):
        """
        Load Q-values from another trainer
        """
        
        with open(save_path, "rb") as f:
            dict = cPickle.load(f)
        # print("Q"+str(dict))
        return Trainer(self, Q=dict)

class Board:
    """ Class that represents the game board of Tic Tac Toe """

    playerX = 1
    playerO = -1

    def __init__(self, rows = 3, cols = 3, win_threshold = 3):
        """
            rows (int)
            cols (int)
            win_threshold (int) - Do not change
        """ 
        self.state = np.zeros((rows, cols), dtype=np.int8)
        self.rows = rows
        self.cols = cols
        self.win_threshold = win_threshold

    def getState(self):
        """ Get state of game """
        return self.state
    
    def getPosition(self, x, y):
        """ Get state at position (x,y) """
        return self.state[x,y]

    def setPosition(self, x, y, value):
        """  Set state at position (x,y) with value """
        self.state[x,y] = value

    def getAvailablePos(self):
        """  Get state positions that have no value (non-zero) """
        return np.argwhere(self.state == 0)

    def getStateHash(self, inverted=False):
        """  Get hash key of state """
        factor = 1
        state_hash = 0
        for i in range(self.rows):
            for j in range(self.cols):
                
                if inverted:
                    state_hash -= self.state[i,j]*factor
                else:
                    state_hash += self.state[i,j]*factor
                
                factor = 10*factor
        return state_hash

    def checkWinner(self):
        """  Get winner, if one exists """
        """ TODO: Not general case, only works for 3x3 board """

        symbols = np.unique(self.state)
        symbols = list(symbols[np.nonzero(symbols)])

        for sym in symbols:

            # Check rows
            row= np.any((np.all(self.state == sym, axis=1)))

            # Check columns
            col = np.any((np.all(self.state == sym, axis=0)))

            # Check diagonals
            diag1 = np.array([self.state[0,0], self.state[1,1], self.state[2,2]])
            diag1 = np.all(diag1 == sym)
            
            diag2 = np.array([self.state[0,2], self.state[1,1], self.state[2,0]])
            diag2 = np.all(diag2 == sym)

            # Check if state has winner and return winner in that case
            if row or col or diag1 or diag2:
                return sym
            
        # No winner found
        return 0 

    def checkWinPossible(self, last_player_value):
        """ 
            Test whether there is a winning move available for the next player.
            Return True if it is available.
            last_player (int)
        """

        # Next player is the negative of last_player_value
        next_player_player = - last_player_value

        winning_move_found = False
        for action in self.getAvailablePos():
            x = action[0]
            y = action[1]

            # Perform action
            self.setPosition(x, y, next_player_player)

            # Check if winning move
            if self.checkWinner() != 0:
                winning_move_found = True

            # Revert action
            self.setPosition(x, y, 0)

            # If found, then return True
            if winning_move_found is True:
                return True

        return False

    def __str__(self):
        return str(self.state)

class Trainer:

    def __init__(self, agent, learning_parameter = 0.1, discount_factor = 0.9, Q = {}):
        """
            agent (Agent)
            learning_parameter (float)
            discount_factor (float)
            Q (dict)
        """

        self.agent = agent
        self.learning_parameter = learning_parameter
        self.discount_factor = discount_factor
        self.Q = Q

    def getStatePairKey(state_hash, action_hash):
        """ Returns state-pair hash key, requires separate state and action hash keys first """
        return state_hash*action_hash

    def getValueQ(self, state_hash, action_hash):
        """ Get expected reward given an action in a given state,
            returns 0 if the state-action pair has not been seen before.
            Input is state and action hash key                          """

        state_action_key = Trainer.getStatePairKey(state_hash, action_hash)
        if state_action_key in self.Q:
            return self.Q.get(state_action_key)
        else:
            self.Q[state_action_key] = 0
            return 0

    def setValueQ(self, state_hash, action_hash, value):
        """ Set value in Q """
        state_action_key = Trainer.getStatePairKey(state_hash, action_hash)

        self.Q[state_action_key] = value

    def getMaxQ(self, state_hash, list_action_hash):
        """ Returns the maximum Q value given a state and list of actions (input is hash keys) """
        maxQ = 0
        for a_hash in list_action_hash:
            tmpQ = self.getValueQ(state_hash, a_hash) 
            if maxQ < tmpQ:
                maxQ = tmpQ
        return maxQ

    def getBestAction(self, state_hash, list_action_hash, list_actions):
        """ Get best action given a set of possible actions in a given state """

        # Pick a random action at first
        random_idx = np.random.choice(list_actions.shape[0])
        best_action = list_actions[random_idx]
        
        #print("best_action:"+str(best_action))

        # Find action that given largest Q in given state
        maxQ = 0
        for a_hash, action in zip(list_action_hash, list_actions):
            tmpQ = self.getValueQ(state_hash, a_hash)
            if maxQ < tmpQ:
                maxQ = tmpQ
                best_action = action

        #print("maxQ:"+str(maxQ))
        return best_action
        

    def updateQ(self, state, action):
        """ Implements Q-learning iterative algorithm """

        state_hash = state.getStateHash()
        action_hash = self.agent.getActionHash(action)

        # Get current Q Value
        currentQ = self.getValueQ(state_hash, action_hash)

        # Find max Q value given the possible set of actions in the next state
        next_state, next_actions = self.agent.getActionHashFromState(action=action, state=state)
        max_nextQ = self.getMaxQ(next_state, next_actions) 
        
        # Update new Q
        newQ = currentQ + self.learning_parameter * (self.agent.rewardFunction(state, action) + self.discount_factor * max_nextQ - currentQ)

        self.setValueQ(state_hash, action_hash, newQ)

## testing
class Play():
    def __init__(self):
        trained_agent = "hw1_3_data.pkl"

        for line in input_file.readlines():
            input_state = line.rstrip("\n").split(" ")
            
            playerX = Board.playerX
            playerO = Board.playerO

            if len(input_state[1:10]) == 9:
                state = input_state[1:10]
                print("state:"+str(state))
            else:
                print("Error State!")

            # Start game
            self.board = Board(rows=3, cols=3, win_threshold=3)
            index = 0
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    if int(state[index]) == 1:
                        self.board.setPosition(i, j, 1)
                    elif int(state[index]) == -1:
                        self.board.setPosition(i, j, -1)
                    index+=1
            print("board: "+str(self.board))

            if input_state[0] == '1':
                self.current_player = playerX
            else:
                self.current_player = playerO
            print("self.current_player: "+str(self.current_player))

            print("Preparing agent")
            self.agent = Agent(self.current_player, self.board, load_trainer = trained_agent)
            self.agent_symbol = self.agent.symbol

            if self.current_player == self.agent_symbol:
                self.agentMove()

        input_file.close() 
        output_file.close()
        print("Finish") 

    def playMove(self, x, y):
        print("x:"+str(x)+", y:"+str(y))
        output_file.write("%d " % x)
        output_file.write("%d\n" % y)

    def agentMove(self):
        move = self.agent.getBestAction()
        time.sleep(random.random()*1 + 0.5)
        self.playMove(move[0], move[1])


BOARD_ROWS = 3
BOARD_COLS = 3

input_file = open("hw1-3_sample_input","r")
output_file = open("hw1-3_output","w")

Play()