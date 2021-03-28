import numpy as np
import _pickle as cPickle
from tqdm import tqdm
import random

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

    def showAgent(self):
        print("symbol: "+str(self.symbol))
        print("current_state: "+str(self.current_state))
        print("actions: "+str(self.actions))
        print("action_history: "+str(self.action_history))

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
        if action.shape != (3,):
            print("Wrong shape " + str(action))

        if state == None:
            state = self.current_state

        # Read action
        x = action[0]
        y = action[1]
        ###
        z = action[2]

        # Update Q as part of Q-learning in the Trainer class
        if updateQ is True:
            self.trainer.updateQ(state, action)

        # Make move
        ###
        state.setPosition(x, y, z, self.symbol)
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
        ###
        z = last_action[2]
        
        # Set to zero
        ###
        state.setPosition(x, y, z, 0) 

        # Update possible actions
        self.updatePossibleActions()

    def getActionHash(self, action):
        """ Get hash key of action """
        action_hash = 10*(action[0]+1) + (action[1]+1) + (action[2]+1)
        ### action_hash = 10*(action[0]+1) + (action[1]+1)
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
        #self.showAgent()

        self.updatePossibleActions()

        # Get hash key for state and actions
        state_hash, actions_hash = self.getActionHashFromState()

        # Return best move (if all are equally good, then it picks one at random)
        return self.trainer.getBestAction(state_hash, actions_hash, self.actions)
    

    def saveTrainer(self, save_path):
        """ Saves agent to save_path (str) """
        dict = self.trainer.Q
        with open(save_path, "wb") as f:
            cPickle.dump(dict, f)
    
    def loadTrainer(self, save_path):
        """
        Load Q-values from another trainer
        """
        
        with open(save_path, "rb") as f:
            dict = cPickle.load(f)
        # print("Q"+str(dict))
        return Trainer(self, Q=dict)

class Board:
    playerX = 1
    playerO = -1

    def __init__(self, rows = 4, cols = 4, heights = 4, win_threshold = 4):
        """
            rows (int)
            cols (int)
            win_threshold (int) - Do not change
        """ 
        self.state = np.zeros((rows, cols, heights), dtype=np.int8)
        self.rows = rows
        self.cols = cols
        self.heights = heights
        self.win_threshold = win_threshold

    def showBoard(self):
        print("state: ")
        print(str(self.state))
        print("rows= "+str(self.rows))
        print("cols= "+str(self.cols))
        print("heights= "+str(self.heights))
        print("win_threshold= "+str(self.win_threshold))

    def getState(self):
        """ Get state of game """
        return self.state
    
    def getPosition(self, x, y, z):
        """ Get state at position (x,y) """
        return self.state[x,y,zip]

    def setPosition(self, x, y, z, value):
        """  Set state at position (x,y) with value """
        self.state[x,y,z] = value

    def getAvailablePos(self):
        """  Get state positions that have no value (non-zero) """
        return np.argwhere(self.state == 0)

    def getStateHash(self, inverted=False):
        """  Get hash key of state """
        factor = 1
        state_hash = 0
        for k in range(self.heights):
            for i in range(self.rows):
                for j in range(self.cols):
                    if inverted:
                        state_hash -= self.state[i,j,k]*factor
                    else:
                        state_hash += self.state[i,j,k]*factor
                    
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

            # Check heights
            height = np.any((np.all(self.state == sym, axis=2)))

            # Check diagonals
            ##
            diag1 = np.array([self.state[0,0,0], self.state[1,1,0], self.state[2,2,0], self.state[3,3,0]])
            diag1 = np.all(diag1 == sym)

            diag2 = np.array([self.state[0,0,1], self.state[1,1,1], self.state[2,2,1], self.state[3,3,1]])
            diag2 = np.all(diag2 == sym)

            diag3 = np.array([self.state[0,0,2], self.state[1,1,2], self.state[2,2,2], self.state[3,3,2]])
            diag3 = np.all(diag3 == sym)
            
            diag4 = np.array([self.state[0,0,3], self.state[1,1,3], self.state[2,2,3], self.state[3,3,3]])
            diag4 = np.all(diag4 == sym)

            diag5 = np.array([self.state[0,3,0], self.state[1,2,0], self.state[2,1,0], self.state[3,0,0]])
            diag5 = np.all(diag5 == sym)
            
            diag6 = np.array([self.state[0,3,1], self.state[1,2,1], self.state[2,1,1], self.state[3,0,1]])
            diag6 = np.all(diag6 == sym)

            diag7 = np.array([self.state[0,3,2], self.state[1,2,2], self.state[2,1,2], self.state[3,0,2]])
            diag7 = np.all(diag7 == sym)

            diag8 = np.array([self.state[0,3,3], self.state[1,2,3], self.state[2,1,3], self.state[3,0,3]])
            diag8 = np.all(diag8 == sym)

            ##

            
            # Check if state has winner and return winner in that case
            if row or col or height or diag1 or diag2 or diag3 or diag4 or diag5 or diag6 or diag7 or diag8:
                return sym
            
        # No winner found
        return 0 

    def checkGameEnded(self):
        """ Check if game has ended by observing if there any possible moves left """
        return len(self.getAvailablePos()) == 0

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
            z = action[2]

            # Perform action
            self.setPosition(x, y, z, next_player_player)

            # Check if winning move
            if self.checkWinner() != 0:
                winning_move_found = True

            # Revert action
            self.setPosition(x, y, z, 0)

            # If found, then return True
            if winning_move_found is True:
                return True

        return False

    def resetGame(self):
        """ Reset game """
        # self.showBoard()
        self.state = np.zeros((self.rows, self.cols, self.heights), dtype=np.int16)

    def getInvertedState(self):
        """ Return state where player O and X have swapped places """
        return -self.state

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
        return state_hash * action_hash

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
        
        # Find action that given largest Q in given state
        maxQ = 0
        for a_hash, action in zip(list_action_hash, list_actions):
            tmpQ = self.getValueQ(state_hash, a_hash)
            if maxQ < tmpQ:
                maxQ = tmpQ
                best_action = action

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


def simulate(iterations, explore_only = False, save_agent = None):
    """
        iterations (int)
        explore_only (bool) - If true, then only explore.
                              Else,  follow an epsilon-greedy policy that lowers the probability to explore over time.
    """

    # Construct game board
    game = Board()
    
    # Initialize players
    agent1 = Agent(player1_symbol, game)
    agent2 = Agent(player2_symbol, game)
    
    # agent1 = Agent(player1_symbol, game, load_trainer = "hw1_3_data.pkl")

    # Counters for wins of each agent and total number of games
    nbr_wins_agent1 = 0
    nbr_wins_agent2 = 0
    nbr_games = 0

    # Pick current player
    current_player = player1_symbol

    # Epsilon-greedy 
    exploration_probability = 1.0

    # Start iterations
    for i in tqdm(range(iterations)):

        # Check if games has ended, reset if True
        if game.checkGameEnded():
            nbr_games += 1
            game.resetGame()
            agent1.updatePossibleActions()
            agent2.updatePossibleActions()

        # Check who is the current player 
        if current_player == agent1.symbol:
            a = agent1
        else:
            a = agent2

        #Explore
        if explore_only is True or random.random() < exploration_probability:
            a.performRandomAction(updateQ=True)
        # Exploit
        else:
            best_action = a.getBestAction()
            a.performAction(best_action, updateQ=(True))


        # Reduce probability to explore during training
        # Do not remove completely
        exploration_probability_lower_bound = 0.2
        if exploration_probability > exploration_probability_lower_bound:
            exploration_probability -= 1/iterations

        # Check if there is a winner
        winner = game.checkWinner() # Returns 0 if there is no winner
        if winner != 0:

            # Reset game and retrieve 
            nbr_games += 1
            game.resetGame()

            # Add to count for corresponding winner
            if winner == agent1.symbol:
                nbr_wins_agent1 += 1
            else:
                nbr_wins_agent2 += 1
        
        # Swap player
        if current_player == player1_symbol:
            current_player = player2_symbol
        else:
            current_player = player1_symbol

        
    # Print outcome
    print(nbr_wins_agent1, nbr_wins_agent2, nbr_games)    
    print("Win percentage: Agent 1 {:.2%}, Agent 2 {:.2%}.".format(nbr_wins_agent1/nbr_games, nbr_wins_agent2/nbr_games))

    if save_agent is not None:
        if (nbr_wins_agent1/nbr_games) > (nbr_wins_agent2/nbr_games):
            print("Saved trainer of agent 1 to {}".format(save_agent))
            agent1.saveTrainer(save_agent)
        else:
            print("Saved trainer of agent 2 to {}".format(save_agent))
            agent2.saveTrainer(save_agent)


player1_symbol = Board.playerX
player2_symbol = Board.playerO

iterations = 50000 #500000
simulate(iterations, explore_only=False, save_agent="hw1_4_data")
