import numpy as np
import _pickle as cPickle
import random 

class Agent:
    def __init__(self, symbol, state, load_trainer = None):
        """
        symbol (string)
        state (Board)
        load_trainer (string)
        """
        self.symbol = symbol
        self.current_state = state

        self.actions = self.current_state.getAvailablePos()
        self.action_history = []

        if load_trainer is None:
            print("Error in loading agent!")
        else:
            # print("Success in loading agent: "+str(load_trainer))
            self.trainer = self.loadTrainer(load_trainer)

    def getPossibleActions(self):
        """ Get possible actions """
        self.updatePossibleActions()
        return self.actions

    def updatePossibleActions(self):
        """ Update possible actions """
        self.actions = self.current_state.getAvailablePos()

    def getActionHash(self, action):
        """ Get hash key of action """
        action_hash = 100*(action[0]+1) + 10*(action[1]+1) + (action[2]+1)
        return action_hash

    def getActionHashFromState(self, action = None, state = None):
        """ Get hash key of actions in a given state, also returns the hash key of that state """
        if state is None:
            state = self.current_state

        next_state_hash = state.getStateHash()
        next_actions_hash = []
        for a in self.actions:
            next_actions_hash.append(self.getActionHash(a))

        return next_state_hash, next_actions_hash

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
        with open(save_path, "rb") as f:
            dict = cPickle.load(f)
        # print("Q: "+str(dict))
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
        for k in range(self.heights):
            for i in range(self.rows):
                for j in range(self.cols):
                    if inverted:
                        state_hash -= self.state[i,j,k]*factor
                    else:
                        state_hash += self.state[i,j,k]*factor
                    
                    factor = 100*factor
                factor = 10*factor
        return state_hash

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

class Play():
    def __init__(self):
        trained_agent = "hw1_4_data"

        for line in input_file.readlines():
            input_state = line.rstrip("\n").split(" ")
            
            playerX = Board.playerX
            playerO = Board.playerO

            if len(input_state[1:65]) == 64:
                state = input_state[1:65]
                # print("state:"+str(state))
            else:
                print("Error State!")

            # Start game
            self.board = Board(rows=4, cols=4, heights=4, win_threshold=4)
            index = 0
            for i in range(BOARD_ROWS):
                for j in range(BOARD_COLS):
                    for k in range(BOARD_HEIGHTS):
                        if int(state[index]) == 1:
                            self.board.setPosition(i, j, k, 1)
                        elif int(state[index]) == -1:
                            self.board.setPosition(i, j, k, -1)
                        index+=1
            # print("board: "+str(self.board))

            if input_state[0] == '1':
                self.current_player = playerX
            else:
                self.current_player = playerO
            # print("self.current_player: "+str(self.current_player))

            self.agent = Agent(self.current_player, self.board, load_trainer = trained_agent)
            self.agent_symbol = self.agent.symbol

            if self.current_player == self.agent_symbol:
                self.agentMove()

        input_file.close() 
        output_file.close()
        # print("Finish!") 

    def playMove(self, x, y, z):
        # print("Output move - x:"+str(x)+", y:"+str(y)+", z:"+str(z))
        print(str(x)+" "+str(y)+" "+str(z))
        output_file.write("%d " % x)
        output_file.write("%d " % y)
        output_file.write("%d\n" % z)

    def agentMove(self):
        move = self.agent.getBestAction()
        self.playMove(move[2], move[1], move[0])


BOARD_ROWS = 4
BOARD_COLS = 4
BOARD_HEIGHTS = 4


input_file = open("hw1-4_sample_input","r")
output_file = open("hw1-4_sample_output","w")

Play()