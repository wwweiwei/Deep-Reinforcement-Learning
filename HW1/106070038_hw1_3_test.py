import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3

class Player:
    def __init__(self):
        self.states_value = {}
    
    # boardHash:[1. 0. 0. 0. 0. 0. 0. 0. 0.]
    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS*BOARD_ROWS))
        return boardHash
    
    def chooseAction(self, positions, current_board, symbol):
        value_max = -999
        for p in positions:
            next_board = current_board.copy()
            next_board[p] = symbol
            next_boardHash = self.getHash(next_board)                
            if self.states_value.get(next_boardHash) is None:
                value = 0
            else:
                self.states_value.get(next_boardHash)
            #print("value:", value)
            if value >= value_max:
                value_max = value
                action = p
        # print("my action:"+str(action))
        return action
    
    def loadPolicy(self, file):
        fr = open(file,'rb')
        self.states_value = pickle.load(fr)
        fr.close()

input_file = open("hw1-3_sample_input","r")
output_file = open("hw1-3_output","w")

for line in input_file.readlines():
    input_state = line.rstrip("\n").split(" ")
    
    symbol = 0 #[1,-1]
    positions = []
    board = np.zeros((BOARD_ROWS, BOARD_COLS))

    if input_state[0] == '1' or input_state[0] == '-1':
        symbol = input_state[0]
        print("symbol:"+str(symbol))   
    else:
        print("Error Player!")
    
    if len(input_state[1:10]) == 9:
        state = input_state[1:10]
        print("state:"+str(state))
    else:
        print("Error State!")

    index = 0
    for i in range(BOARD_ROWS):
        for j in range(BOARD_COLS):
            board[i][j] = state[index]
            if int(state[index]) == 0:
                positions.append((i, j))
            index+=1
    # print("board:"+str(board))
    # print("positions:"+str(positions))

    p1 = Player()
    p1.loadPolicy("policy_p1")

    # type: list, ndarray, int
    action = p1.chooseAction(positions, board, int(symbol))
    # print("action:"+str(action))

    output_file.write("%d " % action[0])
    output_file.write("%d\n" % action[1])


input_file.close() 
output_file.close()
# print("Finish") 
