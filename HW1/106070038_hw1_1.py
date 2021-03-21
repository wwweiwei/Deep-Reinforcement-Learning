import numpy as np

## 4*4 grid world
## Matrix: (to_state, from_state, direction)
Matrix = np.zeros((16,16,4))

## change gamma
gamma = 0.9
theta = 0.05
## initial delta = 0.001
delta = theta + 0.001
counter = 1

# print("** gamma = " + str(gamma) + " **")
# print("** threshold = " + str(theta) + " **")

## generate moving result
def traverse_state(now_state, dir):
    ## create Map and init value as zero(6*6 because of padding)
    Map = np.zeros((4+2,4+2))

    ## calculate the row and col of now location
    loc_row = int(now_state / 4)
    loc_col = now_state - loc_row * 4

    ## now location to Map location(add 1 because of padding)
    Map_row = loc_row + 1
    Map_col = loc_col + 1

    if dir == 0:
        # print("** dir == 0 **")
        # print("now_state: "+str(now_state))
        Map[Map_row-1, Map_col] = 1
        # print("Map: ")
        # print(Map)
    elif dir == 1:
        Map[Map_row, Map_col-1] = 1
    elif dir == 2:
        Map[Map_row+1, Map_col] = 1
    elif dir == 3:
        Map[Map_row, Map_col+1] = 1
    else:
        print(str(dir) + "is not a valid direction.")

    ## if the agent hits the wall
    if np.max(Map[:,0]) > 0:
        # print("Map[:,0]: "+str(Map[:,0]))
        idx = np.argmax(Map[:,0])
        # print("idx: "+str(idx))
        # print("Map[idx,0]: "+str(Map[idx,0]))
        # print("Before Map[idx,1]: "+str(Map[idx,1]))
        Map[idx,1] += Map[idx,0]
        # print("After Map[idx,1]: "+str(Map[idx,1]))

    if np.max(Map[:,5]) > 0:
        idx = np.argmax(Map[:,4+1])
        Map[idx,4] += Map[idx,4+1]

    if np.max(Map[0,:]) > 0:
        idx = np.argmax(Map[0,:])
        Map[1,idx] += Map[0,idx]

    if np.max(Map[5,:]) > 0:
        idx = np.argmax(Map[4+1,:])
        Map[4,idx] += Map[4+1,idx]

    # print("return Map:")
    # print(Map[1:5,1:5])

    return Map[1:5,1:5]

## traverse all grid world
for row in range(4):
    for col in range(4):
        now_state = row * 4 + col
        ## 0 == move up
        Matrix[:, now_state, 0] = traverse_state(now_state, 0).flatten()
        ## 1 == move left
        Matrix[:, now_state, 1] = traverse_state(now_state, 1).flatten()
        ## 2 == move down
        Matrix[:, now_state, 2] = traverse_state(now_state, 2).flatten()
        ## 3 == move right        
        Matrix[:, now_state, 3] = traverse_state(now_state, 3).flatten()

# print("Matrix[1,:,:]:")
# print(Matrix[1,:,:])

## state 0 and 15 is terminal, set the value as 1
state0 = np.zeros([16,4])
state0[0,:] = 1
# print("Matrix[:,state_0,:]: "+str(state0))
Matrix[:,0,:] = state0

state15 = np.zeros([16,4])
state15[15,:] = 1
# print("Matrix[:,state_15,:]: "+str(state15))
Matrix[:,15,:] = state15

# environment setting
# random policy: pi(a|s)
prob = np.full(4,0.25)
func_value = np.zeros(16)
func_reward = np.full(16,-1)
func_reward[0] = 0
func_reward[15] = 0
num_actions = 4
num_states = 16

# iterate policy evaluation
while delta > theta:
    func_value_now = func_value.copy()
    for state in range(1,15):
        ## p(s',r|s,a)
        prob_next_state = prob * Matrix[:, state, :]
        ## r + gamma * V(s')
        future_reward = func_reward + gamma * func_value_now 
        ## sum( prob_next_state * future_reward)
        func_value[state] = np.sum(np.matmul(np.transpose(prob_next_state), future_reward))
    delta = np.max(np.abs(func_value - func_value_now))

    print("Iter " + str(counter))
    # print("delta = " + str(delta))
    print(func_value.reshape(4,4).round(2))
    print("-" * 30)
    counter += 1

## output file
output = func_value.reshape(4,4).flatten()

if gamma == 0.9:
    f = open("106070038_hw1_1_data_gamma_0.9.txt","w+")
elif gamma == 0.1:
    f = open("106070038_hw1_1_data_gamma_0.1.txt","w+")
else:
    f = open("106070038_hw1_1_data.txt","w+")

for i in range(1,15):
     f.write("%.2f " % output[i])

f.close() 