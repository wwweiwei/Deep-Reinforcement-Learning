import numpy as np
import os, time
import matplotlib.pyplot as plt

def TransMat(now_state, action):
    max_row = 4
    max_col = 4
    now_row = int(now_state/max_col)
    now_col = (now_state%max_col)

    if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:
        print("index error")
        return

    col = now_col
    row = now_row
    if action == 0 and now_row > 0:    # up
        row -= 1
    elif action == 1 and now_col > 0:   # left
        col -= 1
    elif action == 2 and (max_row-1) > now_row:   # down
        row += 1
    elif action == 3 and (max_col-1) > now_col:   # right
        col += 1
    next_state = row * max_col + col
    return next_state

def qlearn(action_value, reward, steps, gamma, alpha, epsilon):
    # initialize setting
    record = []
    state = 0
    for step in range(steps):
        # get next information
        action = GetAction(action_value, epsilon, state)
        next_state = TransMat(state, action)
        record.append([state, action, reward[next_state], next_state])
        # update action value
        action_value[state, action] = ValueUpdate('qlearning', action_value, record[step], alpha, gamma)
        # update for next state
        state = next_state
        if state == 5 or state == 9 or state == 12:
            break
    # episode reward
    record = np.array(record,dtype=object)
    map(record[:,0])
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward
    
def sarsa(action_value, reward, steps, gamma, alpha, epsilon):
    # initialize setting
    record = []
    state = 0
    action = GetAction(action_value, epsilon, state)
    for step in range(steps):
        # get next information
        next_state = TransMat(state, action)
        next_action = GetAction(action_value, epsilon, next_state)
        record.append([state, action, reward[next_state], next_state, next_action])
        # update action value
        action_value[state, action] = ValueUpdate('sarsa', action_value, record[step], alpha, gamma)
        # update for next state
        state = next_state
        action = next_action
        if state == 5 or state == 9 or state == 12:
            break
    # episode reward
    record = np.array(record,dtype=object)
    map(record[:,0])
    epi_reward = np.sum(record[:,2])
    return action_value, epi_reward

def GetAction(action_value, epsilon, next_state):
    if np.random.rand(1) >= epsilon:
        policy = np.argmax(action_value, axis = 1)
        action = policy[next_state]
    else:
        action = np.random.randint(0,4,1)
    return action

def ValueUpdate(method, action_value, record, alpha, gamma):
    state = record[0]
    action = record[1]
    reward = record[2]
    next_state = record[3]
    now_value = action_value[state, action]

    # Q(S, A) ← Q(S, A) + α [R + γ max Q(S’, A) - Q(S, A)]
    if method == 'qlearning':
        update_value = alpha * (reward + gamma * np.max(action_value[next_state,:]) - now_value)
    # Q(S, A) ← Q(S, A) + α [R + γ Q(S’, A’) - Q(S, A)]
    elif method == 'sarsa':
        next_action = record[4]
        update_value = alpha * (reward + gamma * action_value[next_state, next_action] - now_value)
    else:
        print("Error method!")
        return

    value = now_value + update_value
    return value

def show():
    cum_step_q = np.load('q-episode-reward.npy')
    cum_step_SARSA = np.load('s-episode-reward.npy')

    plt.plot(cum_step_q, label = "q_learning")
    plt.plot(cum_step_SARSA, label = "SARSA")

    plt.title('SARSA & Q-Learning Learning Curves')
    plt.xlabel('Episodes')
    plt.ylabel('Reward per Episode')
    plt.ylim(-600, 0)
    plt.legend(loc='lower right', ncol=2, borderaxespad=0.)

    plt.show()

def map(state):
    draw_state = np.zeros(16)
    for i in range(len(state)):
        for j in range(16):
            if state[i] == j:
                draw_state[j] = -1
    print("--------------------------------------------------")
    for i in range(0, 4):
            out = '| '
            for j in range(0, 4):
                if draw_state[i*4+j] == -1:
                    token = '*'
                if draw_state[i*4+j] == 0:
                    token = '0'
                if i == 0 and j == 0:
                    token = 'S'
                if i == 3 and j == 0:
                    token = 'G'
                if (i == 1 or i == 2) and j == 1:
                    token = 'X'
                out += token + ' | '
            print(out)


def main(episodes, method):

    ActionValue = np.zeros([16, 4])
    # design map
    Reward = np.full(16, -1)
    Reward[5] = -500
    Reward[9] = -200

    print("Reward")
    print(Reward[0:4])
    print(Reward[4:8])
    print(Reward[8:12])
    print(Reward[12:16])
    
    episodeReward = []

    Gamma = 0.99
    Epsilon = 0.1
    Steps = 1000
    Alpha = 0.05

    # Execute
    if method == 'qlearning':
        for episode in range(episodes):
            print("Q-learning ep "+str(episode))
            ActionValue, Epi_Reward = qlearn(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)
            episodeReward.append(Epi_Reward)
        #print("QLearn ActionValue:")
        #print(ActionValue)
    elif method == 'sarsa':
        for episode in range(episodes):
            print("Sarsa ep "+str(episode))
            ActionValue, Epi_Reward = sarsa(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)
            episodeReward.append(Epi_Reward)
        #print("Sarsa ActionValue:")
        #print(ActionValue)
    else:
        print("Error method!")
        return
    episodeReward = np.array(episodeReward)
    return episodeReward

if __name__ == '__main__':
    q_reward = main(200, 'qlearning')
    s_reward = main(200, 'sarsa')

    np.save('q-episode-reward.npy', q_reward)
    np.save('s-episode-reward.npy', s_reward)
    # show plot
    show()