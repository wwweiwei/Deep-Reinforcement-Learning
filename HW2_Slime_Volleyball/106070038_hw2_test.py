class RandomAgent(object):
    """Agent that acts randomly."""
    def __init__(self, action_space):
        self.action_space = gym.spaces.Discrete(6)

    def act(self, observation, reward, done):
        return self.action_space.sample()
