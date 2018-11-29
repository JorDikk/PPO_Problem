import numpy as np

class multiProcess():
    def __init__(self, envs):

        self.envs = envs

        self.state = []
        self.reward = []
        self.done = []

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

# Step function, returns state, reward, done, for every environment
    def step(self, actions):
        self.state = []
        self.reward = []
        self.done = []

        for i, env in enumerate(self.envs):
            state, reward, done, info = env.step(actions[i])
            self.state.append(state)
            self.reward.append(reward)
            self.done.append(done)
        return np.array(self.state), np.array(self.reward), np.array(self.done), {}


# Reset function, resets all environments and returns the new initial state of every environment
    def reset(self):
        for env in self.envs:
            state = env.reset()
            self.state.append(state)
        return np.array(self.state)


