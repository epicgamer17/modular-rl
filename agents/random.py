import numpy as np


class RandomAgent:
    def __init__(self, name="random", action_space=None):
        self.name = name
        self.action_space = action_space

    def predict(self, observation, info, env=None, *args, **kwargs):
        return observation, info

    def select_actions(self, prediction, info, *args, **kwargs):
        if self.action_space is not None:
            return self.action_space.sample()
        return np.random.choice(info["legal_moves"])
