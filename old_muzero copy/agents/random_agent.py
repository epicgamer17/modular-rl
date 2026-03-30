import numpy as np


class RandomAgent:
    def __init__(self, name="random"):
        self.name = name

    def predict(self, observation, info, env=None, *args, **kwargs):
        return observation, info

    def select_actions(self, prediction, info, *args, **kwargs):
        if "legal_moves" in info and len(info["legal_moves"]) > 0:
            return np.random.choice(info["legal_moves"])
        return 0
