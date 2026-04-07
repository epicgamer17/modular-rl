import random
from tabnanny import check
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pygame
import copy


class WardrobeEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, render_mode=None):  # Observations are planes.
        # The first plane represents player 1s tokens, the second player 2s and the third encodes the current players turn.
        self.number_of_features = 17
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, 2 + self.number_of_features * 16),
            dtype=np.float64,
        )

        # [desired_formality (0 = any, 1 = casual, 2 = smart, 3 = smart_casual, 4 = formal, 5 = athletic), temperature_outside (hot, warm, cool, cold), item_id, r_primary, g_primary, b_primary, r_secondary, g_secondary, b_secondary, r_tertiary, g_tertiary, b_tertiary, patterend (0 = solid, 1 = striped, 2 = repeated_images (floral tee), other), graphic_location (0 = none, 1 = front, 2 = back, 3 = both), collared_shirt (0 = not_collared, 1 = collared), length (0 = short short, 1 = short, 2 = long), body_part (0 = feet, legs, torso, accessory), fit (0 = normal, 1 = tight, 2 = oversized), material = (0 = cotton, 1 = polyester, 2 = leather, 3 = knitted, 4 = denim, 5 = other), items_in_outfit x 15 * number_of_features]
        print(self.observation_space)

        # We have 9 actions, corresponding to each cell
        self.action_space = spaces.Discrete(3)
        # add to outfit, next item, done

    def _get_obs(self):
        observation = np.zeros((1, 2 + self.number_of_features * 16))
        # get desires
        observation[0, 0] = self._desired_formality
        observation[0, 1] = self._temperature_outside
        # from a json file that makes up my wardrobe
        # get current item we are looking at
        # from input from user at start of "game"
        # get current outfit
        self._current_outfit
        return

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._wardrobe = pd.read_csv("wardrobe.csv")
        self._current_outfit = np.zeros(
            (15, self.number_of_features)
        )  # 15 items, 16 features
        self._current_item = random.randint(len(self._wardrobe))
        print("Current item", self._current_item)
        # set player ones and twos tokens
        self.number_of_items = 0
        self.steps = 0

        # ask some questions about temperature and formality
        self._desired_formality = int(
            input(
                "What formality do you want? (0 = any, 1 = casual, 2 = smart, 3 = smart_casual, 4 = formal, 5 = athletic)"
            )
        )
        self._temperature_outside = int(
            input(
                "What is the temperature outside? (0 = hot, 1 = warm, 2 = cool, 3 = cold)"
            )
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if action == 0:
            self.number_of_items += 1
            self._current_outfit[self.number_of_items] = self._wardrobe.iloc[
                self._current_item
            ]
            self._wardrobe.drop(self._current_item)
            self._current_item = self._wardrobe.iloc[
                random.randint(len(self._wardrobe))
            ]  # maybe should go to next item so there is some sort of consistency for the agent
        if action == 1:
            self._current_item = self._wardrobe.iloc[
                random.randint(len(self._wardrobe))
            ]  # maybe should go to next item so there is some sort of consistency for the agent
            pass
        if action == 2:
            terminated = True
        self.steps += 1
        truncated = False
        if self.steps > 100:
            truncated = True
        if self.number_of_items == 15:
            truncated = True

        observation = self._get_obs()
        info = self._get_info()

        reward = self.get_reward_from_user()

        return observation, reward, terminated, truncated, info

    def get_reward_from_user(self):
        reward = 0
        valid_outfit = input("Is this outfit valid?")
        if valid_outfit == "n":
            return -10
        else:
            reward += 0

        rating = int(
            input(
                "What did you think of the outfit? (0 = hate, 1 = dislike, 2 = meh, 3 = like, 4 = love)"
            )
        )
        reward += rating
        confidence = int(
            input(
                "How did you feel in the outfit? (0 = insecure, 1 = uncomfortable, 2 = comfortable, 3 = confident)"
            )
        )
        reward += confidence - 2

        formality = int(
            input(
                "How did you feel about the formality? (0 = too casual, 1 = too smart, 2 = just right)"
            )
        )
        reward += 0 if formality == 2 else -1

        temperature = int(
            input(
                "How did you feel about the temperature? (0 = too hot, 1 = too cold, 2 = just right)"
            )
        )
        reward += 0 if temperature == 2 else -1

        fit = int(
            input(
                "How did you feel about the fit? (0 = too tight, 1 = too loose, 2 = just right)"
            )
        )
        reward += 0 if fit == 2 else -1

        material = int(
            input(
                "How did you feel about the material? (0 = too heavy, 1 = too light, 2 = just right)"
            )
        )
        reward += 0 if material == 2 else -1
        return reward
