import numpy as np
import gymnasium as gym

class AECSequentialWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        first_agent = self.env.possible_agents[0]
        self.action_space = self.env.action_space(first_agent)
        self.observation_space = self.env.observation_space(first_agent)
        self.possible_agents = self.env.possible_agents

    def _get_player_idx(self, agent_str):
        try: return self.possible_agents.index(agent_str)
        except ValueError: return 0

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        obs, reward, term, trunc, info = self.env.last()
        if info is None: info = {}
        info["player"] = self._get_player_idx(self.env.agent_selection)
        info["all_player_rewards"] = dict(self.env.rewards)
        return obs, info

    def step(self, action):
        acting_player_str = self.env.agent_selection
        acting_player_idx = self._get_player_idx(acting_player_str)
        self.env.step(action)
        obs, reward, term, trunc, info = self.env.last()
        done = term or trunc
        step_reward = float(self.env.rewards.get(acting_player_str, 0.0))
        if info is None: info = {}
        info["player"] = self._get_player_idx(self.env.agent_selection)
        info["acting_player"] = acting_player_idx
        info["all_player_rewards"] = dict(self.env.rewards)
        info["was_terminal"] = done
        if done:
            info["terminal_observation"] = obs.copy() if hasattr(obs, "copy") else obs
            info["terminal_legal_moves"] = info.get("legal_moves", [])
            info["terminal_player_id"] = info["player"]
            info["terminal_all_player_rewards"] = dict(self.env.rewards)
            self.env.reset()
            _, _, _, _, new_info = self.env.last()
            if new_info is not None:
                if "legal_moves" in new_info: info["legal_moves"] = new_info["legal_moves"]
                elif "action_mask" in new_info: info["legal_moves"] = [i for i, v in enumerate(new_info["action_mask"]) if v == 1]
            info["player"] = self._get_player_idx(self.env.agent_selection)
            info["all_player_rewards"] = dict(self.env.rewards)
        return obs, step_reward, done, False, info
