from __future__ import annotations

from typing import Any
import gymnasium.spaces
import numpy as np
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from gymnasium.spaces import Box, Discrete
from supersuit.utils.frame_stack import stack_init, stack_obs, stack_obs_space
from collections import deque
from utils.utils import action_mask_to_legal_moves
from pettingzoo.utils.wrappers.base import BaseWrapper
import gymnasium as gym


def action_mask_to_info(state, info, current_player):
    info["legal_moves"] = action_mask_to_legal_moves(state["action_mask"])

    info["player"] = current_player

    if "observation" in state:
        state = state["observation"]

    return state


class ActionMaskInInfoWrapper(BaseWrapper):
    """Wrapper to convert dict observations to processed Box observations."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)

        if isinstance(orig_space, gymnasium.spaces.Dict):
            obs_space = orig_space["observation"]
            shape = obs_space.shape
            return gymnasium.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=shape,
                dtype=obs_space.dtype,
            )
        else:
            return orig_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            self.env.infos[agent] = getattr(self.env, "infos", {}).get(agent, {})
            agent_index = self.env.agents.index(agent)
            _ = action_mask_to_info(obs, self.env.infos[agent], agent_index)

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)
        info = self.env.infos[agent]
        agent_index = self.env.agents.index(agent)
        return action_mask_to_info(obs, info, agent_index)

    def step(self, action: ActionType):
        self.env.step(action)
        # Process observation and info for current agent
        agent = self.env.agent_selection
        agent_index = self.env.agents.index(agent)
        obs = self.env.observe(agent)
        info = self.env.infos[agent]
        _ = action_mask_to_info(obs, info, agent_index)

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def state(self) -> np.ndarray:
        return self.env.state()


class ChannelLastToFirstWrapper(BaseWrapper):
    """Wrapper to convert image observations from HWC to CHW."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)

        if isinstance(orig_space, gymnasium.spaces.Box) and len(orig_space.shape) == 3:
            # HWC -> CHW
            h, w, c = orig_space.shape
            return gymnasium.spaces.Box(
                low=orig_space.low.min(),
                high=orig_space.high.max(),
                shape=(c, h, w),
                dtype=orig_space.dtype,
            )
        elif (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            obs_space = orig_space["observation"]
            if len(obs_space.shape) == 3:
                h, w, c = obs_space.shape
                return gymnasium.spaces.Box(
                    low=obs_space.low.min(),
                    high=obs_space.high.max(),
                    shape=(c, h, w),
                    dtype=obs_space.dtype,
                )
        return orig_space

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)

        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        if isinstance(obs, np.ndarray) and obs.ndim == 3:  # image
            obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW

        return obs

    def step(self, action: ActionType) -> None:
        self.env.step(action)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def state(self) -> np.ndarray:
        return self.env.state()


class AppendAgentSelectionWrapper(BaseWrapper):
    """
    Appends a one-hot vector indicating the currently-selected agent to 1-D vector observations.

    - If observation is a Box with shape (n,), returns shape (n + num_possible_agents,).
    - If observation is a Dict containing "observation", it will take that array,
      append the one-hot vector, and return the appended array (not the full dict).

    NOTE: observation_space() tries to be defensive if the env hasn't populated
    possible_agents yet (it falls back to env.agents or an empty list).
    """

    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        orig_space = self.env.observation_space(agent)

        def _append_box(obs_space: gym.spaces.Box):
            # Only support 1-D vector Boxes here — otherwise return unchanged.
            if len(obs_space.shape) != 1:
                return obs_space

            orig_low = np.asarray(obs_space.low).reshape(-1)
            orig_high = np.asarray(obs_space.high).reshape(-1)

            num_agents = len(self.env.possible_agents)

            # If we can't determine number of agents, don't modify space.
            if num_agents == 0:
                return obs_space

            # For the one-hot appended dimensions, lows are 0 and highs are 1.
            appended_low = np.zeros((num_agents,), dtype=orig_low.dtype)
            appended_high = np.ones((num_agents,), dtype=orig_high.dtype)

            new_low = np.concatenate([orig_low, appended_low])
            new_high = np.concatenate([orig_high, appended_high])

            new_shape = (orig_space.shape[0] + num_agents,)
            return gym.spaces.Box(
                low=new_low, high=new_high, shape=new_shape, dtype=obs_space.dtype
            )

        if (
            isinstance(orig_space, gym.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            return _append_box(orig_space["observation"])
        elif isinstance(orig_space, gym.spaces.Box):
            return _append_box(orig_space)

        # unsupported types are returned unchanged
        return orig_space

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)

        # extract inner observation if dict-style
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        obs = np.asarray(obs)

        # only append for 1-D vectors
        if obs.ndim != 1:
            # fallback: return original observation unchanged
            return obs

        possible_agents = list(self.env.possible_agents)
        num_agents = len(possible_agents)

        # if we can't determine num_agents, fallback to original observation
        if num_agents == 0:
            return obs

        # compute index of the currently-selected agent (who's turn it is).
        try:
            sel_agent = self.env.agent_selection
            selected_index = possible_agents.index(sel_agent)
        except Exception:
            # fallback if for some reason it's unavailable — default to all zeros
            selected_index = None

        # build one-hot vector
        if np.issubdtype(obs.dtype, np.integer):
            oh_dtype = obs.dtype
        elif np.issubdtype(obs.dtype, np.floating):
            oh_dtype = obs.dtype
        else:
            oh_dtype = np.float32

        one_hot = np.zeros((num_agents,), dtype=oh_dtype)
        if selected_index is not None and 0 <= selected_index < num_agents:
            one_hot[selected_index] = 1

        return np.concatenate([obs, one_hot], axis=0)

    def reset(self, seed: int | None = None, options: dict | None = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActionType) -> None:
        return self.env.step(action)

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def state(self) -> np.ndarray:
        return self.env.state()


class TwoPlayerPlayerPlaneWrapper(BaseWrapper):
    """
    Wrapper to add a player plane to observations.

    Adds an extra channel where all entries are 0 if player 1, 1 if player 2 (or agent index > 0),
    compatible with both HWC (channel-last) and CHW (channel-first) observations.
    """

    def __init__(
        self, env: AECEnv[AgentID, ObsType, ActionType], channel_first: bool = True
    ):
        super().__init__(env)
        self.channel_first = channel_first

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)

        if isinstance(orig_space, gymnasium.spaces.Box):
            shape = orig_space.shape
            if len(shape) == 3:
                if self.channel_first:
                    c, h, w = shape if shape[0] != 3 else shape  # CHW
                    new_shape = (c + 1, h, w)
                else:
                    h, w, c = shape
                    new_shape = (h, w, c + 1)
                return gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=new_shape,
                    dtype=orig_space.dtype,
                )
        elif (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            obs_space = orig_space["observation"]
            shape = obs_space.shape
            if len(shape) == 3:
                if self.channel_first:
                    c, h, w = shape if shape[0] != 3 else shape
                    new_shape = (c + 1, h, w)
                else:
                    h, w, c = shape
                    new_shape = (h, w, c + 1)
                return gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=new_shape,
                    dtype=obs_space.dtype,
                )

        return orig_space

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)

        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        # Determine player plane value (0 for first agent, 1 for others)
        plane_val = 0 if agent == self.env.agents[0] else 1

        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            if self.channel_first:
                # CHW
                h, w = obs.shape[1], obs.shape[2]
                plane = np.full((1, h, w), plane_val, dtype=obs.dtype)
                obs = np.concatenate([obs, plane], axis=0)
            else:
                # HWC
                h, w = obs.shape[0], obs.shape[1]
                plane = np.full((h, w, 1), plane_val, dtype=obs.dtype)
                obs = np.concatenate([obs, plane], axis=2)

        return obs

    def step(self, action: ActionType) -> None:
        self.env.step(action)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def state(self) -> np.ndarray:
        return self.env.state()


class FrameStackWrapper(BaseWrapper):
    """
    Wrapper to stack the last k frames along the channel dimension.

    Args:
        env: The PettingZoo AECEnv to wrap.
        k: Number of frames to stack.
        channel_first: Whether the input is channel-first (CHW) or channel-last (HWC).
    """

    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        k: int = 4,
        channel_first: bool = True,
    ):
        super().__init__(env)
        self.k = k
        self.channel_first = channel_first
        self.stacks: dict[AgentID, deque] = {
            agent: deque(maxlen=k) for agent in self.env.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        # Clear stacks
        for agent in self.env.agents:
            self.stacks[agent].clear()
            obs = self.env.observe(agent)
            if isinstance(obs, dict) and "observation" in obs:
                obs = obs["observation"]
            self._init_stack(agent, obs)

    def _init_stack(self, agent: AgentID, obs: np.ndarray):
        """Fill the deque with the initial frame k times."""
        for _ in range(self.k):
            self.stacks[agent].append(obs)

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        self.stacks[agent].append(obs)
        frames = list(self.stacks[agent])[::-1]  # newest first, oldest last

        if self.channel_first:
            # CHW: concatenate along channel axis
            return np.concatenate(frames, axis=0)
        else:
            # HWC: concatenate along last axis
            return np.concatenate(frames, axis=-1)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)
        if (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            obs_space = orig_space["observation"]
        else:
            obs_space = orig_space

        shape = obs_space.shape
        if len(shape) == 3:
            if self.channel_first:
                c, h, w = shape
                new_shape = (c * self.k, h, w)
            else:
                h, w, c = shape
                new_shape = (h, w, c * self.k)

            return gymnasium.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=new_shape,
                dtype=obs_space.dtype,
            )
        elif len(shape) == 1:
            # Only stack 3D frames
            return gymnasium.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=(shape[0] * self.k,),
                dtype=obs_space.dtype,
            )
        else:
            raise NotImplementedError

    def step(self, action: ActionType) -> None:
        self.env.step(action)

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def state(self) -> np.ndarray:
        return self.env.state()


class InitialMovesWrapper(BaseWrapper):
    """
    Wrapper to stack the last k frames along the channel dimension.

    Args:
        env: The PettingZoo AECEnv to wrap.
        k: Number of frames to stack.
        channel_first: Whether the input is channel-first (CHW) or channel-last (HWC).
    """

    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        initial_moves: list,
    ):
        super().__init__(env)
        self.initial_moves = initial_moves

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        # Clear stacks
        for move in self.initial_moves:
            self.step(move)

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def state(self) -> np.ndarray:
        return self.env.state()


import gymnasium as gym
from gymnasium.core import Wrapper
from typing import Any, Tuple, Dict, SupportsFloat as float_t


class CatanatronWrapper(Wrapper):
    """
    A Gymnasium wrapper to rename a specific key within the 'info' dictionary
    returned by the environment's step method.

    This is useful for standardizing environment metadata, such as changing
    'valid_actions' (used in some older or custom environments) to
    'legal_moves' (a more common term in board/turn-based games).

    In this specific implementation, it changes "valid_actions" to "legal_moves".

    If the original key does not exist, the info dictionary is returned unmodified.
    """

    def __init__(self, env: gym.Env):
        """
        Initializes the wrapper.

        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.old_key = "valid_actions"
        self.new_key = "legal_moves"

    def step(self, action: Any) -> Tuple[Any, float_t, bool, bool, Dict[str, Any]]:
        """
        Performs a step in the environment and renames the key in the info dictionary.

        Args:
            action: The action to take.

        Returns:
            obs (Any): The new observation.
            reward (float_t): The reward from the step.
            terminated (bool): Whether the episode terminated.
            truncated (bool): Whether the episode truncated.
            info (Dict[str, Any]): The modified info dictionary.
        """
        # Call the wrapped environment's step method
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if the old key exists in the info dictionary
        if self.old_key in info:
            # 1. Copy the value to the new key
            info[self.new_key] = info[self.old_key]
            # 2. Delete the old key
            del info[self.old_key]

            # Note: For demonstration purposes below, we'll also inject a mock key
            # if the environment doesn't provide one.

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment and renames the key in the initial info dictionary.

        Args:
            **kwargs: Keyword arguments passed to the environment's reset method
                      (e.g., 'seed' or 'options').

        Returns:
            obs (Any): The initial observation.
            info (Dict[str, Any]): The modified info dictionary.
        """
        # Call the wrapped environment's reset method
        obs, info = self.env.reset(**kwargs)

        # Apply the same key renaming logic
        if self.old_key in info:
            info[self.new_key] = info[self.old_key]
            del info[self.old_key]

        return obs, info


import imageio
import numpy as np
import os
from typing import Optional, Dict, Any, Callable
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper


# For pickling and multiprocessing compatibility
class EpisodeTrigger:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, episode_id: int) -> bool:
        return episode_id % self.period == 0


class RecordVideo(BaseWrapper):
    """
    Records video of PettingZoo AEC environment episodes.

    Args:
        env: The PettingZoo AEC environment to wrap
        video_folder: Directory to save videos
        episode_trigger: Function that takes episode_id and returns True if recording should start
        video_length: Maximum number of frames per video (0 for unlimited)
        name_prefix: Prefix for video filenames
        fps: Frames per second for the video
        codec: Video codec (fourcc format)
    """

    def __init__(
        self,
        env: AECEnv,
        video_folder: str = "videos",
        episode_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 0,
        name_prefix: str = "episode",
        fps: int = 30,
    ):
        super().__init__(env)

        self.video_folder = video_folder
        self.episode_trigger = episode_trigger or EpisodeTrigger(1000)
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.fps = fps

        # Create video directory
        os.makedirs(self.video_folder, exist_ok=True)

        # Video recording state
        self.recording = False
        self.video_writer = None
        self.frames_recorded = 0
        self.episode_id = 0
        self.episode_started = False

        # Ensure environment has render capability
        if not hasattr(env, "render"):
            raise ValueError("Environment must support rendering to record video")

    def last(self, observe: bool = True):
        _, reward, term, trunc, info = self.env.last(observe=False)
        obs = self.observe(self.env.agent_selection) if observe else None
        return obs, reward, term, trunc, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """Reset environment and potentially start recording new episode"""
        # Call parent reset
        super().reset(seed=seed, options=options)

        # Check if we should record this episode
        should_record = self.episode_trigger(self.episode_id)

        if should_record and not self.recording:
            self._start_recording()
        elif not should_record and self.recording:
            self._stop_recording()

        # Capture initial frame if recording
        if self.recording:
            self._capture_frame()

        self.episode_started = True
        return None  # PettingZoo AEC reset doesn't return anything

    def step(self, action):
        """Step environment and capture frame if recording"""
        # Call parent step - this updates env state but returns nothing
        super().step(action)

        # Capture frame if recording
        if self.recording:
            self._capture_frame()

        # Check if episode ended
        episode_ended = (
            not self.agents  # No more agents
            or all(self.terminations.values())
            or all(self.truncations.values())
        )

        # Stop recording if episode ended or max frames reached
        if self.recording and (
            episode_ended
            or (self.video_length > 0 and self.frames_recorded >= self.video_length)
        ):
            self._stop_recording()

        # If episode ended, increment episode counter
        if episode_ended and self.episode_started:
            self.episode_id += 1
            self.episode_started = False

        return None  # PettingZoo AEC step doesn't return anything

    def _start_recording(self):
        """Initialize video recording"""
        if self.recording:
            self._stop_recording()

        # Generate filename
        video_name = f"{self.name_prefix}_{self.episode_id:06d}.mp4"
        video_path = os.path.join(self.video_folder, video_name)

        # Get a frame to determine video dimensions
        frame = self._get_frame()
        # Initialize imageio writer
        try:
            self.video_writer = imageio.get_writer(
                video_path, fps=self.fps, macro_block_size=1
            )
            self.recording = True
            self.frames_recorded = 0
            print(f"Started recording episode {self.episode_id} to {video_path}")
        except Exception as e:
            print(f"Warning: Could not open video writer for {video_path}: {e}")
            self.video_writer = None
            return

    def _stop_recording(self):
        """Stop video recording and save file"""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
            print(
                f"Stopped recording episode {self.episode_id}. Recorded {self.frames_recorded} frames."
            )

        self.recording = False
        self.frames_recorded = 0

    def _get_frame(self):
        """Get current frame from environment"""
        try:
            # Try to render as rgb_array
            frame = self.env.render()
            # imageio/ffmpeg expects RGB (PettingZoo/Pygame already provides this usually)
            # If the frame has an alpha channel, we'll keep it or strip it depending on needs,
            # but usually for MP4 we want RGB.
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                # Convert RGBA to RGB
                frame = frame[:, :, :3]

            return frame

        except Exception as e:
            print(f"Warning: Error getting frame: {e}")
            return None

    def _capture_frame(self):
        """Capture and write current frame to video"""
        if not self.recording or self.video_writer is None:
            return

        frame = self._get_frame()
        if frame is not None:
            self.video_writer.append_data(frame)
            self.frames_recorded += 1

    def close(self):
        """Clean up video recording"""
        if self.recording:
            self._stop_recording()
        super().close()

    def __del__(self):
        """Ensure video recording is stopped on deletion"""
        if hasattr(self, "recording") and self.recording:
            self._stop_recording()

    # Convenience function for common use cases


def record_video_wrapper(
    env: AECEnv,
    video_folder: str = "videos",
    record_every: int = 1000,
    max_frames: int = 0,
    fps: int = 30,
):
    """
    Convenience function to wrap environment with video recording.

    Args:
        env: PettingZoo environment
        video_folder: Where to save videos
        record_every: Record every N episodes (default: every 1000)
        max_frames: Maximum frames per video (0 for unlimited)
        fps: Video framerate
    """
    return RecordVideo(
        env=env,
        video_folder=video_folder,
        episode_trigger=EpisodeTrigger(record_every),
        video_length=max_frames,
        fps=fps,
    )


class GymRecordVideo(gym.Wrapper):
    """
    Records video of Gymnasium environment episodes.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str = "videos",
        episode_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 0,
        name_prefix: str = "episode",
        fps: int = 30,
    ):
        super().__init__(env)

        self.video_folder = video_folder
        self.episode_trigger = episode_trigger or EpisodeTrigger(1000)
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.fps = fps

        # Create video directory
        os.makedirs(self.video_folder, exist_ok=True)

        # Video recording state
        self.recording = False
        self.video_writer = None
        self.frames_recorded = 0
        self.episode_id = 0

        # Ensure environment has render capability
        if not hasattr(env, "render"):
            raise ValueError("Environment must support rendering to record video")

    def reset(self, **kwargs):
        """Reset environment and potentially start recording new episode"""
        result = super().reset(**kwargs)

        # Check if we should record this episode
        should_record = self.episode_trigger(self.episode_id)

        if should_record and not self.recording:
            self._start_recording()
        elif not should_record and self.recording:
            self._stop_recording()

        # Capture initial frame if recording
        if self.recording:
            self._capture_frame()

        return result

    def step(self, action):
        """Step environment and capture frame if recording"""
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated

        # Capture frame if recording
        if self.recording:
            self._capture_frame()

        # Stop recording if episode ended or max frames reached
        if self.recording and (
            done
            or (self.video_length > 0 and self.frames_recorded >= self.video_length)
        ):
            self._stop_recording()

        # If episode ended, increment episode counter
        if done:
            self.episode_id += 1

        return obs, reward, terminated, truncated, info

    def _start_recording(self):
        """Initialize video recording"""
        if self.recording:
            self._stop_recording()

        video_name = f"{self.name_prefix}_{self.episode_id:06d}.mp4"
        video_path = os.path.join(self.video_folder, video_name)

        frame = self._get_frame()
        try:
            self.video_writer = imageio.get_writer(
                video_path, fps=self.fps, macro_block_size=1
            )
            self.recording = True
            self.frames_recorded = 0
            print(f"Started recording episode {self.episode_id} to {video_path}")
        except Exception as e:
            print(f"Warning: Could not open video writer for {video_path}: {e}")
            self.video_writer = None

    def _stop_recording(self):
        """Stop video recording and save file"""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
            print(
                f"Stopped recording episode {self.episode_id}. Recorded {self.frames_recorded} frames."
            )

        self.recording = False
        self.frames_recorded = 0

    def _get_frame(self):
        """Get current frame from environment"""
        try:
            frame = self.env.render()
            if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame
        except Exception as e:
            print(f"Warning: Error getting frame: {e}")
            return None

    def _capture_frame(self):
        """Capture and write current frame to video"""
        if not self.recording or self.video_writer is None:
            return

        frame = self._get_frame()
        if frame is not None:
            self.video_writer.append_data(frame)
            self.frames_recorded += 1

    def close(self):
        """Clean up video recording"""
        if self.recording:
            self._stop_recording()
        super().close()


def wrap_recording(env, **kwargs):
    """Factory function to wrap either Gym or PettingZoo env for recording."""
    is_pz = hasattr(env, "possible_agents") and hasattr(env, "agent_selection")
    if not is_pz and hasattr(env, "unwrapped"):
        unwrapped = env.unwrapped
        is_pz = hasattr(unwrapped, "possible_agents") and hasattr(
            unwrapped, "agent_selection"
        )

    if is_pz:
        return RecordVideo(env, **kwargs)
    else:
        return GymRecordVideo(env, **kwargs)


import pufferlib.emulation


def make_puffer_env_creator(config):
    """
    Creates an environment creator function for PufferLib.

    Args:
        config: Configuration object containing env_type, env_name, and env_kwargs.

    Returns:
        A callable that returns a PufferLib-wrapped environment.
    """

    def env_creator():
        if config.env_type == "pettingzoo":
            # For Catan, TicTacToe, etc.
            if config.env_name == "catan":
                from custom_gym_envs_pkg.custom_gym_envs.envs import catan

                raw_env = catan.env(**config.env_kwargs)
            # TODO: getting of raw envs for standard pettingzoo envs
            # PufferLib automatically groups agents into flat batches!
            return pufferlib.emulation.PettingZooPufferEnv(env=raw_env)

        elif config.env_type == "gym":
            import gymnasium as gym

            raw_env = gym.make(config.env_name, **config.env_kwargs)
            return pufferlib.emulation.GymnasiumPufferEnv(env=raw_env)

        elif config.env_type == "ocean":
            import pufferlib.ocean

            return pufferlib.ocean.make(config.env_name)
        else:
            raise ValueError(f"Unsupported environment type: {config.env_type}")

    return env_creator
