from typing import Dict, List, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod
from collections import deque
from data.utils import discounted_cumulative_sums
from utils.utils import legal_moves_mask
from logging import warning

from data.processors.input_processors import InputProcessor


class SequenceTensorProcessor(InputProcessor):
    """
    Converts a complete Sequence object into tensor format for storage.
    Handles observation stacking, action/reward tensors, and legal move masks.
    """

    def __init__(
        self,
        num_actions: int,
        num_players: int,
        player_id_mapping: Dict[str, int],
        device="cpu",
    ):
        self.num_actions = num_actions
        self.num_players = num_players
        self.player_id_mapping = player_id_mapping
        self.device = device

    def _resolve_player_id(self, player_id) -> int:
        if isinstance(player_id, int):
            return player_id
        if player_id not in self.player_id_mapping:
            raise KeyError(player_id)
        return self.player_id_mapping[player_id]

    def process_single(self, **kwargs):
        raise NotImplementedError(
            "SequenceTensorProcessor only supports process_sequence."
        )  # pragma: no cover

    def process_sequence(self, sequence, **kwargs):
        # 1. Prepare Observations
        obs_history = sequence.observation_history
        n_states = len(obs_history)
        n_transitions = len(sequence.action_history)
        if n_transitions + 1 != n_states:
            raise ValueError(
                f"observation_history must have exactly one more entry than action_history: {n_states} vs {n_transitions}"
            )

        try:
            obs_array = np.stack(obs_history)
            if obs_array.dtype == object:
                raise TypeError(
                    f"Observations stacked into object array. Length: {len(obs_history)}"
                )
            obs_tensor = torch.from_numpy(obs_array)
        except Exception as e:
            raise RuntimeError(f"Failed to process observations: {e}")

        # 2. Prepare transition-aligned tensors
        try:
            acts_t = torch.tensor(sequence.action_history, dtype=torch.float16)
        except Exception as e:
            raise RuntimeError(f"Failed to process actions: {e}")

        try:
            rews_t = torch.tensor(sequence.rewards, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to process rewards: {e}")

        if sequence.policy_history:
            try:
                pols_t = (
                    torch.stack(
                        [
                            p.detach() if torch.is_tensor(p) else torch.as_tensor(p)
                            for p in sequence.policy_history
                        ]
                    )
                    .cpu()
                    .float()
                )
            except Exception as e:
                raise RuntimeError(f"Failed to process policies: {e}")
        else:
            pols_t = torch.empty((0, self.num_actions), dtype=torch.float32)

        vals_t = torch.tensor(sequence.value_history, dtype=torch.float32)

        # To Plays
        assert len(sequence.player_id_history) == n_states, (
            f"player_id_history length ({len(sequence.player_id_history)}) "
            f"must match n_states ({n_states})"
        )
        t_plays = [
            self._resolve_player_id(pid) for pid in sequence.player_id_history
        ]
        tps_t = torch.tensor(t_plays, dtype=torch.int16)

        # Chances
        assert len(sequence.chance_history) == n_states or not sequence.chance_history, (
            f"chance_history length ({len(sequence.chance_history)}) "
            f"must match n_states ({n_states}) or be empty"
        )
        if sequence.chance_history:
            chance_t = torch.tensor(sequence.chance_history, dtype=torch.int16).unsqueeze(1)
        else:
            chance_t = torch.zeros((n_states, 1), dtype=torch.int16)

        # Legal Moves Mask
        try:
            legal_masks_t = torch.stack(
                [
                    legal_moves_mask(
                        self.num_actions,
                        (
                            sequence.legal_moves_history[i]
                            if i < len(sequence.legal_moves_history)
                            else []
                        ),
                    )
                    for i in range(n_states)
                ]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process legal masks: {e}")

        terminated_t = torch.tensor(sequence.terminated_history, dtype=torch.bool)
        truncated_t = torch.tensor(sequence.truncated_history, dtype=torch.bool)
        dones_t = torch.tensor(sequence.done_history, dtype=torch.bool)

        return {
            "observations": obs_tensor,
            "actions": acts_t,
            "rewards": rews_t,
            "policies": pols_t,
            "values": vals_t,
            "to_plays": tps_t,
            "chances": chance_t,
            "terminated": terminated_t,
            "truncated": truncated_t,
            "done": dones_t,
            "legal_masks": legal_masks_t,
            "n_states": n_states,
        }
