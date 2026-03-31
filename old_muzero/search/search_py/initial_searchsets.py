from abc import ABC, abstractmethod
from typing import List

import torch


class SearchSet(ABC):
    @abstractmethod
    def create_initial_searchset(
        self, priors, legal_moves, count: int, trajectory_action=None
    ) -> List[int]:
        pass  # pragma: no cover


class SelectAll(SearchSet):
    def create_initial_searchset(
        self, priors, legal_moves, count: int, trajectory_action=None
    ) -> torch.Tensor:
        # Convert list of ints to tensor
        return torch.tensor(legal_moves, dtype=torch.long, device=priors.device)


class SelectTopK(SearchSet):
    def create_initial_searchset(
        self, priors, legal_moves, count: int, trajectory_action=None
    ) -> torch.Tensor:
        """Selects top-k actions from legal actions only."""
        assert legal_moves

        # Determine K
        k = min(count, len(legal_moves))
        legal_actions = torch.as_tensor(
            legal_moves, dtype=torch.long, device=priors.device
        )
        legal_priors = priors[legal_actions]
        topk_local = torch.argsort(legal_priors, descending=True)[:k]
        selected_actions = legal_actions[topk_local]

        # Force include trajectory_action if provided and not selected
        if trajectory_action is not None and trajectory_action not in selected_actions:
            assert (
                trajectory_action in legal_moves
            ), f"trajectory_action {trajectory_action} not in legal actions {legal_moves}"

            # Add it to the list
            # selected_actions.append(trajectory_action)
            selected_actions[-1] = trajectory_action

        # print("selected actions", selected_actions)
        return selected_actions
