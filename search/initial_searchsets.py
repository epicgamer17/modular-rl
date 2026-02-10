from abc import ABC, abstractmethod
from typing import List

import torch


class SearchSet(ABC):
    @abstractmethod
    def create_initial_searchset(
        self, priors, legal_moves, count: int, trajectory_action=None
    ) -> List[int]:
        pass


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
        """using the priors should work the same as using logits, as priors for gumbel would be g + logits softmaxed, so top k are the same as top k of g + logits"""
        assert legal_moves

        # TODO: SOMEHOW MAKE SURE THESE ONLY PICK LEGAL ACTIONS. RIGHT NOW I DONT.

        # Determine K
        k = min(count, len(legal_moves))
        selected_actions = torch.argsort(priors, descending=True)[:k]
        # selected_actions = [legal_moves[i] for i in top_indices]

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
